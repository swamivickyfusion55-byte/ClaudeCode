"""
Phoenix Mobile API adapter for Swamitech Phoenix (v11.2.80 Resume).

Stable Gradio API surface:
  phoenix_submit_video
  phoenix_job_status
  phoenix_download
  phoenix_cancel

Plus plain-HTTP routes (install_http_routes) that survive a phone being
minimised, because none of them depends on one connection staying open:
  POST   /phoenix/upload/init          start, or resume by token, an upload
  GET    /phoenix/upload/{id}          how many bytes the server already has
  PUT    /phoenix/upload/{id}          append a chunk at Upload-Offset
  DELETE /phoenix/upload/{id}          abandon an upload
  GET    /phoenix/status/{job|token}   job status, no Gradio session involved
  GET    /phoenix/download/{job}       the result, with Range / If-Range

The adapter reuses the exact core_pipeline.submit_video() job engine.
No HF token is stored here.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import core_pipeline as cp
from core_pipeline import _lock, jobs, submit_video

try:
    # Module level on purpose: FastAPI resolves a handler's string annotations
    # (this module uses `from __future__ import annotations`) from the module's
    # globals, and a Request imported inside a function would not be found.
    from starlette.requests import Request
except Exception:      # pragma: no cover - the routes then simply are not installed
    Request = None

try:
    from config import VERSION_FULL, ensure_runtime_dirs
except Exception:
    VERSION_FULL = "v11.2.51 (DetZoom)"
    def ensure_runtime_dirs():
        os.makedirs("/tmp/gradio", exist_ok=True)


def _cfg(name, default):
    """A tunable from config.py, with a default so an older config.py still
    imports (a partly-updated Space must degrade, not crash on import)."""
    try:
        import config as _c
        return getattr(_c, name, default)
    except Exception:
        return default


_JOB_RE = re.compile(r"\bJob\s+([A-Z0-9]{8})\b")


def _json_error(message: str, code: str = "invalid_request") -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}


def _settings_dict(settings: Any) -> dict[str, Any]:
    if settings is None or settings == "":
        return {}
    if isinstance(settings, str):
        try:
            settings = json.loads(settings)
        except Exception as exc:
            raise ValueError(f"settings must be valid JSON: {exc}") from exc
    if not isinstance(settings, dict):
        raise ValueError("settings must be a JSON object")
    return settings


def _job_config(settings: Any) -> dict[str, Any]:
    s = _settings_dict(settings)
    allowed = {
        "secs", "fps", "res", "quality", "enhancer", "swap_n", "det_n",
        "det_int", "password", "trim_start", "trim_end", "face_mode",
        "enhance_scope", "device_mode", "smooth_motion", "target_refs",
        "face_enhance",
    }
    return {k: s[k] for k in allowed if k in s}


def _load_target_refs(raw) -> list:
    """Embeddings from Detect-frame crops the user assigned to slots."""
    paths = []
    if isinstance(raw, (list, tuple)):
        paths = list(raw)
    elif isinstance(raw, dict):
        paths = [raw.get(str(i)) or raw.get(i) for i in range(4)]
    out = []
    if not any(paths):
        return out
    try:
        cp._load()
    except Exception:
        logging.exception("load models for target refs")
        return []
    for p in list(paths)[:4]:
        emb = None
        try:
            if isinstance(p, dict):
                p = p.get("path") or p.get("name")
            if p and os.path.isfile(str(p)):
                im = cp._as_bgr_image(str(p))
                f = cp._cached_source_face(im) if im is not None else None
                if f is None and im is not None:
                    fs = cp._fa_get(im) or []
                    f = fs[0] if fs else None
                if f is not None:
                    emb = getattr(f, "normed_embedding", None)
        except Exception:
            logging.exception("target ref %s", p)
        out.append(emb)
    while len(out) < 4:
        out.append(None)
    return out


def _coerce_video(video):
    """Turn a Gradio File payload into a path that still exists.

    Gradio 4 File.preprocess copies into GRADIO_TEMP_DIR. If that folder was
    missing, the copy never happens and we never reach this function — the
    caller still mkdir's first. If we receive bytes / a vanished path, write
    a durable copy under /data (or /tmp).
    """
    try:
        ensure_runtime_dirs()
    except Exception:
        os.makedirs("/tmp/gradio", exist_ok=True)
    if video is None:
        return None
    if isinstance(video, dict):
        video = video.get("path") or video.get("name") or video.get("orig_name")
    if isinstance(video, (list, tuple)) and video:
        video = video[0]
    if hasattr(video, "name") and not isinstance(video, (str, bytes, bytearray)):
        video = getattr(video, "name", None)
    if isinstance(video, (bytes, bytearray)):
        root = Path("/data/phoenix_uploads") if os.path.isdir("/data") else Path("/tmp/phoenix_uploads")
        root.mkdir(parents=True, exist_ok=True)
        dest = root / f"up_{uuid.uuid4().hex[:10]}.mp4"
        dest.write_bytes(video)
        return str(dest)
    if isinstance(video, str) and video and not os.path.isfile(video):
        return None
    return video


# ---------------------------------------------------------------------------
# Idempotent submit (v11.2.80)
#
# Measured on the pinned Gradio 4.44.1 (probe in SDOS-079): a queued event whose
# client disconnects while it is still WAITING is discarded - it never runs. An
# event that is already RUNNING finishes, but its result is thrown away. For
# phoenix_submit_video that means a phone that is minimised mid-submit either
# never starts the job, or starts it and never learns the job id (and then
# submits again, so the same clip is processed twice on a 1-worker CPU Space).
#
# A client_token (12-64 chars, generated by the client ONCE per user action and
# stored before the call) names the request. A retry with the same token gets
# the job the first attempt started, and phoenix_job_status accepts the token in
# place of the job id, so a client that lost the response can ask "did my
# submit land?" and resubmit only if the answer is not_found.
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{12,64}$")
_TOK_LOCK = threading.Lock()
_TOK_INFLIGHT: "dict[str, threading.Event]" = {}


def _job_for_token(token: str):
    """Job id previously started under this client_token, else None."""
    if not token:
        return None
    with _lock:
        for jid, j in list(jobs.items()):
            if isinstance(j, dict) and j.get("client_token") == token:
                return jid
    return None


def _token_enter(token: str):
    """(True, event) for the one caller that should start the job; (False,
    event) for callers that arrived while it is starting and must wait."""
    with _TOK_LOCK:
        ev = _TOK_INFLIGHT.get(token)
        if ev is not None:
            return False, ev
        ev = threading.Event()
        _TOK_INFLIGHT[token] = ev
        return True, ev


def _token_leave(token: str, ev: "threading.Event") -> None:
    with _TOK_LOCK:
        if _TOK_INFLIGHT.get(token) is ev:
            del _TOK_INFLIGHT[token]
    ev.set()


def _job_record(jid: str, deduplicated: bool = False) -> dict[str, Any]:
    with _lock:
        job = dict(jobs.get(jid) or {})
    return {
        "ok": True,
        "job_id": jid,
        "status": job.get("status", "processing"),
        "progress": int(job.get("progress") or 0),
        "message": job.get("message") or "",
        "eta_seconds": job.get("eta_seconds"),
        "device": job.get("api_device") or "",
        "version": VERSION_FULL,
        "engine": "aequus-1.2.3-solid-face",
        "client_token": job.get("client_token") or None,
        "deduplicated": bool(deduplicated),
    }


def _run_once_per_token(token: str, start, rerun_after_wait: bool = False):
    """start() -> dict. Runs at most once per token; every other caller with
    the same token gets the job that run produced. A split call has no single
    job to look up (its parts carry their own tokens), so a caller that waited
    behind it runs start() again: every part then answers from its own token."""
    if not token:
        return start()
    if not rerun_after_wait:
        jid = _job_for_token(token)
        if jid:
            return _job_record(jid, deduplicated=True)
    leader, ev = _token_enter(token)
    if not leader:
        ev.wait(timeout=180)
        if rerun_after_wait:
            return start()
        jid = _job_for_token(token)
        if jid:
            return _job_record(jid, deduplicated=True)
        return _json_error("The first attempt with this client_token did not start a job; "
                           "submit again", "retry")
    try:
        if not rerun_after_wait:
            jid = _job_for_token(token)      # a previous leader may have just finished
            if jid:
                return _job_record(jid, deduplicated=True)
        return start()
    finally:
        _token_leave(token, ev)


def _read_token(raw: dict):
    """(token, error). Absent token is fine; a malformed one is an error, never
    silently ignored - ignoring it would quietly turn retries back into
    duplicate jobs."""
    tok = raw.get("client_token")
    if tok is None or tok == "":
        return "", None
    tok = str(tok)
    if not _TOKEN_RE.fullmatch(tok):
        return "", _json_error("client_token must be 12-64 characters of A-Z a-z 0-9 _ -",
                               "invalid_client_token")
    return tok, None


# ---------------------------------------------------------------------------
# Server-side split (v11.2.80)
#
# The Android client used to cut the clip into parts ON THE PHONE
# (phoenix_trim_*.mp4) and upload each part. That work stops the moment the app
# is minimised. The pipeline already cuts by trim_start/trim_end percent with
# ffmpeg, so the phone can upload the clip once and ask for the split here:
# settings.split_seconds = N cuts the (trimmed) range into equal parts of at
# most N seconds, one job per part, queued back to back on the one worker.
# ---------------------------------------------------------------------------
def _video_seconds(path: str):
    """Duration of a video in seconds, or None."""
    try:
        cap = cv2.VideoCapture(path)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        f = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if n and f and n > 0 and f > 0:
            return float(n) / float(f)
    except Exception:
        logging.exception("video duration")
    return None


def _plan_split(video: str, raw: dict, defaults: dict):
    """([(part_index, trim_start, trim_end, seconds)], error)"""
    try:
        split_s = float(raw.get("split_seconds"))
    except Exception:
        return None, _json_error("split_seconds must be a number", "invalid_split")
    if not (5.0 <= split_s <= 360.0):
        return None, _json_error("split_seconds must be between 5 and 360", "invalid_split")
    dur = _video_seconds(video)
    if not dur:
        return None, _json_error("Could not read the video's duration to split it", "invalid_split")
    ts = min(100.0, max(0.0, float(defaults.get("trim_start") or 0)))
    te = min(100.0, max(0.0, float(defaults.get("trim_end") if defaults.get("trim_end") is not None else 100)))
    if te <= ts:
        return None, _json_error("Invalid trim range", "invalid_split")
    span = dur * (te - ts) / 100.0
    n = max(1, int(math.ceil(span / split_s - 1e-9)))
    cap = int(_cfg("SPLIT_MAX_PARTS", 10))
    if n > cap:
        return None, _json_error(
            f"This would make {n} parts (max {cap}); raise split_seconds to "
            f"{int(math.ceil(span / cap))} or more", "too_many_parts")
    parts = []
    for i in range(n):
        a = ts + (te - ts) * i / n
        b = te if i == n - 1 else ts + (te - ts) * (i + 1) / n   # contiguous: end(i) == start(i+1)
        parts.append((i + 1, a, b, span / n))
    return parts, None


def _defaults_from(raw: dict):
    defaults = {
        "secs": 30,
        "fps": 30,
        "res": "720p (HD)",
        "quality": "Ultra",
        "enhancer": "Cinematic (clarity + smooth)",
        "swap_n": "1",
        "det_n": "1",
        "det_int": 1,
        "password": "",
        "trim_start": 0,
        "trim_end": 100,
        "face_mode": "2 faces",
        "enhance_scope": "All faces",
        "device_mode": "CPU only",
        "smooth_motion": "Fast (blend)",
        "face_enhance": False,
    }
    defaults.update(_job_config(raw))
    return defaults


def _count_face(f):
    if f is None:
        return False
    if isinstance(f, dict):
        return bool(f.get("path") or f.get("name"))
    if isinstance(f, str):
        return bool(f.strip())
    if isinstance(f, (list, tuple)):
        return len(f) > 0 and f[0] is not None
    try:
        import numpy as np
        if isinstance(f, np.ndarray):
            return f.size > 0
    except Exception:
        pass
    return True


def api_submit_video(video, face1, face2, face3, face4, settings=None):
    """Start a Phoenix v11 video job and return a JSON job record.

    settings.client_token  - makes the call idempotent (see above)
    settings.split_seconds - split into server-side parts (see above); the reply
                             then also carries "jobs": one record per part
    """
    video = _coerce_video(video)
    if video is None:
        return _json_error("Upload a target video", "missing_video")
    if all(face is None for face in (face1, face2, face3, face4)):
        return _json_error("Upload at least one replacement face", "missing_face")

    try:
        raw = _settings_dict(settings)
        defaults = _defaults_from(raw)
    except ValueError as exc:
        return _json_error(str(exc), "invalid_settings")
    token, terr = _read_token(raw)
    if terr:
        return terr

    faces = (face1, face2, face3, face4)
    n_faces = sum(1 for f in faces if _count_face(f))
    if n_faces >= 2:
        defaults["face_mode"] = "2 faces" if n_faces == 2 else "Multiple faces"
    fe = defaults.get("face_enhance", False)
    if isinstance(fe, str):
        fe = fe.strip().lower() in ("1", "true", "on", "yes", "gfpgan")
    else:
        fe = bool(fe)
    defaults["face_enhance"] = fe
    if fe:
        defaults["enhancer"] = "GFPGAN"
    logging.info(
        "phoenix_submit_video replacements=%d face_mode=%s smooth_motion=%s face_enhance=%s token=%s",
        n_faces, defaults.get("face_mode"), defaults.get("smooth_motion"), bool(fe),
        (token[:6] + "…") if token else "-",
    )

    if raw.get("split_seconds") not in (None, "", 0):
        return _run_once_per_token(token, lambda: _submit_split(video, faces, defaults, raw, token),
                                   rerun_after_wait=True)
    return _run_once_per_token(
        token, lambda: _start_job(video, faces, defaults, raw, token, None, None, None))


def _submit_split(video, faces, defaults, raw, token):
    parts, err = _plan_split(video, raw, defaults)
    if err:
        return err
    if token and len(token) > 56:
        return _json_error("client_token may be at most 56 characters when split_seconds is used",
                           "invalid_client_token")
    refs = _load_target_refs(defaults.get("target_refs"))
    logging.info("target refs from Detect frames: %s", [r is not None for r in (refs or [])])
    n = len(parts)
    records = []
    for idx, a, b, part_sec in parts:
        d = dict(defaults)
        d["trim_start"], d["trim_end"] = a, b
        # The job's own length cap would otherwise cut a part short.
        d["secs"] = max(int(d.get("secs") or 0), int(math.ceil(part_sec)) + 1)
        ptok = f"{token}-p{idx}" if token else ""
        rec = _run_once_per_token(
            ptok, lambda d=d, ptok=ptok, idx=idx: _start_job(
                video, faces, d, raw, ptok, idx, n, refs))
        if not rec.get("ok"):
            return {"ok": False, "error": rec.get("error") or {"code": "job_not_started",
                    "message": "A part did not start"}, "part_count": n,
                    "jobs": records,
                    "hint": "Call again with the same client_token: parts already started are kept."}
        rec = dict(rec)
        rec.update(part_index=idx, part_count=n, trim_start=round(a, 4), trim_end=round(b, 4))
        records.append(rec)
    first = dict(records[0])
    first.update(split=True, part_count=n, jobs=records)
    logging.info("split: %d parts of ~%.1fs → jobs %s", n, parts[0][3], [r["job_id"] for r in records])
    return first


def _start_job(video, faces, defaults, raw, token, part_index, part_count, refs):
    """Start ONE job and return its record."""
    face1, face2, face3, face4 = faces
    fe = bool(defaults.get("face_enhance"))
    if refs is None:
        refs = _load_target_refs(defaults.get("target_refs"))
        logging.info("target refs from Detect frames: %s", [r is not None for r in (refs or [])])

    # For multi-face mode, the v11 pipeline itself forces detection/swap cadence
    # to 1 frame, so the adapter does not override that safety logic.
    # Deliberately NOT deriving refs here (see swap_engine.py's own comments:
    # "nothing currently populates a genuine per-slot reference" - a known,
    # accepted state, not a bug). The tracker already has a purpose-built
    # left-to-right positional bootstrap for exactly this case, in
    # MultiFaceTracker.assign(): a never-established slot only falls back to
    # position when EVERY candidate face's similarity stays below
    # trk_gate_new (0.30) for it, so every entry in that slot's row of the
    # cost matrix stays at the 1e9 sentinel and optimal_assignment() rejects
    # every pairing that touches it.
    #
    # A prior version of this file "fixed" the wrong-slot problem by calling
    # detect_video() here and forwarding ITS refs. That backfires: those refs
    # come from an independently, arbitrarily chosen frame with no relation
    # to the slot the user actually assigned, so they give _identity() a
    # non-degenerate (not not -1.0) similarity to compare - and ArcFace
    # cross-identity similarity commonly lands close to 0.30 by chance. When
    # that spurious value clears the gate for the WRONG face, a "complete"
    # pairing is found immediately, the correct positional fallback never
    # gets the chance to run, and the wrong assignment locks in and persists
    # (hysteresis makes an established slot hard to correct afterwards).
    # Verified this exact mechanism by running the project's own
    # optimal_assignment() against both cases.
    #
    # refs=[] is what lets _identity() return a clean, always-losing -1.0 for
    # any never-established slot, which is what actually lets the correct
    # fallback engage.
    result = submit_video(
        face1, face2, face3, face4,
        video,
        defaults["secs"], defaults["fps"], defaults["res"], defaults["quality"],
        defaults["enhancer"], defaults["swap_n"], defaults["det_n"], defaults["det_int"],
        defaults["password"], refs, defaults["trim_start"], defaults["trim_end"],
        defaults["face_mode"], defaults["enhance_scope"], defaults["device_mode"],
        bool(fe),
        None,
        smooth_motion=defaults.get("smooth_motion") or "Fast (blend)",
    )

    message = str(result[0]) if result else ""
    match = _JOB_RE.search(message)
    if not match:
        return _json_error(message or "Unable to start Phoenix job", "job_not_started")

    jid = match.group(1)
    if part_index is None:
        part_n, part_count = 0, 1
        try:
            part_count = int(raw.get("part_count") or 1)
            part_n = int(raw.get("part_index") or 0)
        except Exception:
            part_n, part_count = 0, 1
    else:
        part_n = int(part_index)
    # The file name is this Space's own job id. Part 2 of a split is
    # Phoenix_<that job id>-2.mp4, not a separate group token.
    if part_count > 1 and 1 <= part_n <= 10:
        wanted = f"Phoenix_{jid}-{part_n}.mp4"
    else:
        wanted = f"Phoenix_{jid}.mp4"
    with _lock:
        job = jobs.get(jid)
        if isinstance(job, dict):
            job["save_name"] = wanted
            job["api_device"] = str(defaults["device_mode"])
            if token:
                job["client_token"] = token
            logging.info("download name %s → %s", jid, wanted)
    try:
        cp._persist_job(jid, force=True)     # the token must survive a process restart
    except Exception:
        pass
    rec = _job_record(jid)
    rec["message"] = rec["message"] or message
    return rec



def _jpeg_rgb(arr, tag):
    if arr is None:
        return None
    try:
        a = np.ascontiguousarray(np.asarray(arr), dtype=np.uint8)
        if a.ndim == 2:
            a = np.stack([a, a, a], axis=-1)
        if a.ndim != 3 or a.shape[2] < 3:
            return None
        root = Path(os.environ.get("GRADIO_TEMP_DIR") or "/tmp/gradio")
        root.mkdir(parents=True, exist_ok=True)
        dest = root / f"{tag}_{uuid.uuid4().hex[:10]}.jpg"
        bgr = cv2.cvtColor(a[:, :, :3], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(dest), bgr)
        return str(dest)
    except Exception:
        logging.exception("jpeg write %s", tag)
        return None


def api_detect_video_frame(video, position_pct=0):
    """Detect up to four faces at a selected video position for mobile frame picking."""
    video = _coerce_video(video)
    if video is None:
        return None, "Upload a target video first", None, None, None, None
    try:
        import cv2
        cp._load()
        frm, idx, total = cp._grab_frame(video, float(position_pct or 0))
        if frm is None:
            return None, f"Could not read frame at {position_pct}%", None, None, None, None
        faces = cp._sort_faces_left(cp._fa_get(frm))[:4]
        annotated = cp._annotate(frm, faces)
        crops = [cp._crop_face(frm, f) for f in faces]
        while len(crops) < 4:
            crops.append(None)
        msg = f"Frame {idx}/{total} · found {len(faces)} face(s)"
        return (
            _jpeg_rgb(annotated, "frame"),
            msg,
            _jpeg_rgb(crops[0], "f1"),
            _jpeg_rgb(crops[1], "f2"),
            _jpeg_rgb(crops[2], "f3"),
            _jpeg_rgb(crops[3], "f4"),
        )
    except Exception as exc:
        logging.exception("api_detect_video_frame")
        return None, f"Detection failed: {type(exc).__name__}: {str(exc)[:180]}", None, None, None, None


def _resolve_job_ref(ref: str):
    """A job id (8 chars) or a client_token (12-64 chars) -> job id or None."""
    ref = (ref or "").strip()
    if _TOKEN_RE.fullmatch(ref):
        return _job_for_token(ref)
    jid = ref.upper()
    return jid if re.fullmatch(r"[A-Z0-9]{8}", jid) else None


def api_job_status(job_id: str):
    """Return status only; never expose internal filesystem paths.

    job_id may also be the client_token the job was submitted under, so a
    client that lost the submit response can still find its job."""
    ref = (job_id or "").strip()
    if not (_TOKEN_RE.fullmatch(ref) or re.fullmatch(r"[A-Za-z0-9]{8}", ref)):
        return _json_error("Invalid job_id", "invalid_job_id")
    jid = _resolve_job_ref(ref)
    if not jid:
        return _json_error("Job not found", "not_found")

    with _lock:
        job = dict(jobs.get(jid) or {})
    if not job:
        return _json_error("Job not found", "not_found")

    status = job.get("status", "error")
    rp = job.get("result_path")
    ready = bool(status == "done" and rp and os.path.isfile(rp))
    return {
        "ok": True,
        "job_id": jid,
        "status": status,
        "progress": int(job.get("progress") or 0),
        "message": job.get("message") or "",
        "eta_seconds": job.get("eta_seconds"),
        "created_at": job.get("created_at"),
        "done_at": job.get("done_at"),
        "download_ready": ready,
        "server_saved": bool(status == "done" and job.get("server_result_path") and os.path.isfile(job.get("server_result_path"))),
        "expires_at": job.get("expires_at"),
        "client_token": job.get("client_token") or None,
        "version": VERSION_FULL,
    }


def api_download(job_id: str):
    """Return the completed result file for a valid job ID (or client_token)."""
    jid = _resolve_job_ref(job_id)
    if not jid:
        return None
    with _lock:
        job = dict(jobs.get(jid) or {})
    rp = job.get("result_path")
    if job.get("status") != "done" or not rp or not os.path.isfile(rp):
        return None
    return rp


def api_cancel(job_id: str):
    """Request cancellation of one Phoenix job (id or client_token)."""
    jid = _resolve_job_ref(job_id)
    if not jid:
        return _json_error("Invalid job_id", "invalid_job_id")
    with _lock:
        job = jobs.get(jid)
        if not job:
            return _json_error("Job not found", "not_found")
        if job.get("status") in ("done", "error", "cancelled"):
            return api_job_status(jid)
        job["cancel"] = True
        job["message"] = "Cancel requested…"
    return api_job_status(jid)


# ---------------------------------------------------------------------------
# Resumable transport (v11.2.80)
#
# Gradio's own /upload is ONE multipart POST. When a phone is minimised the OS
# freezes or kills the app's sockets; that POST dies at an arbitrary byte, the
# client treats it as failed, and every byte already sent is wasted. (The
# ClientDisconnect tracebacks in the Space log are exactly these.)
#
# The routes below make each step restartable from where it stopped:
#   * upload   - append-only chunks at an explicit offset (tus-style). The
#                server's file size IS the offset, so after any failure the
#                client asks, then continues. A new PUT takes the upload over
#                from a dead one whose socket the server has not noticed yet.
#   * status   - plain GET, no Gradio session, accepts a job id or client_token.
#   * download - Range / If-Range, so a cut download resumes instead of
#                restarting.
# A finished upload lands in Gradio's own upload folder and is returned as a
# FileData-compatible dict, so phoenix_submit_video takes it unchanged.
# ---------------------------------------------------------------------------
_UP_ROOT = Path(tempfile.gettempdir()) / "phoenix_resumable"
_UP_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,40}$")
_UP_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".3gp", ".avi",
           ".jpg", ".jpeg", ".png", ".webp"}
_UP_LOCK = threading.Lock()
_UP: "dict[str, _Upload]" = {}
_HTTP_STATE = {"janitor": False}


def _gradio_upload_folder() -> str:
    # Same rule as gradio.utils.get_upload_folder(); a file outside this folder
    # is rejected by Gradio's preprocess.
    return os.environ.get("GRADIO_TEMP_DIR") or str((Path(tempfile.gettempdir()) / "gradio").resolve())


class _Upload:
    def __init__(self, uid, name, size, token="", sha256="", created=None, final_path=""):
        self.id, self.name, self.size = uid, name, int(size)
        self.token, self.sha256 = token or "", (sha256 or "").lower()
        self.created = created or time.time()
        self.final_path = final_path or ""
        self.gen = 0                      # bumped by every claim; stale writers stop
        self.lock = threading.Lock()      # guards gen + writes (held for one piece)
        self.fin = threading.Lock()       # serialises finalisation (may hash a big file)
        self.finalizing = False           # the file is complete and is being moved

    @property
    def dir(self) -> Path:
        return _UP_ROOT / self.id

    @property
    def part(self) -> Path:
        return self.dir / "data.part"

    def offset(self) -> int:
        if self.final_path or self.finalizing:
            return self.size
        try:
            return os.path.getsize(self.part)
        except OSError:
            return 0

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "meta.tmp"
        tmp.write_text(json.dumps({
            "id": self.id, "name": self.name, "size": self.size, "token": self.token,
            "sha256": self.sha256, "created": self.created, "final_path": self.final_path,
        }, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, self.dir / "meta.json")

    def info(self) -> dict:
        off = self.offset()
        d = {"ok": True, "upload_id": self.id, "name": self.name, "size": self.size,
             "offset": off, "complete": bool(self.final_path),
             "chunk_bytes": int(_cfg("RESUMABLE_CHUNK_BYTES", 4 * 1024 * 1024))}
        if self.final_path:
            d["path"] = self.final_path
            d["file"] = {"path": self.final_path, "orig_name": self.name, "size": self.size,
                         "url": None, "mime_type": None, "is_stream": False,
                         "meta": {"_type": "gradio.FileData"}}
        return d


def _up_get(uid: str):
    """The upload, from memory or (after a process restart) from disk."""
    if not _UP_ID_RE.fullmatch(uid or ""):
        return None
    with _UP_LOCK:
        up = _UP.get(uid)
        if up is not None:
            return up
        meta = _UP_ROOT / uid / "meta.json"
        try:
            d = json.loads(meta.read_text(encoding="utf-8"))
            up = _Upload(d["id"], d["name"], d["size"], d.get("token", ""), d.get("sha256", ""),
                         d.get("created"), d.get("final_path", ""))
            if up.final_path and not os.path.isfile(up.final_path):
                return None             # Gradio's own cleanup removed it
            _UP[uid] = up
            return up
        except Exception:
            return None


def _up_find_token(token: str):
    if not token:
        return None
    with _UP_LOCK:
        for up in _UP.values():
            if up.token == token:
                return up
    try:
        for d in _UP_ROOT.iterdir():
            up = _up_get(d.name)
            if up is not None and up.token == token:
                return up
    except Exception:
        pass
    return None


def _up_drop(up: "_Upload") -> None:
    with _UP_LOCK:
        _UP.pop(up.id, None)
    shutil.rmtree(up.dir, ignore_errors=True)


def _up_sweep(now=None, ttl=None, only_unfinished=False) -> int:
    """Remove uploads nobody has touched for `ttl` seconds (default
    RESUMABLE_PARTIAL_TTL_SEC). only_unfinished leaves completed uploads alone."""
    ttl = float(_cfg("RESUMABLE_PARTIAL_TTL_SEC", 6 * 3600) if ttl is None else ttl)
    now = now or time.time()
    n = 0
    try:
        dirs = list(_UP_ROOT.iterdir())
    except Exception:
        return 0
    for d in dirs:
        try:
            if only_unfinished and _up_final(d.name):
                continue
            latest = max((f.stat().st_mtime for f in d.iterdir()), default=d.stat().st_mtime)
            if now - latest > ttl:
                with _UP_LOCK:
                    _UP.pop(d.name, None)
                shutil.rmtree(d, ignore_errors=True)
                n += 1
        except Exception:
            pass
    if n:
        logging.info("resumable: removed %d stale upload(s)", n)
    return n


def _up_active_count() -> int:
    try:
        return sum(1 for d in _UP_ROOT.iterdir() if not _up_final(d.name))
    except Exception:
        return 0


def _up_final(uid: str) -> bool:
    up = _up_get(uid)
    return bool(up and up.final_path)


def _clean_upload_name(name: str):
    """A safe file name with an allowed extension, or None."""
    base = os.path.basename(str(name or "").replace("\\", "/")).strip()
    stem, ext = os.path.splitext(base)
    ext = ext.lower()
    if ext not in _UP_EXT:
        return None
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")[:60] or "upload"
    return stem + ext


class _ChecksumMismatch(Exception):
    pass


def _finalize_upload(up: "_Upload") -> None:
    """Verify, then move the finished file into Gradio's upload folder."""
    with up.fin:
        if up.final_path:
            return
        with up.lock:
            up.gen += 1                    # any zombie writer is now stale
            have = up.offset()
        if have != up.size:
            raise RuntimeError(f"upload incomplete: {have}/{up.size}")
        up.finalizing = True
        try:
            if up.sha256:
                h = hashlib.sha256()
                with open(up.part, "rb") as f:
                    for blk in iter(lambda: f.read(1 << 20), b""):
                        h.update(blk)
                if h.hexdigest() != up.sha256:
                    raise _ChecksumMismatch()
            dest_dir = Path(_gradio_upload_folder()) / ("rs" + secrets.token_hex(11))
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / up.name
            try:
                os.replace(up.part, dest)
            except OSError:
                shutil.move(str(up.part), str(dest))
            up.final_path = str(dest)
            up.save()
        finally:
            up.finalizing = False
        logging.info("resumable: upload %s complete → %s (%d bytes)", up.id[:8], dest, up.size)


def _parse_range(header: str, size: int):
    """(start, end) inclusive; None = send everything; 'bad' = unsatisfiable."""
    m = re.fullmatch(r"\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*", header or "")
    if not m or "," in (header or ""):
        return None
    a, b = m.group(1), m.group(2)
    if a == "" and b == "":
        return None
    if a == "":                                    # suffix: the last N bytes
        n = int(b)
        if n <= 0:
            return "bad"
        return max(0, size - n), size - 1
    start = int(a)
    end = size - 1 if b == "" else min(int(b), size - 1)
    if start >= size or end < start:
        return "bad"
    return start, end


def _iter_file(path: str, start: int, end: int, step: int = 256 * 1024):
    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            blk = f.read(min(step, remaining))
            if not blk:
                break
            remaining -= len(blk)
            yield blk


def _result_for(ref: str):
    """(jid, job_dict, error_kind). error_kind: None | notfound | notready | gone"""
    jid = _resolve_job_ref(ref)
    if not jid:
        return None, {}, "notfound"
    with _lock:
        job = dict(jobs.get(jid) or {})
    if not job:
        return jid, {}, "notfound"
    rp = job.get("result_path")
    if job.get("status") != "done":
        return jid, job, "notready"
    if not rp or not os.path.isfile(rp):
        return jid, job, "gone"
    return jid, job, None


def install_http_routes(app) -> bool:
    """Add the resumable routes to Gradio's FastAPI app. Never raises: if this
    fails the Space still serves everything it served before."""
    try:
        if getattr(app.state, "phoenix_http_installed", False):
            return True
        if not bool(_cfg("RESUMABLE_ENABLED", True)):
            logging.info("resumable transport disabled (config.RESUMABLE_ENABLED)")
            return False
        if Request is None:
            raise RuntimeError("starlette is not importable")
        from fastapi.responses import JSONResponse, Response, StreamingResponse
        from starlette.concurrency import run_in_threadpool
        from starlette.requests import ClientDisconnect

        chunk_max = int(_cfg("RESUMABLE_MAX_CHUNK_BYTES", 32 * 1024 * 1024))
        file_max = int(_cfg("RESUMABLE_MAX_FILE_BYTES", 2 * 1024 ** 3))
        active_max = int(_cfg("RESUMABLE_MAX_ACTIVE", 16))
        _UP_ROOT.mkdir(parents=True, exist_ok=True)

        def jerr(status, code, message, **extra):
            return JSONResponse({"ok": False, "error": {"code": code, "message": message}, **extra},
                                status_code=status)

        @app.post("/phoenix/upload/init")
        async def phoenix_upload_init(request: Request):
            try:
                body = await request.json()
                if not isinstance(body, dict):
                    raise ValueError
            except Exception:
                return jerr(400, "invalid_request", "Body must be a JSON object")
            name = _clean_upload_name(body.get("name"))
            if name is None:
                return jerr(400, "invalid_name", "name needs a video or image extension "
                            "(mp4 mov m4v mkv webm 3gp avi jpg png webp)")
            try:
                size = int(body.get("size"))
            except Exception:
                return jerr(400, "invalid_size", "size must be the file size in bytes")
            if size < 1:
                return jerr(400, "invalid_size", "size must be at least 1")
            if size > file_max:
                return jerr(413, "too_large", f"max upload is {file_max} bytes")
            token = str(body.get("token") or "")
            if token and not _TOKEN_RE.fullmatch(token):
                return jerr(400, "invalid_token", "token must be 12-64 characters of A-Z a-z 0-9 _ -")
            sha = str(body.get("sha256") or "").lower()
            if sha and not re.fullmatch(r"[0-9a-f]{64}", sha):
                return jerr(400, "invalid_sha256", "sha256 must be 64 hex characters")

            up = await run_in_threadpool(_up_find_token, token) if token else None
            if up is not None:
                if up.name != name or up.size != size:
                    return jerr(409, "token_conflict",
                                "this token already names a different upload", upload_id=up.id)
                with up.lock:
                    up.gen += 1
                return JSONResponse(up.info())          # resume: same upload, current offset

            await run_in_threadpool(_up_sweep)
            if await run_in_threadpool(_up_active_count) >= active_max:
                # A phone killed mid-upload leaves its partial behind; do not let
                # a handful of those lock the owner out for hours. Anything idle
                # this long is not coming back.
                await run_in_threadpool(
                    lambda: _up_sweep(ttl=float(_cfg("RESUMABLE_EVICT_IDLE_SEC", 900)),
                                      only_unfinished=True))
            if await run_in_threadpool(_up_active_count) >= active_max:
                return jerr(429, "too_many_uploads", "too many unfinished uploads; try again shortly")
            free = shutil.disk_usage(tempfile.gettempdir()).free
            if free < size + 256 * 1024 * 1024:
                return jerr(507, "insufficient_storage", "the Space does not have room for this file")
            up = _Upload(secrets.token_urlsafe(18), name, size, token, sha)
            await run_in_threadpool(up.save)
            with _UP_LOCK:
                _UP[up.id] = up
            return JSONResponse(up.info())

        @app.get("/phoenix/upload/{upload_id}")
        async def phoenix_upload_status(upload_id: str):
            up = _up_get(upload_id)
            if up is None:
                return jerr(404, "not_found", "unknown or expired upload")
            return JSONResponse(up.info(), headers={"Upload-Offset": str(up.offset())})

        @app.delete("/phoenix/upload/{upload_id}")
        async def phoenix_upload_abort(upload_id: str):
            up = _up_get(upload_id)
            if up is None:
                return jerr(404, "not_found", "unknown or expired upload")
            with up.lock:
                up.gen += 1
            if not up.final_path:                       # a finished file is Gradio's now
                await run_in_threadpool(_up_drop, up)
            return JSONResponse({"ok": True})

        @app.put("/phoenix/upload/{upload_id}")
        async def phoenix_upload_put(upload_id: str, request: Request):
            up = _up_get(upload_id)
            if up is None:
                return jerr(404, "not_found", "unknown or expired upload")
            if up.final_path:
                return JSONResponse(up.info())          # a lost final response, retried
            raw_off = request.headers.get("upload-offset", request.query_params.get("offset"))
            try:
                want = int(raw_off)
                if want < 0:
                    raise ValueError
            except Exception:
                return jerr(400, "invalid_offset", "send the byte offset in the Upload-Offset header")
            cl = request.headers.get("content-length")
            if cl and cl.isdigit() and int(cl) > chunk_max:
                return jerr(413, "chunk_too_large", f"max chunk is {chunk_max} bytes")

            with up.lock:
                up.gen += 1                              # take over from any dead writer
                gen = up.gen
                cur = up.offset()
            if want != cur:
                return jerr(409, "offset_mismatch", "continue from the offset in this reply",
                            offset=cur, size=up.size)
            if cl and cl.isdigit() and cur + int(cl) > up.size:
                return jerr(413, "past_end", "chunk runs past the declared size",
                            offset=cur, size=up.size)

            written = 0
            problem = None
            try:
                # buffering=0: every piece is in the file (and visible to
                # offset()) before the lock is released.
                with open(up.part, "ab", buffering=0) as f:
                    async for piece in request.stream():
                        if not piece:
                            continue
                        with up.lock:
                            if up.gen != gen:
                                problem = ("stale", None)
                                break
                            if cur + written + len(piece) > up.size:
                                os.truncate(up.part, cur)   # a malformed request leaves nothing behind
                                written = 0
                                problem = ("past_end", None)
                                break
                            if written + len(piece) > chunk_max:
                                problem = ("big", None)
                                break
                            f.write(piece)
                            written += len(piece)
                    if problem is None:
                        os.fsync(f.fileno())
            except ClientDisconnect:
                return Response(status_code=499)          # what arrived is kept
            if problem:
                kind = problem[0]
                if kind == "stale":
                    return jerr(409, "superseded", "another request took this upload over",
                                offset=up.offset(), size=up.size)
                if kind == "past_end":
                    return jerr(413, "past_end", "chunk runs past the declared size",
                                offset=up.offset(), size=up.size)
                return jerr(413, "chunk_too_large", f"max chunk is {chunk_max} bytes",
                            offset=up.offset(), size=up.size)
            if up.offset() >= up.size:
                try:
                    await run_in_threadpool(_finalize_upload, up)
                except _ChecksumMismatch:
                    await run_in_threadpool(_up_drop, up)
                    return jerr(422, "checksum_mismatch",
                                "the file arrived damaged; start the upload again")
                except Exception as exc:
                    logging.exception("resumable: finalise failed")
                    return jerr(500, "finalize_failed", f"{type(exc).__name__}: {str(exc)[:160]}")
            return JSONResponse(up.info())

        @app.get("/phoenix/status/{ref}")
        async def phoenix_http_status(ref: str):
            out = api_job_status(ref)
            if out.get("ok"):
                return JSONResponse(out)
            code = (out.get("error") or {}).get("code")
            return JSONResponse(out, status_code=404 if code == "not_found" else 400)

        @app.api_route("/phoenix/download/{ref}", methods=["GET", "HEAD"])
        async def phoenix_http_download(ref: str, request: Request):
            jid, job, bad = _result_for(ref)
            if bad == "notfound":
                return jerr(404, "not_found", "unknown job")
            if bad == "notready":
                return jerr(409, "not_ready", "the job has not finished", job_status=job.get("status"),
                            progress=int(job.get("progress") or 0))
            if bad == "gone":
                return jerr(410, "expired", "the result was removed; submit the job again")
            rp = str(job["result_path"])
            st = os.stat(rp)
            size = st.st_size
            etag = '"%x-%x"' % (size, st.st_mtime_ns)
            name = os.path.basename(str(job.get("save_name") or "")) or os.path.basename(rp)
            ctype = "video/mp4" if rp.lower().endswith(".mp4") else (
                "application/zip" if rp.lower().endswith(".zip") else "application/octet-stream")
            base = {"Accept-Ranges": "bytes", "ETag": etag, "Cache-Control": "private, no-store",
                    "Content-Disposition": f'attachment; filename="{name}"'}
            rng = None
            hdr = request.headers.get("range")
            if hdr:
                ifr = request.headers.get("if-range")
                if not ifr or ifr.strip() == etag:       # a stale If-Range means: start over
                    rng = _parse_range(hdr, size)
            if rng == "bad":
                return Response(status_code=416, headers={**base, "Content-Range": f"bytes */{size}"})
            if rng is None:
                start, end, status = 0, size - 1, 200
                hdrs = {**base, "Content-Length": str(size)}
            else:
                start, end = rng
                status = 206
                hdrs = {**base, "Content-Length": str(end - start + 1),
                        "Content-Range": f"bytes {start}-{end}/{size}"}
            if request.method == "HEAD":
                return Response(status_code=status, headers={**hdrs, "Content-Type": ctype})
            return StreamingResponse(_iter_file(rp, start, end), status_code=status,
                                     media_type=ctype, headers=hdrs)

        if not _HTTP_STATE["janitor"]:
            _HTTP_STATE["janitor"] = True

            def _janitor():
                while True:
                    time.sleep(600)
                    try:
                        _up_sweep()
                    except Exception:
                        pass
            threading.Thread(target=_janitor, name="phoenix-up-janitor", daemon=True).start()

        app.state.phoenix_http_installed = True
        logging.info("resumable transport on: /phoenix/upload, /phoenix/status, /phoenix/download "
                     "(chunk %d B, max file %d B)", int(_cfg("RESUMABLE_CHUNK_BYTES", 4 * 1024 * 1024)), file_max)
        return True
    except Exception:
        logging.exception("resumable transport NOT installed - uploads fall back to Gradio's /upload")
        return False
