"""
Phoenix Mobile API adapter for Swamitech Phoenix v11.2.51 DetZoom.

Stable Gradio API surface:
  phoenix_submit_video
  phoenix_job_status
  phoenix_download
  phoenix_cancel

The adapter reuses the exact core_pipeline.submit_video() job engine.
No HF token is stored here.
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import core_pipeline as cp
from core_pipeline import _lock, jobs, submit_video

try:
    from config import VERSION_FULL, ensure_runtime_dirs
except Exception:
    VERSION_FULL = "v11.2.51 (DetZoom)"
    def ensure_runtime_dirs():
        os.makedirs("/tmp/gradio", exist_ok=True)

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


def api_submit_video(video, face1, face2, face3, face4, settings=None):
    """Start a Phoenix v11 video job and return a JSON job record."""
    video = _coerce_video(video)
    if video is None:
        return _json_error("Upload a target video", "missing_video")
    if all(face is None for face in (face1, face2, face3, face4)):
        return _json_error("Upload at least one replacement face", "missing_face")

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
    try:
        defaults.update(_job_config(settings))
    except ValueError as exc:
        return _json_error(str(exc), "invalid_settings")

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

    n_faces = sum(1 for f in (face1, face2, face3, face4) if _count_face(f))
    fm = str(defaults.get("face_mode") or "")
    if n_faces >= 2:
        defaults["face_mode"] = "2 faces" if n_faces == 2 else "Multiple faces"
    fe = defaults.get("face_enhance", False)
    if isinstance(fe, str):
        fe = fe.strip().lower() in ("1", "true", "on", "yes", "gfpgan")
    else:
        fe = bool(fe)
    if fe:
        defaults["enhancer"] = "GFPGAN"
    logging.info(
        "phoenix_submit_video replacements=%d face_mode=%s smooth_motion=%s face_enhance=%s",
        n_faces, defaults.get("face_mode"), defaults.get("smooth_motion"), bool(fe),
    )

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
    part_n = 0
    part_count = 1
    try:
        raw = _settings_dict(settings)
        part_count = int(raw.get("part_count") or 1)
        part_n = int(raw.get("part_index") or 0)
    except Exception:
        part_n = 0
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
            logging.info("download name %s → %s", jid, wanted)
        job = dict(jobs.get(jid) or {})
    return {
        "ok": True,
        "job_id": jid,
        "status": job.get("status", "processing"),
        "progress": int(job.get("progress") or 0),
        "message": job.get("message") or message,
        "eta_seconds": job.get("eta_seconds"),
        "device": defaults["device_mode"],
        "version": VERSION_FULL,
        "engine": "aequus-1.2.3-solid-face",
    }


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


def api_job_status(job_id: str):
    """Return status only; never expose internal filesystem paths."""
    jid = (job_id or "").strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{8}", jid):
        return _json_error("Invalid job_id", "invalid_job_id")

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
        "version": VERSION_FULL,
    }


def api_download(job_id: str):
    """Return the completed result file for a valid job ID."""
    jid = (job_id or "").strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{8}", jid):
        return None
    with _lock:
        job = dict(jobs.get(jid) or {})
    rp = job.get("result_path")
    if job.get("status") != "done" or not rp or not os.path.isfile(rp):
        return None
    return rp


def api_cancel(job_id: str):
    """Request cancellation of one Phoenix job."""
    jid = (job_id or "").strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{8}", jid):
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
