"""
Phoenix Mobile API adapter for Swamitech Phoenix v11.0.4 SoftStable.

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
import os
import re
from typing import Any

import core_pipeline as cp
from core_pipeline import _lock, jobs, submit_video

try:
    from config import VERSION_FULL
except Exception:
    VERSION_FULL = "v11.0.4 (SoftStable)"

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
        "enhance_scope", "device_mode",
    }
    return {k: s[k] for k in allowed if k in s}


def api_submit_video(video, face1, face2, face3, face4, settings=None):
    """Start a Phoenix v11 video job and return a JSON job record."""
    if video is None:
        return _json_error("Upload a target video", "missing_video")
    if all(face is None for face in (face1, face2, face3, face4)):
        return _json_error("Upload at least one replacement face", "missing_face")

    defaults = {
        "secs": 30,
        "fps": 30,
        "res": "720p (HD)",
        "quality": "Ultra",
        "enhancer": "None",
        "swap_n": "1",
        "det_n": "1",
        "det_int": 1,
        "password": "",
        "trim_start": 0,
        "trim_end": 100,
        "face_mode": "1 face (fastest)",
        "enhance_scope": "Primary face only (faster)",
        "device_mode": "CPU only",
    }
    try:
        defaults.update(_job_config(settings))
    except ValueError as exc:
        return _json_error(str(exc), "invalid_settings")

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
        defaults["password"], [], defaults["trim_start"], defaults["trim_end"],
        defaults["face_mode"], defaults["enhance_scope"], defaults["device_mode"],
        None,
    )

    message = str(result[0]) if result else ""
    match = _JOB_RE.search(message)
    if not match:
        return _json_error(message or "Unable to start Phoenix job", "job_not_started")

    jid = match.group(1)
    with _lock:
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
        "engine": "aequus-1.0",
    }


def api_detect_video_frame(video, position_pct=0):
    """Detect up to four faces at a selected video position for mobile frame picking."""
    if video is None:
        return None, "Upload a target video first", None, None, None, None
    try:
        cp._load()
        frm, idx, total = cp._grab_frame(video, float(position_pct or 0))
        if frm is None:
            return None, f"Could not read frame at {position_pct}%", None, None, None, None
        faces = sorted(cp._fa.get(frm), key=lambda f: f.bbox[0])[:4]
        annotated = cp._annotate(frm, faces)
        crops = [cp._crop_face(frm, f) for f in faces]
        while len(crops) < 4:
            crops.append(None)
        return annotated, f"Frame {idx}/{total} · found {len(faces)} face(s)", crops[0], crops[1], crops[2], crops[3]
    except Exception as exc:
        return None, f"Detection failed: {str(exc)[:140]}", None, None, None, None


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
