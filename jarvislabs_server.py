"""Run THIS on the Jarvislabs GPU instance, not on your Hugging Face Space.

This is the other half of the remote-GPU setup: a thin Flask wrapper
around this repo's own phoenix_api_adapter.py, exposed as plain HTTP/JSON
so the Space (still running on CPU) can hand it a whole video job and get
a finished video back. It does not reimplement any pipeline logic -
core_pipeline.py and swap_engine.py run completely unmodified here, the
only difference from your Space is that this machine has an NVIDIA GPU
and onnxruntime-gpu installed, so _cuda_available() returns True and the
existing device_mode="GPU" path in submit_video() actually uses it.

Setup on the Jarvislabs instance (see README.md's Jarvislabs section for
the full walkthrough):
    git clone <this repo>
    cd ClaudeCode
    pip install -r requirements.txt
    pip uninstall -y onnxruntime && pip install onnxruntime-gpu
    pip install flask
    python jarvislabs_server.py
Then, per Jarvislabs' own Flask-API docs (docs.jarvislabs.ai/deploy/flask-api),
expose port 6006 as an API endpoint from the instance's dashboard - that
public URL is what you put in JARVISLABS_ENDPOINT_URL on the Space side.

Security note: this server has NO authentication of its own beyond
whatever Jarvislabs' own API-endpoint gateway provides. It is meant for
one person's personal instance, not a public service - do not expose it
more broadly without adding your own auth in front of it.
"""
from __future__ import annotations

import os
import tempfile
import traceback

from flask import Flask, request, jsonify, send_file

import core_pipeline as cp  # noqa: F401 - triggers module-level setup (env defaults, etc.)
from phoenix_api_adapter import api_submit_video, api_job_status, api_download, api_cancel

try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "opencv is required (it's already in requirements.txt) - "
        "did you run `pip install -r requirements.txt` on this instance?"
    ) from exc

app = Flask(__name__)

UPLOAD_DIR = os.environ.get("PHOENIX_REMOTE_UPLOAD_DIR", "/tmp/phoenix_remote_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _save_upload(file_storage, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, dir=UPLOAD_DIR)
    os.close(fd)
    file_storage.save(path)
    return path


def _load_face_image(file_storage):
    """Mirrors what a Gradio Image component hands submit_video(): a BGR
    numpy array, not a file path - api_submit_video()/submit_video()
    immediately do cv2.imwrite() on whatever is passed for a face image."""
    if file_storage is None or file_storage.filename == "":
        return None
    path = _save_upload(file_storage, os.path.splitext(file_storage.filename)[1] or ".jpg")
    try:
        img = cv2.imread(path)
        return img
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


@app.route("/health", methods=["GET"])
def health():
    try:
        cuda = cp._cuda_available()
    except Exception:
        cuda = None
    return jsonify({"ok": True, "cuda_available": cuda})


@app.route("/submit_video", methods=["POST"])
def submit_video_route():
    try:
        video_file = request.files.get("video")
        if video_file is None or video_file.filename == "":
            return jsonify({"ok": False, "error": {"code": "missing_video", "message": "Upload a target video"}}), 400
        video_path = _save_upload(video_file, os.path.splitext(video_file.filename)[1] or ".mp4")

        faces = [_load_face_image(request.files.get(f"face{i}")) for i in (1, 2, 3, 4)]

        settings_raw = request.form.get("settings")
        # device_mode is forced to GPU here regardless of what the client
        # sent - the whole point of this box is to run the swap network on
        # its GPU; a client-provided "CPU only" would defeat that silently.
        import json as _json
        settings = _json.loads(settings_raw) if settings_raw else {}
        settings["device_mode"] = "GPU"

        result = api_submit_video(video_path, *faces, settings=settings)
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"ok": False, "error": {"code": "server_error", "message": str(exc)[:300]}}), 500


@app.route("/job_status/<job_id>", methods=["GET"])
def job_status_route(job_id):
    return jsonify(api_job_status(job_id))


@app.route("/download/<job_id>", methods=["GET"])
def download_route(job_id):
    path = api_download(job_id)
    if not path:
        return jsonify({"ok": False, "error": {"code": "not_ready", "message": "Result not ready or job not found"}}), 404
    return send_file(path, as_attachment=True)


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel_route(job_id):
    return jsonify(api_cancel(job_id))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "6006"))
    print(f"Phoenix remote GPU server starting on 0.0.0.0:{port}")
    print(f"CUDA available: {cp._cuda_available()}")
    app.run(host="0.0.0.0", port=port, threaded=True)
