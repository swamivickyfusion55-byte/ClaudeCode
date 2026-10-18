"""The Space's real resumable upload routes (phoenix_api_adapter.install_http_routes from v11.2.90), served by FastAPI/uvicorn.
core_pipeline is replaced by a stub (the routes only need jobs/_lock for the status route)."""
import sys, types, threading, os
pack = sys.argv[1]; port = int(sys.argv[2])
cp = types.ModuleType("core_pipeline"); cp._lock = threading.Lock(); cp.jobs = {}; cp.submit_video = lambda *a, **k: None
sys.modules["core_pipeline"] = cp
sys.path.insert(0, pack)
os.environ.setdefault("GRADIO_TEMP_DIR", "/tmp/claude-0/wire_gradio")
import phoenix_api_adapter as A
from fastapi import FastAPI
import uvicorn
app = FastAPI()
print("routes installed:", A.install_http_routes(app), flush=True)
uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
