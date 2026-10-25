"""The Space's real HTTP routes (phoenix_api_adapter.install_http_routes of PACK) for BusyTest.kt, plus one test-only
route that sets a fake job's status. core_pipeline is a stub (the routes only read jobs/_lock).
usage: busy_server.py PACK PORT"""
import sys, types, threading, os, time
pack = sys.argv[1]; port = int(sys.argv[2])
cp = types.ModuleType("core_pipeline"); cp._lock = threading.RLock(); cp.jobs = {}; cp.submit_video = lambda *a, **k: None
sys.modules["core_pipeline"] = cp
sys.path.insert(0, pack)
os.environ.setdefault("GRADIO_TEMP_DIR", "/tmp/claude-0/wire_gradio")
import phoenix_api_adapter as A
from fastapi import FastAPI
import uvicorn
app = FastAPI()
print("routes installed:", A.install_http_routes(app), flush=True)

@app.post("/test/job/{jid}/{status}")
def set_job(jid: str, status: str):
    with cp._lock:
        cp.jobs[jid] = dict(status=status, progress=50 if status == "processing" else 100, message=status,
                            created_at=time.time(), result_path=None, eta_seconds=None, cancel=False)
    return {"ok": True}

uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
