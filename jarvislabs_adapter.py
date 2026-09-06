"""Bridges a Phoenix video job to a GPU box rented on Jarvislabs.

Two independent pieces, deliberately kept separable:

1. RemoteJobClient - a plain HTTP client against jarvislabs_server.py
   (see that file - you deploy it ON the Jarvislabs instance). This part
   is fully real and locally tested against an actual Flask server in
   this repo's test script; it does not depend on anything about
   Jarvislabs specifically, only on the server contract this repo also
   defines.

2. InstanceManager - an OPTIONAL, best-effort wrapper around Jarvislabs'
   own `jarvislabs` Python SDK for auto-creating/resuming/pausing the GPU
   instance itself, so a job can be submitted without you manually
   starting the instance and pasting its URL in first. This part is
   built from Jarvislabs' published SDK usage examples, not verified
   against a live account from this environment (no network access to
   Jarvislabs, no account credentials here) - see its class docstring for
   exactly what to check once you have a real account. If it fails for
   any reason, everything falls back to MANUAL mode: you start the
   instance yourself from the Jarvislabs dashboard, copy its API-endpoint
   URL into JARVISLABS_ENDPOINT_URL, and only RemoteJobClient is used.

The usage cap (gpu_usage_governor.GPUUsageGovernor) gates BOTH modes
identically - manual or automatic, no remote-GPU session starts without
the governor's ok.
"""
from __future__ import annotations

import os
import time
import logging

import requests

from gpu_usage_governor import GPUUsageGovernor

log = logging.getLogger("swamitech.jarvislabs_adapter")

JARVISLABS_ENDPOINT_URL = os.environ.get("JARVISLABS_ENDPOINT_URL", "").strip().rstrip("/")
JARVISLABS_API_KEY = os.environ.get("JARVISLABS_API_KEY", "").strip()
JARVISLABS_GPU_TYPE = os.environ.get("JARVISLABS_GPU_TYPE", "A30").strip()
JARVISLABS_MACHINE_ID = os.environ.get("JARVISLABS_MACHINE_ID", "").strip()

# How long a submitted job is assumed to need, for the governor's
# before-you-start budget check (can_start's `estimated_hours`). Kept
# deliberately conservative (high) so the check errs toward refusing a
# job that might blow the cap rather than starting one that will -
# actual usage recorded at the end (via end_session) is always the exact
# real elapsed time regardless of this estimate; this number only ever
# affects the PRE-flight refusal decision, never what gets billed against
# the ledger.
DEFAULT_JOB_ESTIMATE_HOURS = float(os.environ.get("PHOENIX_REMOTE_JOB_ESTIMATE_HOURS", "0.5"))

POLL_INTERVAL_SEC = float(os.environ.get("PHOENIX_REMOTE_POLL_INTERVAL_SEC", "5"))
INSTANCE_READY_TIMEOUT_SEC = float(os.environ.get("PHOENIX_REMOTE_READY_TIMEOUT_SEC", "180"))

GOVERNOR = GPUUsageGovernor()


class RemoteJobFailed(Exception):
    pass


class RemoteJobClient:
    """HTTP client for jarvislabs_server.py's endpoints."""

    def __init__(self, endpoint_url: str, timeout: float = 30.0):
        if not endpoint_url:
            raise ValueError("endpoint_url is required (JARVISLABS_ENDPOINT_URL not set?)")
        self.base = endpoint_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        r = requests.get(f"{self.base}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def submit(self, video_path: str, face_paths: dict, settings: dict) -> dict:
        files = {"video": open(video_path, "rb")}
        try:
            for idx in (1, 2, 3, 4):
                p = face_paths.get(idx - 1)  # src_paths is keyed 0..3 in core_pipeline
                if p:
                    files[f"face{idx}"] = open(p, "rb")
            import json
            data = {"settings": json.dumps(settings)}
            r = requests.post(f"{self.base}/submit_video", files=files, data=data,
                               timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        finally:
            for f in files.values():
                try:
                    f.close()
                except Exception:
                    pass

    def status(self, job_id: str) -> dict:
        r = requests.get(f"{self.base}/job_status/{job_id}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def download(self, job_id: str, dest_path: str) -> str:
        r = requests.get(f"{self.base}/download/{job_id}", timeout=120, stream=True)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        return dest_path

    def cancel(self, job_id: str) -> dict:
        r = requests.post(f"{self.base}/cancel/{job_id}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()


class InstanceManager:
    """Best-effort Jarvislabs instance lifecycle via the official `jarvislabs`
    SDK (PyPI package `jarvislabs`, `from jarvislabs import Client`).

    NOT VERIFIED against a live Jarvislabs account from this environment -
    built from the SDK's own published usage example:

        from jarvislabs import Client
        with Client() as client:
            inst = client.instances.create(gpu_type="A100", name="my-run")
            client.instances.pause(inst.machine_id)

    Before relying on this in production, confirm against YOUR account
    and the current SDK docs (docs.jarvislabs.ai, jarvislabs.ai/sdk):
      - The exact `gpu_type` strings accepted (this module defaults to
        "A30" per Jarvislabs' own India pricing page - confirm the SDK
        accepts that exact string, it may want a different casing/alias).
      - Whether `create()` needs more required arguments on your account
        (an image/template id, a disk size, etc.) than shown above.
      - The attribute that carries the instance's public API-endpoint URL
        once running - `inst.ssh_command` is confirmed from the example
        above, but the HTTP endpoint URL (what RemoteJobClient needs) is
        NOT shown in any example this module's author found. You may need
        to read it from a different SDK attribute/call, or from the
        Jarvislabs dashboard by hand the first time and store it in
        JARVISLABS_MACHINE_ID/JARVISLABS_ENDPOINT_URL instead of relying
        on auto-discovery here.
      - Whether `instances.resume(machine_id)` (for a previously-created,
        paused instance - reusing it avoids re-downloading models every
        session) exists under that name; only create/pause are confirmed.

    If any SDK call here raises or behaves unexpectedly, run_job() catches
    it and falls back to requiring JARVISLABS_ENDPOINT_URL to already be
    set (manual mode) - this class is an optimization, never a hard
    dependency for using the rest of this module.
    """

    def __init__(self, api_key: str = JARVISLABS_API_KEY, gpu_type: str = JARVISLABS_GPU_TYPE,
                 machine_id: str = JARVISLABS_MACHINE_ID):
        self.api_key = api_key
        self.gpu_type = gpu_type
        self.machine_id = machine_id or None

    def _client(self):
        from jarvislabs import Client  # local import: optional dependency
        return Client(api_key=self.api_key) if self.api_key else Client()

    def ensure_running(self) -> str:
        """Returns an endpoint URL, creating/resuming an instance if needed.
        Raises on any failure - caller must catch and fall back to manual
        JARVISLABS_ENDPOINT_URL mode."""
        with self._client() as client:
            if self.machine_id:
                try:
                    client.instances.resume(self.machine_id)
                except Exception as exc:
                    raise RuntimeError(
                        f"could not resume existing instance {self.machine_id}: {exc}. "
                        "SDK method name/behavior unverified - see InstanceManager docstring."
                    ) from exc
            else:
                inst = client.instances.create(gpu_type=self.gpu_type, name="phoenix-remote-gpu")
                self.machine_id = getattr(inst, "machine_id", None)
                log.info("Jarvislabs instance created: machine_id=%s ssh=%s",
                         self.machine_id, getattr(inst, "ssh_command", "?"))

            # Best-effort: look for a URL-shaped attribute. Unverified - see
            # docstring. If this doesn't find one, the caller falls back to
            # JARVISLABS_ENDPOINT_URL, which is why this raises rather than
            # guessing at a URL shape.
            for attr in ("endpoint_url", "api_endpoint", "url", "http_endpoint"):
                val = getattr(client.instances.get(self.machine_id), attr, None) if self.machine_id else None
                if val:
                    return str(val)
            raise RuntimeError(
                "Could not determine the instance's HTTP endpoint URL from the SDK "
                "(no known attribute matched) - set JARVISLABS_ENDPOINT_URL manually "
                "from the Jarvislabs dashboard instead. See InstanceManager docstring."
            )

    def pause(self):
        if not self.machine_id:
            return
        try:
            with self._client() as client:
                client.instances.pause(self.machine_id)
        except Exception as exc:
            log.warning("Jarvislabs instance pause failed (billing may continue "
                        "until you pause it manually from the dashboard): %s", exc)


def _resolve_endpoint() -> tuple[str, InstanceManager | None]:
    """Returns (endpoint_url, manager_or_None). manager is only non-None if
    auto-provisioning was actually used, so the caller knows whether it is
    responsible for pausing the instance afterward."""
    if JARVISLABS_API_KEY:
        mgr = InstanceManager()
        try:
            url = mgr.ensure_running()
            return url, mgr
        except Exception as exc:
            log.warning("Jarvislabs auto-provisioning failed (%s); "
                        "falling back to JARVISLABS_ENDPOINT_URL manual mode.", exc)
    if JARVISLABS_ENDPOINT_URL:
        return JARVISLABS_ENDPOINT_URL, None
    raise RemoteJobFailed(
        "No Jarvislabs endpoint available: set JARVISLABS_API_KEY for auto-provisioning "
        "(best-effort, see InstanceManager docstring) or JARVISLABS_ENDPOINT_URL to an "
        "already-running instance's API endpoint (start it from the Jarvislabs dashboard "
        "and copy the URL it gives you)."
    )


def _build_remote_settings(cfg: dict) -> dict:
    """Translate core_pipeline's internal job cfg (max_seconds/resolution/...)
    into the settings dict api_submit_video()/_job_config() expect
    (secs/res/...). Key names genuinely differ between the two - see
    core_pipeline.submit_video()'s own `cfg = dict(...)` construction and
    phoenix_api_adapter._job_config()'s `allowed` set."""
    return {
        "secs": cfg.get("max_seconds"),
        "fps": cfg.get("fps"),
        "res": cfg.get("resolution"),
        "quality": cfg.get("quality"),
        "enhancer": cfg.get("enhancer"),
        "swap_n": cfg.get("swap_n"),
        "det_n": cfg.get("det_n"),
        "det_int": cfg.get("det_int"),
        "password": cfg.get("password"),
        "trim_start": cfg.get("trim_start"),
        "trim_end": cfg.get("trim_end"),
        "face_mode": cfg.get("face_mode"),
        "enhance_scope": cfg.get("enhance_scope"),
        # device_mode is forced to "GPU" server-side regardless (see
        # jarvislabs_server.py) - included here only for completeness.
        "device_mode": "GPU",
    }


def run_job_on_jarvislabs(jid: str, src_paths: dict, vp: str, cfg: dict,
                           jobs: dict, lock, estimate_hours: float = DEFAULT_JOB_ESTIMATE_HOURS):
    """Mirrors core_pipeline._run_job_on_gpu()'s shape/signature so it can be
    called the same way from submit_video(): runs the ENTIRE job on a
    remote Jarvislabs GPU instance and mirrors progress into the same
    `jobs[jid]` dict the rest of the app already reads from (history,
    progress bar, download links), so nothing downstream needs to know or
    care whether a job ran locally or remotely.

    Raises on any failure (governor refusal, no endpoint, network error,
    remote job error) - the caller (submit_video) is expected to catch
    this and fall back to local CPU processing, exactly like the existing
    ZeroGPU branch already does for its own failures.
    """
    ok, reason = GOVERNOR.can_start(estimated_hours=estimate_hours)
    if not ok:
        raise RemoteJobFailed(reason)

    session_id = GOVERNOR.start_session()
    manager = None
    try:
        with lock:
            if jid in jobs:
                jobs[jid].update(status="processing", progress=1,
                                  message="Connecting to remote GPU…")

        endpoint, manager = _resolve_endpoint()
        client = RemoteJobClient(endpoint)
        client.health()  # fail fast with a clear error if unreachable

        remote_settings = _build_remote_settings(cfg)
        submit_resp = client.submit(vp, src_paths, remote_settings)
        if not submit_resp.get("ok"):
            err = (submit_resp.get("error") or {}).get("message", "remote submit failed")
            raise RemoteJobFailed(err)
        remote_jid = submit_resp["job_id"]

        with lock:
            if jid in jobs:
                jobs[jid].update(message=f"Running on remote GPU (Jarvislabs job {remote_jid})…")

        deadline_notice_shown = False
        while True:
            with lock:
                cancelled = bool(jobs.get(jid, {}).get("cancel"))
            if cancelled:
                try:
                    client.cancel(remote_jid)
                except Exception:
                    pass
                with lock:
                    if jid in jobs:
                        jobs[jid].update(status="cancelled", done_at=time.time(),
                                          message="✋ Cancelled")
                return

            st = client.status(remote_jid)
            if not st.get("ok"):
                raise RemoteJobFailed((st.get("error") or {}).get("message", "remote status failed"))

            with lock:
                if jid in jobs:
                    jobs[jid].update(
                        progress=int(st.get("progress") or 0),
                        message=st.get("message") or jobs[jid].get("message"),
                        eta_seconds=st.get("eta_seconds"),
                    )

            status = st.get("status")
            if status == "done":
                break
            if status in ("error", "cancelled"):
                raise RemoteJobFailed(st.get("message") or f"remote job ended with status={status}")

            # Governor cap crossed WHILE a job is in flight: let it finish
            # (see GPUUsageGovernor.can_start's docstring on why), just stop
            # silently pretending nothing changed - the elapsed time is
            # still ticking on GOVERNOR via this very session, and it will
            # correctly block the NEXT job once end_session below records it.
            time.sleep(POLL_INTERVAL_SEC)

        local_result_path = f"/tmp/remote_result_{jid}.mp4"
        client.download(remote_jid, local_result_path)

        import core_pipeline as cp
        server_path, expires_at = cp._server_save_result(jid, local_result_path)
        if os.path.abspath(server_path) != os.path.abspath(local_result_path):
            try:
                os.remove(local_result_path)
            except Exception:
                pass

        with lock:
            if jid in jobs:
                jobs[jid].update(
                    status="done", result_path=server_path, server_result_path=server_path,
                    progress=100, done_at=time.time(), expires_at=expires_at, eta_seconds=None,
                    message=(st.get("message") or "Done") + " · ☁ Jarvislabs remote GPU",
                )
        try:
            cp._persist_job(jid, force=True)
        except Exception:
            pass

    finally:
        GOVERNOR.end_session(session_id)
        if manager is not None:
            manager.pause()
