"""
Background render jobs: submit, walk away, come back.

A render takes minutes. Running it inside the HTTP request that started it -
which is what a Gradio generator does - means the work belongs to the browser
tab: minimise the window, lock the phone, lose the wifi, and the request is
dropped and the render dies with it. That is the wrong lifetime for the work.

So a render is a job here. The request only submits it; a worker thread owns
it from there and keeps going whether or not anyone is watching. The UI polls
for state by id, so any tab - including one opened an hour later on a
different device - can pick the job up, watch it finish, and download it.

State lives in an index file next to the outputs, so the job list survives a
page reload and a process restart (a job that was mid-render when the process
died comes back as "interrupted" rather than pretending to still be running).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field

from .pipeline import Cancelled, render_video
from .settings import Settings

log = logging.getLogger(__name__)

TERMINAL = ("done", "failed", "cancelled", "interrupted", "expired")


def jobs_root() -> str:
    root = os.environ.get("BEAUTY_JOBS_DIR") or os.path.join(
        tempfile.gettempdir(), "beauty_jobs")
    os.makedirs(root, exist_ok=True)
    return root


@dataclass
class Job:
    id: str
    source_name: str
    preset: str
    status: str = "queued"
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    done_frames: int = 0
    total_frames: int = 0
    message: str = "queued"
    error: str | None = None
    out_path: str | None = None
    source_path: str | None = None
    width: int = 0
    height: int = 0
    seconds: float = 0.0
    face_pct: float = 0.0
    body_pct: float = 0.0
    face_shift: float = 0.0
    body_shift: float = 0.0
    notes: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=dict)

    @property
    def progress(self) -> float:
        if self.status == "done":
            return 1.0
        if not self.total_frames:
            return 0.0
        return min(1.0, self.done_frames / float(self.total_frames))

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")

    def age_text(self) -> str:
        secs = time.time() - self.created
        if secs < 90:
            return f"{secs:.0f}s ago"
        if secs < 5400:
            return f"{secs / 60:.0f} min ago"
        return f"{secs / 3600:.1f} h ago"

    def label(self) -> str:
        icon = {"done": "✅", "running": "⏳", "queued": "…", "failed": "⚠️",
                "cancelled": "✖", "interrupted": "⚠️", "expired": "🗑"}.get(self.status, "·")
        return f"{icon} {self.source_name} · {self.preset} · {self.age_text()} [{self.id[:6]}]"

    def size_mb(self) -> float:
        try:
            return os.path.getsize(self.out_path) / 1e6 if self.out_path else 0.0
        except OSError:
            return 0.0


class JobManager:
    """
    One worker, a queue, and an index file.

    One worker on purpose: a render saturates the CPU it runs on, so a second
    concurrent render does not finish two jobs sooner - it makes both slower
    and doubles the peak memory. Extra submissions queue.
    """

    def __init__(self, root: str | None = None):
        self.root = root or jobs_root()
        self.index_path = os.path.join(self.root, "index.json")
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._queue: list[str] = []
        self._cancelled: set[str] = set()
        self._worker: threading.Thread | None = None
        self._load()

    # ------------------------------------------------------------- persistence
    def _load(self):
        try:
            with open(self.index_path) as fh:
                raw = json.load(fh)
        except Exception:
            return
        for d in raw.get("jobs", []):
            try:
                job = Job(**d)
            except TypeError:
                continue
            # Anything the index claims is in flight cannot be: this is a
            # fresh process, so whatever was running died with the old one.
            if job.active:
                job.status = "interrupted"
                job.message = "the app restarted while this was rendering"
            self._jobs[job.id] = job
        log.info("job index: %d previous jobs loaded from %s", len(self._jobs), self.root)

    def _save_locked(self):
        tmp = self.index_path + ".tmp"
        try:
            with open(tmp, "w") as fh:
                json.dump({"jobs": [asdict(j) for j in self._jobs.values()]}, fh)
            os.replace(tmp, self.index_path)
        except Exception as e:
            log.debug("could not write the job index: %s", e)

    def _save(self):
        with self._lock:
            self._save_locked()

    # ------------------------------------------------------------------ submit
    def submit(self, source_path: str, settings: Settings, preset: str,
               start: float = 0.0, end: float = 1.0) -> Job:
        job_id = uuid.uuid4().hex
        job_dir = os.path.join(self.root, job_id)
        os.makedirs(job_dir, exist_ok=True)

        # The upload lives in Gradio's cache, which the retention sweeper owns
        # and which a later upload may replace. Copy it in, so the job holds
        # everything it needs for as long as it needs it.
        name = os.path.basename(source_path)
        local_src = os.path.join(job_dir, name)
        try:
            shutil.copyfile(source_path, local_src)
        except Exception:
            local_src = source_path

        job = Job(id=job_id, source_name=name, preset=preset,
                  source_path=local_src,
                  out_path=os.path.join(job_dir, "beauty_studio_output.mp4"),
                  settings={"start": start, "end": end, **settings.to_dict()})
        with self._lock:
            self._jobs[job_id] = job
            self._queue.append(job_id)
            self._save_locked()
            self._ensure_worker()
        log.info("job %s queued: %s (%s)", job_id[:6], name, preset)
        return job

    def _ensure_worker(self):
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run_loop,
                                            name="beauty-render", daemon=True)
            self._worker.start()

    # ------------------------------------------------------------------ worker
    def _run_loop(self):
        while True:
            with self._lock:
                if not self._queue:
                    self._worker = None
                    return
                job_id = self._queue.pop(0)
                job = self._jobs.get(job_id)
                if job is None or job_id in self._cancelled:
                    continue
                job.status = "running"
                job.started = time.time()
                job.message = "starting…"
                self._save_locked()
            self._render(job)

    def _render(self, job: Job):
        def progress(done, total, message):
            job.done_frames = done
            job.total_frames = total
            job.message = message
            # The index is written every few seconds, not every frame: a
            # reload should show roughly where a job is, and that does not
            # justify a file write per frame.
            if done % 24 == 0:
                self._save()

        try:
            settings = Settings.from_dict(job.settings)
            res = render_video(job.source_path, settings, out_path=job.out_path,
                               start=float(job.settings.get("start", 0.0)),
                               end=float(job.settings.get("end", 1.0)),
                               progress=progress,
                               should_cancel=lambda: job.id in self._cancelled)
        except Cancelled:
            with self._lock:
                job.status = "cancelled"
                job.message = "stopped"
                job.finished = time.time()
                self._cancelled.discard(job.id)
                self._save_locked()
            return
        except Exception as e:
            log.error("job %s failed: %s", job.id[:6], traceback.format_exc())
            with self._lock:
                job.status = "failed"
                job.error = str(e)
                job.message = f"failed: {e}"
                job.finished = time.time()
                self._save_locked()
            return

        seen = max(res["frames_seen"], 1)
        with self._lock:
            job.status = "done"
            job.finished = time.time()
            job.out_path = res["path"]
            job.done_frames = res["frames"]
            job.total_frames = res["frames"]
            job.width, job.height = res["size"]
            job.seconds = res["seconds"]
            job.face_pct = 100.0 * res["faces_seen"] / seen
            job.body_pct = 100.0 * res["persons_seen"] / seen
            job.face_shift = res["max_face_shift_px"]
            job.body_shift = res["max_body_shift_px"]
            job.notes = list(res["notes"])
            job.message = "done"
            self._save_locked()
        log.info("job %s done in %.1fs -> %s", job.id[:6], job.seconds, job.out_path)

    # ------------------------------------------------------------------ access
    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not job.active:
                return False
            self._cancelled.add(job_id)
            if job_id in self._queue:
                # Never started: retire it here, the worker will not see it.
                self._queue.remove(job_id)
                job.status = "cancelled"
                job.message = "stopped before it started"
                job.finished = time.time()
                self._cancelled.discard(job_id)
                self._save_locked()
        return True

    def get(self, job_id: str | None) -> Job | None:
        if not job_id:
            return None
        with self._lock:
            return self._jobs.get(job_id)

    def latest_active(self) -> Job | None:
        with self._lock:
            active = [j for j in self._jobs.values() if j.active]
        return max(active, key=lambda j: j.created) if active else None

    def list_jobs(self, limit: int = 50) -> list[Job]:
        """Newest first, with any job whose output has been swept marked."""
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)
            changed = False
            for job in jobs:
                if job.status == "done" and (not job.out_path
                                             or not os.path.exists(job.out_path)):
                    job.status = "expired"
                    job.message = "deleted by the retention policy"
                    changed = True
            if changed:
                self._save_locked()
            return jobs[:limit]

    def forget_missing(self):
        """Drop index entries whose directory is gone, so the history does not
        grow a tail of rows pointing at nothing."""
        with self._lock:
            gone = [jid for jid, job in self._jobs.items()
                    if not os.path.isdir(os.path.join(self.root, jid))]
            for jid in gone:
                self._jobs.pop(jid, None)
            if gone:
                self._save_locked()
        return len(gone)


_manager: JobManager | None = None


def manager() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager
