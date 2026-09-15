"""
Data retention: uploads, renders and working files live for a few hours, then go.

A video of someone's face is the most personal thing this app touches, and on
a hosted Space it lands in a temp directory that nothing would otherwise clean
until the container is recycled - which might be days. So a janitor thread
sweeps every ten minutes and deletes anything older than the retention window
(three hours by default, `BEAUTY_RETENTION_HOURS` to change it).

What is swept: the render directories this app creates, and Gradio's own cache,
which is where uploads and the files served back to the browser are kept. Both
are addressed by name rather than by sweeping the whole temp directory, because
deleting another process's files would be a worse bug than keeping ours.

What is NOT swept by default: the MediaPipe model cache. Those are weights, not
anyone's data, and dropping them only forces a 10 MB re-download on the next
render. `BEAUTY_PURGE_MODELS=1` includes them anyway.
"""
from __future__ import annotations

import glob
import logging
import os
import shutil
import tempfile
import threading
import time

log = logging.getLogger(__name__)

DEFAULT_HOURS = 3.0
SWEEP_INTERVAL_S = 600


def retention_hours() -> float:
    try:
        v = float(os.environ.get("BEAUTY_RETENTION_HOURS", DEFAULT_HOURS))
    except ValueError:
        return DEFAULT_HOURS
    # A zero or negative window would delete a render the moment it finished.
    return max(v, 0.05)


def _model_dirs() -> list[str]:
    try:
        from .mp_backend import _cache_dir
        return [_cache_dir()]
    except Exception:
        return []


def target_dirs(include_models: bool = False) -> list[str]:
    """Directories this app owns, that are safe to delete from."""
    tmp = tempfile.gettempdir()
    dirs = sorted(glob.glob(os.path.join(tmp, "beauty_*")))
    dirs += sorted(glob.glob(os.path.join(tmp, "beauty_selftest_*")))
    gradio = os.environ.get("GRADIO_TEMP_DIR") or os.path.join(tmp, "gradio")
    if os.path.isdir(gradio):
        dirs.append(gradio)
    if include_models:
        dirs += _model_dirs()
    seen, out = set(), []
    for d in dirs:
        real = os.path.realpath(d)
        if real not in seen and os.path.isdir(real):
            seen.add(real)
            out.append(real)
    return out


def _delete(path: str) -> int:
    """Delete a file or directory tree; returns the bytes it was using."""
    size = 0
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            for root, _, files in os.walk(path):
                for name in files:
                    try:
                        size += os.path.getsize(os.path.join(root, name))
                    except OSError:
                        pass
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            os.remove(path)
    except OSError as e:
        log.debug("could not delete %s: %s", path, e)
        return 0
    return size


def sweep(hours: float | None = None, include_models: bool = False) -> tuple[int, int]:
    """Delete everything older than the window. Returns (items, bytes)."""
    window = (retention_hours() if hours is None else hours) * 3600.0
    cutoff = time.time() - window
    items = 0
    freed = 0
    for base in target_dirs(include_models):
        # A whole render directory older than the window goes in one piece.
        try:
            entries = os.listdir(base)
        except OSError:
            continue
        if base.startswith(os.path.join(tempfile.gettempdir(), "beauty_")):
            if _age_ok(base, cutoff):
                freed += _delete(base)
                items += 1
            continue
        for name in entries:
            path = os.path.join(base, name)
            if _age_ok(path, cutoff):
                freed += _delete(path)
                items += 1
    if items:
        log.info("retention sweep: removed %d items, %.1f MB (older than %.1f h)",
                 items, freed / 1e6, window / 3600.0)
    return items, freed


def _age_ok(path: str, cutoff: float) -> bool:
    """True when the path was last touched before the cutoff.

    Directories report the mtime of the directory itself, which does not
    change when a file deep inside it does, so the newest mtime anywhere in
    the tree decides - otherwise an in-progress render whose folder was
    created hours ago would be deleted out from under itself.
    """
    try:
        newest = os.path.getmtime(path)
    except OSError:
        return False
    if os.path.isdir(path) and not os.path.islink(path):
        for root, _, files in os.walk(path):
            for name in files:
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(root, name)))
                except OSError:
                    pass
    return newest < cutoff


def purge_now(include_models: bool = False) -> tuple[int, int]:
    """Delete everything regardless of age - the "clear it now" button."""
    return sweep(hours=0.0, include_models=include_models)


class Janitor(threading.Thread):
    """Daemon sweeper. Daemon so it never holds the process open on shutdown."""

    def __init__(self, interval_s: int = SWEEP_INTERVAL_S):
        super().__init__(name="beauty-retention", daemon=True)
        self.interval = int(interval_s)
        self._stop = threading.Event()

    def run(self):
        include_models = os.environ.get("BEAUTY_PURGE_MODELS", "").strip().lower() in (
            "1", "true", "yes", "on")
        while not self._stop.is_set():
            try:
                sweep(include_models=include_models)
            except Exception as e:                      # never kill the thread
                log.warning("retention sweep failed: %s", e)
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


_janitor: Janitor | None = None


def start(interval_s: int = SWEEP_INTERVAL_S) -> Janitor:
    """Start the sweeper once per process."""
    global _janitor
    if _janitor is None or not _janitor.is_alive():
        _janitor = Janitor(interval_s)
        _janitor.start()
        log.info("retention: uploads and renders are deleted after %.1f hours",
                 retention_hours())
    return _janitor


def policy_text() -> str:
    h = retention_hours()
    unit = "hour" if abs(h - 1.0) < 1e-6 else "hours"
    return (f"Uploads, renders and working files are deleted automatically "
            f"{h:g} {unit} after they are last touched.")
