"""
Phoenix core pipeline — models, swap, temporal, jobs, cleanup.
UI lives in app.py. Constants live in config.py.
"""
import os
for _k, _v in {
    "GRADIO_ANALYTICS_ENABLED": "False",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "GRADIO_TEMP_DIR": "/tmp/gradio",
}.items():
    os.environ.setdefault(_k, _v)

import gradio as gr
import cv2, numpy as np, subprocess, threading, uuid, time, shutil, bisect, logging, queue, tempfile
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

# PHASE 1: Optional boundary validation (safe fallback if unavailable)
try:
    from swap_engine import validate_detection_confidence, TrackingState, FaceTrackState
    HAS_PHASE1 = True
except ImportError:
    HAS_PHASE1 = False
    def validate_detection_confidence(bbox, frame_shape, landmarks=None):
        return 1.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Fix Gradio client schema crash on HF
try:
    import gradio_client.utils as _gc_utils
    _orig_get_type = _gc_utils.get_type
    def _safe_get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_get_type(schema)
    _gc_utils.get_type = _safe_get_type
    if hasattr(_gc_utils, "_json_schema_to_python_type"):
        _orig_js = _gc_utils._json_schema_to_python_type
        def _safe_js(schema, defs=None):
            if not isinstance(schema, dict):
                return "Any"
            try:
                return _orig_js(schema, defs)
            except TypeError:
                return "Any"
        _gc_utils._json_schema_to_python_type = _safe_js
        def _safe_js_public(schema):
            defs = schema.get("$defs") if isinstance(schema, dict) else None
            return _safe_js(schema, defs)
        _gc_utils.json_schema_to_python_type = _safe_js_public
except Exception as _patch_err:
    logging.warning("gradio schema patch skipped: %s", _patch_err)

_cpu_n = max(1, os.cpu_count() or 1)
# CPU-only policy: one small native thread budget shared by ORT/OpenCV/BLAS.
# Cap the default at 4 so an 8-vCPU HF Pro box does not thrash when ffmpeg
# also runs. Override with PHOENIX_NATIVE_THREADS=1..8 for A/B tests.
try:
    _default_native = max(1, min(4, _cpu_n))
    _native_threads = max(1, min(8, int(os.environ.get("PHOENIX_NATIVE_THREADS", str(_default_native)))))
except Exception:
    _native_threads = max(1, min(4, _cpu_n))
os.environ.setdefault("OMP_NUM_THREADS", str(_native_threads))
os.environ.setdefault("ORT_NUM_THREADS", str(_native_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(_native_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_native_threads))
try:
    cv2.setNumThreads(_native_threads)
    cv2.setUseOptimized(True)
except Exception:
    pass

_tmp_registry, _reg_lock = set(), threading.Lock()
def _reg(path):
    with _reg_lock: _tmp_registry.add(path)
    return path

def _cleanup():
    try:
        from config import RETAIN_SEC as _RS, ORPHAN_SEC as _OS
        RETAIN, ORPHAN = int(_RS), int(_OS)
    except Exception:
        RETAIN, ORPHAN = 10800, 86400  # 3h history files, 24h orphan tmp
    cycle = 0
    while True:
        now = time.time(); cycle += 1
        try:
            with _lock:
                known = set(jobs.keys())
                expired = []
                for j in jobs.values():
                    if j.get('done_at') and now - j['done_at'] > RETAIN and j.get('result_path'):
                        expired.append(j['result_path'])
                        j['result_path'] = None
                        if j.get('status') == 'done':
                            j['message'] = (j.get('message','') or '') + " · file auto-deleted"
            for rp in expired:
                try:
                    _server_delete_result(rp) if str(rp).startswith(str(PERSISTENT_OUTPUT_DIR)) else os.remove(rp)
                except Exception: pass
                with _reg_lock: _tmp_registry.discard(rp)
            # Remove expired authoritative server outputs even if the in-memory
            # job registry was lost during a Space restart.
            try:
                root = _persistent_output_dir()
                if root is not None:
                    for meta in root.glob("*/metadata.json"):
                        try:
                            import json
                            d = json.loads(meta.read_text(encoding="utf-8"))
                            if float(d.get("expires_at") or 0) <= now:
                                job_dir = meta.parent
                                shutil.rmtree(job_dir, ignore_errors=True)
                                logging.info("Expired server output deleted → %s", job_dir)
                        except Exception:
                            pass
            except Exception:
                pass
            with _reg_lock: _reg_snapshot = list(_tmp_registry)
            for fp in _reg_snapshot:
                try:
                    p = Path(fp)
                    if not p.exists():
                        with _reg_lock: _tmp_registry.discard(fp)
                    elif (now - p.stat().st_mtime > ORPHAN and not any(jid in p.name for jid in known)):
                        p.unlink()
                        with _reg_lock: _tmp_registry.discard(fp)
                except Exception: pass
            if cycle % 3:
                time.sleep(180)
                continue
            for pat in ["face_*","src_*","vid_*","raw_*","result_*","image_swap_*"]:
                for f in Path("/tmp").glob(pat):
                    try:
                        if any(jid in f.name for jid in known): continue
                        if now - f.stat().st_mtime > ORPHAN: f.unlink()
                    except Exception: pass
            gdir = Path(os.environ.get("GRADIO_TEMP_DIR","/tmp/gradio"))
            if gdir.exists():
                for f in gdir.rglob("*"):
                    try:
                        if f.is_file() and now - f.stat().st_mtime > ORPHAN: f.unlink()
                    except Exception: pass
            # Expired per-session dirs (audit NSDOS-016)
            try:
                sroot = Path("/tmp/swamitech_sessions")
                if sroot.is_dir():
                    for d in sroot.iterdir():
                        if not d.is_dir():
                            continue
                        sj = d / "session.json"
                        exp = None
                        if sj.is_file():
                            try:
                                import json
                                with open(sj) as f:
                                    exp = float((json.load(f) or {}).get("expires_at") or 0)
                            except Exception:
                                exp = None
                        mtime = d.stat().st_mtime
                        if (exp and now > exp) or (now - mtime > ORPHAN):
                            shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
        except Exception: pass
        time.sleep(180)

jobs, _lock = {}, threading.RLock()
# Bounded video workers (audit NSDOS-003) — do not spawn unbounded threads
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
VIDEO_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="phoenix-video")
_FFMPEG_ENCODER = None  # cached "libx264" / "h264_nvenc" probe
_JOB_FUTURES = {}

# Refresh-safe job snapshots. UI state is not the source of truth for jobs.
try:
    from config import PERSISTENT_OUTPUT_DIR, PERSISTENT_JOB_STATE_DIR, SERVER_OUTPUT_TTL_SEC
except Exception:
    PERSISTENT_OUTPUT_DIR = os.environ.get("PHOENIX_PERSISTENT_OUTPUT_DIR", "/data/phoenix_outputs")
    PERSISTENT_JOB_STATE_DIR = os.environ.get("PHOENIX_PERSISTENT_JOB_STATE_DIR", "/data/phoenix_jobs")
    SERVER_OUTPUT_TTL_SEC = 10800

# Prefer the mounted persistent volume for authoritative job state. Fall back
# to /tmp only when the volume is unavailable; this is explicitly logged so a
# deployment never silently claims persistence that it does not have.
_PERSIST_ROOT = Path(os.environ.get("PHOENIX_PERSISTENT_ROOT", PERSISTENT_JOB_STATE_DIR))
try:
    _PERSIST_ROOT.mkdir(parents=True, exist_ok=True)
    _PERSIST_OK = os.access(_PERSIST_ROOT, os.W_OK)
except Exception:
    _PERSIST_OK = False
if not _PERSIST_OK:
    _PERSIST_ROOT = Path("/tmp/swamitech_jobs")
    _PERSIST_ROOT.mkdir(parents=True, exist_ok=True)
    logging.warning("Persistent job volume unavailable; job metadata will be ephemeral at %s", _PERSIST_ROOT)
else:
    logging.info("Persistent job metadata directory: %s", _PERSIST_ROOT)
JOB_STATE_ROOT = _PERSIST_ROOT
_JOB_PERSIST_LAST = {}
_JOB_PERSIST_MIN_INTERVAL = 0.75
try:
    JOB_STATE_ROOT.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

def _job_state_path(jid):
    return JOB_STATE_ROOT / f"{jid}.json"

def _persist_job(jid, force=False):
    """Write a small atomic job snapshot without turning progress I/O into a
    per-frame bottleneck. Terminal state is always persisted immediately."""
    try:
        import json
        now = time.monotonic()
        with _lock:
            j = jobs.get(jid)
            if not j:
                return
            terminal = j.get("status") in ("done", "error", "cancelled")
            if not force and not terminal and now - _JOB_PERSIST_LAST.get(jid, 0.0) < _JOB_PERSIST_MIN_INTERVAL:
                return
            data = {k:v for k,v in j.items() if isinstance(v,(str,int,float,bool)) or v is None}
            _JOB_PERSIST_LAST[jid] = now
        tmp = _job_state_path(jid).with_suffix(".tmp")
        tmp.write_text(json.dumps(data, separators=(",",":")), encoding="utf-8")
        os.replace(tmp, _job_state_path(jid))
    except Exception as e:
        logging.debug("job snapshot failed for %s: %s", jid, e)

def _remove_job_snapshot(jid):
    try:
        _job_state_path(jid).unlink(missing_ok=True)
    except Exception:
        pass

def _restore_job_snapshots():
    """Restore job metadata after a process restart; never falsely resume work."""
    import json
    try:
        for fp in JOB_STATE_ROOT.glob("*.json"):
            try:
                data=json.loads(fp.read_text(encoding="utf-8"))
                jid=fp.stem
                if not isinstance(data,dict): continue
                old=data.get("status")
                if old not in ("done","error","cancelled"):
                    data.update(status="error", progress=int(data.get("progress") or 0),
                                 message="Processing interrupted by Space restart; please resubmit.",
                                 done_at=time.time(), eta_seconds=None)
                elif old == "done" and (not data.get("result_path") or not os.path.exists(data.get("result_path"))):
                    data.update(status="error", message="Completed result is no longer available after Space restart.", done_at=time.time())
                with _lock: jobs[jid]=data
            except Exception as e:
                logging.debug("job restore failed for %s: %s", fp, e)
    except Exception as e:
        logging.debug("job restore scan failed: %s", e)
_restore_job_snapshots()
threading.Thread(target=_cleanup, daemon=True).start()

try:
    import pyzipper
    _PYZIP_OK = True
except Exception:
    _PYZIP_OK = False

def _encrypt_zip(src_path, password, out_path):
    if not password:
        raise ValueError("no password supplied")
    # Prefer pyzipper — avoids password on process cmdline (audit NSDOS-013)
    if not _PYZIP_OK:
        raise RuntimeError("pyzipper is required for encrypted ZIP (install pyzipper)")
    with pyzipper.AESZipFile(out_path, 'w', compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES) as z:
        z.setpassword(password.encode('utf-8'))
        z.setencryption(pyzipper.WZ_AES, nbits=256)
        z.write(src_path, os.path.basename(src_path))
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError("encrypted archive was not created")
    return out_path


def _autosave_cfg():
    try:
        from config import (
            AUTO_SAVE_ENABLED,
            AUTO_SAVE_DIR,
            AUTO_SAVE_RETAIN_SEC,
            HF_OUTPUT_REPO,
            HF_UPLOAD_ENABLED,
            RETAIN_SEC,
            ORPHAN_SEC,
        )
        return {
            "enabled": bool(AUTO_SAVE_ENABLED),
            "dir": AUTO_SAVE_DIR or "/tmp/.swamitech_autosave",
            "retain": int(AUTO_SAVE_RETAIN_SEC),
            "hf_repo": HF_OUTPUT_REPO or "",
            "hf_upload": bool(HF_UPLOAD_ENABLED),
            "retain_hist": int(RETAIN_SEC),
            "orphan": int(ORPHAN_SEC),
        }
    except Exception:
        return {
            "enabled": True,
            "dir": "/tmp/.swamitech_autosave",
            "retain": 86400 * 3,
            "hf_repo": os.environ.get("SWAMITECH_HF_OUTPUT_REPO", "").strip(),
            "hf_upload": bool(os.environ.get("HF_TOKEN")) and bool(
                os.environ.get("SWAMITECH_HF_OUTPUT_REPO", "").strip()
            ),
            "retain_hist": 21600,
            "orphan": 86400,
        }



def _persistent_output_dir():
    """Return/create the authoritative server output directory."""
    root = Path(os.environ.get("PHOENIX_PERSISTENT_OUTPUT_DIR", PERSISTENT_OUTPUT_DIR))
    try:
        root.mkdir(parents=True, exist_ok=True)
        if not os.access(root, os.W_OK):
            raise PermissionError(f"not writable: {root}")
        return root
    except Exception as e:
        logging.warning("Persistent output volume unavailable: %s", e)
        return None

def _server_save_result(jid, result_path):
    """Atomically copy the completed result to authoritative server storage.

    This happens BEFORE the job is marked done. The Android client is therefore
    never the owner of the only completed copy. The server copy expires three
    hours after completion and is removed by the cleanup thread.
    """
    root = _persistent_output_dir()
    if root is None or not result_path or not os.path.isfile(result_path):
        raise RuntimeError("Persistent server storage is not available")
    job_dir = root / jid
    job_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(result_path).suffix.lower() or ".mp4"
    dest = job_dir / f"final{suffix}"
    tmp = job_dir / f"final{suffix}.part"
    shutil.copy2(result_path, tmp)
    if not tmp.is_file() or tmp.stat().st_size < 1000:
        try: tmp.unlink()
        except Exception: pass
        raise RuntimeError("Server-side result copy is invalid or empty")
    os.replace(tmp, dest)
    meta = job_dir / "metadata.json"
    now = time.time()
    try:
        import json
        meta.write_text(json.dumps({
            "job_id": jid,
            "completed_at": now,
            "expires_at": now + int(SERVER_OUTPUT_TTL_SEC),
            "result": dest.name,
            "size_bytes": dest.stat().st_size,
        }, separators=(",", ":")), encoding="utf-8")
    except Exception as e:
        logging.warning("Server result metadata write failed for %s: %s", jid, e)
    logging.info("Authoritative server save → %s · expires in %ds", dest, int(SERVER_OUTPUT_TTL_SEC))
    return str(dest), now + int(SERVER_OUTPUT_TTL_SEC)

def _server_delete_result(result_path):
    if not result_path:
        return
    try:
        p = Path(result_path)
        if p.exists(): p.unlink()
        if p.parent.name and p.parent.name not in ("/", "phoenix_outputs"):
            # Only remove an empty per-job directory under the configured root.
            root = Path(os.environ.get("PHOENIX_PERSISTENT_OUTPUT_DIR", PERSISTENT_OUTPUT_DIR)).resolve()
            try:
                if p.parent.parent.resolve() == root and not any(p.parent.iterdir()):
                    p.parent.rmdir()
            except Exception:
                pass
    except Exception as e:
        logging.debug("server result delete failed: %s", e)

def _auto_save_result(jid, result_path, password=""):
    cfg = _autosave_cfg()
    notes = []
    dest = None
    pw = (password or "").strip()
    if not cfg["enabled"] or not result_path or not os.path.isfile(result_path):
        return ""
    if not pw:
        return " · 💾 autosave skipped (set Password for encrypted ZIP)"

    try:
        out_dir = cfg["dir"]
        os.makedirs(out_dir, mode=0o700, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(out_dir, f"{stamp}_{jid[:8]}.zip")
        _encrypt_zip(result_path, pw, dest)
        notes.append("💾 encrypted autosave")
        logging.info("Encrypted auto-save → %s", dest)
    except Exception as e:
        logging.warning("Encrypted auto-save failed: %s", e)
        notes.append("autosave failed")
        dest = None

    if dest and cfg["hf_upload"] and cfg["hf_repo"]:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=os.environ.get("HF_TOKEN"))
            path_in_repo = f"outputs/{time.strftime('%Y/%m')}/{os.path.basename(dest)}"
            api.upload_file(
                path_or_fileobj=dest,
                path_in_repo=path_in_repo,
                repo_id=cfg["hf_repo"],
                repo_type="dataset",
            )
            notes.append(f"☁ hub:{cfg['hf_repo']}")
            logging.info("Uploaded encrypted zip to %s/%s", cfg["hf_repo"], path_in_repo)
        except Exception as e:
            logging.warning("HF auto-upload failed: %s", e)
            notes.append("hub upload failed")

    return (" · " + " · ".join(notes)) if notes else ""


_fa = _sw = None
_ov_active = False
_loaded_device = [None]  # "gpu" | "cpu"
_device_pref = ["cpu"]   # safe default
_gpu_try_cached = [None]

try:
    import spaces as _spaces
    HAS_SPACES = True
except Exception:
    _spaces = None
    HAS_SPACES = False


def _is_zerogpu_space():
    for key in (
        "SPACES_ZERO_GPU", "SPACE_ZERO_GPU", "ZERO_GPU",
        "HF_SPACES_ZERO_GPU", "SPACES_GPU", "SPACE_GPU",
    ):
        val = (os.environ.get(key) or "").strip().lower()
        if val in ("1", "true", "yes", "on"):
            return True
    hw = (
        os.environ.get("SPACE_HARDWARE")
        or os.environ.get("SPACES_HARDWARE")
        or os.environ.get("HARDWARE")
        or os.environ.get("SPACE_HARDWARE_TARGET")
        or ""
    ).strip().lower()
    if any(x in hw for x in ("zerogpu", "zero-gpu", "zero_gpu", "zero gpu")):
        return True
    if any(x in hw for x in ("a10g", "a100", "t4", "l4", "zero")):
        if "cpu" not in hw:
            return True
    return False


if HAS_SPACES:
    @_spaces.GPU(duration=60)
    def _spaces_gpu_ping():
        return True
else:
    def _spaces_gpu_ping():
        return False


def _cuda_available():
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    return False


def _gpu_worth_trying(prefer_gpu=True):
    if not prefer_gpu:
        return False
    if _gpu_try_cached[0] is False:
        return False
    if _cuda_available():
        return True
    if HAS_SPACES and _is_zerogpu_space():
        return True
    return False


def _device_status_text():
    pref = _device_pref[0]
    cuda = _cuda_available()
    zg = _is_zerogpu_space()
    active = _loaded_device[0] or "—"
    if pref == "jarvislabs":
        try:
            from jarvislabs_adapter import GOVERNOR as _JL_GOVERNOR
            snap = _JL_GOVERNOR.usage_snapshot()
            mode = (
                "Jarvislabs GPU (remote) — budget left: "
                f"{snap['day']['remaining']:.1f}h today, "
                f"{snap['week']['remaining']:.1f}h this week, "
                f"{snap['month']['remaining']:.1f}h this month"
            )
        except Exception as e:
            mode = f"Jarvislabs GPU (remote) — adapter unavailable ({str(e)[:60]})"
        return f"🖥 Device: {mode}"
    if pref == "cpu":
        mode = "CPU only (selected)"
    elif cuda:
        mode = "GPU active (CUDA attached)"
    elif zg and pref == "gpu":
        mode = "ZeroGPU Space — GPU attaches when job runs"
    elif pref == "gpu":
        mode = "CPU (no GPU on this Space — select CPU only)"
    else:
        mode = "CPU"
    extra = f" · models={active.upper()}" if _fa is not None else " · models not loaded yet"
    hw = " · ZeroGPU hw" if zg else " · CPU Space"
    return f"🖥 Device: {mode}{extra}{hw}"


def _get_providers(prefer_gpu=False):
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except Exception:
        return ["CPUExecutionProvider"]

    if prefer_gpu and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if "OpenVINOExecutionProvider" in available:
        return [
            (
                "OpenVINOExecutionProvider",
                {
                    "device_type": "CPU_FP32",
                    "num_of_threads": int(_native_threads),
                    "performance_hint": "LATENCY",
                },
            ),
            "CPUExecutionProvider",
        ]
    if "CPUExecutionProvider" in available:
        return ["CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


from collections import OrderedDict
import hashlib

_SRC_CACHE_MAX = 128
_MASK_CACHE_MAX = 512

class _LRUCache:
    def __init__(self, maxsize=128):
        self.maxsize = maxsize
        self._d = OrderedDict()
        self._lock = threading.Lock()
    def get(self, key, default=None):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
            return default
    def set(self, key, value):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
            self._d[key] = value
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)
    def __len__(self):
        return len(self._d)
    def clear(self):
        with self._lock:
            self._d.clear()

_src_face_cache = _LRUCache(_SRC_CACHE_MAX)
_mask_cache = _LRUCache(_MASK_CACHE_MAX)

# ---------------------------------------------------------------------------
# v11 "Aequus" aligned-space engine
# ---------------------------------------------------------------------------
import swap_engine as _E

try:
    from config import ENGINE_TUNABLES as _CFG_ENGINE
    _E.configure(**dict(_CFG_ENGINE or {}))
    logging.info("swap_engine %s configured from config.py", _E.ENGINE_VERSION)
except Exception as _e:
    logging.debug("engine tunables not overridden: %s", _e)

_COMPOSITOR = None


def _compositor():
    """Lazy AlignedCompositor bound to the currently loaded swapper."""
    global _COMPOSITOR
    if _COMPOSITOR is None or getattr(_COMPOSITOR, "swapper", None) is not _sw:
        _COMPOSITOR = _E.AlignedCompositor(_sw) if _sw is not None else None
    return _COMPOSITOR


def _img_sha(im):
    try:
        small = cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)
        return hashlib.sha256(small.tobytes()).hexdigest()
    except Exception:
        return None

def _cached_source_face(im):
    k = _img_sha(im)
    if k is not None:
        hit = _src_face_cache.get(k)
        if hit is not None:
            return hit
    fs = _fa.get(im)
    f = max(fs, key=_area) if fs else None
    if k is not None and f is not None:
        _src_face_cache.set(k, f)
    return f

RES = {
    "540p (Fastest)": (960, 540),
    "640p (Fast)": (1136, 640),
    "680p": (1208, 680),
    "720p (HD)": (1280, 720),
    "900p (HD+)": (1600, 900),
    "1080p (Full HD)": (1920, 1080),
}

DET_MAX_W = 720
# Use the detector at its normal 640x640 preparation size. The video frame is
# still downscaled for speed, with an adaptive high-resolution probe for small
# or profile faces. This is materially safer than a permanently tiny 256 detector.
DET_SIZE = (640, 640)
try:
    from config import DET_THRESH as _CFG_DET_THRESH
    DET_THRESH = float(_CFG_DET_THRESH)
except Exception:
    DET_THRESH = 0.32

def _make_det_frame(frm, max_w=DET_MAX_W):
    h, w = frm.shape[:2]
    if w <= max_w:
        return frm, 1.0, 1.0
    scale = max_w / float(w)
    nw, nh = max_w, max(2, int(round(h * scale)))
    det = cv2.resize(frm, (nw, nh), interpolation=cv2.INTER_AREA)
    return det, w / float(nw), h / float(nh)

def _scale_faces(faces, sx, sy):
    if sx == 1.0 and sy == 1.0:
        return faces
    for f in faces:
        f.bbox = f.bbox.astype(np.float32) * np.array([sx, sy, sx, sy], dtype=np.float32)
        if getattr(f, 'kps', None) is not None:
            f.kps = f.kps.astype(np.float32) * np.array([sx, sy], dtype=np.float32)
        lmk = getattr(f, 'landmark_2d_106', None)
        if lmk is not None and len(lmk):
            f.landmark_2d_106 = lmk.astype(np.float32) * np.array([sx, sy], dtype=np.float32)
    return faces


def _bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _box_center_dist_norm(b1, b2):
    """Normalized distance between two bounding box centers."""
    if b1 is None or b2 is None: return 999.0
    c1 = np.array([(float(b1[0])+float(b1[2]))*0.5, (float(b1[1])+float(b1[3]))*0.5])
    c2 = np.array([(float(b2[0])+float(b2[2]))*0.5, (float(b2[1])+float(b2[3]))*0.5])
    diag = max(1.0, float((float(b1[2])-float(b1[0]) + float(b1[3])-float(b1[1]))*0.5))
    return float(np.linalg.norm(c1 - c2) / diag)


def _motion_class(motion_val):
    if motion_val < 2.5: return "STATIC"
    if motion_val < 6.0: return "LOW"
    if motion_val < 14.0: return "MEDIUM"
    return "HIGH"

_FACE_STATIC_THR = 2.5
_FACE_ROI_PAD = 0.22
_FACE_FEATHER = 0.32
_FACE_EMA_ALPHA = 0.40

def _smooth_faces(faces, ema_state):
    """Deliberately a pass-through in v11.

    The old implementation kept an EMA list keyed by *sorted position*, so when
    a face entered or left the shot every subsequent face inherited a different
    face's history. It also never wrote the smoothed value back onto the face
    object, so the smoothing had no visual effect at all — only the identity
    cross-contamination was real.

    Geometry smoothing now lives in ``swap_engine.TrackState``, keyed by
    identity slot rather than by position, which is where it belongs.
    """
    if not faces:
        return faces, []
    return sorted(faces, key=lambda f: float(f.bbox[0])), []


def _pad_bbox(bbox, h, w, pad=_FACE_ROI_PAD):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad, bh * pad
    x1 = int(max(0, np.floor(x1 - px)))
    y1 = int(max(0, np.floor(y1 - py)))
    x2 = int(min(w, np.ceil(x2 + px)))
    y2 = int(min(h, np.ceil(y2 + py)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Per-output-frame geometry.
#
# Detection and the swap network run on key frames; the frames in between used
# to be filled by copying a face ROI out of a NEIGHBOURING frame and stamping
# it onto the current one. That is the origin of most of the visible defects:
#
#   * the stamped ROI carries the neighbour's head position, so during fast
#     movement the face lags the body and snaps forward on the next key frame;
#   * it also carries the neighbour's background inside the ellipse, and its
#     own exposure, so every transition between a real swap frame and a filled
#     one is a brightness step;
#   * two of the fill branches cross-faded whole FRAMES, which ghosts the
#     entire image whenever the camera or the subject moves;
#   * and several branches simply returned the untouched original frame, which
#     is the real face appearing for one to four frames.
#
# Instead, geometry is interpolated between key frames per identity slot, and
# every output frame is composited for real from the cached aligned swap - see
# swap_engine.AlignedCompositor.reuse(). Interpolating five keypoints is
# essentially free; it is the ONNX forward pass that is expensive, and that is
# the only thing still restricted to key frames.
# ---------------------------------------------------------------------------
def _touches_frame_edge(bbox, shape, margin_frac: float = 0.012) -> bool:
    """True if the box is in contact with a frame border.

    Containment alone does not catch someone walking out of shot: the detector
    reports the visible SLIVER of the face, which is a small box sitting
    entirely inside the frame, so it scores ~1.0 containment right up until the
    person is gone. Contact with the border is the signal that actually tracks
    "this face is on its way out".
    """
    if bbox is None or shape is None:
        return False
    try:
        H, W = int(shape[0]), int(shape[1])
        m = max(1.0, margin_frac * min(W, H))
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return bool(x1 <= m or y1 <= m or x2 >= W - m or y2 >= H - m)
    except Exception:
        return False


def _frame_containment(bbox, shape) -> float:
    """Fraction of ``bbox`` that lies inside the frame, 0..1."""
    if bbox is None or shape is None:
        return 1.0
    try:
        H, W = int(shape[0]), int(shape[1])
        x1, y1, x2, y2 = [float(v) for v in bbox]
        area = max(1.0, (x2 - x1) * (y2 - y1))
        ix = max(0.0, min(x2, W) - max(x1, 0.0))
        iy = max(0.0, min(y2, H) - max(y1, 0.0))
        return float(max(0.0, min(1.0, (ix * iy) / area)))
    except Exception:
        return 1.0


def _geom_record(face, track, alpha, guard, src, lmk=None):
    """Freeze the geometry a face will be RENDERED with, at detection time.

    Reading ``track.bbox`` / ``track.kps`` at swap time instead - which is what
    the pipeline used to do - reads shared, mutable tracker state that the
    detection loop has already advanced to the END of the chunk, because the
    whole chunk is detected before any swap runs. Every swapped frame in a
    chunk therefore rendered at the last key frame's head position. Snapshotting
    here binds the geometry to the frame it belongs to.
    """
    kps = getattr(track, "kps", None) if track is not None else None
    if kps is None:
        kps = getattr(face, "kps", None)
    bbox = getattr(track, "bbox", None) if track is not None else None
    if bbox is None:
        bbox = getattr(face, "bbox", None)
    if lmk is None:
        lmk = getattr(track, "lmk", None) if track is not None else None
        if lmk is None:
            lmk = getattr(face, "landmark_2d_106", None)
    if kps is None or bbox is None:
        return None
    return {
        "kps": np.asarray(kps, np.float32).copy(),
        "bbox": np.asarray(bbox, np.float32).reshape(4).copy(),
        "lmk": None if lmk is None else np.asarray(lmk, np.float32).copy(),
        "alpha": float(np.clip(alpha, 0.0, 1.0)),
        "guard": float(np.clip(guard, 0.0, 1.0)),
        "src": src,
        "track": track,
        # The last box the detector actually reported for this identity. The
        # smoothed/predicted bbox lags and stalls, so it is useless for asking
        # "was this face on its way out of frame?" - this is not.
        "hit_bbox": (None if track is None or getattr(track, "last_hit_bbox", None) is None
                     else np.asarray(track.last_hit_bbox, np.float32).copy()),
        # True when this came from a real detection rather than from the
        # tracker predicting forward. _geom_for_frame() prefers to interpolate
        # between two real observations: between them, interpolation is exact
        # for any motion the tracker can model, whereas extrapolation trails
        # the subject and then snaps at the next detection.
        "det": not bool(getattr(face, "predicted", False)),
    }


def _geom_lerp(a, b, t):
    """Blend two geometry records. ``t`` in [0,1], 0 = a, 1 = b."""
    t = float(np.clip(t, 0.0, 1.0))
    out = dict(a)
    out["kps"] = (a["kps"] * (1.0 - t) + b["kps"] * t).astype(np.float32)
    out["bbox"] = (a["bbox"] * (1.0 - t) + b["bbox"] * t).astype(np.float32)
    if a["lmk"] is not None and b["lmk"] is not None and a["lmk"].shape == b["lmk"].shape:
        out["lmk"] = (a["lmk"] * (1.0 - t) + b["lmk"] * t).astype(np.float32)
    else:
        out["lmk"] = b["lmk"] if t >= 0.5 else a["lmk"]
    out["alpha"] = float(a["alpha"] * (1.0 - t) + b["alpha"] * t)
    out["guard"] = float(a["guard"] * (1.0 - t) + b["guard"] * t)
    out["src"] = b["src"] if t >= 0.5 else a["src"]
    out["track"] = b["track"] if t >= 0.5 else a["track"]
    out["det"] = bool(a.get("det")) and bool(b.get("det"))
    out["hit_bbox"] = b.get("hit_bbox") if t >= 0.5 else a.get("hit_bbox")
    return out


def _geom_for_frame(timeline, g, taper, max_bracket=None):
    """Geometry for output frame ``g`` from a slot's key-frame timeline.

    ``timeline`` is an ordered list of ``(global_frame_index, record)``. A slot
    that is only anchored on one side (the face has just entered, or has just
    been lost) holds its last known geometry and fades out over ``taper``
    frames rather than disappearing between one frame and the next.

    Staleness (how long ago this identity was actually seen, which drives
    both the ``taper`` cutoff and the fade) is always measured against the
    last REAL detection, never against however recently the tracker merely
    EXTRAPOLATED one. That distinction is the fix for a reported "ghost face"
    defect: while a subject is turned away for longer than a brief occlusion,
    _carry_pairs() keeps producing a fresh-looking predicted entry on almost
    every detector call, for as long as its own, much larger miss budget
    (trk_max_missed) allows. Each of those entries is a NAIVE CONSTANT-
    VELOCITY extrapolation, which has no way to know it is wrong and simply
    keeps compounding once the subject's real motion stops being a straight
    line (a head turn, rolling over). An earlier version of this function
    measured staleness against the newest entry in the timeline regardless of
    whether it was real or extrapolated - so every fresh (but by then
    thoroughly wrong) predicted entry reset the fade to full alpha, and the
    face was rendered with high confidence at a position that had long since
    parted ways with the subject: a face floating in empty space, disconnected
    from any body. Reproduced directly: with that version, a synthetic 100
    frame "turned away" gap rendered at alpha=1.00 throughout with position
    error growing UNBOUNDED (350px+ and climbing). Anchoring staleness to the
    last real sighting instead caps both the exposure time and the drift.

    ``max_bracket`` bounds a DIFFERENT case: two real detections bracketing a
    gap (obs_lo and obs_hi both present). Interpolating between two real,
    confirmed points is normally safe regardless of gap length - both ends
    are true. It stops being safe once the gap is long enough that the
    subject's real path in between is no longer well approximated by a
    straight line - a turn, a roll, a round trip back to nearly the starting
    position. There is no way to detect that from the two endpoints alone: a
    round trip's average velocity looks identical to "barely moved" (measured
    directly - a 130-frame turn-and-back scored the same near-zero velocity
    mismatch as an 18-frame linear dropout). Frame count is therefore the
    only signal available, but it cannot be a fixed number of frames: under a
    sparse detection cadence (e.g. the Optimized preset's ~10-frame keyframe
    spacing) a routine one-second occlusion and a several-second deliberate
    turn-away can produce the SAME raw span - measured directly, an 18-frame
    real dropout produced brackets up to 30 frames wide purely from cadence
    spacing. `max_bracket` is therefore expressed in OUTPUT FRAMES already
    converted from a fixed TIME budget (seconds) at the call site, so it
    scales with fps/quality instead of being tuned against one preset's
    cadence and breaking on another. Beyond it, this falls back to the same
    near-edge hold and fade as the one-sided case, rather than a confident
    full-span interpolation.
    """
    if not timeline:
        return None

    def _bracket(entries):
        lo = hi = None
        for gi, rec in entries:
            if gi <= g:
                lo = (gi, rec)
            elif hi is None:
                hi = (gi, rec)
                break
        return lo, hi

    # Prefer a bracket made of real detections. A predicted anchor sitting
    # between two real ones only drags the interpolation toward the tracker's
    # lag; the real pair on either side describes the motion better.
    observed = [e for e in timeline if e[1].get("det")]
    obs_lo, obs_hi = _bracket(observed)

    if obs_lo is not None and obs_hi is not None:
        span = float(obs_hi[0] - obs_lo[0])
        # A short dip between two real detections is safe to interpolate
        # across in full: the two endpoints anchor it, and real motion over a
        # fraction of a second is well approximated by a straight line - this
        # is what keeps a brief detector dropout smooth. A LONG dip is not:
        # nothing constrains the subject's actual path in between, and once a
        # real detection eventually resumes, this branch previously bridged
        # however long that gap was with a confident, full-alpha straight
        # line - which is the reported "ghost" defect from a THIRD angle:
        # rather than drifting via extrapolation (the one-sided case above)
        # or lingering past a chunk boundary (the trim above), it glides in a
        # straight line between two real sightings while the subject's actual
        # motion in between - a turn, a roll - is anything but straight.
        #
        # Bounded the same way the one-sided case is bounded: render only
        # within `taper` frames of EITHER real endpoint, using a hold near
        # whichever endpoint is closer (not a blend across the unconstrained
        # middle) with the same quadratic fade. The deep middle of a long
        # gap renders nothing - the original frame - rather than a guess.
        budget = float(max_bracket) if max_bracket else float("inf")
        if span > budget:
            dist_lo = g - obs_lo[0]
            dist_hi = obs_hi[0] - g
            if dist_lo <= dist_hi:
                edge, side = dist_lo, obs_lo
            else:
                edge, side = dist_hi, obs_hi
            if taper > 0 and edge > taper:
                return None
            rec = dict(side[1])
            rec["det"] = False
            if taper > 0:
                rec["alpha"] = float(rec["alpha"] * max(0.0, 1.0 - (edge / float(taper)) ** 2))
            return rec if rec["alpha"] > 0.02 else None
        t = 0.0 if span <= 0 else (g - obs_lo[0]) / span
        return _geom_lerp(obs_lo[1], obs_hi[1], t)

    if obs_lo is not None:
        # obs_hi is None: this identity has not been seen for REAL since
        # obs_lo, and has not been confirmed again yet (an ongoing gap, not a
        # bracketed dip). taper/fade are computed from obs_lo - the last real
        # sighting - not from whatever the tracker most recently guessed.
        real_dist = g - obs_lo[0]
        if taper > 0 and real_dist > taper:
            return None
        # Still inside the short grace window: use the MOST RECENT entry
        # (which may be an extrapolated one, and is typically a better
        # position estimate for these few frames than the stale real
        # observation alone) for placement, but drive alpha from real_dist,
        # not from that entry's own recency - so a fresh extrapolation cannot
        # look "just seen" and stay at full opacity indefinitely.
        lo, _hi_all = _bracket(timeline)
        side = lo if lo is not None else obs_lo
        rec = dict(side[1])
        rec["det"] = False
        rec["alpha"] = float(rec["alpha"] * max(0.0, 1.0 - (real_dist / float(taper)) ** 2))
        return rec if rec["alpha"] > 0.02 else None

    # No real detection anywhere in this timeline yet - never established, or
    # history was trimmed past it. Fall back to whatever is available; this
    # is the pre-existing cold-start path and is unchanged.
    lo, hi = _bracket(timeline)
    if lo is not None and hi is not None:
        span = float(hi[0] - lo[0])
        t = 0.0 if span <= 0 else (g - lo[0]) / span
        return _geom_lerp(lo[1], hi[1], t)
    side = lo if lo is not None else hi
    if side is None:
        return None
    dist = abs(g - side[0])
    if taper > 0 and dist > taper:
        return None
    rec = dict(side[1])
    if dist > 0:
        rec["det"] = False
        rec["alpha"] = float(rec["alpha"] * max(0.0, 1.0 - (dist / float(taper)) ** 2))
    return rec if rec["alpha"] > 0.02 else None


def _rival_landmarks(records, slot):
    """Landmarks of the faces painted AFTER ``slot`` (i.e. nearer the camera).

    During a kiss or a hug the nearer person's cheek lands inside this face's
    aligned crop. Their chroma is essentially identical, so the skin-confidence
    guard cannot separate them - but their own landmarks can, and they project
    into this crop through the same affine.
    """
    out = []
    me = records.get(slot)
    if me is None:
        return out
    my_area = float(max(1.0, (me["bbox"][2] - me["bbox"][0]) * (me["bbox"][3] - me["bbox"][1])))
    for s2, r2 in records.items():
        if s2 == slot or r2 is None or r2.get("lmk") is None:
            continue
        area2 = float(max(1.0, (r2["bbox"][2] - r2["bbox"][0]) * (r2["bbox"][3] - r2["bbox"][1])))
        if area2 <= my_area:
            continue                      # painted before us; not an occluder
        if _bbox_iou(me["bbox"], r2["bbox"]) < 0.06:
            continue
        out.append(r2["lmk"])
    return out


def _session_probe(session):
    feed = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        feed[inp.name] = np.zeros(shape, dtype=np.float32)
    session.run(None, feed)

def _warmup(fa, sw):
    for _m in fa.models.values():
        _session_probe(_m.session)
    _session_probe(sw.session)

def _ort_cpu_load_context():
    """Temporarily tune ORT session construction for CPU-only loading.

    InsightFace constructs its own InferenceSession objects, so the usual
    SessionOptions knob is otherwise inaccessible. We deliberately use
    ORT_ENABLE_EXTENDED during model construction to avoid the very expensive
    full graph-optimization pass observed on CPU Spaces; inference remains
    fully CPU-backed. The patch is restored immediately after model loading.

    NEW (opt-in, off by default): if PHOENIX_ORT_OPTIMIZED_CACHE_DIR is set,
    the graph-optimized model is persisted there on first load, and loaded
    directly from that cached file on subsequent loads - skipping the
    EXTENDED-level optimization pass entirely on cache hits.

    CAVEAT - verify this actually helps on your Space before relying on it:
    a plain HF Space's /tmp (and most of its filesystem) is ephemeral and
    does NOT survive a full Space restart/rebuild - it only persists for
    the lifetime of the running container. This caching only pays off if
    _load() can be triggered more than once within the SAME container
    lifetime (uncommon here, since _fa/_sw are already cached in-process),
    or if PHOENIX_ORT_OPTIMIZED_CACHE_DIR points at your Space's Persistent
    Storage mount (a paid HF feature) rather than /tmp. If neither applies,
    this will build the cache once and never hit it again before the
    container recycles - i.e. no measurable benefit. Test with your actual
    restart pattern before assuming this helps.
    """
    try:
        import onnxruntime as ort
        original = ort.InferenceSession
        level = os.environ.get("PHOENIX_ORT_GRAPH_LEVEL", "extended").strip().lower()
        if level in ("all", "enable_all"):
            return ort, original, False

        cache_dir = os.environ.get("PHOENIX_ORT_OPTIMIZED_CACHE_DIR", "").strip()

        def _make_opts():
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            opts.intra_op_num_threads = int(_native_threads)
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            return opts

        so = _make_opts()

        def _tuned_session(path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
            opts = sess_options or so
            load_path = path_or_bytes
            cached_path = None

            if cache_dir and isinstance(path_or_bytes, str):
                os.makedirs(cache_dir, exist_ok=True)
                base = os.path.basename(path_or_bytes)
                cached_path = os.path.join(cache_dir, f"optimized_{base}")
                if os.path.isfile(cached_path):
                    # Cache hit: load the already-optimized graph directly,
                    # skip re-running EXTENDED optimization on it.
                    load_path = cached_path
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = int(_native_threads)
                    opts.inter_op_num_threads = 1
                    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                elif sess_options is None:
                    # Cache miss: run the normal optimization pass, but also
                    # write the result out for next time.
                    opts = _make_opts()
                    opts.optimized_model_filepath = cached_path

            return original(load_path, opts, providers=providers, provider_options=provider_options, **kwargs)

        ort.InferenceSession = _tuned_session
        return ort, original, True
    except Exception as e:
        logging.debug("ORT CPU session tuning unavailable: %s", e)
        return None, None, False

def _load(prefer_gpu=None):
    global _fa, _sw, _ov_active
    if prefer_gpu is None:
        prefer_gpu = (_device_pref[0] == "gpu")
    use_gpu = bool(prefer_gpu) and _cuda_available()
    target = "gpu" if use_gpu else "cpu"
    if _fa is not None and _loaded_device[0] == target:
        return

    from insightface.app import FaceAnalysis
    import insightface
    from huggingface_hub import hf_hub_download
    try:
        from config import FACE_MODEL_NAME, DET_SIZE, DET_SIZE_BALANCED, DET_SIZE_BEST
    except Exception:
        FACE_MODEL_NAME, DET_SIZE, DET_SIZE_BALANCED, DET_SIZE_BEST = "buffalo_l", (640,640), (512,512), (640,640)
    _ort_mod, _ort_original, _ort_patched = _ort_cpu_load_context() if not use_gpu else (None, None, False)
    _load_t0 = time.perf_counter()

    _fa = None
    _sw = None
    _loaded_device[0] = None

    prov = _get_providers(prefer_gpu=use_gpu)
    ctx_id = 0 if use_gpu else -1

    def _mk_fa(p, ctx):
        fa = FaceAnalysis(
            name=FACE_MODEL_NAME,
            providers=p,
            allowed_modules=["detection", "recognition", "landmark_2d_106"],
        )
        # v11: lower the detector acceptance threshold. Profile and distant
        # faces routinely score 0.3-0.5 with RetinaFace; at the 0.5 default they
        # are simply not returned, and "no detection" was the trigger for every
        # revert-to-original path. Accepting them and letting the tracker's
        # identity gating decide is strictly better -- a weak detection that the
        # tracker rejects costs nothing, a missed detection costs a visible
        # flash of the original face.
        try:
            det_size = DET_SIZE
            try:
                # CPU speed path: use the configured compact detector input.
                if not use_gpu and os.environ.get("PHOENIX_DET_SIZE", "").strip():
                    n = int(os.environ["PHOENIX_DET_SIZE"].split(",")[0])
                    det_size = (n, n)
            except Exception:
                pass
            fa.prepare(ctx_id=ctx, det_size=det_size, det_thresh=DET_THRESH)
        except TypeError:
            fa.prepare(ctx_id=ctx, det_size=DET_SIZE)
            try:
                fa.det_model.det_thresh = float(DET_THRESH)
            except Exception:
                pass
        return fa

    try:
        _fa = _mk_fa(prov, ctx_id)
    except Exception as e:
        if FACE_MODEL_NAME != "buffalo_l":
            logging.warning("%s load failed: %s — retrying buffalo_l", FACE_MODEL_NAME, e)
            FACE_MODEL_NAME = "buffalo_l"
            _fa = _mk_fa(prov, ctx_id)
        else:
            logging.warning(f"FaceAnalysis load failed ({prov}): {e} — falling back to CPU")
            prov = ["CPUExecutionProvider"]
            use_gpu = False
            target = "cpu"
            _fa = _mk_fa(prov, -1)

    _fp32 = [
        ("deepinsight/inswapper", "inswapper_128.onnx"),
        ("ezioruan/inswapper_128.onnx", "inswapper_128.onnx"),
        ("Devia/inswapper_128", "inswapper_128.onnx"),
    ]
    mp_ = None
    for repo, fn in _fp32:
        try:
            mp_ = hf_hub_download(repo, fn, cache_dir="/tmp/models")
            break
        except Exception:
            pass
    if mp_ is None:
        raise RuntimeError("Could not download inswapper_128.onnx")

    try:
        _sw = insightface.model_zoo.get_model(mp_, providers=prov)
        if use_gpu:
            try:
                _warmup(_fa, _sw)
            except Exception:
                pass
    except Exception as e:
        logging.warning(f"Swapper GPU load failed: {e} — CPU fallback")
        prov = ["CPUExecutionProvider"]
        use_gpu = False
        target = "cpu"
        _fa = _mk_fa(prov, -1)
        _sw = insightface.model_zoo.get_model(mp_, providers=prov)

    if _ort_patched and _ort_mod is not None:
        try:
            _ort_mod.InferenceSession = _ort_original
        except Exception:
            pass
        logging.info("ORT CPU session construction: %.1fs · graph=EXTENDED · intra=%d inter=1", time.perf_counter() - _load_t0, _native_threads)
    try:
        _live = _sw.session.get_providers()
        _ov_active = "OpenVINOExecutionProvider" in str(_live)
        logging.info(f"Models ready on {target} · providers={_live}")
    except Exception:
        _ov_active = False

    _loaded_device[0] = target

# Face Enhancers
_enhancers = {}
_enh_failed = set()

def _ensure_tv_shim():
    import sys
    if 'torchvision.transforms.functional_tensor' not in sys.modules:
        try:
            import torchvision.transforms.functional as _tvf
            import types
            _shim = types.ModuleType('torchvision.transforms.functional_tensor')
            _shim.rgb_to_grayscale = _tvf.rgb_to_grayscale
            sys.modules['torchvision.transforms.functional_tensor'] = _shim
        except Exception:
            pass



class ModelRegistry:
    """Thread-safe model holder (audit NSDOS-004)."""
    def __init__(self):
        self._lock = threading.RLock()
        self.face_analysis = None
        self.swapper = None
        self.device = None  # "gpu" | "cpu"

    def get(self, prefer_gpu=False):
        global _fa, _sw, _loaded_device, _ov_active
        with self._lock:
            target = "gpu" if (prefer_gpu and _cuda_available()) else "cpu"
            if (
                self.face_analysis is not None
                and self.swapper is not None
                and self.device == target
            ):
                _fa, _sw = self.face_analysis, self.swapper
                _loaded_device[0] = self.device
                return self.face_analysis, self.swapper
            # Load via existing _load path under registry lock
            _load(prefer_gpu=(target == "gpu"))
            self.face_analysis, self.swapper = _fa, _sw
            self.device = _loaded_device[0] or target
            return self.face_analysis, self.swapper


MODELS = ModelRegistry()


def _load_gfpgan():
    if "GFPGAN" in _enhancers: return _enhancers["GFPGAN"]
    if "GFPGAN" in _enh_failed: return None
    try:
        _ensure_tv_shim()
        from gfpgan import GFPGANer
        from huggingface_hub import hf_hub_download
        wp = None
        for repo, fn in [("leonelhs/gfpgan","GFPGANv1.4.pth"), ("gmk123/GFPGAN","GFPGANv1.4.pth")]:
            try:
                wp = hf_hub_download(repo, fn, cache_dir="/tmp/models")
                break
            except Exception:
                pass
        if not wp:
            _enh_failed.add("GFPGAN")
            return None
        model = GFPGANer(model_path=wp, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
        _enhancers["GFPGAN"] = model
        logging.info("GFPGAN loaded")
        return model
    except Exception as e:
        logging.warning(f"GFPGAN load failed: {e}")
        _enh_failed.add("GFPGAN")
        return None

def _load_realesrgan():
    if "Real-ESRGAN (face)" in _enhancers: return _enhancers["Real-ESRGAN (face)"]
    if "Real-ESRGAN (face)" in _enh_failed: return None
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        from huggingface_hub import hf_hub_download
        wp = None
        for repo, fn in [
            ("ai-forever/Real-ESRGAN", "RealESRGAN_x2.pth"),
            ("ai-forever/Real-ESRGAN", "RealESRGAN_x4.pth"),
        ]:
            try:
                wp = hf_hub_download(repo, fn, cache_dir="/tmp/models")
                break
            except Exception:
                pass
        if not wp:
            _enh_failed.add("Real-ESRGAN (face)")
            return None
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        upsampler = RealESRGANer(scale=2, model_path=wp, model=model, tile=0, tile_pad=10, pre_pad=0, half=False)
        _enhancers["Real-ESRGAN (face)"] = upsampler
        logging.info("Real-ESRGAN loaded")
        return upsampler
    except Exception as e:
        logging.warning(f"Real-ESRGAN load failed: {e}")
        _enh_failed.add("Real-ESRGAN (face)")
        return None

def _load_codeformer():
    if "CodeFormer" in _enhancers: return _enhancers["CodeFormer"]
    if "CodeFormer" in _enh_failed: return None
    try:
        from gfpgan import GFPGANer
        from huggingface_hub import hf_hub_download
        wp = None
        for repo, fn in [
            ("sczhou/CodeFormer", "codeformer.pth"),
            ("leonelhs/codeformer", "codeformer.pth"),
            ("gmk123/CodeFormer", "codeformer.pth"),
        ]:
            try:
                wp = hf_hub_download(repo_id=repo, filename=fn)
                if wp: break
            except Exception: continue
        if not wp:
            _enh_failed.add("CodeFormer")
            return None
        try:
            model = GFPGANer(model_path=wp, upscale=1, arch="CodeFormer", channel_multiplier=2, bg_upsampler=None)
        except TypeError:
            model = GFPGANer(model_path=wp, upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None)
        _enhancers["CodeFormer"] = model
        logging.info("CodeFormer loaded from %s", wp)
        return model
    except Exception as e:
        logging.warning("CodeFormer load failed: %s", e)
        _enh_failed.add("CodeFormer")
        return None


def _get_enhancer(name):
    if not name or name == "None": return None
    if name == "GFPGAN": return _load_gfpgan()
    if name == "Real-ESRGAN (face)": return _load_realesrgan()
    if name == "CodeFormer": return _load_codeformer()
    return None


def _is_soft_polish(name):
    if not name: return False
    s = str(name).lower()
    return ("soft polish" in s or "cinematic" in s
            or s in ("bilateral smooth", "mild sharpen", "detail boost")
            or s.startswith("soft"))


def _unsharp_luma(bgr, amount=0.55, radius=1.4, threshold=2):
    """Unsharp mask applied to luma only.

    Sharpening all three BGR channels independently amplifies chroma noise and
    produces coloured fringes on skin. Working in YCrCb and touching only Y
    keeps edge crispness without shifting skin tone.
    """
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y = ycc[:, :, 0].astype(np.float32)
    blur = cv2.GaussianBlur(y, (0, 0), radius)
    diff = y - blur
    # Soft noise gate, not a hard cutoff. A binary "abs(diff) < threshold ->
    # 0" step is discontinuous: a pixel whose true edge strength sits near
    # that boundary gets fully toggled on/off as ordinary frame-to-frame
    # compression noise nudges it across the line - each frame independently,
    # since this runs per-frame with no memory of the previous one. Measured
    # that toggling amplifying realistic frame-to-frame noise by ~1.6x on real
    # footage - visible as shimmer on skin texture even with the enhancer
    # itself fully brightness-neutral. Ramping gain from 0 at `threshold` to 1
    # at `2*threshold` keeps the same noise-suppression intent (small diffs
    # still end up near-zero) without a hard edge for noise to straddle.
    mag = np.abs(diff)
    gate = np.clip((mag - threshold) / max(threshold, 1e-6), 0.0, 1.0)
    diff = diff * gate
    ycc[:, :, 0] = np.clip(y + amount * diff, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def _fixed_contrast_lut(strength=0.07):
    """Deterministic S-curve LUT for the 'clarity' lift.

    Deliberately not CLAHE: CLAHE rebuilds a histogram per tile per frame, so
    near-identical consecutive frames get slightly different mappings, which
    reads as brightness pulsing on video. A fixed LUT always maps a given
    input value to the same output.

    Strength is kept low on purpose — see _local_contrast for why large
    luminance shifts are dangerous here.
    """
    x = np.arange(256, dtype=np.float32) / 255.0
    s = x - strength * np.sin(2.0 * np.pi * x)
    return np.clip(s * 255.0, 0, 255).astype(np.uint8)


_CONTRAST_LUT = None


def _local_contrast(bgr, clip=None, grid=None):
    """Brightness-preserving midtone contrast on L only.

    Critical for video: the enhancer runs only on frames where a swap actually
    happened, so with any frame-skip > 1 enhanced and unenhanced frames
    alternate. If enhancement changes mean luminance, that alternation shows up
    as the face pulsing brighter/darker several times a second. Rescaling L
    back to its original mean makes an enhanced frame photometrically
    interchangeable with an unenhanced one — contrast is redistributed, overall
    exposure is not touched.

    `clip`/`grid` are accepted and ignored (retained so existing CLAHE-era call
    sites keep working).
    """
    global _CONTRAST_LUT
    if _CONTRAST_LUT is None:
        _CONTRAST_LUT = _fixed_contrast_lut()
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]
    before = float(l.mean())
    curved = cv2.LUT(l, _CONTRAST_LUT).astype(np.float32)
    after = float(curved.mean())
    if after > 1.0:
        curved *= (before / after)          # restore original mean luminance
    lab[:, :, 0] = np.clip(curved, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _soft_polish(frame, mode="Soft polish (fast)"):
    """CPU-local enhancement family.

    Previously this ignored `mode` entirely and always ran the same bilateral
    filter — meaning 'Mild sharpen' and 'Detail boost' actually SOFTENED the
    image, and every option in this family looked identical. Each mode now
    does what its label says.
    """
    if frame is None or frame.size < 100: return frame
    s = str(mode or "").lower()
    try:
        h, w = frame.shape[:2]
        small = min(h, w) < 220
        d = 5 if small else 7

        if "cinematic" in s:
            # Smooth → clarity → crispness, in that order: denoise first so the
            # later two amplify real detail rather than amplifying noise.
            out = cv2.bilateralFilter(frame, d=d, sigmaColor=20, sigmaSpace=20)
            out = _local_contrast(out, clip=1.4, grid=8)
            out = _unsharp_luma(out, amount=0.32 if small else 0.40,
                                radius=1.2 if small else 1.5, threshold=3)
            # Hold back slightly from the original so the result reads as
            # graded rather than processed.
            return cv2.addWeighted(out, 0.85, frame, 0.15, 0)

        if "mild sharpen" in s:
            return _unsharp_luma(frame, amount=0.45, radius=1.2, threshold=3)

        if "detail boost" in s:
            out = _unsharp_luma(frame, amount=0.75, radius=1.6, threshold=2)
            return _local_contrast(out, clip=1.2, grid=8)

        # "Bilateral smooth" and "Soft polish (fast)" — original behaviour.
        return cv2.bilateralFilter(frame, d=d, sigmaColor=22, sigmaSpace=22)
    except Exception:
        return frame


def _enhance(frame, enhancer_name="GFPGAN"):
    if not enhancer_name or enhancer_name == "None": return frame
    if _is_soft_polish(enhancer_name): return _soft_polish(frame, enhancer_name)
    model = _get_enhancer(enhancer_name)
    if model is None: return _soft_polish(frame, "Soft polish (fast)")
    try:
        if enhancer_name in ("GFPGAN", "CodeFormer"):
            try:
                _, _, out = model.enhance(frame, has_aligned=False, only_center_face=False, paste_back=True, weight=0.7)
            except TypeError:
                _, _, out = model.enhance(frame, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5)
            return out if out is not None else frame
        if enhancer_name == "Real-ESRGAN (face)":
            out, _ = model.enhance(frame, outscale=1)
            if out is None: return frame
            if out.shape[:2] != frame.shape[:2]:
                out = cv2.resize(out, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
            return out
        return frame
    except Exception:
        return _soft_polish(frame, "Soft polish (fast)")


def _color_match_fast(swapped, original, bbox):
    x1,y1,x2,y2 = [max(0,int(v)) for v in bbox]
    x2,y2 = min(swapped.shape[1],x2), min(swapped.shape[0],y2)
    if x2<=x1 or y2<=y1: return swapped
    sw, og = swapped[y1:y2,x1:x2], original[y1:y2,x1:x2]
    if sw.size < 100 or og.size < 100: return swapped
    ms, mo = cv2.mean(sw), cv2.mean(og)
    if max(abs(ms[0]-mo[0]), abs(ms[1]-mo[1]), abs(ms[2]-mo[2])) < 3.0:
        return swapped
    s = cv2.cvtColor(sw, cv2.COLOR_BGR2LAB).astype(np.float32)
    o = cv2.cvtColor(og, cv2.COLOR_BGR2LAB)
    sm, ss = cv2.meanStdDev(s)
    om, osd = cv2.meanStdDev(o)
    # v11 FIX. This line used to read ``* 0.20``, which multiplied every pixel's
    # DEVIATION FROM THE MEAN by 0.2 — i.e. it crushed face contrast to a fifth
    # and pulled the whole crop to the bbox mean. On a side turn the bbox mean is
    # mostly hair/neck/background, so the face collapsed into a flat brown slab.
    # Measured on a synthetic crop: L std 19.8 -> 8.2, L mean 169 -> 121.
    # A true Reinhard transfer keeps the ratio; we only clamp it and blend.
    scale = np.clip((osd / (ss + 1e-6)).reshape(1, 1, 3), 0.72, 1.45).astype(np.float32)
    matched = (s - sm.reshape(1, 1, 3)) * scale + om.reshape(1, 1, 3)
    w = 0.55  # lerp toward the match instead of replacing outright
    s = s * (1.0 - w) + matched * w
    np.clip(s, 0, 255, out=s)
    swapped[y1:y2,x1:x2] = cv2.cvtColor(s.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return swapped

class _CancelledJob(Exception): pass


def _profile_safe_composite(res, orig, face, alpha=1.0):
    """Composite the swap through a face-shaped mask without fading back to original.

    The v10.9.3 mask deliberately reduced alpha to ~0.20 on side poses. That avoided
    rectangular paste-back artifacts, but it also created the exact symptom users see
    as "original face returning" or an original/replacement mix. The new compositor
    keeps replacement opacity high and changes the *mask shape*, not the identity
    strength, as pose changes. When 106-point landmarks are available, their convex
    hull follows an asymmetric/profile face much better than a centered ellipse.
    """
    if res is None or orig is None or face is None:
        return res
    if res.shape[:2] != orig.shape[:2]:
        return orig.copy()
    try:
        H, W = orig.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < 12 or bh < 12:
            return orig.copy()

        # A tight ROI prevents hair/background from being replaced.
        rx1 = max(0, int(x1 - bw * 0.10))
        ry1 = max(0, int(y1 - bh * 0.08))
        rx2 = min(W, int(x2 + bw * 0.10))
        ry2 = min(H, int(y2 + bh * 0.10))
        rw, rh = rx2 - rx1, ry2 - ry1
        if rw < 10 or rh < 10:
            return orig.copy()

        mask = np.zeros((rh, rw), np.uint8)
        lmk = getattr(face, "landmark_2d_106", None)
        used_landmarks = False
        if lmk is not None:
            try:
                pts = np.asarray(lmk, dtype=np.float32).reshape(-1, 2)
                pts[:, 0] -= rx1
                pts[:, 1] -= ry1
                valid = (pts[:, 0] >= -2) & (pts[:, 0] <= rw + 2) & (pts[:, 1] >= -2) & (pts[:, 1] <= rh + 2)
                pts = pts[valid]
                if len(pts) >= 8:
                    hull = cv2.convexHull(np.round(pts).astype(np.int32))
                    cv2.fillConvexPoly(mask, hull, 255)
                    used_landmarks = True
            except Exception:
                used_landmarks = False

        if not used_landmarks:
            cx, cy = rw // 2, rh // 2
            ax = max(5, int(rw * 0.40))
            ay = max(7, int(rh * 0.45))
            cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

        # A very small dilation closes tiny gaps around profile contours without
        # bringing the old rectangular paste-back edge back.
        dk = max(1, int(min(rw, rh) * 0.018))
        if dk > 0:
            kernel = np.ones((dk * 2 + 1, dk * 2 + 1), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        fk = max(5, (int(min(rw, rh) * 0.07) | 1))
        if fk % 2 == 0:
            fk += 1
        mask = cv2.GaussianBlur(mask, (fk, fk), 0).astype(np.float32) / 255.0

        # v11.2.1 SolidFace: pose changes mask geometry only — replacement stays
        # fully opaque when gates say YES (no 0.94 soft pose fade).
        final_alpha = float(np.clip(alpha, 0.0, 1.0))
        m = (mask * final_alpha)[..., None]

        a = res[ry1:ry2, rx1:rx2].astype(np.float32)
        b = orig[ry1:ry2, rx1:rx2].astype(np.float32)
        out = orig.copy()
        out[ry1:ry2, rx1:rx2] = (a * m + b * (1.0 - m)).astype(np.uint8)
        return out
    except Exception:
        return orig.copy()

_CM_STRENGTH = {"Fast": 0.70, "Balanced": 0.85, "Optimized": 0.90, "Best": 1.00, "Ultra": 1.00}


def _swap_one(work, orig, face, src_face, quality, alpha=1.0, *,
              rivals=None, frame_ord=0, post=None):
    """Swap and composite in ArcFace-aligned space. Returns (image, cached).

    ``cached`` reports whether the aligned crop was stored on the track, i.e.
    whether later frames can reuse it instead of falling back to the original.


    Everything — the mask, the colour statistics, the temporal EMAs — is built
    inside the 128x128 aligned crop that inswapper produces internally. That
    crop is pose-normalised by construction, so the mask has identical geometry
    whether the head is frontal or rolled 40 degrees, and colour statistics can
    never see hair, neck or background.

    That single change removes the two defects that image-space compositing
    kept reintroducing: the brown patch (bbox-wide colour stats contaminated by
    the background on a profile turn) and the shiny/rectangular patch (an
    axis-aligned or upright-elliptical mask over a rolled head).

    If the installed inswapper build will not return its affine matrix, we fall
    back to the legacy image-space path so the app still works.
    """
    if face is None or src_face is None:
        return work, False

    comp = _compositor()
    track = getattr(face, "_track", None)
    guard = float(getattr(face, "_occlusion_guard", 0.0) or 0.0)
    strength = _CM_STRENGTH.get(str(quality), 1.0)

    if comp is not None:
        try:
            out, ok = comp.run(
                work, orig, face, src_face,
                track=track, alpha=float(alpha),
                colour_strength=strength, occlusion_guard=guard,
                rivals=rivals, frame_ord=int(frame_ord), post=post,
            )
            if ok:
                return out, True
        except Exception as e:
            logging.debug("aligned compositor failed, falling back: %s", e)

    return _swap_one_legacy(work, orig, face, src_face, quality, alpha=alpha), False


def _reuse_one(work, orig, rec, quality, *, rivals=None, cached=None):
    """Composite an already-computed aligned swap onto THIS frame.

    Called for every output frame the swap network did not run on. Geometry,
    mask, background and colour match all come from the current frame; only
    the (expensive) aligned swap texture is reused.
    """
    comp = _compositor()
    if comp is None or rec is None:
        return work, False
    track = rec.get("track")
    if track is None:
        return work, False
    if cached is not None:
        track.fake, track.fake_corr, track.fake_size = cached[0], cached[1], int(cached[0].shape[0])
    if track.fake is None:
        return work, False
    face = _E.PredictedFace(rec["bbox"], rec["kps"], rec["lmk"], None, 0.5)
    face._track = track
    try:
        return comp.reuse(
            work, orig, rec["kps"], face,
            track=track, alpha=float(rec["alpha"]),
            colour_strength=_CM_STRENGTH.get(str(quality), 1.0),
            occlusion_guard=float(rec.get("guard", 0.0) or 0.0),
            rivals=rivals,
        )
    except Exception as e:
        logging.debug("aligned reuse failed: %s", e)
        return work, False


def _swap_one_legacy(work, orig, face, src_face, quality, alpha=1.0):
    """Image-space fallback for inswapper builds without an exposed affine."""
    try:
        raw = _sw.get(work, face, src_face, paste_back=True)
    except Exception:
        return work
    if raw is None:
        return work

    res = raw
    try:
        x1, y1, x2, y2 = [max(0, int(v)) for v in face.bbox]
        x2, y2 = min(orig.shape[1], x2), min(orig.shape[0], y2)
        if x2 > x1 and y2 > y1:
            roi0 = orig[y1:y2, x1:x2]
            if float(np.mean(roi0)) >= 55 and float(np.std(roi0)) >= 12:
                res = _color_match_fast(res, orig, face.bbox)
    except Exception:
        pass

    # v11.2.1 SolidFace: full-strength paste for all qualities (was Fast 0.92…).
    quality_alpha = {
        "Fast": 1.00, "Balanced": 1.00, "Optimized": 1.00, "Best": 1.00, "Ultra": 1.00,
    }.get(str(quality), 1.0)
    return _profile_safe_composite(res, orig, face,
                                   alpha=quality_alpha * float(alpha))

def _to_bgr(img): return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
def _area(f): return (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])

def _build_sources(imgs):
    smap = {}
    for i,im in enumerate(imgs):
        if im is None: continue
        f = _cached_source_face(_to_bgr(im))
        if f is not None:
            smap[i] = f
    return smap

def _parse_face_mode(mode):
    if mode is None: return 99
    s = str(mode).lower()
    if s.startswith("1") or "1 face" in s: return 1
    if s.startswith("2") or "2 face" in s: return 2
    return 99


def _limit_smap(smap, max_faces):
    if not smap or max_faces >= 99: return smap
    out = {}
    for i in range(4):
        if i in smap and len(out) < max_faces:
            out[i] = smap[i]
    if len(out) < max_faces:
        for k, v in smap.items():
            if k not in out and len(out) < max_faces:
                out[k] = v
    return out


def _frontal_score(face):
    kps = getattr(face, "kps", None)
    if kps is None or len(kps) < 5:
        return 0.5
    try:
        le, re, nose = kps[0], kps[1], kps[2]
        eye_dist = float(np.linalg.norm(le - re)) + 1e-6
        mid = (le + re) * 0.5
        offset = abs(float(nose[0] - mid[0])) / eye_dist
        return float(np.clip(1.0 - offset * 1.8, 0.0, 1.0))
    except Exception:
        return 0.5


def _pitch_score(face):
    """0..1. Low when the head is pitched down / away so the detector box
    is hair or skull, not a paintable face. Horizontal-only frontal_score
    cannot see this — that was the looking-down failure mode."""
    kps = getattr(face, "kps", None)
    if kps is None or len(kps) < 3:
        return 0.4
    try:
        le, re, nose = np.asarray(kps[0], np.float32), np.asarray(kps[1], np.float32), np.asarray(kps[2], np.float32)
        eye_dist = float(np.linalg.norm(le - re)) + 1e-6
        mid = (le + re) * 0.5
        vert = float(nose[1] - mid[1]) / eye_dist
        # Typical frontal: nose sits ~0.25-0.85 eye-distances below the eyes.
        # Looking down / back-of-head: nose collapses toward or above the eyes.
        if vert < 0.10:
            return 0.08
        if vert < 0.18:
            return 0.28
        if vert > 1.55:
            return 0.30
        return 1.0
    except Exception:
        return 0.4


def _bbox_drift_too_far(curr_bb, anchor_bb, limit=0.58) -> bool:
    """True when the predicted box has slid off the last real face
    (shoulder / arm / torso). Lying-down faces that stay on the head
    keep a small drift and must still swap."""
    if curr_bb is None or anchor_bb is None:
        return False
    try:
        c = np.asarray(curr_bb, np.float32)
        a = np.asarray(anchor_bb, np.float32)
        ch = max(8.0, float(a[3] - a[1]))
        cw = max(8.0, float(a[2] - a[0]))
        dx = abs(((c[0] + c[2]) * 0.5) - ((a[0] + a[2]) * 0.5)) / cw
        dy = abs(((c[1] + c[3]) * 0.5) - ((a[1] + a[3]) * 0.5)) / ch
        return (dx * dx + dy * dy) ** 0.5 > limit
    except Exception:
        return False


def _predicted_miss_budget() -> int:
    """Paste budget for predicted faces.

    v11.1.9 ReentrySafe capped this at a flat 10 frames regardless of
    detector cadence, reasoning "prefer skip over arm/body paste after
    exit." Measured against real footage (v11.2.2): that budget is spent
    in well under one detector cycle on Fast/Optimized cadence (SKIP_N of
    5-6 frames between detector calls, and ``missed`` advances by the
    frames actually elapsed, not by calls) - two ordinary, non-exit
    detector misses in a row exhausts it, dropping the paste to the
    original face for the rest of the clip until the next clean hit. That
    is what "arm/body paste after exit" degenerated into: not a rare edge
    case, ordinary cadence gaps.
    The genuine "face actually left the frame" case this budget was meant
    to guard is now caught directly by TrackState._paste_frozen (geometric
    containment against the real frame bounds, set in predict()) - that
    check runs BEFORE this budget is ever consulted (_face_swap_allowed
    returns False immediately while frozen), independent of frame count.
    So this can trust trk_max_missed again without reopening the arm-paste
    bug the flat cap was reacting to.
    """
    try:
        return int(_E._P.get("trk_max_missed", 24) or 24)
    except Exception:
        try:
            from config import ENGINE_TUNABLES as _ET
            return int((_ET or {}).get("trk_max_missed", 24) or 24)
        except Exception:
            return 24


def _kps_reliable(f) -> bool:
    """False when the 5 landmarks do not describe a paintable face.

    v11.1.10 HairGate: FAIL CLOSED when kps is missing (was fail-open
    ``return True`` — that let hair/skull detections with no usable landmarks
    sail into paste). Also rejects classic InsightFace motion-blur hair blobs:
    eye midline too low in the bbox along the face up-axis, or eye spacing
    tiny vs the box.

    This is the gate _pitch_score()'s own docstring promises ("not a
    paintable face") but that was never actually wired into a decision to
    skip painting - only into the occlusion-guard's mask trimming, which
    softens edges but cannot stop a misaligned paste. A live detection with
    det_score above the floor and 5 keypoints present sailed straight through
    _face_swap_allowed() regardless of what those keypoints described.

    Two independent signals, because either alone can be dodged:

      * _pitch_score()'s lowest bucket (<=0.10) is its own documented
        "looking down/away, box is hair or skull" case;
      * landmark_fit_error() is a direct geometric consistency check (do the
        5 points admit ANY single rigid pose), not a heuristic on where
        individual points sit, so it also catches configurations that do not
        trip the pitch/frontal thresholds.

    Thresholds were set with margin below the legitimate ceiling: swept over
    profile turns to yaw 0.9 and lying-down poses to +-95 degrees roll (every
    roll tested, since the vertical check below is roll-corrected), the
    worst legitimate case measured roll-corrected-vert=0.357 and
    fit_error=0.103. A synthetic "looking down" detector-confusion signature -
    the nose collapsed to/above the eye line, which is the actual failure
    reported: a rotated, misplaced patch specifically when the subject looks
    away - scored roll-corrected-vert<=0 and fit_error 0.47-0.59 at every
    severity and every roll tested, comfortably on the reject side of both
    thresholds (0.12 and 0.20 respectively).

    Fails toward REJECT on error, not toward "trust it": unlike most gates in
    this module (which fail open, because their failure mode is an
    unnecessary hold), a computation error here has the opposite failure
    mode - risking the exact visible corruption this function exists to
    prevent. A spurious hold is the already-accepted, designed-for fallback
    everywhere else in this engine; a bad paste is not.
    """
    kps = getattr(f, "kps", None)
    # v11.1.10: fail CLOSED — missing landmarks are not paintable.
    if kps is None or len(kps) < 5:
        return False
    try:
        le, re = np.asarray(kps[0], np.float32), np.asarray(kps[1], np.float32)
        nose = np.asarray(kps[2], np.float32)
        eye_vec = re - le
        eye_dist = float(np.linalg.norm(eye_vec)) + 1e-6

        # Roll-corrected vertical pitch signal. _pitch_score() computes this
        # along the IMAGE y-axis, which only means "up/down on the face" when
        # roll is near zero. At a genuine ~90 degree roll (lying down - a
        # pose this engine explicitly supports and was verified against) the
        # face's own vertical axis IS the image's horizontal axis, so the
        # image-axis version collapses toward zero and misreads a perfectly
        # good lying-down pose as "looking down, box is hair/skull". This is
        # the same class of blind spot the engine's own mask/enhancer
        # orientation code elsewhere already had to correct for with the
        # measured head roll - re-derived here from the eye line rather than
        # importing that machinery, to keep this check self-contained.
        roll = float(np.arctan2(eye_vec[1], eye_vec[0]))
        mid = (le + re) * 0.5
        rel = nose - mid
        c, sn = float(np.cos(-roll)), float(np.sin(-roll))
        vert = (rel[0] * sn + rel[1] * c) / eye_dist
        if vert < 0.12:
            return False

        # Hair/skull bbox: eye midline must not sit in the bottom third of
        # the detection box along the face's roll-corrected "up" axis.
        # Classic motion-blur hair blob puts eyes near the chin end of a
        # tall box that mostly covers scalp/hair - that signature already
        # trips the vert<0.12 check above in every measured case (see this
        # function's docstring: the confusion signature scores
        # roll-corrected-vert<=0), so this is a defensive second signal,
        # not the primary one. v11.2.2: loosened 0.45→0.65 after real
        # footage showed ordinary frontal/chin-down poses routinely place
        # the eye line at t=0.40-0.55 of an InsightFace detector box - the
        # tighter threshold was rejecting normal frames throughout most of
        # a clip, not just genuine hair/skull confusion, which is what
        # made the swap look absent/weak almost everywhere instead of only
        # during an actual hair sweep.
        bb = getattr(f, "bbox", None)
        if bb is not None:
            x1, y1, x2, y2 = [float(v) for v in np.asarray(bb, np.float32).reshape(4)]
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            # Features tiny vs box → hair clump / oversized scalp box.
            if eye_dist / min(bw, bh) < 0.12:
                return False
            # Face-down unit vector in image coords (nose direction from eyes).
            down = np.asarray([sn, c], np.float32)
            dn = float(np.linalg.norm(down)) + 1e-6
            down = down / dn
            corners = np.asarray(
                [[x1, y1], [x2, y1], [x1, y2], [x2, y2]], np.float32
            )
            projs = corners @ down
            p_min = float(np.min(projs))
            p_max = float(np.max(projs))
            p_eye = float(np.dot(mid, down))
            # t=0 at face-top (forehead end of box), t=1 at face-bottom (chin).
            t = (p_eye - p_min) / (p_max - p_min + 1e-6)
            # Reject only if eyes sit in the bottom third of the box.
            if t > 0.65:
                return False

        fit = _E.landmark_fit_error(kps, 128)
        # None means the fit could not even be attempted (a degenerate point
        # set) - itself evidence of an unreliable read, not a reason to pass.
        # In practice this branch is defensive: the only input that makes the
        # Umeyama fit degenerate (all 5 points coincident) already collapses
        # `rel` to zero and is rejected by the vert check above first.
        # v11.1.10 tightened 0.20 -> 0.15 "for live paint"; v11.2.2 reverts
        # that. Measured directly: a face whose detection box narrows (a
        # normal yaw turn - width shrinks, height does not) produces a
        # rising landmark_fit_error purely because a SIMILARITY transform
        # cannot separately rescale width and height to match the fixed-
        # aspect canonical template - nothing about the face itself became
        # less reliable. At 0.15 this fired continuously through an
        # ordinary profile-turn clip (13 of 28 detector calls rejected vs
        # 0 of 28 at 0.20), and each rejection is exactly what triggers
        # HairGate's reacquire-and-freeze path below - which then could not
        # collect two consecutive clean hits often enough to ever un-freeze,
        # dropping the paste to the original face for the rest of the clip.
        # 0.20 is this project's own previously-measured number, with the
        # stated margin above every legitimate pose this docstring already
        # swept (worst case 0.103) still intact.
        if fit is None or fit > 0.20:
            return False
        return True
    except Exception:
        return False


def _face_swap_allowed(f) -> bool:
    """Balanced gate.

    Live detections (including lying-down / 3-quarter) are trusted.
    Predicted / carried boxes are allowed through the engine miss budget so
    profile turns do not flash the original face. Prefer alpha fade in work()
    over hard reject until the miss budget is exhausted or geometry drifts
    far off the last real face (arm-paste failure).
    """
    if f is None or getattr(f, "bbox", None) is None:
        return False
    predicted = bool(getattr(f, "predicted", False))
    tr = getattr(f, "_track", None)
    missed = int(getattr(tr, "missed", 0) or 0) if tr is not None else (4 if predicted else 0)
    det = float(getattr(f, "det_score", 0.5) or 0.5)

    # v11.1.10 HairGate: while paste-frozen (exit OR reacquire confirmation),
    # disallow BOTH predicted and live paste — show original, never hair.
    if tr is not None and getattr(tr, "_paste_frozen", False):
        return False

    if not predicted:
        # Real detector hit. Keep lying-down and soft-profile faces.
        if det < 0.20:
            return False
        # Fail-closed landmark gate (was: missing kps allowed at det>=0.45).
        if not _kps_reliable(f):
            return False
        return True

    # Predicted (v11.1.9 ReentrySafe): hold briefly through a head turn, but
    # prefer skip over arm/body paste after exit / long miss. Soft alpha fade
    # in work() still tapers; hard-drop earlier than Continuum 1.1.8.
    miss_budget = _predicted_miss_budget()  # min(trk_max_missed, 10)
    if missed > miss_budget:
        return False
    anchor = None
    if tr is not None:
        anchor = getattr(tr, "last_hit_bbox", None)
        if anchor is None:
            anchor = getattr(tr, "obs_bbox", None)
    # 0.55 (was 0.72): prefer skip over arm-paste; user confirmed only this bug remains.
    if _bbox_drift_too_far(getattr(f, "bbox", None), anchor, limit=0.55):
        return False
    # A predicted box that has largely left the frame is not a face any more.
    # Without this the tracker happily extrapolates someone who walked out of
    # shot, and the compositor paints them onto the frame edge.
    # Containment 0.70 (was 0.55): stop paste sooner when leaving frame.
    shape = getattr(f, "_frame_shape", None)
    if shape is None and tr is not None:
        wh = getattr(tr, "_frame_wh", None)
        if wh is not None:
            shape = (int(wh[1]), int(wh[0]))  # (H, W)
    if shape is not None and _frame_containment(getattr(f, "bbox", None), shape) < 0.70:
        return False
    return True


def _render_alpha(f, track):
    """Composite opacity for one face, smoothed on the track.

    v11.2.1 SolidFace policy (binary paste):
      gates NO  → caller skips / shows original (unchanged HairGate/ReentrySafe)
      gates YES → composite at full strength (~1.0), never a soft mix with original
                  during stable tracking.
    """
    target = 1.0
    predicted = bool(getattr(f, "predicted", False))
    if predicted and track is not None:
        missed = int(getattr(track, "missed", 0) or 0)
        budget = max(8, _predicted_miss_budget())
        # Only fade in the last 2 frames of the miss budget; floor 0.94 (was 0.70).
        fade_start = max(0, budget - 2)
        denom = max(1.0, float(budget - fade_start))
        target = float(np.clip(1.0 - max(0, missed - fade_start) / denom, 0.94, 1.0))
    else:
        # Live detection already passed gates (det << 0.20 never reaches here).
        target = 1.0
    # v11.2.1: confirm soft ease / _first_confirm_soft caps removed — full strength.
    if track is not None:
        return float(track.smooth_alpha(target))
    return target


def _face_looks_marginal(f) -> bool:
    """True when a freshly detected face is small, low-confidence, or
    non-frontal enough that its detected region likely includes some
    non-face content - a hand or object over part of the face, hair at
    a steep turn-away angle, etc. Reuses the exact thresholds already
    tuned for the high-resolution detection probe above rather than
    inventing new ones, so this doesn't add a second, uncoordinated
    notion of "marginal" to the codebase.
    """
    try:
        return (
            _area(f) < 4500.0
            or min(float(f.bbox[2] - f.bbox[0]), float(f.bbox[3] - f.bbox[1])) < 58.0
            or float(getattr(f, "det_score", 0.5) or 0.5) < 0.28
            # 0.22 (was 0.38): moderate profile / 3-quarter (0.25–0.38) is
            # legitimate single-face content; treating it as marginal over-trimmed
            # via occlusion_guard and looked like flicker. Keep this for severe
            # edge-on / degenerate detections only.
            or _frontal_score(f) < 0.22
            or _pitch_score(f) < 0.28
        )
    except Exception:
        return False


def _persistent_track_pairs(faces, smap, refs, tracker, max_faces, frame_shape=None,
                            dt_frames=1.0):
    """Associate detections to replacement slots via the v11 MultiFaceTracker.

    Replaces the old greedy slot-by-slot loop. Three properties matter here:

    * assignment is globally optimal, so slot #0 can no longer grab slot #1's
      face merely because it was iterated first;
    * an established slot keeps its binding unless a rival is better by a real
      margin for several consecutive frames, so a 1-3 frame embedding wobble on
      a profile turn cannot cause a label flip;
    * while two tracks overlap (a hug or a kiss) embeddings are distrusted
      entirely and motion decides, because ArcFace is least reliable exactly
      when two faces are cheek to cheek.

    Each returned face carries the ``_track`` that owns its temporal state, so
    the compositor can apply that identity's mask and colour EMAs.
    """
    if not smap:
        return []
    if not isinstance(tracker, _E.MultiFaceTracker):
        return []

    ref_map = {}
    for j, src in smap.items():
        r = refs[j] if refs and j < len(refs) and refs[j] is not None else None
        if r is None:
            r = getattr(src, "normed_embedding", None)
        ref_map[j] = r

    assigned = tracker.assign(list(faces or []), ref_map, dt_frames=dt_frames)

    # v11.1.9: stamp frame size onto every live track so predict() can freeze
    # once the bbox leaves the shot.
    if frame_shape is not None:
        try:
            _fh, _fw = int(frame_shape[0]), int(frame_shape[1])
            for _tr in tracker.tracks.values():
                if _tr is not None:
                    _tr._frame_wh = (_fw, _fh)
        except Exception:
            pass

    pairs = []
    for slot in sorted(assigned.keys()):
        if slot not in smap:
            continue
        f = assigned[slot]
        tr = tracker.tracks.get(slot)
        try:
            f._track = tr
            f._slot = slot
            if frame_shape is not None:
                f._frame_shape = (int(frame_shape[0]), int(frame_shape[1]))
            # Phase 1: Add boundary confidence penalty to occlusion guard
            is_marginal = _face_looks_marginal(f)
            boundary_conf = 1.0
            if HAS_PHASE1 and frame_shape is not None:
                boundary_conf = validate_detection_confidence(
                    f.bbox, frame_shape,
                    getattr(f, "landmark_2d_106", None)
                )
            # Occlusion guard for multi-face crossings / frame-boundary clips.
            # Moderate profile alone must NOT trip the guard on single-face
            # (that over-trimmed and looked like flicker). is_marginal still
            # applies when tracks are crossing or multiple slots are live.
            # n_live used to count tracker.tracks entries, which are created
            # for every slot up front and are never None - so it was a constant
            # equal to the slot count, and `multi_or_cross` was simply always
            # True whenever more than one face was configured. Count tracks
            # that are actually live.
            n_live = 0
            try:
                n_live = sum(1 for _t in tracker.tracks.values()
                             if _t is not None and _t.established
                             and int(getattr(_t, "missed", 0) or 0) <= 2)
            except Exception:
                n_live = 0
            multi_or_cross = bool(tr is not None and tr.crossing) or n_live >= 2

            # Continuous, not boolean. A guard that snaps between 0 and 1
            # changes the mask silhouette - and therefore the colour statistics
            # weighted by that mask - in a single frame, so each toggle moved
            # both the outline and the brightness. TrackState.ramp_occlusion()
            # slews it instead.
            want = 0.0
            if tr is not None and tr.crossing:
                want = max(want, 1.0)
            if is_marginal and multi_or_cross:
                want = max(want, 0.75)
            if boundary_conf < 0.75:
                want = max(want, float(np.clip((0.75 - boundary_conf) / 0.35, 0.0, 1.0)))
            f._occlusion_guard = (tr.ramp_occlusion(want) if tr is not None else want)
        except Exception:
            pass
        # v11.1.10: gate live too — frozen reacquire must not paint hair.
        if not _face_swap_allowed(f):
            continue
        pairs.append((f, smap[slot]))
        if len(pairs) >= max_faces:
            break
    return pairs


def _carry_pairs(tracker, smap, max_faces, advance=True, dt_frames=1.0):
    """Swap pairs for slots the detector did not see this frame.

    This is what replaces "fall back to the original frame". A single original
    frame dropped between swapped frames is the most visible artefact a face
    swap can produce — it reads as the real face flashing back. A predicted
    face still carries valid 5-point kps, which is all inswapper needs, so the
    swap keeps running through detector gaps and short occlusions instead of
    blinking. Prediction is bounded: once a track has been missing too long,
    carry stops rather than hallucinating a face indefinitely.
    """
    if not smap or not isinstance(tracker, _E.MultiFaceTracker):
        return []
    if advance:
        for tr in tracker.tracks.values():
            tr.predict(dt_frames)
    carried = tracker.carry(slots=set(smap.keys()))
    pairs = []
    for slot in sorted(carried.keys()):
        pf = carried[slot]
        tr = tracker.tracks.get(slot)
        try:
            pf._track = tr
            pf._slot = slot
            pf._occlusion_guard = tr.ramp_occlusion(0.85) if tr is not None else 0.85
            # v11.1.9: attach frame shape so containment / drift gates work on
            # predicted faces (TrackState._frame_wh set on assign/bind).
            if tr is not None and getattr(tr, "_frame_wh", None) is not None:
                _W, _H = tr._frame_wh
                pf._frame_shape = (int(_H), int(_W))
        except Exception as e:
            logging.warning("could not tag carried face for slot %s: %s", slot, e)
        # Predicted geometry is only used for a short detector blink.
        # Longer gaps / looking-down / out-of-frame must NOT paste a face.
        if not _face_swap_allowed(pf):
            continue
        pairs.append((pf, smap[slot]))
        if len(pairs) >= max_faces:
            break
    return pairs


def _pairs_for_frame(
    faces,
    smap,
    refs,
    max_faces=99,
    prev_bbox=None,
    frame_bgr=None,
    prev_bboxes=None,
    locked_emb=None,
):
    """Assign detected target faces to replacement slots without identity cross-over.

    For 2+ faces, identity similarity is deliberately dominant and ambiguous/low-
    confidence assignments are rejected rather than guessing. ``prev_bboxes`` is
    ordered by replacement slot, not by detector/face order.
    """
    if not faces or not smap:
        return []
    smap = _limit_smap(smap, max_faces)
    max_faces = min(int(max_faces), max(1, len(smap)))
    faces = sorted(faces, key=lambda f: float(f.bbox[0]))

    if max_faces == 1:
        only = smap.get(0) or next(iter(smap.values()))
        best = None
        ref0 = locked_emb if locked_emb is not None else (refs[0] if refs and len(refs) > 0 else None)
        if ref0 is not None and prev_bbox is not None:
            def _score(f):
                sim = float(np.dot(f.normed_embedding, ref0))
                iou = _bbox_iou(prev_bbox, f.bbox)
                cdist = _box_center_dist_norm(prev_bbox, f.bbox)
                spatial = max(iou, max(0.0, 1.0 - cdist * 0.75))
                return sim * 0.60 + spatial * 0.40
            ranked = sorted(faces, key=_score, reverse=True)
            if _score(ranked[0]) >= 0.10:
                best = ranked[0]
        elif ref0 is not None:
            ranked = sorted(faces, key=lambda f: float(np.dot(f.normed_embedding, ref0)), reverse=True)
            # No prev_bbox to lean on (before the first successful bind) -
            # identity similarity alone has to clear a real bar. Unbounded
            # here meant this branch, like the _open_score fallback below,
            # would hand the swap to whatever face-shaped thing the detector
            # found with the highest (however low) similarity - see the
            # comment on `if ref0 is not None:` below for why that matters
            # once an identity exists.
            if float(np.dot(ranked[0].normed_embedding, ref0)) >= 0.10:
                best = ranked[0]
        elif prev_bbox is not None:
            ranked = sorted(faces, key=lambda f: _box_center_dist_norm(prev_bbox, f.bbox))
            if _box_center_dist_norm(prev_bbox, ranked[0].bbox) < 0.70:
                best = ranked[0]
        if best is None:
            if ref0 is not None:
                # An identity IS already established, but nothing this frame
                # cleared minimum confidence against it. The old behaviour
                # fell through to _open_score below regardless - a fallback
                # meant for the one-time cold-start bootstrap (no reference
                # to check against yet), which does not look at identity or
                # position AT ALL, only how frontal/confident/large a
                # detection is. Once ref0 exists, that fallback will happily
                # hand the swap to ANY OTHER face-shaped thing the detector
                # reports - a false-positive detection elsewhere in frame,
                # a hand, a face-shaped shadow - whenever the real face is
                # merely at a hard angle or motion-blurred for a moment and
                # scores below the bar. Measured directly against a reported
                # clip: this is what put the swap on a spurious detection
                # near a pillow while the subject's real, unmodified face
                # kept showing normally a few inches away - not a duplicate
                # paste, one identity painted in the wrong place because
                # nothing here required it to actually BE that identity.
                # No reliable candidate this frame is not a reason to guess
                # at the most face-like blob in frame; it is exactly the
                # existing "no detection" case, which already holds the last
                # good geometry and fades rather than painting a guess.
                return []
            def _open_score(f):
                area = _area(f)
                front = _frontal_score(f)
                det = float(getattr(f, "det_score", 0.5) or 0.5)
                return front * 0.45 + det * 0.30 + min(area / 80000.0, 1.0) * 0.25
            best = max(faces, key=_open_score)
        return [(best, only)]

    # Multi-face: slot identity is tied to the target embedding captured in the UI.
    # Never use detector order as identity. This is the key protection against the
    # first replacement appearing on the second person when people cross/occlude.
    src_items = []
    slot_prev = list(prev_bboxes) if prev_bboxes else []
    for j, src in smap.items():
        ref_e = refs[j] if refs and j < len(refs) and refs[j] is not None else getattr(src, "normed_embedding", None)
        src_items.append((j, src, ref_e))
    if not src_items:
        return []

    n_f, n_s = len(faces), len(src_items)
    cost = [[1e6] * n_s for _ in range(n_f)]
    sims = [[-1.0] * n_s for _ in range(n_f)]
    spatials = [[0.0] * n_s for _ in range(n_f)]

    for fi, f in enumerate(faces):
        f_emb = getattr(f, "normed_embedding", None)
        for si, (j, src, ref_e) in enumerate(src_items):
            sim = float(np.dot(f_emb, ref_e)) if f_emb is not None and ref_e is not None else -1.0
            sims[fi][si] = sim
            pb = slot_prev[si] if si < len(slot_prev) else None
            if pb is None and si == 0 and prev_bbox is not None:
                pb = prev_bbox
            if pb is not None:
                iou = _bbox_iou(pb, f.bbox)
                cdist = _box_center_dist_norm(pb, f.bbox)
                spatial = max(iou, max(0.0, 1.0 - cdist * 0.75))
            else:
                spatial = 0.0
            spatials[fi][si] = spatial
            # A target reference is authoritative. Reject weak identity matches.
            # The spatial term helps through head turns without being allowed to
            # override a clearly better identity match.
            # Profile / partial faces crush ArcFace similarity. If the box
            # is still on the same head as last frame, keep the assignment.
            if ref_e is not None and sim < 0.18 and spatial < 0.22:
                continue
            cost[fi][si] = 0.50 * (1.0 - max(sim, -1.0)) + 0.44 * (1.0 - spatial) + 0.06 * (1.0 - spatial)

    # Same solver as the video tracker: exhaustive over column choices only,
    # which visits each distinct assignment once instead of k! times over.
    assign = sorted(_E.optimal_assignment(cost).items())
    pairs = []
    used_src, used_face = set(), set()
    for fi, si in assign:
        if fi in used_face or si in used_src:
            continue
        if cost[fi][si] >= 1e5:
            continue
        j, src, ref_e = src_items[si]
        sim = sims[fi][si]

        # Ambiguity guard: if another slot has a materially better identity score,
        # do not let Hungarian's spatial tie-breaker steal this face.
        if ref_e is not None:
            other_sims = [sims[fi][sj] for sj in range(n_s) if sj != si and sims[fi][sj] > -1.0]
            best_other = max(other_sims) if other_sims else -1.0
            if (sim < 0.16 and spatials[fi][si] < 0.22) or (best_other > -1.0 and sim + 0.050 < best_other and spatials[fi][si] < 0.40):
                continue

        pairs.append((faces[fi], src))
        used_src.add(si)
        used_face.add(fi)
        if len(pairs) >= max_faces:
            break

    # IMPORTANT: no area-based fallback in multi-face mode. Guessing here is what
    # causes one person's replacement to jump onto the other person's face.
    return pairs[:max_faces]

def _crop_face(img_bgr, face, pad=0.32, size=140):
    x1,y1,x2,y2 = face.bbox.astype(int)
    w,h = x2-x1, y2-y1
    px,py = int(w*pad), int(h*pad)
    cx1,cy1 = max(0,x1-px), max(0,y1-py)
    cx2,cy2 = min(img_bgr.shape[1],x2+px), min(img_bgr.shape[0],y2+py)
    crop = img_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0: return None
    return cv2.cvtColor(cv2.resize(crop,(size,size)), cv2.COLOR_BGR2RGB)

def _annotate(img, faces):
    prev = img.copy()
    for i,f in enumerate(faces):
        x1,y1,x2,y2 = f.bbox.astype(int)
        cv2.rectangle(prev,(x1,y1),(x2,y2),(176,88,21),3)
        cv2.putText(prev, f"#{i+1}", (x1, max(22,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, .9, (176,88,21), 2)
    return cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)

def _detect_return(img_bgr, faces):
    faces = sorted(faces, key=lambda f: f.bbox[0])[:4]
    refs = [f.normed_embedding for f in faces] + [None]*(4-len(faces))
    crops = [_crop_face(img_bgr, f) for f in faces] + [None]*(4-len(faces))
    msg = f"Found {len(faces)} face(s) — each detected face is shown below; upload its replacement in the box beside it."
    return _annotate(img_bgr, faces), msg, refs, crops[0], crops[1], crops[2], crops[3]

def _grab_frame(video, pos_pct):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = int(max(0, min(total-1, round((pos_pct/100.0)*(total-1)))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frm = cap.read(); cap.release()
    return (frm if ok else None), idx, total

def show_frame(video, pos_pct):
    if video is None: return None, "Upload a target video first"
    _load()
    frm, idx, total = _grab_frame(video, pos_pct)
    if frm is None: return None, "Could not read that frame"
    try:
        det_frm, sx, sy = _make_det_frame(frm, min(DET_MAX_W, 640))
        raw = _fa.get(det_frm) or []
        faces = _scale_faces(raw, sx, sy) if raw else []
    except Exception:
        faces = _fa.get(frm) or []
    return _annotate(frm, sorted(faces, key=lambda f: f.bbox[0])), \
           f"Frame {idx}/{total} · {len(faces)} face(s)"

def capture_face(video, pos_pct, slot_idx, refs_state):
    refs_state = list(refs_state) if refs_state else [None]*4
    while len(refs_state) < 4: refs_state.append(None)
    if video is None: return refs_state, None, "Upload a target video first"
    _load()
    frm, idx, total = _grab_frame(video, pos_pct)
    if frm is None: return refs_state, None, "Could not read that frame"
    try:
        det_frm, sx, sy = _make_det_frame(frm, min(DET_MAX_W, 640))
        raw = _fa.get(det_frm) or []
        faces = _scale_faces(raw, sx, sy) if raw else []
    except Exception:
        faces = _fa.get(frm) or []
    if not faces: return refs_state, None, f"No face found at frame {idx}"
    face = max(faces, key=_area)
    refs_state[slot_idx] = face.normed_embedding
    return refs_state, _crop_face(frm, face), f"✓ Captured face into slot #{slot_idx+1}"

def detect_image(target):
    if target is None: return None, "Upload a target image first", [], None, None, None, None
    _load()
    img = _to_bgr(target)
    faces = _fa.get(img)
    if not faces: return None, "No faces detected", [], None, None, None, None
    return _detect_return(img, faces)

def detect_video(video):
    if video is None:
        return None, "Upload a target video first", [], None, None, None, None
    _load()
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return None, "Could not open video", [], None, None, None, None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    probes = [0]
    if total > 5: probes.append(min(total - 1, max(1, int(fps * 1.0))))
    if total > 20: probes.append(min(total - 1, max(2, int(fps * 3.0))))
    seen = set()
    probes = [i for i in probes if not (i in seen or seen.add(i))]

    best, best_faces = None, []
    for fi in probes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frm = cap.read()
        if not ok or frm is None: continue
        try:
            det_frm, sx, sy = _make_det_frame(frm, min(DET_MAX_W, 640))
            raw = _fa.get(det_frm) or []
            fs = _scale_faces(raw, sx, sy) if raw else []
        except Exception:
            fs = _fa.get(frm) or []
        if len(fs) > len(best_faces):
            best_faces, best = fs, frm
        if best_faces: break
    cap.release()
    if not best_faces:
        return None, "No faces detected in the first frames", [], None, None, None, None
    return _detect_return(best, best_faces)

def swap_image(target, s1, s2, s3, s4, quality, refs):
    try:
        _load()
        if target is None: return None, "❌ Upload a target image"
        smap = _build_sources([s1,s2,s3,s4])
        if not smap: return None, "❌ Upload at least one replacement face"
        work = _to_bgr(target)
        orig = work if quality == "Fast" else work.copy()
        faces = _fa.get(work)
        if not faces: return None, "❌ No face detected"
        # Same reliability gate as the video path: a face whose 5 keypoints do
        # not describe a paintable pose is left as the original rather than
        # swapped, since a still image has no "hold the last good frame"
        # fallback to fall back to. See _kps_reliable() for why this check
        # exists - it is the case behind the "face pasted at the wrong angle
        # when looking away" report.
        faces = [f for f in faces if _kps_reliable(f)]
        if not faces: return None, "❌ No reliably-aligned face detected"
        pairs = _pairs_for_frame(faces, smap, refs or [], frame_bgr=work)
        for f, src in pairs:
            work, _ = _swap_one(work, orig, f, src, quality)
        if quality in ("Best", "Ultra"):
            work = _enhance(work, "Soft polish (fast)")
        out = f"/tmp/image_swap_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(out, work, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return out, f"✓ Done — {len(faces)} face(s)"
    except Exception as e:
        return None, f"❌ {str(e)[:90]}"

# CPU-only speed tiers (base intervals). Prefer config.SKIP_N so UI docs and
# runtime stay aligned; _adaptive_swap_gap() still stretches/shrinks by motion.
# Best/Ultra stay at 1 so Stable/HQ never silently skip AI swap frames.
try:
    from config import SKIP_N as _CFG_SKIP_N
    SKIP_N = dict(_CFG_SKIP_N)
except Exception:
    SKIP_N = {"Fast": 6, "Balanced": 4, "Optimized": 5, "Best": 1, "Ultra": 1}

def _skip_n(quality): return SKIP_N.get(quality, 5)

def _adaptive_swap_gap(base_gap, motion_class, quality):
    base = max(1, int(base_gap))
    # When the user/preset asked for every-frame swap (swap_n=1 / Ultra / Best),
    # never expand the gap — STATIC used to bump gap to 2 and flash original.
    if base <= 1 or quality in ("Ultra", "Best"):
        return 1
    # Head turns read as MEDIUM/HIGH motion. Forcing gap >= base made the
    # swap too sparse and the original face flashed through on profile.
    #
    # STATIC and LOW no longer stretch the gap PAST the base. They used to
    # (x1.6 and x1.2), and that is the one place this trade goes wrong: the
    # motion estimate is a whole-frame grey difference at 160x90, so a talking
    # head in front of a locked-off camera reads STATIC while the mouth is
    # moving. Stretching a 5-frame cadence to 8 there is a third of a second of
    # stale mouth on exactly the shot where lip movement is most watched.
    # Shortening on motion is still free, so MEDIUM/HIGH keep their multipliers.
    # This also makes the "Swap every N" control mean what it says: N is a
    # ceiling the adaptive logic may tighten, never loosen.
    mult = {"STATIC": 1.0, "LOW": 1.0, "MEDIUM": 0.75, "HIGH": 0.50}.get(motion_class, 1.0)
    return max(1, min(10, int(round(base * mult))))

def _fit_box(box_wh, W, H):
    ow, oh = box_wh
    ar = W / max(H, 1)
    if ow / max(oh, 1) > ar: ow = int(oh * ar)
    else: oh = int(ow / max(ar, .001))
    return max(2, ow - ow % 2), max(2, oh - oh % 2)

def _run_job(jid, src_paths, vp, cfg):
    # Phoenix CPU-only production build.  Do not silently enter a ZeroGPU path
    # or spend time probing for CUDA; the deployment constraint is CPU-only.
    cfg = dict(cfg)
    cfg["use_gpu"] = False
    _device_pref[0] = "cpu"
    return _run_job_body(jid, src_paths, vp, cfg)


def _face_region_ok(img, bbox, min_std=6.0, reference=None):
    """Reject flat / corrupted / solid patches.

    ``reference`` is the pre-swap frame. Judging the result against an ABSOLUTE
    std floor punishes legitimately low-detail regions - a motion-blurred face
    during exactly the rapid movement this build is meant to handle, a soft
    shallow-depth-of-field shot, a face in deep shadow - and every rejection
    drops that face back to the original for one frame. Comparing against the
    source region instead only rejects output that is degenerate RELATIVE to
    what was there before, which is the actual failure this guards against.
    """
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        if x2 - x1 < 8 or y2 - y1 < 8: return False
        roi = img[y1:y2, x1:x2]
        std = float(np.std(roi))
        mean = float(np.mean(roi))
        if reference is not None and reference.shape[:2] == img.shape[:2]:
            ref_std = float(np.std(reference[y1:y2, x1:x2]))
            return std >= max(2.0, min(float(min_std), ref_std * 0.35))
        if std < min_std: return False
        # v11: the old "brown patch" heuristic that used to live here (reject any
        # ROI whose channel means looked warm and flat) was a downstream band-aid
        # for the colour-match contrast crush. It fired on legitimately warm or
        # dim faces, and each time it fired the frame reverted to the original
        # face -- converting a colour defect into a flicker defect. The colour
        # bug is now fixed at source, so this only guards genuinely degenerate
        # output (a solid or near-solid patch).
        if mean < 12 and std < 8: return False
        return True
    except Exception:
        return True


def _run_job_body(jid, src_paths, vp, cfg):
    def u(p, m):
        with _lock:
            if jid in jobs: jobs[jid].update(progress=p, message=m)
        _persist_job(jid)
    t0 = time.time()
    ex = None
    stop_io = threading.Event()
    result_q = None
    enc_proc = None
    try:
        use_gpu = False
        # Device locked for this job.  CPU-only is intentional for this build.
        u(3, "Loading AI models (CPU optimized)…")
        _model_t0 = time.perf_counter()
        MODELS.get(prefer_gpu=False)
        logging.info("CPU model load: %.1fs · ORT providers=%s · native_threads=%d",
                     time.perf_counter() - _model_t0,
                     getattr(MODELS.face_analysis, "models", None) and "CPUExecutionProvider",
                     _native_threads)
        u(6, "Reading replacement faces…")
        smap = {}
        for idx, p in src_paths.items():
            im = cv2.imread(p)
            if im is None: continue
            f = _cached_source_face(im)
            if f is not None: smap[idx] = f
        if not smap:
            with _lock:
                if jid in jobs: jobs[jid].update(status='error', message="No valid replacement face", done_at=time.time())
            return

        max_faces = _parse_face_mode(cfg.get("face_mode", "1 face"))
        smap = _limit_smap(smap, max_faces)
        max_faces = min(max_faces, max(1, len(smap)))
        multi_face_safe = max_faces >= 2
        refs = cfg.get('refs') or []
        quality = cfg['quality']
        u(8, f"Face mode: {max_faces if max_faces < 99 else 'multi'} · sources={len(smap)}")

        def _cxl():
            with _lock: return jobs.get(jid, {}).get('cancel', False)

        fps = cfg['fps']
        # SKIP_N and the Optimized tier's cadence are tuned as a RAW FRAME
        # COUNT - e.g. "detect every 5 frames" - with no reference to how
        # much real time 5 frames actually spans. That is silently wrong the
        # moment the source isn't the fps the number was tuned against: at
        # 24fps, 5 frames is 208ms between detector calls; at 30fps it is
        # 167ms - 25% more real time for the same nominal "Optimized"
        # quality, and therefore 25% more opportunity for genuine motion to
        # invalidate the linear interpolation/carry/consistency-veto math in
        # between two real detections, none of which is itself fps-aware.
        # taper and max_bracket_frames were already converted to a real-time
        # budget for exactly this reason (v11.1.3) - this is the same fix
        # applied one level earlier, to how often the detector is asked to
        # look at all, not just how long a gap between two of its answers
        # may be trusted. Rescaled relative to 30fps, the fps this table's
        # numbers were tuned against (also this project's synthetic test
        # harness default - every existing regression test up to this point
        # ran at 30fps and so could not have caught this).  Only the
        # TABLE-DRIVEN cadence is rescaled; an explicit numeric override
        # (a user literally typing a frame count) means exactly that number
        # of frames and is left alone.
        _REF_FPS = 30.0
        def _fps_scaled(n):
            return max(1, int(round(int(n) * float(fps) / _REF_FPS)))

        if quality == "Optimized":
            # Self-managed CPU tier: ignore literal "1" dropdown defaults so
            # skip/det actually engage. det_int is in KEYFRAME space.
            skip_n = _fps_scaled(SKIP_N["Optimized"])
            base_det_int = 2
        else:
            skip_n = _fps_scaled(_skip_n(quality)) if cfg.get('swap_n') == 'Auto' else int(cfg.get('swap_n', 5))
            if cfg.get('det_n') not in (None, '', 'Auto'):
                try: base_det_int = max(1, int(cfg.get('det_n', 2)))
                except Exception: base_det_int = 2
            elif cfg.get('det_int'):
                try: base_det_int = max(1, int(cfg['det_int']))
                except Exception: base_det_int = 2
            else:
                # Auto: use DET_SKIP_INTERVAL as the default cadence (now 1).
                try:
                    base_det_int = max(1, int(getattr(__import__('config'), 'DET_SKIP_INTERVAL', 1) or 1))
                except Exception:
                    base_det_int = 1

        # Two cadences, deliberately separate:
        #
        #   skip_n / base_det_int  -> how often DETECTION and tracking run. This
        #       is the identity-critical one: association, the crossing lock and
        #       the per-slot embedding all depend on it. Multi-face still runs it
        #       every single frame, exactly as before.
        #
        #   swap_gap_base          -> how often the SWAP NETWORK runs. This one
        #       is not identity-critical any more. Since v11.1.0 every output
        #       frame is composited from the cached aligned crop using its own
        #       interpolated keypoints, mask, background and lighting, so a wider
        #       swap gap costs expression freshness and nothing else.
        #
        # These used to be the same number, which is why a two-face job ran the
        # ONNX forward pass twice on every frame. The original comment here said
        # multi-face "requires correctness over temporal shortcuts" because the
        # old fill path stamped a face ROI copied from a neighbouring frame and
        # could land it on the other person. That path no longer exists.
        swap_gap_base = skip_n
        if multi_face_safe:
            skip_n = 1
            base_det_int = 1
            # Best/Ultra map to 1 here, so those tiers keep swapping every frame.
            swap_gap_base = int(SKIP_N.get(quality, 1) or 1)

        cpu_n = os.cpu_count() or 4
        # Detection/tracking run sequentially before workers; ORT+OpenCV+x264
        # already consume _native_threads. Default VIDEO_WORKERS=1 on HF CPU.
        # PHOENIX_VIDEO_WORKERS may raise to 2 for A/B — never above 2.
        try:
            from config import VIDEO_WORKERS as _CFG_VW
            _default_workers = max(1, min(2, int(_CFG_VW)))
        except Exception:
            _default_workers = 1
        _worker_override = os.environ.get("PHOENIX_VIDEO_WORKERS", "").strip()
        if _worker_override.isdigit():
            workers = max(1, min(2, int(_worker_override)))
        else:
            workers = _default_workers

        stats = {
            "detector_calls": 0, "tracker_hits": 0, "gfpgan_calls": 0,
            "swap_calls": 0, "swap_skips": 0, "frames_in": 0, "frames_out": 0,
            "det_times": [], "swap_times": [],
            "frames_filled": 0, "kps_rejected": 0,
            "encode_seconds": 0.0, "processing_seconds": 0.0,
        }

        _det_cache, _det_counter = [], [0]
        _no_face_streak = [0]
        _kps_veto_streak = [0]
        _last_gray = [None]
        _last_gray_full = [None]
        _last_motion_class = ["MEDIUM"]
        _face_ema = []
        _last_swap_bboxes = []
        # Previous bbox for each replacement slot. Never infer slot identity from
        # detector ordering because detector order changes during crossings.
        _slot_prev_bboxes = {j: None for j in smap.keys()}
        # v11: one tracker owns every slot's identity, geometry, mask EMA and
        # colour EMA. Predictions ARE rendered now — carrying a swap through a
        # detector gap looks far better than blinking back to the original face.
        _tracker = _E.MultiFaceTracker(sorted(smap.keys()))
        # ---- continuous-composite state (persists across chunk boundaries) --
        # Chunk boundaries used to be visible: everything after a chunk's last
        # key frame had no following key frame to interpolate toward, so it fell
        # through every guard and emitted the untouched original frame - four
        # consecutive real-face frames roughly every five seconds, plus the
        # same at the head of each chunk. Geometry and aligned-swap history now
        # span chunks, and any frame that is not yet bracketed is deferred to
        # the next chunk instead of being emitted unswapped.
        _geom_hist = {j: [] for j in smap.keys()}      # slot -> [(g, record)]
        _aligned_hist = {j: [] for j in smap.keys()}   # slot -> [(g, fake, corr)]
        _pending_tail = []                             # [(g, frame)] not yet emittable
        _det_dt = [1.0]                                # frames since the last detection
        _last_det_frame = [-1]
        # Per identity slot: did the most recent detector probe actually see
        # this face? Tracked per slot, not per frame - in a two-person shot one
        # face can be plainly visible while the other is behind a shoulder, and
        # refreshing the hidden one's crop from a guessed position is exactly
        # what puts a shoulder into its colour statistics.
        _seen_ok = {j: True for j in smap.keys()}
        _legacy_mode = [False]   # inswapper build exposes no affine

        # v11.2.2: how far back a reacquire event is allowed to scrub. The
        # detection pass runs a whole CHUNK ahead of rendering (see
        # _record_geometry's own docstring) - wiping a slot's ENTIRE
        # timeline on reacquire, as v11.2.0 CinemaQA did, does not just drop
        # the few pre-exit entries close enough to the gap to bracket-
        # interpolate a ghost glide across it; it also erases every already-
        # recorded, already-valid entry for every EARLIER frame in the same
        # chunk that has not been rendered yet. Measured directly: a single
        # reacquire at output frame ~108 (recovering from an 18-frame
        # dropout) wiped frames 0-89's perfectly good records, and the
        # renderer - which reads this same list afterward - had nothing to
        # composite from until the timeline rebuilt past frame 114, showing
        # the original face for the first 115 frames of a 240-frame clip.
        #
        # What actually needs protecting is only the handful of entries
        # close enough to the gap to bracket across it - the same
        # max_bracket_frames the renderer itself uses (out_fps * 0.55,
        # taper+1 floor). taper is not available this early (computed per-
        # chunk, after this nested def already exists), but out_fps is
        # (assigned once, above, before any call to this function) and
        # dominates the real formula at every fps this project exposes
        # (taper caps at 12, so taper+1 <= 13 <= round(24*0.55) - the
        # lowest fps offered). A flat +5 frames covers that fixed-floor
        # case at low fps without depending on taper's exact value.
        # A first attempt used a flat 120-frame constant, reasoning
        # "generous but still much smaller than a whole clip" - measured
        # directly and found USELESS for exactly the case above: a wipe at
        # frame 108 with a 120-frame window keeps nothing back to frame 0
        # either (108 - 0 = 108 < 120), degenerating to the same full wipe.
        #
        # A function, not a value computed here: `out_fps` (referenced
        # below) is assigned later in this same enclosing function's own
        # execution, so a plain assignment at this point - before that line
        # has run - would raise UnboundLocalError. Deferring the read into
        # a nested function is safe because Python closures resolve names
        # at CALL time, by which point out_fps already holds its value
        # (every call site is well after that assignment).
        def _reacquire_scrub_window():
            return int(round(out_fps * 0.55)) + 5

        def _scrub_reacquire_timelines(g=None):
            """v11.2.0: wipe RECENT geom timeline entries for tracks that
            just reacquired (see _reacquire_scrub_window() above for why not
            the whole timeline).

            Must run even when pairs are empty (HairGate still paste-frozen),
            otherwise sticky `_reacquired` is cleared on the confirming update
            before any record runs and pre-exit anchors survive into emission.
            """
            try:
                tracks = getattr(_tracker, "tracks", None) or {}
            except Exception:
                tracks = {}
            for slot, tr in list(tracks.items()):
                if slot not in _geom_hist or tr is None:
                    continue
                if not getattr(tr, "_reacquired", False):
                    continue
                if g is None:
                    _geom_hist[slot] = []
                else:
                    _geom_hist[slot] = [
                        (gg, rr) for (gg, rr) in _geom_hist[slot]
                        if (g - gg) > _reacquire_scrub_window()
                    ]
                try:
                    tr._reacquired = False
                except Exception:
                    pass

        def _record_geometry(pairs, g):
            """Snapshot, per identity slot, the geometry key frame ``g`` will
            be rendered with. Bound to the frame here rather than read off the
            shared tracker at swap time - by then the detection pass has
            already advanced every track to the end of the chunk."""
            for f, src in (pairs or []):
                slot = getattr(f, "_slot", None)
                if slot is None or slot not in _geom_hist:
                    continue
                tr = getattr(f, "_track", None)
                # v11.2.0 CinemaQA wiped the WHOLE slot timeline here on
                # reacquire. v11.2.2: only scrub entries within
                # _reacquire_scrub_window() of this frame - see that
                # function's comment above _scrub_reacquire_timelines for
                # why a full wipe was destroying already-rendered-worthy
                # history from earlier in the same detection chunk, not
                # just the few entries that could actually bracket a ghost
                # glide across this gap.
                if (tr is not None
                        and getattr(tr, "_reacquired", False)
                        and not bool(getattr(f, "predicted", False))):
                    _geom_hist[slot] = [
                        (gg, rr) for (gg, rr) in _geom_hist[slot]
                        if (g - gg) > _reacquire_scrub_window()
                    ]
                    try:
                        tr._reacquired = False
                    except Exception:
                        pass
                rec = _geom_record(
                    f, tr,
                    _render_alpha(f, tr),
                    float(getattr(f, "_occlusion_guard", 0.0) or 0.0),
                    src,
                )
                if rec is not None:
                    # Predicted entries must not look "just seen": force det=False
                    # already set; additionally clamp alpha via _render_alpha.
                    _geom_hist[slot].append((int(g), rec))

        _startup_emb_buf = []
        _locked_emb = [None]
        _startup_confirm = [0]
        if refs and len(refs) > 0 and refs[0] is not None:
            _locked_emb[0] = refs[0]
            _startup_confirm[0] = 3
        # Couple: keep per-slot ref embeddings for Hungarian (already passed via refs)

        cap = cv2.VideoCapture(vp)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        W, H = int(cap.get(3)), int(cap.get(4))
        ow, oh = _fit_box(RES.get(cfg['resolution'], (1280, 720)), W, H)
        # Requested FPS should match output when possible (audit NSDOS-006)
        requested_fps = float(fps) if fps else float(src_fps)
        effective_fps = min(max(1.0, requested_fps), float(src_fps) if src_fps > 0 else requested_fps)
        step = max(1, int(round(float(src_fps) / max(effective_fps, 1.0))))
        out_fps = float(effective_fps)

        trim_start = float(cfg.get('trim_start', 0) or 0)
        trim_end = float(cfg.get('trim_end', 100) or 100)
        trim_start = min(100.0, max(0.0, trim_start))
        trim_end = min(100.0, max(0.0, trim_end))
        if trim_end <= trim_start:
            raise ValueError(f"Invalid trim range: start={trim_start:.1f}% end={trim_end:.1f}%")
        start_frame = int((trim_start / 100.0) * (total_src_frames - 1))
        end_frame = int((trim_end / 100.0) * (total_src_frames - 1))
        usable = max(1, end_frame - start_frame + 1)
        available_output_frames = max(1, (usable + step - 1) // step)
        lim = min(int(max(1.0, float(out_fps) * float(cfg['max_seconds']))), available_output_frames)
        if start_frame > 0: cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        src_fi = start_frame

        chunk_n = max(120, min(600, int(400_000_000 / max(ow * oh * 3, 1))))
        interp = cv2.INTER_AREA if ow < W else cv2.INTER_LINEAR

        # Stream processed frames directly into the final encoder. The old path
        # encoded MP4V with OpenCV, wrote it to disk, then FFmpeg decoded and
        # re-encoded the same frames as H.264. Eliminating that intermediate
        # encode/decode pass is one of the largest CPU-side optimizations.
        final = f"/tmp/result_{jid}.mp4"
        # Encoder probe is process-global — avoid spawning ffmpeg every job.
        global _FFMPEG_ENCODER
        if _FFMPEG_ENCODER is None:
            _enc = "libx264"
            try:
                _enc_probe = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-encoders"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    timeout=8, check=False,
                ).stdout or ""
                if "h264_nvenc" in _enc_probe and os.path.exists("/dev/nvidia0"):
                    _enc = "h264_nvenc"
            except Exception:
                pass
            _FFMPEG_ENCODER = _enc
        _video_encoder = _FFMPEG_ENCODER
        if _video_encoder == "libx264":
            # Optimized: ultrafast+CRF23 — large CPU win, minor bitrate cost.
            _encoder_args = ["-c:v", "libx264", "-crf",
                             {"Fast":"28","Balanced":"24","Optimized":"23",
                              "Best":"20","Ultra":"18"}.get(quality, "20"),
                             "-preset", {"Fast":"ultrafast","Balanced":"veryfast",
                                          "Optimized":"ultrafast","Best":"faster",
                                          "Ultra":"medium"}.get(quality, "faster"),
                             "-threads", str(int(_native_threads))]
        else:
            _cq = {"Fast":"30","Balanced":"25","Optimized":"23",
                   "Best":"21","Ultra":"19"}.get(quality, "21")
            _encoder_args = ["-c:v", "h264_nvenc", "-cq", _cq, "-preset", "p4"]

        _src_fps = float(src_fps) if src_fps else 30.0
        trim_start_sec = (trim_start / 100.0) * (total_src_frames / max(_src_fps, 0.001))
        trim_duration_sec = ((trim_end - trim_start) / 100.0) * (total_src_frames / max(_src_fps, 0.001))
        trim_duration_sec = max(0.1, float(trim_duration_sec))
        enc_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}",
            "-r", f"{out_fps:.6f}", "-i", "pipe:0",
            "-ss", f"{trim_start_sec:.3f}", "-t", f"{trim_duration_sec:.3f}", "-i", vp,
            "-map", "0:v:0", "-map", "1:a:0?",
            *_encoder_args, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-shortest", final,
        ]
        _enc_log_path = f"/tmp/ffmpeg_{jid}.log"
        _enc_log = None
        try:
            _enc_log = open(_enc_log_path, "wb")
            enc_proc = subprocess.Popen(
                enc_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=_enc_log
            )
        except Exception as e:
            try:
                if _enc_log is not None:
                    _enc_log.close()
            except Exception:
                pass
            try: cap.release()
            except Exception: pass
            raise RuntimeError(f"Could not start FFmpeg encoder: {e}")
        wr = None

        fi = produced = keys_done = 0
        _adaptive_last_swap_g = [-10**9]
        _adaptive_last_det_frame = [-10**9]
        # Release-audited progress model:
        #   0-12  startup / model loading
        #  12-30  face analysis
        #  30-84  actual frame processing
        #  84-99  ffmpeg encoding
        # 100     completed output
        #
        # ETA is derived only from completed measurable work. A slow first frame
        # therefore does not make ETA grow indefinitely, and the estimate is
        # exponentially smoothed to avoid large jumps between updates.
        eof = False
        progress_last_emit = [0.0]
        eta_ema_rate = [None]
        eta_last_done = [0.0]
        eta_last_ts = [t0]
        last_progress = [12]

        def _format_eta(seconds):
            if seconds is None or seconds < 0:
                return "ETA calculating…"
            sec = int(max(0, seconds))
            m, s = divmod(sec, 60)
            if m >= 60:
                h, m = divmod(m, 60)
                return f"ETA {h}h {m}m"
            if m:
                return f"ETA {m}m {s}s"
            return f"ETA {s}s"

        def _set_phase_progress(base, span, msg, completed=None, total=None, phase="processing"):
            now = time.time()
            eta_seconds = None
            p = int(base)
            if completed is not None and total:
                total_i = max(1, int(total))
                completed_i = max(0, min(total_i, int(completed)))
                frac = completed_i / total_i
                p = int(base + frac * span)
                # Only advance the ETA clock when measurable work completed.
                # Waiting on a long-running inference therefore cannot make ETA
                # increase merely because wall-clock time passed.
                prev_done, prev_ts = eta_last_done[0], eta_last_ts[0]
                dt = now - prev_ts
                dd = completed_i - prev_done
                if dd > 0 and dt >= 0.25:
                    inst_rate = dd / dt
                    if eta_ema_rate[0] is None:
                        eta_ema_rate[0] = inst_rate
                    else:
                        eta_ema_rate[0] = 0.20 * inst_rate + 0.80 * eta_ema_rate[0]
                    eta_last_done[0], eta_last_ts[0] = completed_i, now
                rate = eta_ema_rate[0]
                if rate and rate > 0.01 and completed_i < total_i:
                    eta_seconds = (total_i - completed_i) / rate
                    msg = f"{completed_i}/{total_i} frames · {_format_eta(eta_seconds)}"
                else:
                    msg = f"{completed_i}/{total_i} frames · ETA calculating…"
            # Never allow a displayed percentage to regress.
            p = max(last_progress[0], min(99, p))
            if p > last_progress[0]:
                last_progress[0] = p
            if now - progress_last_emit[0] >= 0.20 or p >= 99:
                progress_last_emit[0] = now
                with _lock:
                    if jid in jobs:
                        jobs[jid]["eta_seconds"] = eta_seconds
                        _persist_job(jid)
                u(p, msg)

        u(12, f"Processing {ow}×{oh} · det={skip_n} · swap≈{swap_gap_base} · det≤{DET_MAX_W}px · Phoenix CPU Speed V3 · adaptive AI cadence · CPU-only · profiled ETA…")

        FRAME_Q_MAX = max(48, min(chunk_n * 2, 240))
        RESULT_Q_MAX = max(48, min(chunk_n * 2, 240))
        frame_q = queue.Queue(maxsize=FRAME_Q_MAX)
        result_q = queue.Queue(maxsize=RESULT_Q_MAX)
        stop_io = threading.Event()

        def _reader_thread():
            local_fi = 0
            local_src = start_frame
            local_produced = 0
            # Single resize to output size. An INPUT_MAX_W pre-downscale before
            # ow×oh caused crush→upscale (e.g. 1080p→720w→1280w) on CPU.
            try:
                while local_produced < lim and not stop_io.is_set():
                    if _cxl() or local_src > end_frame: break
                    # Skip frames at the decoder level when possible. read()
                    # fully decodes a frame that may immediately be discarded;
                    # grab() advances the decoder without materialising pixels.
                    if local_fi % step != 0:
                        if not cap.grab():
                            break
                        local_fi += 1
                        local_src += 1
                        continue

                    ok, frm = cap.read()
                    if not ok or frm is None: break
                    if frm.shape[1] == ow and frm.shape[0] == oh:
                        resized = frm
                    else:
                        resized = cv2.resize(frm, (ow, oh), interpolation=interp)
                    while not stop_io.is_set():
                        try:
                            frame_q.put((local_produced, resized), timeout=0.4)
                            break
                        except queue.Full:
                            if _cxl(): break
                    local_produced += 1
                    local_fi += 1
                    local_src += 1
            except Exception as e:
                logging.warning(f"reader thread: {e}")
            finally:
                try: frame_q.put(None, timeout=2)
                except Exception: pass

        def _writer_thread():
            try:
                while True:
                    item = result_q.get()
                    if item is None:
                        result_q.task_done()
                        break
                    try:
                        if enc_proc.stdin is None:
                            raise RuntimeError("FFmpeg encoder stdin is unavailable")
                        # Prefer buffer view over bytes() alloc every frame.
                        if not item.flags['C_CONTIGUOUS']:
                            item = np.ascontiguousarray(item)
                        enc_proc.stdin.write(item.data)
                    except Exception as e:
                        logging.error(f"writer fatal: {e}")
                        result_q.task_done()
                        break
                    else: result_q.task_done()
            except Exception as e:
                logging.warning(f"writer thread: {e}")

        reader_t = threading.Thread(target=_reader_thread, name=f"reader-{jid}", daemon=True)
        writer_t = threading.Thread(target=_writer_thread, name=f"writer-{jid}", daemon=True)
        reader_t.start()
        writer_t.start()

        ex = ThreadPoolExecutor(max_workers=workers)

        # `or _pending_tail`: frames deferred from the previous chunk were
        # already counted in `produced`, so once `produced` reaches `lim` the
        # loop would otherwise exit with them still unemitted - silently
        # truncating the output by up to skip_n-1 frames.
        while (produced < lim or _pending_tail) and not eof:
            cframes = [f for _g, f in _pending_tail]
            gidx = [_g for _g, _f in _pending_tail]
            _pending_tail = []
            while len(cframes) < chunk_n and produced < lim:
                if _cxl(): raise _CancelledJob()
                try: item = frame_q.get(timeout=8.0)
                except queue.Empty:
                    if not reader_t.is_alive(): eof = True; break
                    continue
                if item is None: eof = True; break
                idx, frm = item
                cframes.append(frm)
                gidx.append(idx)
                produced += 1
            if not cframes: break

            n = len(cframes)
            stats["frames_in"] += n
            keyf = [(gidx[k] % skip_n == 0) for k in range(n)]
            results = {}

            key_indices = [k for k in range(n) if keyf[k]]
            key_motion_map = {}
            # Whether the detector can currently SEE this face. Only then may
            # the cached aligned crop be refreshed: during a genuine occlusion
            # the tracker still supplies plausible geometry, but a crop cut
            # there is a crop of whatever is covering the face, and the colour
            # match then locks onto that. Re-projecting the last crop taken
            # while the face was visible is the right thing to keep doing.
            key_swap_ok = {}
            # No separate per-chunk "last position" state here on purpose.
            # _tracker (a single MultiFaceTracker created once for the whole
            # job, not per chunk) already IS the one place this identity's
            # last-seen geometry lives - _tracker.tracks[0].obs_bbox is the
            # last RAW real detection, correctly available or correctly
            # absent across a chunk boundary with no extra bookkeeping. A
            # second, separately-maintained "previous bbox" here was a stale
            # copy of the same fact, on its own reseeding schedule, and had
            # to be independently re-invalidated at both a chunk boundary
            # and a detector gap (see the removed history below). One source
            # of truth instead of two that can silently disagree.

            def _tick():
                # Each completed key frame represents skip_n output frames of
                # measurable AI work. This gives useful movement while workers
                # complete out of order without pretending an in-flight frame is
                # finished.
                completed_est = min(lim, keys_done * skip_n)
                _set_phase_progress(30, 54, "Processing video…", completed_est, lim, phase="processing")

            # v11.0.1 FIX: honour user/preset det_n exactly. The old
            # `det_n_val = max(det_n_val, DET_SKIP_INTERVAL)` floor forced
            # re-detect every ≥3 keyframes even when Stable set det_n=1 —
            # a primary cause of flicker / original-face fallback.
            # DET_SKIP_INTERVAL remains a documented Auto default in config.py;
            # it must never raise an explicit cadence.
            det_n_val = max(1, int(base_det_int) if base_det_int else 1)
            initial_force_until = max(3, int(round(out_fps * 1.5)))
            last_pairs_ref = [None]
            last_bboxes_ref = [list(_last_swap_bboxes) if _last_swap_bboxes else []]
            last_det_g = _adaptive_last_det_frame
            last_swap_g = _adaptive_last_swap_g

            for key_ord, k in enumerate(key_indices):
                if _cxl(): raise _CancelledJob()
                frm = cframes[k]
                g = gidx[k]
                # v11.1.9: keep TrackState._frame_wh current so predict() can
                # freeze when the face leaves the shot (even on miss/carry).
                try:
                    _fw = (int(frm.shape[1]), int(frm.shape[0]))
                    for _tr in _tracker.tracks.values():
                        if _tr is not None:
                            _tr._frame_wh = _fw
                except Exception:
                    pass
                gray_s = cv2.resize(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY), (160, 90))
                motion = float(np.mean(cv2.absdiff(gray_s, _last_gray[0]))) if _last_gray[0] is not None else 999.0
                _last_gray[0] = gray_s
                mclass = _motion_class(motion)
                key_motion_map[k] = mclass

                force_initial = (g < initial_force_until)
                # Schedule detection in KEY-FRAME space (not output-frame index).
                # Static scenes may hold longer; motion keeps det_mult=1.
                det_mult = {"STATIC": 2, "LOW": 1, "MEDIUM": 1, "HIGH": 1}.get(mclass, 1)
                # Scheduled in OUTPUT-FRAME space. It used to compare `key_ord`,
                # which is an index within the current chunk and restarts at 0
                # at every chunk boundary, so the cadence silently reset there.
                det_due = (g - last_det_g[0]) >= det_n_val * det_mult * max(1, skip_n)
                need_det = force_initial or det_due or (last_pairs_ref[0] is None)

                if need_det:
                    _set_phase_progress(12, 18, "Analysing faces…", min(lim, max(0, g)), lim, phase="detection")
                    t_det0 = time.perf_counter()
                    # Main detector pass. Multi-face scenes get a little more
                    # source resolution because profile/small faces are the hard case.
                    primary_det_w = 840 if multi_face_safe else (560 if quality == "Optimized" else DET_MAX_W)
                    det_frm, sx, sy = _make_det_frame(frm, min(frm.shape[1], primary_det_w))
                    raw_faces = _fa.get(det_frm)
                    faces = _scale_faces(raw_faces, sx, sy) if raw_faces else []
                    faces = sorted(faces, key=lambda f: f.bbox[0])
                    # Frames elapsed since the previous detection. Velocity and
                    # the One-Euro filter are both expressed per OUTPUT frame,
                    # so they need the real interval, not an assumed 1.
                    _det_dt[0] = float(max(1, g - _last_det_frame[0])) if _last_det_frame[0] >= 0 else 1.0
                    _last_det_frame[0] = g

                    # Adaptive high-resolution probe. Do NOT wait for total face
                    # loss: small or low-confidence faces are exactly the cases where
                    # the old implementation reverted to the original. The second
                    # pass is only invoked when needed, so normal frontal processing
                    # keeps the fast path.
                    # Default OFF (config ENABLE_HI_DET_PROBE). Total miss still
                    # probes via needs_hi = not faces below.
                    try:
                        from config import ENABLE_HI_DET_PROBE as _hi_probe_enabled
                    except Exception:
                        _hi_probe_enabled = os.environ.get("PHOENIX_HI_DET_PROBE", "0").strip().lower() in ("1", "true", "yes", "on")
                    needs_hi = not faces
                    if faces:
                        needs_hi = any(
                            _area(f) < 4500.0
                            or min(float(f.bbox[2]-f.bbox[0]), float(f.bbox[3]-f.bbox[1])) < 58.0
                            or float(getattr(f, "det_score", 0.5) or 0.5) < 0.28
                            or _frontal_score(f) < 0.50
                            for f in faces
                        ) and _hi_probe_enabled
                    if needs_hi or (not faces and _no_face_streak[0] > 0):
                        try:
                            hi_w = min(frm.shape[1], 1280 if multi_face_safe else 1024)
                            det2, sx2, sy2 = _make_det_frame(frm, hi_w)
                            raw2 = _fa.get(det2) or []
                            faces2 = _scale_faces(raw2, sx2, sy2) if raw2 else []
                            if faces2:
                                # Merge the high-resolution result, preferring it
                                # for overlapping detections and adding genuinely new
                                # small faces.
                                merged = list(faces)
                                for f2 in faces2:
                                    overlaps = [(_bbox_iou(f2.bbox, f1.bbox), i) for i, f1 in enumerate(merged)]
                                    best_iou, best_i = max(overlaps, default=(0.0, -1))
                                    if best_i >= 0 and best_iou >= 0.20:
                                        f1 = merged[best_i]
                                        score1 = float(getattr(f1, "det_score", 0.0) or 0.0)
                                        score2 = float(getattr(f2, "det_score", 0.0) or 0.0)
                                        area1, area2 = _area(f1), _area(f2)
                                        if score2 > score1 + 0.03 or area2 > area1 * 1.05:
                                            merged[best_i] = f2
                                    else:
                                        merged.append(f2)
                                faces = merged
                            stats["detector_calls"] += 1
                        except Exception as e:
                            logging.debug("high-resolution detector probe skipped: %s", e)
                    faces, _face_ema = _smooth_faces(faces, _face_ema)

                    # Reject detections whose own 5 keypoints do not describe a
                    # paintable face (see _kps_reliable). This runs AFTER the
                    # hi-res probe rescue above, so a real face that is merely
                    # small or partly turned still gets its second chance
                    # first; only genuinely inconsistent reads are dropped
                    # here. A rejected identity falls straight into the
                    # existing "no detection this frame" path below, which
                    # already holds the last good geometry through a bounded
                    # gap instead of painting - exactly the behaviour wanted
                    # here, reused rather than reinvented.
                    _n_before_kps_gate = len(faces)
                    faces = [f for f in faces if _kps_reliable(f)]
                    if len(faces) < _n_before_kps_gate:
                        stats["kps_rejected"] = stats.get("kps_rejected", 0) + (_n_before_kps_gate - len(faces))

                    stats["detector_calls"] += 1
                    stats["det_times"].append((time.perf_counter() - t_det0) * 1000)
                    last_det_g[0] = g

                    if not faces:
                        _no_face_streak[0] += 1
                        # v11: a detector miss is not a reason to show the real
                        # face again. Carry the tracked geometry for a bounded
                        # number of frames so the swap rides through the gap.
                        # Advance every track by the frames actually elapsed.
                        _tracker.assign([], {}, dt_frames=_det_dt[0])
                        carried = _carry_pairs(_tracker, smap, max_faces,
                                               advance=False)
                        _record_geometry(carried, g)
                        for _j in _seen_ok:
                            _seen_ok[_j] = False
                        key_swap_ok[k] = dict(_seen_ok)
                        if carried:
                            last_pairs_ref[0] = carried
                        # Detection itself is visible as a heartbeat, so a slow
                        # first chunk does not look frozen.
                        det_done = min(lim, max(0, g + 1))
                        _set_phase_progress(12, 18, "Analysing faces…", det_done, lim, phase="detection")
                        continue
                    _after_real_gap = _no_face_streak[0] > 0
                    _no_face_streak[0] = 0

                    # v11.1.10 HairGate: after a real gap, still skip the
                    # motion-veto vs pre-gap position (below), but refuse to
                    # bind a weak / unreliable first return (hair/skull/neck
                    # under motion blur). Prefer original this frame.
                    if _after_real_gap:
                        _n_gap = len(faces)
                        faces = [
                            f for f in faces
                            if _kps_reliable(f)
                            and float(getattr(f, "det_score", 0.0) or 0.0) >= 0.32
                        ]
                        if len(faces) < _n_gap:
                            stats["hair_gate_gap"] = stats.get("hair_gate_gap", 0) + (
                                _n_gap - len(faces)
                            )
                        if not faces:
                            _tracker.assign([], {}, dt_frames=_det_dt[0])
                            carried = _carry_pairs(_tracker, smap, max_faces,
                                                   advance=False)
                            _record_geometry(carried, g)
                            for _j in _seen_ok:
                                _seen_ok[_j] = False
                            key_swap_ok[k] = dict(_seen_ok)
                            if carried:
                                last_pairs_ref[0] = carried
                            det_done = min(lim, max(0, g + 1))
                            _set_phase_progress(12, 18, "Analysing faces…",
                                                det_done, lim, phase="detection")
                            continue

                    if multi_face_safe:
                        pairs = _persistent_track_pairs(
                            faces, smap, refs, _tracker, max_faces,
                            frame_shape=frm.shape if frm is not None else None,
                            dt_frames=_det_dt[0],
                        )
                    else:
                        # prev_bbox is used only to help pick the right
                        # candidate when the detector reports more than one
                        # face-like region this frame - NOT to smooth or
                        # blend the geometry that gets painted (see the
                        # single source of truth note at this chunk's start).
                        # tr.obs_bbox is the track's own last RAW real
                        # detection, correctly present or correctly None
                        # across any gap with nothing extra to invalidate.
                        _tr0_ = _tracker.tracks.get(0) if hasattr(_tracker, "tracks") else None
                        pairs = _pairs_for_frame(
                            faces, smap, refs,
                            max_faces=max_faces,
                            prev_bbox=_tr0_.obs_bbox if _tr0_ is not None else None,
                            frame_bgr=frm,
                            prev_bboxes=[_slot_prev_bboxes.get(j) for j in sorted(smap.keys())],
                            locked_emb=_locked_emb[0],
                        )
                        # Motion-consistency veto: reject a live detection
                        # whose 5 keypoints land far from where THIS
                        # identity's own tracked motion says they should be,
                        # before it reaches tracker.bind() and poisons that
                        # history. _kps_reliable() upstream already rejects
                        # keypoints that are not mutually consistent with ANY
                        # rigid pose - a degenerate/impossible read. It
                        # cannot catch a read that fits a pose fine but is
                        # the WRONG pose for this identity right now: hair
                        # sweeping across the face mid-turn can pull the
                        # landmark regression onto a plausible-looking
                        # configuration that is not where the eyes and mouth
                        # actually are, for as long as the hair keeps
                        # confusing the same few reads in a similar way.
                        #
                        # This reads and writes ONLY TrackState fields
                        # (tr.kps/.vel_kps/.last_hit_kps/.missed) - the same,
                        # single state _tracker.bind() below updates and
                        # _pairs_for_frame's selection above already reads
                        # (tr.obs_bbox). Earlier attempts at this check
                        # instead compared against _stabilize_face_geometry's
                        # OWN, separately-blended "previous position" - a
                        # second smoothing pass downstream of this one, fed
                        # different inputs on a different schedule, that
                        # could itself go stale and disagree with whatever
                        # this check had just approved. That divergence, not
                        # any single threshold, was the root of the hard-
                        # edged/frozen/duplicate-looking paste reports; that
                        # second pass is gone now (a real detection is
                        # painted with its own raw geometry once accepted,
                        # exactly like the multi-face path already does), so
                        # there is only one position for this check, the
                        # bind() call right after it, and the compositor to
                        # ever agree or disagree about.
                        #
                        # Two failure modes were measured directly while
                        # building this and are guarded against explicitly:
                        # (1) comparing against tr.kps advanced by
                        # TrackState.predict()'s deliberately damped,
                        # under-committing extrapolation - correct for a
                        # cautious carry-forward guess, wrong for judging a
                        # NEW detection, since real constant-velocity motion
                        # then falls further behind every veto cycle and the
                        # measured deviation grows unbounded even though
                        # nothing is wrong. Fixed by projecting from
                        # last_hit_kps (the actual last real detection) over
                        # the full elapsed time, undamped, and by advancing
                        # the track's own prediction on every veto so it
                        # keeps pace. (2) comparing a detection right after a
                        # GENUINE gap against the pre-gap position, which
                        # carries no information about where the subject is
                        # after a real absence - skipped outright below.
                        # Bounding how many consecutive detector calls the
                        # veto may override before conceding reuses
                        # `trk_flip_frames`, the same constant the multi-face
                        # tracker already uses for "how long may a competing
                        # signal override the established one before
                        # conceding" - the same hysteresis idiom, not a new
                        # tuned number.
                        if _after_real_gap:
                            _kps_veto_streak[0] = 0
                        if pairs and len(pairs) == 1 and not _after_real_gap:
                            _pf0, _psrc0 = pairs[0]
                            _tr0 = _tracker.tracks.get(0) if hasattr(_tracker, "tracks") else None
                            _new_kps = getattr(_pf0, "kps", None)
                            # v11.2.3: a track that just reacquired (see
                            # TrackState.update()'s "ReentrySafe snap") has
                            # vel_kps deliberately zeroed - "we don't yet
                            # know this identity's motion, don't extrapolate
                            # a guess" is correct for THAT frame's own
                            # predict(), but here it means the very next
                            # frame's expected position is the frozen snap
                            # point itself, with no velocity credit at all.
                            # During sustained rapid motion (the case this
                            # veto exists to protect, not reject) the
                            # subject has plainly kept moving since the
                            # snap, so a real, correct detection reads as a
                            # huge "deviation" against a zero-velocity
                            # expectation and gets vetoed - which never lets
                            # update() run to record real velocity or clear
                            # HairGate's confirm_hits, so the SAME zero-
                            # velocity state persists and the very next
                            # candidate gets vetoed too. Measured directly:
                            # a face oscillating at up to ~55px/frame
                            # deadlocked in exactly this cycle and never
                            # painted again for the rest of a 200-frame
                            # clip. No reliable velocity yet is the same
                            # "nothing to compare against" case
                            # _after_real_gap already skips this veto for -
                            # reused here rather than reinvented.
                            _no_vel_basis = (
                                _tr0 is not None
                                and (_tr0.vel_kps is None
                                     or not bool(np.any(np.abs(_tr0.vel_kps) > 1e-6)))
                            )
                            if (_tr0 is not None and _tr0.established and not _no_vel_basis and
                                    _tr0.kps is not None and _new_kps is not None and
                                    _tr0.kps.shape == np.asarray(_new_kps).shape and
                                    _kps_veto_streak[0] < int(_E._P.get("trk_flip_frames", 5))):
                                _elapsed = float(_tr0.missed) + float(_det_dt[0])
                                _expected = _tr0.kps
                                _est_speed = 0.0
                                if _tr0.vel_kps is not None and _tr0.last_hit_kps is not None:
                                    _expected = _tr0.last_hit_kps + _tr0.vel_kps * _elapsed
                                    _est_speed = float(np.mean(np.linalg.norm(_tr0.vel_kps, axis=1))) * _elapsed
                                _face_w = max(1.0, float(_pf0.bbox[2] - _pf0.bbox[0]))
                                _dev = float(np.mean(np.linalg.norm(
                                    np.asarray(_new_kps, np.float32) - _expected, axis=1)))
                                # Budget scales with how much this identity's
                                # OWN tracked velocity says it is actually
                                # moving, not with elapsed frames directly -
                                # scaling by elapsed frames alone let the
                                # budget balloon past 200px on a 150px-wide
                                # face after just a 10-frame cadence gap,
                                # loose enough to accept almost anything
                                # exactly when a sustained confusion or
                                # occlusion had already widened the cadence.
                                _budget = _face_w * 0.22 + 0.5 * _est_speed
                                if _dev > _budget:
                                    _tr0.predict(float(_det_dt[0]))
                                    _kps_veto_streak[0] += 1
                                    pairs = []
                        if pairs:
                            _kps_veto_streak[0] = 0
                        # The single-face path keeps its own well-tested pairing
                        # (startup identity lock, reference embeddings), but it
                        # still gets the track's temporal memory so its mask and
                        # colour correction are EMA-stabilised the same way.
                        for _pf, _psrc in pairs:
                            for _sj, _sface in smap.items():
                                if _psrc is _sface:
                                    _tr = _tracker.bind(
                                        _sj, _pf, dt_frames=_det_dt[0],
                                        kps_ok=_kps_reliable(_pf),
                                    )
                                    try:
                                        _pf._track = _tr
                                        _pf._slot = _sj
                                        if frm is not None:
                                            _pf._frame_shape = frm.shape
                                            if _tr is not None:
                                                _tr._frame_wh = (int(frm.shape[1]), int(frm.shape[0]))
                                        # FIX (debug session): single-face path previously
                                        # hardcoded _occlusion_guard=False unconditionally,
                                        # which meant a face partially outside the frame
                                        # (or otherwise low boundary confidence) still got
                                        # pasted with a full, untrimmed mask. Multi-face
                                        # already computes this; single-face never did.
                                        # Gate ONLY on frame-boundary confidence here (not
                                        # is_marginal / profile score) so ordinary profile
                                        # turns are NOT affected — that was the specific
                                        # regression the previous hardcoded False avoided.
                                        _boundary_conf = 1.0
                                        if HAS_PHASE1:
                                            try:
                                                _boundary_conf = validate_detection_confidence(
                                                    _pf.bbox,
                                                    frm.shape if frm is not None else None,
                                                    getattr(_pf, "landmark_2d_106", None),
                                                )
                                            except Exception:
                                                _boundary_conf = 1.0
                                        _want = float(np.clip((0.75 - _boundary_conf) / 0.35, 0.0, 1.0))
                                        # v11.2.0: light occlusion trim on marginal
                                        # single-face (hair/hand in box) — was
                                        # multi-only; full mask covered ears/hair.
                                        try:
                                            if _face_looks_marginal(_pf):
                                                _want = max(_want, 0.40)
                                        except Exception:
                                            pass
                                        _pf._occlusion_guard = (
                                            _tr.ramp_occlusion(_want) if _tr is not None else _want
                                        )
                                    except Exception:
                                        pass
                                    break

                    # v11.1.10: drop live pairs still paste-frozen (reacquire
                    # confirmation). Multi-face already gates inside
                    # _persistent_track_pairs; single-face bind can leave a
                    # frozen track attached — do not record/paint it.
                    if pairs:
                        pairs = [(f, s) for f, s in pairs if _face_swap_allowed(f)]

                    if pairs and max_faces == 1 and _locked_emb[0] is None:
                        cand_emb = pairs[0][0].normed_embedding
                        if _startup_emb_buf:
                            sim = float(np.dot(_startup_emb_buf[-1], cand_emb))
                            if sim >= 0.45:
                                _startup_confirm[0] += 1
                                _startup_emb_buf.append(cand_emb)
                            else:
                                _startup_confirm[0] = 1
                                _startup_emb_buf = [cand_emb]
                        else:
                            _startup_confirm[0] = 1
                            _startup_emb_buf = [cand_emb]

                        if _startup_confirm[0] >= 3:
                            _locked_emb[0] = np.mean(np.stack(_startup_emb_buf[-3:], axis=0), axis=0)
                            nrm = float(np.linalg.norm(_locked_emb[0])) + 1e-6
                            _locked_emb[0] = _locked_emb[0] / nrm
                            logging.info("Startup identity locked after %d consistent frames", _startup_confirm[0])
                        # Do NOT clear pairs while locking. That flashed the
                        # original face on every profile/partial frame until
                        # three high-sim embeddings arrived — they never do
                        # on a side view.

                    # A real, accepted detection is painted with its own RAW
                    # geometry - no extra blend-toward-history step here.
                    # That step (_stabilize_face_geometry, since removed) was
                    # a second, independently-maintained "smoothed position"
                    # alongside TrackState's own (tr.bbox/tr.kps, EMA and
                    # One-Euro-filtered in swap_engine.py), fed slightly
                    # different inputs in a different order, with no
                    # synchronization between the two. That divergence - not
                    # any single threshold in either one - was the root of
                    # the hard-edged/frozen/duplicate-looking paste reports:
                    # whichever of the two happened to still hold stale data
                    # could silently outvote the other. Trusting the raw,
                    # gated detection directly matches how the multi-face
                    # path already works, which has not needed a second
                    # smoothing layer.
                    if pairs:
                        # Update the slot that actually received the source face.
                        # This prevents bbox history from swapping when detector
                        # ordering changes.
                        for pf, psrc in pairs:
                            for sj, sface in smap.items():
                                if psrc is sface:
                                    _slot_prev_bboxes[sj] = pf.bbox.astype(np.float32).copy()
                                    break
                        _last_swap_bboxes.clear()
                        for sj in sorted(smap.keys()):
                            bb = _slot_prev_bboxes.get(sj)
                            if bb is not None:
                                _last_swap_bboxes.append(bb.copy())
                        last_bboxes_ref[0] = list(_last_swap_bboxes)

                    # Carry forward any slot that pairing did NOT cover this
                    # frame - not only when EVERY slot failed to pair.
                    #
                    # The previous check ("if not pairs") only carried when the
                    # WHOLE list came back empty, so a slot that fails to pair
                    # while a DIFFERENT slot in the same frame succeeds got
                    # nothing at all: no real pair, no carried one either - a
                    # silent hole in the one guarantee this engine is built
                    # around ("a slot always gets carried through a miss").
                    # Measured directly: a heavily-occluded identity in a
                    # two-face scene (a near-total overlap, its own visible
                    # sliver too narrow for a reliable landmark read) went 82
                    # CONSECUTIVE frames with NOTHING recorded for it, purely
                    # because the OTHER identity kept pairing successfully
                    # every single frame and so the all-or-nothing check never
                    # tripped. That is what a downstream fix (bounding how long
                    # a gap may be bridged) surfaced as a visible defect - the
                    # gap this closes was always there, just never this long
                    # before a slot's own kps could get rejected outright.
                    pairs = list(pairs or [])
                    covered = {sj for _pf, psrc in pairs
                              for sj, sface in smap.items() if psrc is sface}
                    missing = set(smap.keys()) - covered
                    if missing:
                        have_srcs = {id(psrc) for _pf, psrc in pairs}
                        for pf, psrc in _carry_pairs(_tracker, smap, max_faces, advance=False):
                            if getattr(pf, "_slot", None) in missing and id(psrc) not in have_srcs:
                                pairs.append((pf, psrc))
                    # Wipe exit-era timeline even while confirm-frozen (no pairs).
                    _scrub_reacquire_timelines(g)
                    _record_geometry(pairs or [], g)
                    # A pair that comes back predicted means the detector ran
                    # and this identity was not among what it found.
                    for _j in _seen_ok:
                        _seen_ok[_j] = False
                    for _f, _ in (pairs or []):
                        _j = getattr(_f, "_slot", None)
                        if _j in _seen_ok and not bool(getattr(_f, "predicted", False)):
                            _seen_ok[_j] = True
                    key_swap_ok[k] = dict(_seen_ok)
                    if pairs:
                        last_pairs_ref[0] = pairs
                    det_done = min(lim, max(0, g + 1))
                    _set_phase_progress(12, 18, "Analysing faces…", det_done, lim, phase="detection")
                else:
                    stats["tracker_hits"] += 1
                    # Detector deliberately skipped this key frame. Predict every
                    # slot forward and swap the predicted geometry — this is the
                    # difference between a continuous swap and one that pulses in
                    # time with the detection interval.
                    carried = _carry_pairs(_tracker, smap, max_faces,
                                           dt_frames=float(max(1, skip_n)))
                    _record_geometry(carried, g)
                    # Detector deliberately skipped: its most recent verdict on
                    # whether each face is visible still stands.
                    key_swap_ok[k] = dict(_seen_ok)

            # One-sided hold: only long enough to bridge to the next key frame.
            # The tracker's own carry budget already decides how long a lost
            # face may be predicted for; stacking a further ~0.8 s of held
            # geometry on top of it is how a face ends up painted on a body
            # after its owner has left the shot.
            taper = int(max(3, min(2 * max(1, skip_n, swap_gap_base), 12)))
            # How long a gap BETWEEN TWO REAL DETECTIONS is still safe to
            # bridge with a full, confident interpolation - expressed as a
            # TIME budget (see _geom_for_frame's docstring for why a fixed
            # frame count cannot work here) and converted to output frames at
            # the actual output fps, so it does not have to be re-tuned
            # whenever fps or the detection cadence preset changes.
            # v11.2.0 CinemaQA: 0.55s (was 0.7s ReentrySafe / 1.5s Continuum).
            # Reacquire wipes timelines; this bounds any residual real–real gap.
            max_bracket_frames = max(taper + 1, int(round(out_fps * 0.55)))

            def _records_at(g):
                """Every slot's geometry for output frame ``g``."""
                out = {}
                for slot in sorted(smap.keys()):
                    rec = _geom_for_frame(_geom_hist.get(slot) or [], g, taper,
                                          max_bracket=max_bracket_frames)
                    if rec is not None:
                        out[slot] = rec
                return out

            # Phase 2 — swap-network scheduling. Detection and tracking can run
            # on sparse key frames; the ONNX forward pass is far more expensive
            # still, so it gets its own, motion-adaptive interval on top.
            selected = []
            for k in key_indices:
                visible = key_swap_ok.get(k) or {}
                if not any(visible.get(sl) for sl in _records_at(gidx[k])):
                    continue
                mclass = key_motion_map.get(k, "MEDIUM")
                gap = _adaptive_swap_gap(swap_gap_base, mclass, quality)
                g = gidx[k]
                if g < initial_force_until or last_swap_g[0] <= -10**8 or (g - last_swap_g[0]) >= gap:
                    selected.append(k)
                    last_swap_g[0] = g
            key_indices_for_swap = selected

            # ---------------------------------------------------------------
            # Phase 2a (parallel) — run ONLY the swap network on the selected
            # key frames and keep each identity's 128x128 aligned result.
            #
            # Compositing is deliberately NOT done here. The aligned crop is the
            # expensive part; placement, masking, background and colour match
            # are cheap, frame-specific, and must happen in strict frame order
            # so the mask and colour EMAs advance monotonically in time. Doing
            # them inside out-of-order workers is what let a frame be toned by
            # statistics belonging to a frame several tenths of a second away.
            # ---------------------------------------------------------------
            enh = cfg.get("enhancer") or "None"
            enh_scope = (cfg.get("enhance_scope") or "primary").lower()
            enh_all = enh_scope.startswith("all")

            def _aligned_post(is_primary):
                """Enhancer applied ONCE, to the aligned crop.

                Running an enhancer on the output frame meant only frames the
                swapper ran on were enhanced, so enhanced and unenhanced frames
                alternated at the swap cadence - the face pulsing in texture and
                tone several times a second. Enhancing the cached aligned crop
                means every frame that reuses it is enhanced identically, at a
                fraction of the cost (128x128 instead of a padded face ROI).
                """
                if not enh or enh == "None":
                    return None
                if not (enh_all or is_primary):
                    return None

                def _post(crop):
                    try:
                        if _is_soft_polish(enh):
                            return _soft_polish(crop, enh)
                        big = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC)
                        out = _enhance(big, enh)
                        if out is None:
                            return None
                        stats["gfpgan_calls"] += 1
                        return cv2.resize(out, (crop.shape[1], crop.shape[0]),
                                          interpolation=cv2.INTER_AREA)
                    except Exception as e:
                        logging.debug("aligned enhancer skipped: %s", e)
                        return None
                return _post

            def swap_frame(k):
                """Produce the aligned swap crop for every slot on key frame k.

                Geometry comes from the SAME interpolated timeline the emission
                pass will use, not from whatever the tracker happened to be
                holding. Two reasons that matters:

                  * between two real detections, interpolation is exact for any
                    motion the tracker can model, while the tracker's own
                    forward prediction trails the subject - so a crop cut at
                    predicted keypoints is cut slightly off the face, and that
                    offset is then baked into every frame that reuses it;
                  * the crop and the composite that re-projects it are then
                    described by one and the same geometry, so the hand-over
                    between a freshly swapped frame and a reused one is exact.
                """
                frm = cframes[k]
                records = _records_at(gidx[k])
                if not records:
                    return k, []
                t_swap0 = time.perf_counter()

                # Largest face is the "primary" for enhancer scope.
                def _area_of(rec):
                    b = rec["bbox"]
                    return float(max(1.0, (b[2] - b[0]) * (b[3] - b[1])))
                primary_slot = max(records, key=lambda sl: _area_of(records[sl]))

                comp = _compositor()
                visible = key_swap_ok.get(k) or {}
                produced_crops = []
                seen_src = set()
                for slot, rec in records.items():
                    # Only refresh a crop for a face the detector can currently
                    # see. A crop cut at a guessed position during an occlusion
                    # is a crop of whatever is covering the face.
                    if not visible.get(slot):
                        continue
                    src = rec.get("src")
                    if src is None or id(src) in seen_src:
                        continue
                    seen_src.add(id(src))
                    face = _E.PredictedFace(rec["bbox"], rec["kps"], rec["lmk"], None, 0.5)
                    # Every crop is cut from the UNMODIFIED frame, so one
                    # person's swap can never be fed into the next person's
                    # alignment. Overlap between two faces in contact is
                    # resolved at composite time, by paint order and by the
                    # rival-hull cut, where the geometry to do it properly is
                    # actually available.
                    try:
                        fake, M = comp._raw_swap(frm, face, src) if comp is not None else (None, None)
                    except Exception as e:
                        logging.debug("swap failed on slot %s: %s", slot, e)
                        fake, M = None, None
                    stats["swap_calls"] += 1
                    if fake is None or M is None:
                        if M is None:
                            _legacy_mode[0] = True
                        continue
                    # Degenerate output (a solid or near-solid patch) is a
                    # property of the CROP, so test it once here rather than
                    # re-testing the composited frame every time the crop is
                    # reused - and judge it against the region it replaces, not
                    # an absolute floor. An absolute floor rejected legitimately
                    # low-detail faces: motion blur during exactly the fast
                    # movement this build has to handle, shallow depth of field,
                    # deep shadow. Every rejection put the real face back for a
                    # frame, converting a non-problem into a visible flick.
                    try:
                        tgt = cv2.warpAffine(frm, M, (fake.shape[1], fake.shape[0]),
                                             borderMode=cv2.BORDER_REPLICATE)
                        if not _face_region_ok(fake, (0, 0, fake.shape[1], fake.shape[0]),
                                               reference=tgt):
                            stats["swap_skips"] += 1
                            continue
                    except Exception:
                        pass
                    post = _aligned_post(slot == primary_slot)
                    if post is not None:
                        try:
                            pf = post(fake)
                            if pf is not None and pf.shape == fake.shape:
                                fake = pf
                        except Exception:
                            pass
                    corr = _E.aligned_correction(rec["kps"], M, int(fake.shape[0]))
                    produced_crops.append((slot, fake, corr))

                stats["swap_times"].append((time.perf_counter() - t_swap0) * 1000)
                return k, produced_crops

            todo = list(key_indices_for_swap)
            # No-face / no-pair keys still count toward progress (audit NSDOS-012).
            keys_done += max(0, len(key_indices) - len(todo))
            _tick()

            pending = {ex.submit(swap_frame, k) for k in todo}
            while pending:
                if _cxl():
                    for fut in pending:
                        fut.cancel()
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise _CancelledJob()
                done_futs, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                if not done_futs:
                    _tick()
                    continue
                for fut in done_futs:
                    try:
                        k, crops = fut.result()
                        for slot, fake, corr in crops:
                            if slot in _aligned_hist:
                                _aligned_hist[slot].append((int(gidx[k]), fake, corr))
                    except Exception as e:
                        logging.warning("swap worker failed: %s", e)
                    keys_done += 1
                    _tick()
            for slot in _aligned_hist:
                _aligned_hist[slot].sort(key=lambda t: t[0])

            def _safe_result_put(item):
                while True:
                    if _cxl(): raise _CancelledJob()
                    try:
                        result_q.put(item, timeout=8.0)
                        return True
                    except queue.Full:
                        if not writer_t.is_alive():
                            logging.error("Writer thread died — aborting result_q puts")
                            return False
                        continue

            # ---------------------------------------------------------------
            # Phase 2b (sequential, strict frame order) — composite every
            # output frame.
            #
            # This is the change that removes the flicker family. Previously
            # only key frames were composited and the frames between them were
            # filled by copying a face ROI out of a neighbouring frame, cross-
            # fading two whole frames, or - in several branches - emitting the
            # untouched original. Now every frame gets a real composite: the
            # aligned swap texture is reused, but the placement, the mask, the
            # background and the colour match are this frame's own. The face
            # therefore tracks the head continuously through fast movement,
            # never carries a neighbouring frame's exposure, and the original
            # face is never re-exposed mid-shot.
            # ---------------------------------------------------------------
            emit_upto = n - 1
            # No more input can arrive, so nothing is left to bracket toward:
            # emit everything now instead of deferring it forever.
            if not (eof or produced >= lim):
                last_key_g = max((gidx[k] for k in key_indices), default=None)
                if last_key_g is not None:
                    # Frames past the last key frame have nothing to interpolate
                    # toward yet. Hand them to the next chunk rather than
                    # guessing - guessing here is what produced the periodic
                    # original-face flash at every chunk boundary.
                    emit_upto = max(-1, max((k for k in range(n) if gidx[k] <= last_key_g),
                                            default=-1))
                    _pending_tail = [(gidx[k], cframes[k]) for k in range(emit_upto + 1, n)]

            def _nearest_aligned(slot, g):
                hist = _aligned_hist.get(slot) or []
                if not hist:
                    return None
                best, bd = None, None
                for gi, fake, corr in hist:
                    d = abs(gi - g)
                    if bd is None or d < bd:
                        best, bd = (fake, corr), d
                return best

            swap_keys = set(key_indices_for_swap)
            writer_ok = True
            for k in range(emit_upto + 1):
                if not writer_ok: break
                if _cxl(): raise _CancelledJob()
                g = gidx[k]
                frm = cframes[k]

                records = _records_at(g)
                if not records:
                    # Genuinely nothing tracked here (before the first face
                    # appears, or long after the last one left).
                    writer_ok = _safe_result_put(frm)
                    if writer_ok: stats["frames_out"] += 1
                    continue

                orig = frm
                out = frm.copy()
                # Painter's order: furthest (smallest) first, nearest last, so
                # the nearer person wins the contested pixels where two faces
                # touch.
                order = sorted(records.keys(),
                               key=lambda sl: float((records[sl]["bbox"][2] - records[sl]["bbox"][0]) *
                                                    (records[sl]["bbox"][3] - records[sl]["bbox"][1])))
                any_ok = False
                for slot in order:
                    rec = records[slot]
                    # A face the detector can still SEE is swapped wherever it
                    # is, including half out of shot - the detector vouches for
                    # it. A face that is only being HELD or predicted is a
                    # different matter: once someone walks out of frame the
                    # detector stops reporting them, the tracker keeps
                    # extrapolating, and the paste ends up pinned against the
                    # frame edge on top of whatever is there. Require a held
                    # face to still be substantially inside the frame, and ramp
                    # its opacity down rather than letting it pop.
                    if not rec.get("det"):
                        # This face is being HELD, not seen. Two very different
                        # situations produce that, and they need opposite
                        # treatment:
                        #
                        #   occluded mid-frame (a hand, a turn) -> hold, which
                        #       is exactly what stops the original face
                        #       flashing back;
                        #   walked out of shot -> stop, because the detector
                        #       will never report them again and the tracker
                        #       will happily extrapolate a face onto whatever
                        #       is left behind.
                        #
                        # The last REAL detection tells them apart: if it was
                        # already in contact with a frame border, the subject
                        # was on their way out. The smoothed bbox cannot be
                        # used for this - it lags the subject and then stalls
                        # short of the edge, which is precisely how a face ends
                        # up painted mid-frame over an empty background.
                        if _touches_frame_edge(rec.get("hit_bbox"), frm.shape):
                            continue
                        if _touches_frame_edge(rec["bbox"], frm.shape):
                            continue
                        keep = _frame_containment(rec["bbox"], frm.shape)
                        if keep < 0.60:
                            continue
                        if keep < 0.85:
                            rec = dict(rec)
                            rec["alpha"] *= (keep - 0.60) / 0.25
                            if rec["alpha"] <= 0.02:
                                continue
                    rivals = _rival_landmarks(records, slot)
                    cached = _nearest_aligned(slot, g)
                    if cached is not None:
                        trial, ok = _reuse_one(out, orig, rec, quality,
                                               rivals=rivals, cached=cached)
                    elif _legacy_mode[0]:
                        # This inswapper build does not expose its affine, so no
                        # aligned result can be cached or re-projected. Run the
                        # legacy image-space swap on THIS frame directly:
                        # slower, but still every frame, so the artefact profile
                        # does not regress to "some frames show the real face".
                        face = _E.PredictedFace(rec["bbox"], rec["kps"], rec["lmk"], None, 0.5)
                        trial = _swap_one_legacy(out, orig, face, rec["src"], quality,
                                                 alpha=rec["alpha"])
                        ok = trial is not None
                    else:
                        trial, ok = None, False
                    if not ok or trial is None:
                        continue
                    out = trial
                    any_ok = True

                if not any_ok:
                    stats["swap_skips"] += 1
                    out = frm
                elif k not in swap_keys:
                    stats["frames_filled"] += 1

                writer_ok = _safe_result_put(out)
                if writer_ok: stats["frames_out"] += 1

                # Drop aligned crops that no later frame can still be nearest
                # to, so peak memory stays a few crops rather than one per
                # swapped frame in the chunk.
                horizon = g - 2 * max(1, skip_n, swap_gap_base)
                for _sl, _h in _aligned_hist.items():
                    if len(_h) > 2:
                        _aligned_hist[_sl] = [e for e in _h if e[0] >= horizon] or _h[-1:]

            # Trim history: keep the last anchor on each side of the boundary so
            # the next chunk's leading frames are still bracketed.
            keep_from = None
            if _pending_tail:
                keep_from = _pending_tail[0][0]
            elif gidx:
                keep_from = gidx[-1]
            if keep_from is not None:
                for slot in _geom_hist:
                    h = _geom_hist[slot]
                    idx = max((i for i, (gi, _r) in enumerate(h) if gi <= keep_from), default=None)
                    trimmed = h[idx:] if idx is not None else h[-1:]
                    # ALWAYS keep the single most recent REAL (det=True) entry
                    # too, however far back it falls, even once everything
                    # else around it has been trimmed away. Without this, a
                    # long run of carried/predicted entries (a subject turned
                    # away for longer than one chunk) leaves the most recent
                    # survivor of the trim above as a CARRIED entry, and the
                    # last real sighting is discarded entirely. The next
                    # chunk's _geom_for_frame then has no "observed" anchor to
                    # measure staleness against, falls through to its
                    # cold-start fallback, and can bracket that stale carried
                    # position against a distant FUTURE real detection -
                    # interpolating confidently across the whole remaining
                    # gap. That is the same reported "ghost face" defect
                    # re-entering through the one boundary the render-time fix
                    # (_geom_for_frame's real_dist check) cannot see across,
                    # because by then the real anchor it depends on is simply
                    # gone. A single retained dict per slot costs nothing.
                    last_real_idx = max(
                        (i for i, (_gi, r) in enumerate(h) if r.get("det")), default=None)
                    if last_real_idx is not None and (
                            not trimmed or h[last_real_idx][0] < trimmed[0][0]):
                        trimmed = [h[last_real_idx]] + trimmed
                    _geom_hist[slot] = trimmed
                for slot in _aligned_hist:
                    _aligned_hist[slot] = (_aligned_hist[slot] or [])[-1:]

            if not writer_ok:
                logging.error("Writer failed mid-job — stopping further chunks")
                break

        stop_io.set()
        try: result_q.put(None, timeout=5)
        except Exception: pass
        try: reader_t.join(timeout=8)
        except Exception: pass
        try: writer_t.join(timeout=60)
        except Exception: pass

        if ex: ex.shutdown(wait=True)
        try:
            if wr is not None: wr.release()
        except Exception: pass
        try: cap.release()
        except Exception: pass

        frame_total = time.time() - t0
        logging.info(f"Frame processing complete in {frame_total:.1f}s — processed {stats['frames_out']} frames; finalizing FFmpeg")

        with _lock:
            if jid in jobs:
                jobs[jid]["eta_seconds"] = None
                _persist_job(jid)
        # IMPORTANT: do not kill FFmpeg here. The writer has finished feeding
        # frames, so closing stdin signals EOF and lets the encoder flush/finalize
        # the MP4. The previous V4 cleanup killed a still-running encoder and then
        # waited on the already-killed process, which surfaced as
        # "FFmpeg encoding failed" even when the frames themselves were valid.
        u(99, "Processing complete · finalizing video…")
        enc_t0 = time.time()
        try:
            if enc_proc.stdin is not None:
                enc_proc.stdin.close()
        except Exception:
            pass
        try:
            enc_rc = enc_proc.wait(timeout=max(60, int(trim_duration_sec * 20)))
        except subprocess.TimeoutExpired:
            try: enc_proc.kill()
            except Exception: pass
            try: enc_proc.wait(timeout=10)
            except Exception: pass
            raise RuntimeError("FFmpeg encoding timed out")
        try:
            if _enc_log is not None:
                _enc_log.close()
        except Exception:
            pass
        try:
            with open(_enc_log_path, "rb") as _ef:
                enc_err = _ef.read()[-4000:].decode("utf-8", "replace")
        except Exception:
            enc_err = ""
        stats["encode_seconds"] = max(0.0, time.time() - enc_t0)
        if enc_rc != 0:
            raise RuntimeError("FFmpeg encoding failed" + (f" (exit={enc_rc}): {enc_err.strip()}" if enc_err.strip() else f" (exit={enc_rc})"))
        if not os.path.isfile(final) or os.path.getsize(final) < 1000:
            raise RuntimeError("Output video invalid or empty")
        try:
            os.remove(_enc_log_path)
        except Exception:
            pass
        stats["processing_seconds"] = max(0.0, time.time() - t0 - stats["encode_seconds"])
        # Keep the final output path explicit before optional encryption/autosave.
        # The previous V4 runtime referenced `res` before assigning it, which
        # caused the otherwise successful job to emit:
        # "local variable 'res' referenced before assignment".
        res = final
        logging.info(
            "Done in %.1fs — frames=%d; detector_calls=%d; swap_calls=%d; detector_avg=%.1fms; swap_avg=%.1fms; encode=%.1fs; reused=%d; kps_rejected=%d",
            time.time() - t0, stats["frames_out"], stats["detector_calls"], stats["swap_calls"],
            (sum(stats["det_times"]) / len(stats["det_times"])) if stats["det_times"] else 0.0,
            (sum(stats["swap_times"]) / len(stats["swap_times"])) if stats["swap_times"] else 0.0,
            stats["encode_seconds"], stats["frames_filled"], stats["kps_rejected"],
        )

        pw = (cfg.get('password') or "").strip()
        enc_note = ""
        if pw:
            zpath = f"/tmp/result_{jid}.zip"
            try:
                _encrypt_zip(res, pw, zpath)
                try: os.remove(res)
                except: pass
                res = zpath
                enc_note = " · 🔒 encrypted"
            except Exception as e:
                raise RuntimeError(f"Encryption failed: {e}")

        el = int(time.time() - t0)
        m, s = el // 60, el % 60
        # AUTHORITATIVE SERVER SAVE — must succeed before status=done.
        server_path, expires_at = _server_save_result(jid, res)
        # The temporary processing copy is no longer needed; keep the server
        # copy as the canonical result. Android may download it whenever it
        # becomes active again, including after a process/app restart.
        if os.path.abspath(server_path) != os.path.abspath(res):
            try: os.remove(res)
            except Exception: pass
        res = server_path
        save_note = f" · ☁ server saved · expires {time.strftime('%H:%M:%S', time.localtime(expires_at))}"
        try:
            # Optional encrypted autosave remains supplementary, never authoritative.
            extra = _auto_save_result(jid, res, password=pw)
            if extra: save_note += extra
        except Exception as e:
            logging.warning("optional auto-save: %s", e)
        with _lock:
            if jid in jobs:
                jobs[jid].update(
                    status='done', result_path=res, server_result_path=res, progress=100, done_at=time.time(), expires_at=expires_at, eta_seconds=None,
                    message=f"Done — {produced} frames · swaps={stats['swap_calls']} · detects={stats['detector_calls']} · {ow}×{oh} · {quality} · "
                            f"{f'{m}m {s}s' if m else f'{s}s'} · CPU V4{enc_note}{save_note}"
                )
        _persist_job(jid, force=True)
        _touch_session_expiry()
    except _CancelledJob:
        try: stop_io.set()
        except Exception: pass
        try:
            if ex: ex.shutdown(wait=False, cancel_futures=True)
        except Exception: pass
        try:
            if result_q is not None: result_q.put(None, timeout=1)
        except Exception: pass
        try: wr.release()
        except Exception: pass
        try: cap.release()
        except Exception: pass
        with _lock:
            if jid in jobs:
                jobs[jid].update(status='cancelled', done_at=time.time(), message="✋ Cancelled")
                _persist_job(jid, force=True)
    except Exception as e:
        try: stop_io.set()
        except Exception: pass
        try:
            if enc_proc is not None:
                if enc_proc.stdin is not None:
                    enc_proc.stdin.close()
                if enc_proc.poll() is None:
                    enc_proc.kill()
                    enc_proc.wait(timeout=10)
        except Exception: pass
        with _lock:
            if jid in jobs:
                jobs[jid].update(status='error', message=str(e)[:90], done_at=time.time())
                _persist_job(jid, force=True)
    finally:
        try: stop_io.set()
        except Exception: pass
        if ex:
            try: ex.shutdown(wait=True)
            except: pass
        for p in list(src_paths.values()) + [vp]:
            try: os.remove(p)
            except: pass

def _age(ts):
    a = int(time.time() - ts)
    return f"{a}s ago" if a < 60 else (f"{a//60}m ago" if a < 3600 else f"{a//3600}h ago")


def _eta_str(j):
    try:
        p = int(j.get("progress") or 0)
        if p <= 0 or p >= 100:
            return ""
        eta = j.get("eta_seconds")
        if eta is None:
            return ""
        eta = max(0.0, float(eta))
        m, s = int(eta) // 60, int(eta) % 60
        if m >= 60:
            h, m = divmod(m, 60)
            return f"ETA {h}h {m}m"
        if m:
            return f"ETA {m}m {s}s"
        return f"ETA {s}s"
    except Exception:
        return ""


def _owned_job_ids(session_id=None):
    """Return all Phoenix jobs visible to this private Space UI.

    Gradio session ids change on browser refresh. Jobs therefore must not be
    hidden merely because the browser received a new session id. The session
    id is retained on each job for diagnostics/future multi-user isolation.
    """
    with _lock:
        return list(jobs.keys())


def _hist_html(session_id=None):
    ids = _owned_job_ids(session_id)
    if not ids:
        return '<div class="empty"><p class="emoji">📭</p><p>No jobs yet</p></div>'
    rows = []
    for jid in reversed(ids):
        with _lock:
            if jid not in jobs: continue
            j = jobs[jid].copy()
        s, p, msg, ts = j['status'], j['progress'], j.get('message',''), j.get('created_at', time.time())
        if s == 'done':
            badge = '<span class="bdg bdg-ok">✓ Done</span>'; edge = '#10B981'
        elif s == 'error':
            badge = '<span class="bdg bdg-err">✗ Error</span>'; edge = '#EF4444'
        elif s == 'cancelled':
            badge = '<span class="bdg bdg-err" style="background:#F59E0B">✋ Cancelled</span>'; edge = '#F59E0B'
        else:
            eta = _eta_str(j)
            eta_bit = f" · {eta}" if eta else ""
            badge = f'<span class="bdg bdg-run">⏳ {p}%{eta_bit}</span>'; edge = '#1558B0'
        rows.append(f'<div class="hcard" style="border-left-color:{edge}"><div class="hrow"><code class="hid">{jid}</code>{badge}'
                    f'<span class="hage">{_age(ts)}</span></div><div class="hmsg">{msg}</div></div>')
    return '<div>' + ''.join(rows) + '</div>'

def _get_done_choices(session_id=None):
    ids = _owned_job_ids(session_id)
    with _lock:
        return [jid for jid in ids if jobs.get(jid, {}).get('status') == 'done' and jobs.get(jid, {}).get('result_path')]

def load_result(jid):
    hide = gr.update(visible=False)
    if not jid: return hide, hide, "Select a job"
    with _lock:
        if jid not in jobs: return hide, hide, "Not found"
        j = jobs[jid].copy()
    if j['status'] == 'done':
        rp = j.get('result_path')
        if not rp or not os.path.exists(rp): return hide, hide, "File deleted"
        if rp.endswith('.zip'):
            return hide, gr.update(value=rp, visible=True), j.get('message', 'Ready to download')
        else:
            return (
                gr.update(value=rp, visible=True),
                gr.update(value=rp, visible=True),
                j.get('message', 'Video loaded — use the Download button below')
            )
    return hide, hide, f"{j['status']}"

def _running_job_ids():
    with _lock:
        return [jid for jid, j in jobs.items() if j.get("status") not in ("done", "error", "cancelled")]


def _running_job_choices(session_id=None):
    out = []
    for jid in _owned_job_ids(session_id):
        with _lock:
            j = jobs.get(jid) or {}
        if j.get("status") in ("done", "error", "cancelled"):
            continue
        p = int(j.get("progress", 0))
        msg = (j.get("message") or "")[:40]
        out.append(f"{jid} · {p}% · {msg}")
    return out


def _ids_from_cancel_selection(selected):
    if not selected: return []
    if isinstance(selected, str): selected = [selected]
    ids = []
    for item in selected:
        s = str(item).strip()
        if not s: continue
        jid = s.split("·")[0].strip().split()[0].strip()
        if jid: ids.append(jid)
    return ids


def cancel_running(selected=None, session_id=None):
    want = _ids_from_cancel_selection(selected)
    n = 0
    owned = set(_owned_job_ids(session_id))
    with _lock:
        targets = want if want else list(owned)
        for jid in targets:
            if jid not in owned:
                continue
            j = jobs.get(jid)
            if not j:
                continue
            if j.get("status") not in ("done", "error", "cancelled"):
                j["cancel"] = True
                j["message"] = "Cancel requested…"
                n += 1
    choices = _running_job_choices()
    msg = f"✋ Cancel requested for {n} job(s)" if n else ("No matching running job" if want else "No running job")
    return (msg, _hist_html(), _video_progress_html(), _status_banner_html(), gr.update(choices=choices, value=[]))


def cancel_all_running(): return cancel_running(None)

def delete_now(jid, session_id=None):
    if not jid:
        return "Select a job", _hist_html(session_id), gr.update(choices=_get_done_choices(session_id)), gr.update(visible=False), _video_progress_html()
    owned = set(_owned_job_ids(session_id))
    if jid not in owned:
        return "Job not found", _hist_html(session_id), gr.update(choices=_get_done_choices(session_id)), gr.update(visible=False), _video_progress_html()
    with _lock:
        j = jobs.pop(jid, None)
        _remove_job_snapshot(jid)
        _JOB_PERSIST_LAST.pop(jid, None)
        if j and j.get('result_path'):
            try:
                os.remove(j['result_path'])
            except Exception:
                pass
    return f"Deleted {jid}", _hist_html(session_id), gr.update(choices=_get_done_choices(session_id)), gr.update(visible=False), _video_progress_html()

def clear_history(session_id=None):
    removed = 0
    owned = set(_owned_job_ids(session_id))
    with _lock:
        for k in list(owned):
            j = jobs.get(k)
            if not j:
                continue
            if j['status'] in ('done', 'error', 'cancelled'):
                rp = j.get('result_path')
                if rp:
                    try:
                        os.remove(rp)
                    except Exception:
                        pass
                del jobs[k]
                _remove_job_snapshot(k)
                removed += 1
    return f"Cleared {removed} jobs", _hist_html(session_id), gr.update(choices=_get_done_choices(session_id)), _video_progress_html()

def _pick_focus_job(session_id=None):
    ids = _owned_job_ids(session_id)
    with _lock:
        if not ids: return None, {}
        for jid in reversed(ids):
            j = jobs.get(jid) or {}
            if j.get("status") not in ("done", "error", "cancelled"):
                return jid, j.copy()
        jid = ids[-1]
        return jid, (jobs.get(jid) or {}).copy()


def _status_banner_html(session_id=None):
    jid, j = _pick_focus_job(session_id)
    if not jid:
        return (
            '<div class="stat-banner idle">'
            '<div class="sb-left"><span class="sb-pct">Idle</span>'
            '<span class="sb-msg">No active job</span></div></div>'
        )
    s = j.get("status")
    p = int(j.get("progress", 0))
    msg = (j.get("message") or "").replace("<", "&lt;")
    if len(msg) > 64: msg = msg[:61] + "…"
    if s == "done":
        pct, fill, badge = "100%", "width:100%;background:linear-gradient(90deg,#10B981,#34D399)", "✓ Done"
    elif s == "error":
        pct, fill, badge = "—", "width:100%;background:#EF4444", "✗ Error"
    elif s == "cancelled":
        pct, fill, badge = "—", "width:100%;background:#F59E0B", "✋ Cancelled"
    else:
        eta = _eta_str(j)
        pct = f"{p}%" + (f" · {eta}" if eta else "")
        fill, badge = f"width:{max(p, 3)}%", "Processing"
    return (
        f'<div class="stat-banner">'
        f'<div class="sb-left">'
        f'<span class="sb-pct">{pct}</span>'
        f'<span class="sb-id">{jid}</span>'
        f'<span class="bdg bdg-run">{badge}</span>'
        f'</div>'
        f'<span class="sb-msg">{msg}</span>'
        f'<div class="sb-bar"><div class="sb-fill" style="{fill}"></div></div>'
        f'</div>'
    )


def _video_progress_html(session_id=None):
    jid, j = _pick_focus_job(session_id)
    if not jid:
        return '<div class="vprog idle"><span class="vp-dot"></span> Idle — no jobs</div>'
    s, p, msg = j.get("status"), int(j.get("progress", 0)), (j.get("message") or "").replace("<", "&lt;")
    if len(msg) > 72: msg = msg[:69] + "…"
    if s == "done":
        head = f'<span class="vp-id">{jid}</span><span class="bdg bdg-ok">✓ Done</span>'
        fill = "width:100%;background:linear-gradient(90deg,#10B981,#34D399)"
        pct = "100%"
    elif s == "error":
        head = f'<span class="vp-id">{jid}</span><span class="bdg bdg-err">✗ Error</span>'
        fill = "width:100%;background:#EF4444"
        pct = "ERR"
    elif s == "cancelled":
        head = f'<span class="vp-id">{jid}</span><span class="bdg bdg-err" style="background:#F59E0B">✋ Cancelled</span>'
        fill = "width:100%;background:#F59E0B"
        pct = "STOP"
    else:
        eta = _eta_str(j)
        head = f'<span class="vp-id">{jid}</span><span class="bdg bdg-run">Processing</span>'
        fill = f"width:{max(p, 3)}%"
        pct = f"{p}%" + (f" · {eta}" if eta else "")
    return (
        f'<div class="vprog"><div class="vp-head">{head}</div>'
        f'<div class="vp-pct">{pct}</div>'
        f'<div class="vprog-bar"><div class="vprog-fill" style="{fill}"></div></div>'
        f'<div class="vp-msg">{msg}</div></div>'
    )


SESSION_ROOT = Path("/tmp/swamitech_sessions")
_SESSION_TTL_SEC = 86400

def _safe_session_id(request=None) -> str:
    """Isolate users by Gradio session_hash when available."""
    import re as _re
    sid = None
    if request is not None:
        sid = getattr(request, "session_hash", None) or getattr(request, "session_id", None)
    if not sid:
        sid = "anon"
    return _re.sub(r"[^A-Za-z0-9_-]", "_", str(sid))[:80]


def _session_dir(request=None) -> Path:
    path = SESSION_ROOT / _safe_session_id(request)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _session_json(request=None) -> Path:
    return _session_dir(request) / "session.json"


def _touch_session_expiry(request=None, extra=None):
    data = {}
    sj = _session_json(request)
    try:
        if sj.is_file():
            import json
            with open(sj, "r") as f:
                data = json.load(f)
    except Exception:
        data = {}
    if extra:
        data.update(extra)
    data["expires_at"] = time.time() + _SESSION_TTL_SEC
    data["saved_at"] = time.time()
    try:
        import json
        with open(sj, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logging.warning("session save failed: %s", e)


def _save_ui_session(vid, s1, s2, s3, s4, settings: dict, request=None):
    try:
        import json
        session_dir = _session_dir(request)
        media = {}
        if vid is not None and isinstance(vid, str) and os.path.isfile(vid):
            dest = str(session_dir / "target_video.mp4")
            shutil.copy2(vid, dest)
            media["video"] = dest
        for i, im in enumerate([s1, s2, s3, s4]):
            if im is None:
                continue
            dest = str(session_dir / f"face_{i}.jpg")
            try:
                if isinstance(im, str) and os.path.isfile(im):
                    shutil.copy2(im, dest)
                else:
                    arr = im
                    if isinstance(arr, np.ndarray):
                        if arr.dtype != np.uint8:
                            arr = np.clip(arr, 0, 255).astype(np.uint8)
                        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if (arr.ndim == 3 and arr.shape[2] == 3) else arr
                        cv2.imwrite(dest, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                media[f"face_{i}"] = dest
            except Exception as e:
                logging.warning("session face %s save failed: %s", i, e)
        payload = {
            "settings": settings,
            "media": media,
            "saved_at": time.time(),
            "expires_at": time.time() + _SESSION_TTL_SEC,
            "session_id": _safe_session_id(request),
        }
        with open(_session_json(request), "w") as f:
            json.dump(payload, f)
    except Exception as e:
        logging.warning("_save_ui_session: %s", e)


def _load_ui_session(request=None):
    import json
    empty = (None, None, None, None, None, {})
    try:
        sj = _session_json(request)
        if not sj.is_file():
            return empty
        with open(sj, "r") as f:
            data = json.load(f)
        if time.time() > float(data.get("expires_at") or 0):
            # Expired — delete session media (audit NSDOS-016)
            try:
                shutil.rmtree(_session_dir(request), ignore_errors=True)
            except Exception:
                pass
            return empty
        media = data.get("media") or {}
        settings = data.get("settings") or {}
        vid = media.get("video") if media.get("video") and os.path.isfile(media["video"]) else None
        faces = []
        for i in range(4):
            pth = media.get(f"face_{i}")
            if pth and os.path.isfile(pth):
                img = cv2.imread(pth)
                faces.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None)
            else:
                faces.append(None)
        return vid, faces[0], faces[1], faces[2], faces[3], settings
    except Exception as e:
        logging.warning("_load_ui_session: %s", e)
        return empty


def _parse_device_mode(mode):
    s = (mode or "").lower()
    if "jarvislabs" in s:
        return "jarvislabs"
    return "cpu" if ("cpu only" in s or s.strip() == "cpu") else "gpu"


def submit_video(s1, s2, s3, s4, vid, secs, fps, res, quality, enhancer, swap_n, det_n, det_int, password, refs, trim_start, trim_end, face_mode, enhance_scope, device_mode, request: gr.Request = None):
    if vid is None:
        return "❌ Upload a target video", _hist_html(), gr.update(choices=_get_done_choices()), _video_progress_html(), _device_status_text()
    if s1 is None and s2 is None and s3 is None and s4 is None:
        return "❌ Upload at least one replacement face", _hist_html(), gr.update(choices=_get_done_choices()), _video_progress_html(), _device_status_text()

    pref = _parse_device_mode(device_mode)
    _device_pref[0] = pref
    want_gpu = pref == "gpu"
    use_gpu = _gpu_worth_trying(want_gpu)
    want_jarvislabs = pref == "jarvislabs"

    # Cheap, local-only, no-network check - decides whether it is even worth
    # trying the remote path below, and lets the immediate return message
    # accurately say which device will actually run the job instead of
    # promising remote GPU and silently falling back.
    jarvislabs_allowed, jarvislabs_reason = (False, "not selected")
    if want_jarvislabs:
        try:
            from jarvislabs_adapter import GOVERNOR as _JL_GOVERNOR
            jarvislabs_allowed, jarvislabs_reason = _JL_GOVERNOR.can_start()
        except Exception as e:
            jarvislabs_allowed, jarvislabs_reason = False, f"adapter unavailable: {e}"

    if not use_gpu and not (want_jarvislabs and jarvislabs_allowed):
        try: MODELS.get(prefer_gpu=False)
        except Exception as e:
            return f"❌ Model load failed: {e}", _hist_html(), gr.update(choices=_get_done_choices()), _video_progress_html(), _device_status_text()

    jid = str(uuid.uuid4())[:8].upper()
    src_paths = {}
    for idx, im in enumerate([s1, s2, s3, s4]):
        if im is not None:
            p = f"/tmp/src_{jid}_{idx}.jpg"
            cv2.imwrite(p, _to_bgr(im), [cv2.IMWRITE_JPEG_QUALITY, 95])
            src_paths[idx] = p
    vp = f"/tmp/vid_{jid}.mp4"
    shutil.copy(vid, vp)
    sid = _safe_session_id(request)
    with _lock:
        jobs[jid] = dict(
            status='processing', progress=0, message='Queued…',
            result_path=None, created_at=time.time(), cancel=False, eta_seconds=None,
            session_id=sid,
        )
    _persist_job(jid)
    scope = "all" if enhance_scope and str(enhance_scope).lower().startswith("all") else "primary"
    cfg = dict(max_seconds=int(secs), fps=int(fps), resolution=res, quality=quality,
               enhancer=enhancer or "None",
               enhance_scope=scope,
               refs=refs or [], swap_n=swap_n, det_n=det_n, det_int=det_int,
               password=(password or "").strip(),
               trim_start=float(trim_start or 0),
               trim_end=float(trim_end or 100),
               face_mode=face_mode or "1 face (fastest)",
               device_mode=pref,
               use_gpu=use_gpu)

    try:
        _save_ui_session(
            vid, s1, s2, s3, s4,
            dict(
                secs=int(secs), fps=int(fps), res=res, quality=quality,
                enhancer=enhancer or "None", swap_n=str(swap_n), det_n=str(det_n),
                det_int=str(det_int), face_mode=face_mode or "1 face (fastest)",
                enhance_scope=enhance_scope or "Primary face only (faster)",
                device_mode=device_mode or "CPU only",
                trim_start=float(trim_start or 0), trim_end=float(trim_end or 100),
            ),
            request=request,
        )
    except Exception as e:
        logging.warning(f"session persist: {e}")

    if want_jarvislabs:
        if not jarvislabs_allowed:
            # Cap already exhausted (or adapter unavailable) - go straight to
            # local CPU, exactly like the normal path below, but say why in
            # the message so "why did this run on CPU" has an answer.
            VIDEO_EXECUTOR.submit(_run_job_body, jid, src_paths, vp, cfg)
            return (
                f"✓ Job {jid} on CPU · {face_mode} · Jarvislabs GPU unavailable ({jarvislabs_reason})",
                _hist_html(),
                gr.update(choices=_get_done_choices()),
                _video_progress_html(),
                _device_status_text(),
            )
        VIDEO_EXECUTOR.submit(_run_job_remote_or_fallback, jid, src_paths, vp, cfg)
        return (
            f"✓ Job {jid} started · {face_mode} · device=Jarvislabs GPU (remote)",
            _hist_html(),
            gr.update(choices=_get_done_choices()),
            _video_progress_html(),
            _device_status_text(),
        )

    if use_gpu and HAS_SPACES and _is_zerogpu_space():
        try:
            _run_job_on_gpu(jid, src_paths, vp, cfg)
            _gpu_try_cached[0] = True
            with _lock: j = jobs.get(jid, {})
            msg = j.get("message") or f"✓ Job {jid} finished on GPU"
            st = j.get("status", "done")
            return (
                f"{'✓' if st == 'done' else '✗'} {msg}",
                _hist_html(),
                gr.update(choices=_get_done_choices()),
                _video_progress_html(),
                _device_status_text(),
            )
        except Exception as e:
            err = str(e)
            if any(x in err.lower() for x in ("not supported", "no gpu", "cpu only", "not a zero")):
                _gpu_try_cached[0] = False
            cfg = dict(cfg)
            cfg["use_gpu"] = False
            with _lock:
                if jid in jobs:
                    jobs[jid].update(status="processing", message=f"GPU failed — continuing on CPU…", progress=1)
            VIDEO_EXECUTOR.submit(_run_job_body, jid, src_paths, vp, cfg)
            return (
                f"✓ Job {jid} on CPU · {face_mode}",
                _hist_html(),
                gr.update(choices=_get_done_choices()),
                _video_progress_html(),
                _device_status_text(),
            )

    VIDEO_EXECUTOR.submit(_run_job_body, jid, src_paths, vp, cfg)
    dev = "GPU" if use_gpu else "CPU"
    return (
        f"✓ Job {jid} started · {face_mode} · enhance={scope} · device={dev}",
        _hist_html(),
        gr.update(choices=_get_done_choices()),
        _video_progress_html(),
        _device_status_text(),
    )


if HAS_SPACES:
    @_spaces.GPU(duration=600)
    def _run_job_on_gpu(jid, src_paths, vp, cfg):
        cfg = dict(cfg)
        cfg["use_gpu"] = True
        return _run_job_body(jid, src_paths, vp, cfg)
else:
    def _run_job_on_gpu(jid, src_paths, vp, cfg):
        return _run_job_body(jid, src_paths, vp, cfg)


def _run_job_remote_or_fallback(jid, src_paths, vp, cfg):
    """Runs on VIDEO_EXECUTOR's background thread (submitted from
    submit_video()'s "jarvislabs" branch) - tries the remote Jarvislabs GPU
    job end to end (governor check already passed synchronously before this
    was even submitted; this call still re-checks, since time has passed),
    and on ANY failure (network, remote error, cap crossed between the sync
    pre-check and now) falls back to ordinary local CPU processing rather
    than leaving the job stuck. Mirrors _run_job_on_gpu's fallback
    philosophy ("GPU failed - continuing on CPU"), done asynchronously here
    instead of blocking submit_video()'s return, because a remote video job
    can run for minutes and the rest of this app's UI already knows how to
    show live progress for a job running on VIDEO_EXECUTOR.
    """
    try:
        from jarvislabs_adapter import run_job_on_jarvislabs
        run_job_on_jarvislabs(jid, src_paths, vp, cfg, jobs, _lock)
    except Exception as e:
        logging.warning("Jarvislabs remote job failed for %s, falling back to local CPU: %s", jid, e)
        with _lock:
            if jid in jobs:
                jobs[jid].update(
                    status="processing", progress=1,
                    message=f"Remote GPU unavailable ({str(e)[:140]}) — continuing on CPU…",
                )
        cfg2 = dict(cfg)
        cfg2["use_gpu"] = False
        _run_job_body(jid, src_paths, vp, cfg2)
