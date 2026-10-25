"""Run the REAL core_pipeline._run_job_body on a synthetic clip, with per-stage wall-time accounting.
Usage: [taskset -c 0,1] python bench_pipeline.py PACKDIR LABEL [key=value ...]   (values via env-like kwargs)"""
import os, sys, types, time, json, threading, shutil, logging, collections
PACK = sys.argv[1]; LABEL = sys.argv[2]
KW = dict(a.split("=", 1) for a in sys.argv[3:])
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = f"/tmp/claude-0/bench_work/{LABEL}"; shutil.rmtree(WORK, ignore_errors=True); os.makedirs(WORK + "/out", exist_ok=True); os.makedirs(WORK + "/jobs", exist_ok=True)
os.environ["PHOENIX_PERSISTENT_OUTPUT_DIR"] = WORK + "/out"; os.environ["PHOENIX_PERSISTENT_JOB_STATE_DIR"] = WORK + "/jobs"; os.environ["PHOENIX_PERSISTENT_ROOT"] = WORK + "/jobs"
os.environ["PHOENIX_MODEL_CACHE"] = WORK + "/models"
os.environ["PATH"] = "/tmp/claude-0/bin:" + os.environ["PATH"]
if "threads" in KW: os.environ["PHOENIX_NATIVE_THREADS"] = KW["threads"]
if "detsize" in KW: os.environ["PHOENIX_DET_SIZE"] = KW["detsize"]
for k, v in KW.items():
    if k.startswith("env_"): os.environ[k[4:]] = v

# --- stubs for the two libraries this sandbox does not have -------------------------------
g = types.ModuleType("gradio"); g.Request = type("Request", (), {}); g.update = lambda **k: k; g.Error = Exception
sys.modules["gradio"] = g
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: "/tmp/claude-0/swapmodel/inswapper_128.onnx"; sys.modules["huggingface_hub"] = hf
sys.path.insert(0, PACK)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=open(WORK + "/job.log", "w"))
import cv2, numpy as np
import core_pipeline as cp
import swap_engine
if os.environ.get("MASK_DUMP"):
    import atexit
    _DUMP = []; _orig_bm = swap_engine.AlignedCompositor.build_mask
    _MS = []; _G = [0]
    def _bm(self, face, M, size, **kw):
        m = _orig_bm(self, face, M, size, **kw)
        _DUMP.append(cv2.resize(m, (128, 128), interpolation=cv2.INTER_AREA))
        tr_ = kw.get("track"); _MS.append((-1 if tr_ is None else int(getattr(tr_, "slot", -1)), np.asarray(M, np.float32).reshape(2, 3).copy(), int(size), int(_G[0])))
        return m
    swap_engine.AlignedCompositor.build_mask = _bm
    atexit.register(lambda: np.savez_compressed(WORK + "/mask_dump.npz", m=np.array(_DUMP), slot=np.array([a for a, _, _, _ in _MS]), M=np.array([b for _, b, _, _ in _MS]), size=np.array([c for _, _, c, _ in _MS]), g=np.array([d for _, _, _, d in _MS])))

# --- per-function wall-time accounting (all threads) ---------------------------------------
T = collections.defaultdict(float); C = collections.defaultdict(int); LK = threading.Lock()
def timed(mod, name, label=None):
    f = getattr(mod, name, None)
    if f is None: return
    def w(*a, **k):
        t0 = time.perf_counter()
        try: return f(*a, **k)
        finally:
            dt = time.perf_counter() - t0
            with LK: T[label or name] += dt; C[label or name] += 1
    w.__name__ = name; setattr(mod, name, w)
for n in ("_fa_get", "_embed_faces", "_enhance", "_soft_polish", "_face_region_ok", "_detect_want", "_dedupe_faces", "_keep_n_faces",
          "_pairs_for_frame", "_cached_source_face", "_grab_frame"):
    timed(cp, n)
for n in ("occluder_keep", "paste_back", "aligned_correction", "scale_affine"):
    timed(cp._E if hasattr(cp, "_E") else swap_engine, n, "E." + n)
if hasattr(cp, "_timeline_strip"):
    _orig_tl = cp._timeline_strip
    def _tl_wrap(tl, fps, nslots, bucket_sec=2.0):
        json.dump(list(tl), open(WORK + "/tl.json", "w")); return _orig_tl(tl, fps, nslots, bucket_sec)
    cp._timeline_strip = _tl_wrap
t_load = time.perf_counter(); cp._load(prefer_gpu=False); t_load = time.perf_counter() - t_load
# wrap the ORT-level models once they exist
fa, sw = cp._fa, cp._sw
def wrap_method(obj, meth, label):
    f = getattr(obj, meth)
    def w(*a, **k):
        t0 = time.perf_counter()
        try: return f(*a, **k)
        finally:
            dt = time.perf_counter() - t0
            with LK: T[label] += dt; C[label] += 1
    setattr(obj, meth, w)
wrap_method(fa.det_model, "detect", "ort.SCRFD detect")
wrap_method(sw, "get", "ort.inswapper get (incl. paste/blend if paste_back)")
for tn, m in fa.models.items():
    if tn != "detection": wrap_method(m, "get", f"ort.{tn}")
    if tn == "recognition" and hasattr(m, "get_feat"): wrap_method(m, "get_feat", "ort.recognition get_feat")

# --- the job ----------------------------------------------------------------------------------
clip = WORK + "/clip.mp4"; shutil.copy(os.path.join(HERE, KW.get("clip", "clip90.mp4")), clip)
src = {0: WORK + "/s0.jpg", 1: WORK + "/s1.jpg"}; shutil.copy(HERE + "/src_a.jpg", src[0]); shutil.copy(HERE + "/src_b.jpg", src[1])
if KW.get("one"): src = {0: src[0]}
jid = "BENCH" + LABEL[:3].upper().replace("-", "0")
if os.environ.get("BENCH_UNIQUE_JID"):
    # parallel runs: the pipeline writes /tmp/result_<jid>.mp4 and /tmp/src_<jid>_N.jpg, so two runs sharing a jid
    # overwrite each other's video and faces
    import hashlib as _hl; jid = jid + _hl.sha1(LABEL.encode()).hexdigest()[:6].upper()
cp.jobs[jid] = dict(status="processing", progress=0, message="Queued", result_path=None, created_at=time.time(), cancel=False, eta_seconds=None, session_id="bench")
cfg = dict(max_seconds=int(KW.get("secs", 360)), fps=int(KW.get("fps", 30)), resolution=KW.get("res", "720p (HD)"), quality=KW.get("quality", "Ultra"),
           enhancer=KW.get("enh", "Cinematic (clarity + smooth)"), enhance_scope=KW.get("scope", "all"), refs=[], swap_n=KW.get("swap_n", "1"),
           det_n=KW.get("det_n", "1"), det_int=int(KW.get("det_int", 4)), password="", trim_start=0.0, trim_end=100.0,
           face_mode=KW.get("face_mode", "2 faces"), device_mode="cpu", use_gpu=False, smooth_motion=KW.get("smooth", "Fast (blend)"))
t0 = time.perf_counter()
for _kv in filter(None, os.environ.get("OVR", "").split(",")):
    _k, _v = _kv.split("="); swap_engine._P[_k] = float(_v)
cp._run_job_body(jid, src, clip, cfg)
wall = time.perf_counter() - t0
j = cp.jobs[jid]
res = j.get("result_path")
print(json.dumps({"label": LABEL, "status": j.get("status"), "wall_s": round(wall, 1), "model_load_s": round(t_load, 1),
                  "threads": cp._native_threads, "cpu_count": os.cpu_count(), "affinity": len(os.sched_getaffinity(0)),
                  "out_bytes": os.path.getsize(res) if res and os.path.isfile(res) else None, "message": (j.get("message") or "")[:230]}))
rows = sorted(T.items(), key=lambda kv: -kv[1])
print("%-52s %9s %8s %9s" % ("function (wall seconds, summed over threads)", "total_s", "calls", "ms/call"))
for k, v in rows: print("%-52s %9.1f %8d %9.1f" % (k, v, C[k], 1000 * v / max(1, C[k])))
for l in open(WORK + "/job.log"):
    if "Done in" in l or "detector input" in l or "native_threads" in l or "SEARCH-BACKOFF" in l: print(l.strip()[:600])
