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
if not hasattr(cp, "_ORDER_NOW") and hasattr(cp, "_geom_for_frame"):
    # v11.2.176: packs before v11.2.95 have no frame counter at paint time - take it from the render-time geometry call, so the
    # mask dump's frame numbers (identity_check.py, contact_lag.py) line up with the clip
    cp._ORDER_NOW = [0]; _orig_gff = cp._geom_for_frame
    def _gff(timeline, g, *a, **k):
        cp._ORDER_NOW[0] = int(g); return _orig_gff(timeline, g, *a, **k)
    cp._geom_for_frame = _gff
DROP_STATS = {}
if KW.get("drop"):
    # v11.2.144: the detector "loses" a person on chosen frames - drop=A:30-100 (person A, frames 30-100), drop=A:30-100:alt (every
    # other frame of that span: a face found and lost by turns), several separated by ";" - to test what the job does when the face is
    # not found during contact (a kiss in profile: SCRFD loses one face, or both, for seconds). The frame is recognised by its
    # pixels (the frame given to _make_det_frame against the clip's own frames); every face the job's detection returns for that
    # frame (main pass, hi-res probe, zoom crops: all end in _sort_faces_left) whose box centre lies within 0.35 of a face width of
    # the person's TRUE 5-point centre is removed. Persons: A = gt index 0 (left at start), B = 1.
    _clip_src = KW.get("clip")
    _gt = np.load(os.path.splitext(_clip_src)[0] + "_gt.npz")["kps"]
    _cap = cv2.VideoCapture(_clip_src); _sigs = []
    while True:
        _ok, _f = _cap.read()
        if not _ok:
            break
        _sigs.append(cv2.resize(cv2.cvtColor(_f, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32))
    _cap.release(); _sigs = np.array(_sigs)
    _drops = {}
    for _spec in KW["drop"].split(";"):
        _pp = _spec.split(":"); _who = {"A": 0, "B": 1}[_pp[0]]; _a, _b = (int(v) for v in _pp[1].split("-"))
        _alt = len(_pp) > 2 and _pp[2] == "alt"
        for _i in range(_a, _b + 1):
            if not _alt or (_i - _a) % 2 == 0:
                _drops.setdefault(_i, set()).add(_who)
    _CUR = {"idx": -1, "sx": 1.0}
    _orig_mdf, _orig_sfl = cp._make_det_frame, cp._sort_faces_left
    def _mdf(frm, *a, **k):
        try:
            s = cv2.resize(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
            _CUR["idx"] = int(np.argmin(np.abs(_sigs - s[None]).mean(axis=(1, 2)))); _CUR["sx"] = float(frm.shape[1]) / 1280.0
        except Exception:
            _CUR["idx"] = -1
        return _orig_mdf(frm, *a, **k)
    def _sfl(faces):
        faces = _orig_sfl(faces)
        who = _drops.get(_CUR["idx"])
        if not who or not faces:
            return faces
        keep = []
        for f in faces:
            cxy = np.array([(f.bbox[0] + f.bbox[2]) / 2.0, (f.bbox[1] + f.bbox[3]) / 2.0])
            gone = False
            for w_ in who:
                k5 = _gt[_CUR["idx"], w_] * _CUR["sx"]; fw = 2.6 * float(np.linalg.norm(k5[1] - k5[0]))
                if float(np.linalg.norm(cxy - k5.mean(axis=0))) < 0.35 * fw:          # (0.75 reached the partner in contact)
                    gone = True
            if gone:
                DROP_STATS[_CUR["idx"]] = DROP_STATS.get(_CUR["idx"], 0) + 1
            else:
                keep.append(f)
        return keep
    cp._make_det_frame, cp._sort_faces_left = _mdf, _sfl
if os.environ.get("MASK_DUMP"):
    import atexit
    _DUMP = []; _orig_bm = swap_engine.AlignedCompositor.build_mask
    _MS = []; _G = [0]
    def _bm(self, face, M, size, **kw):
        m = _orig_bm(self, face, M, size, **kw)
        _DUMP.append(cv2.resize(m, (128, 128), interpolation=cv2.INTER_AREA))
        _G[0] = int(getattr(cp, "_ORDER_NOW", [0])[0])        # v11.2.144: the output frame being painted (frames without paint have no entry)
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
_OD = []
if os.environ.get("ORDER_DUMP") and hasattr(cp, "_rival_landmarks"):
    # v11.2.144: every front / behind decision of the one-face-per-pixel rule - per frame and face: whether it and the other face
    # count as seen, the outline-edge evidence (key[0] over key[1], -1 without), and whether the other face was put in front
    _orig_rl = cp._rival_landmarks
    def _rl_wrap(records, slot):
        out_ = _orig_rl(records, slot)
        me = records.get(slot) or {}
        for s2, r2 in records.items():
            if s2 == slot or r2 is None:
                continue
            key = (slot, s2) if slot < s2 else (s2, slot)
            fr = [o.get("front") for o in out_ if isinstance(o, dict) and o.get("lmk") is r2.get("lmk")]
            _OD.append({"g": int(getattr(cp, "_ORDER_NOW", [0])[0]), "slot": int(slot), "other": int(s2),
                        "me_seen": me.get("seen_near", me.get("seen")), "o_seen": r2.get("seen_near", r2.get("seen")),
                        "me_det": bool(me.get("det")), "o_det": bool(r2.get("det")), "follow": bool(me.get("follow")),
                        "cue": float(getattr(cp, "_ORDER_CUE", {}).get(key, -1.0)), "other_front": (fr[0] if fr else None)})
        return out_
    cp._rival_landmarks = _rl_wrap
_FD = []
if os.environ.get("FRONT_DUMP") and hasattr(cp, "_front_relations"):
    # v11.2.176: every front / behind decision of _front_relations - per frame and pair: the decision, the box areas, whether each
    # face counts as seen, its identity cue (xid_near), and the decision memory
    _orig_fr = cp._front_relations
    def _fr_wrap(records, g):
        rel = _orig_fr(records, g)
        for key, v in rel.items():
            ra, rb = records.get(key[0]) or {}, records.get(key[1]) or {}
            def _ar(r):
                b = r.get("bbox")
                return None if b is None else float((b[2] - b[0]) * (b[3] - b[1]))
            st = cp._FRONT_MEM.get(key)
            _FD.append({"g": int(g), "pair": list(key), "lo_front": bool(v), "area": [_ar(ra), _ar(rb)],
                        "seen": [ra.get("seen_near", ra.get("det")), rb.get("seen_near", rb.get("det"))],
                        "xid": [ra.get("xid_near"), rb.get("xid_near")], "mem": [x if not isinstance(x, bool) else int(x) for x in (st or [])]})
        return rel
    cp._front_relations = _fr_wrap
    atexit.register(lambda: json.dump(_FD, open(WORK + "/front_dump.json", "w"), default=str))
if hasattr(cp, "_timeline_strip"):
    _orig_tl = cp._timeline_strip
    def _tl_wrap(tl, fps, nslots, bucket_sec=2.0):
        json.dump(list(tl), open(WORK + "/tl.json", "w")); return _orig_tl(tl, fps, nslots, bucket_sec)
    cp._timeline_strip = _tl_wrap
t_load = time.perf_counter(); cp._load(prefer_gpu=False); t_load = time.perf_counter() - t_load
# v11.2.99: emulate the detector losing faces while two faces are in contact (a kiss): DROP_CONTACT=b drops the right-hand
# face, =both drops both, on full-frame detector calls k of each contact run with lo <= k % per < hi (DROP_RUN="lo:hi:per");
# while a full-frame call is being dropped, the pipeline's crop re-detections in that frame find nothing either.
if os.environ.get("DROP_CONTACT"):
    _orig_fa_get = cp._fa_get
    _DROP = {"k": 0, "dropped": 0, "calls": 0, "active": False, "crop_dropped": 0, "full": 0, "right_x": None}
    _lo, _hi, _per = (int(v) for v in os.environ.get("DROP_RUN", "8:26:60").split(":"))
    def _fa_get_drop(img, *a, **k):
        faces = _orig_fa_get(img, *a, **k)
        try:
            fl = list(faces or [])
        except Exception:
            return faces
        full = img is not None and img.shape[0] * img.shape[1] >= _DROP["full"]
        if full:
            _DROP["full"] = max(_DROP["full"], img.shape[0] * img.shape[1])
            _DROP["active"] = False
            if len(fl) >= 2:
                bb = sorted(fl, key=lambda f: float(f.bbox[0]))
                l, r = bb[0].bbox, bb[-1].bbox
                if float(r[0] - l[2]) < 0.15 * float(l[2] - l[0]):          # touching / overlapping
                    _DROP["calls"] += 1
                    kk = _DROP["k"]; _DROP["k"] += 1
                    if _lo <= kk % _per < _hi:
                        _DROP["dropped"] += 1
                        _DROP["active"] = True
                        if os.environ["DROP_CONTACT"] == "both":
                            return [] if not isinstance(faces, tuple) else tuple()
                        keep = [f for f in fl if f is not bb[-1]]
                        return keep if not isinstance(faces, tuple) else tuple(keep)
            return faces
        if _DROP["active"]:                                                   # a crop re-detection in a dropped frame
            _DROP["crop_dropped"] += 1
            if os.environ["DROP_CONTACT"] == "both" or not fl:
                return [] if not isinstance(faces, tuple) else tuple()
        return faces
    cp._fa_get = _fa_get_drop
    import atexit
    atexit.register(lambda: print("DROP_CONTACT", os.environ["DROP_CONTACT"], _DROP, flush=True))
# v11.2.112: BENCH_NO_FACES=1 - no face is found in the clip's frames (the replacement photos, read first, still are): for clips
# made from a real person's picture, whose face must not be swapped (the belly-clean test on MediaPipe's own pose picture)
if os.environ.get("BENCH_NO_FACES"):
    _orig_fa_get_nf = cp._fa_get
    _NF = {"on": False, "blind": 0}
    def _fa_get_nf(img, *a, **k):
        if img is not None and img.shape[0] * img.shape[1] > 120000:
            _NF["on"] = True
        if _NF["on"]:
            _NF["blind"] += 1
            return []
        return _orig_fa_get_nf(img, *a, **k)
    cp._fa_get = _fa_get_nf
    import atexit
    atexit.register(lambda: print("BENCH_NO_FACES", _NF, flush=True))
# wrap the ORT-level models once they exist
fa, sw = cp._fa, cp._sw
if os.environ.get("BENCH_NO_FACES"):            # (and the detector itself, for the pipeline's other detection paths)
    _orig_det_nf = fa.det_model.detect
    def _det_nf(img, *a, **k):
        if _NF["on"] or (img is not None and img.shape[0] * img.shape[1] > 120000):
            _NF["on"] = True; _NF["blind"] += 1
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        return _orig_det_nf(img, *a, **k)
    fa.det_model.detect = _det_nf
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
if KW.get("refs"):             # v11.2.207: Detect-frame pictures, one image path per slot ("-" = none), as the app sends them
    _rf = []
    for _p in KW["refs"].split(","):
        _e = None
        if _p and _p != "-":
            _f = cp._cached_source_face(cv2.imread(_p))
            _e = None if _f is None else getattr(_f, "normed_embedding", None)
            print("bench ref", _p, "->", "embedding" if _e is not None else "NO FACE", flush=True)
        _rf.append(_e)
    cfg["refs"] = _rf
if KW.get("patches"):          # Patch Removal (v11.2.96): make_patchclip.py ground truth -> the app's frame + mask
    _g = np.load(KW["patches"]); _pi, _pm = WORK + "/patch_frame.jpg", WORK + "/patch_mask.png"
    cv2.imwrite(_pi, _g["ref"], [cv2.IMWRITE_JPEG_QUALITY, 95]); cv2.imwrite(_pm, _g["refmask"])
    cfg["patches"] = [dict(image=_pi, mask=_pm, t=float(_g["ref_t"]), group="m1")]
    if "nmarks" in _g.files:                          # v11.2.100: make_patchclip.py multi - one mark per painted mask, all on the same frame
        cfg["patches"] = []
        for _k in range(int(_g["nmarks"])):
            _pmk = WORK + f"/patch_mask_{_k}.png"; cv2.imwrite(_pmk, _g[f"refmask_{_k}"])
            cfg["patches"].append(dict(image=_pi, mask=_pmk, t=float(_g["ref_t"]), group=f"m{_k}"))
    if KW.get("views") == "2" and "ref2" in _g:     # v11.2.97: the same mark painted at a second position
        _pi2, _pm2 = WORK + "/patch_frame2.jpg", WORK + "/patch_mask2.png"
        cv2.imwrite(_pi2, _g["ref2"], [cv2.IMWRITE_JPEG_QUALITY, 95]); cv2.imwrite(_pm2, _g["refmask2"])
        cfg["patches"].append(dict(image=_pi2, mask=_pm2, t=float(_g["ref_t2"]), group="m1"))
if KW.get("patches") and KW.get("app_mask") == "1" and "nmarks" not in np.load(KW["patches"]).files:
    # v11.2.179: send the mark exactly as the app does - frame JPEG + an OPAQUE ARGB mask PNG (black, white paint), base64 -
    # through the Space adapter's own _load_patches, instead of writing a grey mask file directly
    import base64 as _b64, phoenix_api_adapter as _PA
    _g = np.load(KW["patches"]); _am = np.zeros(_g["refmask"].shape[:2] + (4,), np.uint8); _am[..., 3] = 255
    _am[_g["refmask"] > 127, :3] = 255
    _ok1, _jb = cv2.imencode(".jpg", _g["ref"], [cv2.IMWRITE_JPEG_QUALITY, 95]); _ok2, _mb = cv2.imencode(".png", _am)
    _items, _err = _PA._load_patches({"patches": [{"image_b64": _b64.b64encode(_jb.tobytes()).decode(),
                                                  "mask_b64": _b64.b64encode(_mb.tobytes()).decode(), "t": float(_g["ref_t"])}]})
    print("APP-MASK patches:", _items, _err, flush=True)
    cfg["patches"] = _items or []
if KW.get("skin"):             # v11.2.205: skin tone per slot, e.g. skin=0.5,0
    cfg["skin_tone"] = [float(v) for v in KW["skin"].split(",")]
if KW.get("pg"):               # v11.2.141: the app's M / F per slot (settings.person_gender), e.g. pg=0,1
    cfg["person_gender"] = [None if v in ("", "x") else int(v) for v in KW["pg"].split(",")]
if KW.get("belly"):            # v11.2.110: lower-belly clear skin
    cfg["belly_clear"] = True
    cfg["belly_length"] = float(KW.get("belly_len", 1.0)); cfg["belly_width"] = float(KW.get("belly_wid", 1.0))
t0 = time.perf_counter()
for _kv in filter(None, os.environ.get("OVR", "").split(",")):
    _k, _v = _kv.split("="); swap_engine._P[_k] = float(_v)
cp._run_job_body(jid, src, clip, cfg)
wall = time.perf_counter() - t0
if _OD:
    json.dump(_OD, open(WORK + "/order_dump.json", "w"))
j = cp.jobs[jid]
res = j.get("result_path")
print(json.dumps({"label": LABEL, "status": j.get("status"), "wall_s": round(wall, 1), "model_load_s": round(t_load, 1),
                  "threads": cp._native_threads, "cpu_count": os.cpu_count(), "affinity": len(os.sched_getaffinity(0)),
                  "out_bytes": os.path.getsize(res) if res and os.path.isfile(res) else None, "message": (j.get("message") or "")[:230]}))
rows = sorted(T.items(), key=lambda kv: -kv[1])
print("%-52s %9s %8s %9s" % ("function (wall seconds, summed over threads)", "total_s", "calls", "ms/call"))
for k, v in rows: print("%-52s %9.1f %8d %9.1f" % (k, v, C[k], 1000 * v / max(1, C[k])))
if KW.get("drop"):
    print("DROP: %d detections removed on %d frames (%s)" % (sum(DROP_STATS.values()), len(DROP_STATS), KW["drop"]))
for l in open(WORK + "/job.log"):
    if "Done in" in l or "detector input" in l or "native_threads" in l or "SEARCH-BACKOFF" in l: print(l.strip()[:600])
