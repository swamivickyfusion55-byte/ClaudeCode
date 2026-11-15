"""cost210.py PACK209 PACK210 LICK.mp4 SMOOCH.mp4 GT DETERR XSEG - added time of v11.2.210's new steps, on 2 threads, nothing else running"""
import sys, os, types, importlib.util, time
P9, P0, LICK, SM, GT, DE, XSEG = sys.argv[1:8]
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception
sys.modules["gradio"] = g_
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: ""; sys.modules["huggingface_hub"] = hf
import logging; logging.disable(logging.CRITICAL)
import numpy as np, cv2
cv2.setNumThreads(2)
def load(name, pack):
    spec = importlib.util.spec_from_file_location(name, os.path.join(pack, "swap_engine.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E9, E0 = load("se209", P9), load("se210", P0)
for E in (E9, E0): assert E.occ_net_load(XSEG, threads=2)
def frames_of(p):
    c = cv2.VideoCapture(p); o = []
    while True:
        ok, f = c.read()
        if not ok: return o
        o.append(f)
fr = frames_of(LICK); K0 = np.load(LICK[:-4] + "_gt.npz")["kps"]
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "landmark_2d_106"], providers=["CPUExecutionProvider"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
d = min(fa.get(fr[0]), key=lambda q: float(np.linalg.norm(q.kps.mean(0) - K0.mean(0)))); size = 256
for E, nm in ((E9, "v11.2.209"), (E0, "v11.2.210")):
    comp = E.AlignedCompositor.__new__(E.AlignedCompositor); tr = E.TrackState(slot=0); M = E.estimate_norm(d.kps, size); ts = []
    for f in fr[20:110]:
        at = cv2.warpAffine(f, M, (size, size), borderMode=cv2.BORDER_REPLICATE)
        t0 = time.perf_counter(); comp.build_mask(d, M, size, track=tr, aligned_target=at, frame_wh=(f.shape[1], f.shape[0])); ts.append(time.perf_counter() - t0)
    print("%s: the face mask on the eating clip (object at the mouth on most frames): %.1f ms per face-frame (median), %.1f mean" % (nm, 1000 * np.median(ts), 1000 * np.mean(ts)))
sys.path.insert(0, P0)
import core_pipeline as cp
sf = frames_of(SM); kps = np.load(GT)["kps"]; z = np.load(DE)
cp._front_relations = lambda records, g, mem=None, stt=None: ({(0, 1): False} if (0 in records and 1 in records and cp._bbox_iou(records[0]["bbox"], records[1]["bbox"]) >= 0.06) else {})
H = cp._HiddenHold(); last = {}; ts = []; car = []
for t in range(len(sf)):
    recs = {}
    for p in range(2):
        if not np.isnan(z["err"][t, p]):
            last[p] = {"kps": z["dkps"][t, p].astype(np.float32), "lmk": z["lmk"][t, p].astype(np.float32), "bbox": z["bbox"][t, p].astype(np.float32), "det": True}
            recs[p] = dict(last[p])
        elif p in last: recs[p] = dict(last[p]); recs[p]["det"] = False
    t0 = time.perf_counter(); H.step(t, sf[t], recs); dt = time.perf_counter() - t0
    (car if t in H.over else ts).append(dt)
print("v11.2.210: the carry step per frame of a two-person job - faces apart %.2f ms, while a face is carried %.1f ms (median; %d and %d frames)" % (1000 * np.median(ts), 1000 * np.median(car), len(ts), len(car)))
