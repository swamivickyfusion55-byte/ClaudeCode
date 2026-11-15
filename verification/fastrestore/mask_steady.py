"""v11.2.211 (SDOS-211): how steady is the paint's edge, frame after frame? A synthetic clip (make_kiss.py / make_objects.py, optionally
degraded by degrade_clip.py to look like phone footage; public photo, NOT user footage) is run through the REAL detector, the REAL
tracker (TrackState.update: the same One Euro / smoothing as the job), the REAL carry step of a face going behind another
(core_pipeline._HiddenHold) and the REAL mask builder (AlignedCompositor.build_mask with its memory), exactly as the job composes them,
without the swap network. Output: the paint map of each person, in the picture, for every frame.
  shimmer   a frame's paint minus the median of its five neighbours, over the pixels where the paint is soft (an edge): the part of an
            edge's movement that is NOT smooth motion - what the eye sees as shaking or "waving". Reported as the mean over the edge
            band of the picture (px of edge position: the soft band is about 2 x 8 px wide, so 1.0 here is about a pixel of shake),
            for the band along the seam where the two faces touch (within 25 px of the other face) and for the rest of the outline.
  seam      the paint's edge along the true seam, against the true seam (signed px), and its wobble (high-pass).
  place     the paint's placement against the true 5 points: mean error (px).
usage: mask_steady.py PACK CLIP.mp4 GT.npz XSEG.onnx [a:b] [key=value ...]    (key=value: engine switches, set before the run;
       guard_pred=0.85: the job's colour test on faces not detected on the frame; guard_hid=x: another value for carried ones)
       one-person object clips (obj_*.mp4): pass GT with only 'a' (the object's alpha) and add one=1"""
import sys, os, types, importlib.util, json
args = [a for a in sys.argv[1:] if "=" not in a]
KW = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
PACK, CLIP, GT, XSEG = args[:4]
SPAN = args[4] if len(args) > 4 else "26:100"
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception
sys.modules["gradio"] = g_
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: ""; sys.modules["huggingface_hub"] = hf
sys.path.insert(0, PACK)
import logging
logging.disable(logging.CRITICAL)
import numpy as np, cv2
import core_pipeline as cp
E = cp._E
assert E.occ_net_load(XSEG, threads=2), "occlusion model did not load"
for k, v in KW.items():
    if k not in ("one", "save", "series", "guard_pred", "guard_hid"):
        E._P[k] = float(v)
ONE = KW.get("one") == "1"
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "landmark_2d_106"], providers=["CPUExecutionProvider"])
fa.prepare(ctx_id=-1, det_size=(640, 640))
cap = cv2.VideoCapture(CLIP); frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
gt = np.load(GT)
H, W = frames[0].shape[:2]
n = len(frames)
a0, a1 = (int(x) for x in SPAN.split(":"))
a1 = min(a1, n)
if ONE:
    # the face of the object clips: the centre face, found once on the first frame
    f0 = min(fa.get(frames[0]), key=lambda q: abs((q.bbox[0] + q.bbox[2]) / 2 - 640))
    TRUE = np.stack([f0.kps] * n)[:, None]
    VIS = np.zeros((n, 1, H, W), np.uint8)
else:
    TRUE, VIS = gt["kps"], gt["a"]
NP = 1 if ONE else 2
cp._front_relations = lambda records, g, mem=None, stt=None: (
    {(0, 1): False} if (0 in records and 1 in records and cp._bbox_iou(records[0]["bbox"], records[1]["bbox"]) >= 0.06) else {})
tracks = {p: E.TrackState(slot=p) for p in range(NP)}
comps = {p: E.AlignedCompositor.__new__(E.AlignedCompositor) for p in range(NP)}
HH = cp._HiddenHold() if hasattr(cp, "_HiddenHold") else None
paint = {p: np.zeros((n, H // 2, W // 2), np.float16) for p in range(NP)}
place = {p: [] for p in range(NP)}
size = 128
last_rec = {}
for t in range(n):
    fs = fa.get(frames[t])
    recs = {}
    # which detection belongs to which person: for two people the JOINT best match on all five true points (the faces of a
    # side-view kiss are only 10-20 px apart, so matching one person after the other by the centre swapped the two for
    # whole stretches of frames - a harness error that looked like the paint jumping to the other face)
    pick = {}
    if not ONE and NP == 2:
        import itertools
        _cost = lambda f, p: float(np.linalg.norm(np.asarray(f.kps, np.float32) - TRUE[t, p], axis=1).mean())
        _best, _bc = None, 1e18
        for _combo in itertools.product(list(range(len(fs))) + [None], repeat=NP):
            _used = [c_ for c_ in _combo if c_ is not None]
            if len(set(_used)) != len(_used):
                continue
            _tot, _ok = 0.0, True
            for _p, _c in enumerate(_combo):
                if _c is None:
                    _tot += 150.0
                else:
                    _cc = _cost(fs[_c], _p)
                    if _cc > 150.0:
                        _ok = False
                        break
                    _tot += _cc
            if _ok and _tot < _bc:
                _best, _bc = _combo, _tot
        for _p, _c in enumerate(_best or (None,) * NP):
            if _c is not None:
                pick[_p] = fs[_c]
    for p in range(NP):
        tr = tracks[p]
        if pick:
            c = [pick[p]] if p in pick else []
        else:
            c = [f for f in fs if np.linalg.norm(f.kps.mean(0) - TRUE[t, p].mean(0)) < 150]
        if c:
            f = c[0] if pick else min(c, key=lambda q: float(np.linalg.norm(q.kps.mean(0) - TRUE[t, p].mean(0))))
            fs = [q for q in fs if q is not f]
            tr.update(f, update_embedding=False)
            det = True
        else:
            tr.predict(1.0)
            det = False
        if tr.kps is None:
            continue
        recs[p] = {"kps": np.asarray(tr.kps, np.float32).copy(), "bbox": np.asarray(tr.bbox, np.float32).copy(),
                   "lmk": None if tr.lmk is None else np.asarray(tr.lmk, np.float32).copy(), "det": det, "alpha": 1.0}
    if not ONE and HH is not None and HH.enabled():
        HH.step(t, frames[t], recs)
        recs = HH.apply(recs, t)
    cp._FRONT_NOW.clear()
    if len(recs) == 2:
        cp._FRONT_NOW[(0, 1)] = False
    for p in sorted(recs):
        r = recs[p]
        face = E.PredictedFace(r["bbox"], r["kps"], r["lmk"], None, 0.5)
        face._track = tracks[p]
        try:
            face._hid = bool(r.get("hid"))
            face._partners = cp._partner_info(recs, p) if len(recs) == 2 else []
        except (AttributeError, TypeError):
            pass                                    # (11.2.209 and older: no such fields)
        rivals = cp._rival_landmarks(recs, p) if len(recs) == 2 else []
        M = E.estimate_norm(r["kps"], size)
        at = cv2.warpAffine(frames[t], M, (size, size), borderMode=cv2.BORDER_REPLICATE)
        # (the job gives a face that was not detected on this frame - one it predicts or carries - the colour test at 0.85: guard_pred=0.85
        # does the same here; guard_hid replaces it for a face whose position is the carried one)
        _g = float(KW.get("guard_pred", 0.0)) if not r.get("det") else 0.0
        if r.get("hid") and "guard_hid" in KW:
            _g = float(KW["guard_hid"])
        m = comps[p].build_mask(face, M, size, track=tracks[p], aligned_target=at, rivals=rivals, frame_wh=(W, H), occlusion_guard=_g)
        full = cv2.warpAffine(m, cv2.invertAffineTransform(M), (W, H))
        paint[p][t] = cv2.resize(full, (W // 2, H // 2), interpolation=cv2.INTER_AREA).astype(np.float16)
        if not ONE:
            place[p].append(float(np.linalg.norm(r["kps"] - TRUE[t, p], axis=1).mean()))


def hp_stack(P, t):
    """frame t minus the median of its 5 neighbours (t-2 .. t+2), float32"""
    w = np.stack([P[t + d].astype(np.float32) for d in range(-2, 3)])
    return P[t].astype(np.float32) - np.median(w, axis=0), np.median(w, axis=0)


out = {"clip": os.path.basename(CLIP), "pack": os.path.basename(PACK), "keys": KW}
bands = {"seam": [], "outline": [], "all": []}
SER = {}                                           # (for debugging: the shimmer of every frame, per person and band)
seam_series = {p: [] for p in range(NP)}
for p in range(NP):
    acc = {"seam": [], "outline": [], "all": []}
    for t in range(max(a0, 2), min(a1, n - 2)):
        dh, med = hp_stack(paint[p], t)
        edge = (med > 0.03) & (med < 0.97)
        if not edge.any():
            continue
        # distance to the OTHER person's visible face (half resolution)
        if not ONE:
            o = cv2.resize((VIS[t, 1 - p] > 127).astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
            d = cv2.distanceTransform(1 - o, cv2.DIST_L2, 3) * 2.0
            seam = edge & (d < 25)
            rest = edge & (d >= 40)
        else:
            seam = edge
            rest = edge & False
        # the object clip: the band around the true object's edge is what counts there ('seam'); the face outline is 'outline'
        if ONE:
            ao = cv2.resize((VIS_OBJ[t] > 127).astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST) if False else None
        for nm, m_ in (("seam", seam), ("outline", rest), ("all", edge)):
            if m_.sum() > 20:
                acc[nm].append(float(np.abs(dh[m_]).sum()) / float(m_.sum()))
                SER.setdefault("%d_%s" % (p, nm), {})[t] = round(acc[nm][-1], 4)
    for nm in acc:
        bands[nm].append(float(np.mean(acc[nm])) if acc[nm] else float("nan"))
if KW.get("series"):
    out["series"] = SER
out["shimmer_seam"] = bands["seam"]
out["shimmer_outline"] = bands["outline"]
out["shimmer_all"] = bands["all"]
if not ONE:
    out["place_mean"] = [float(np.mean(place[p][a0:a1])) for p in range(NP)]
    out["carried_frames"] = int(sum(1 for t in HH.over if a0 <= t < a1)) if (HH is not None and HH.enabled()) else 0
    # frames where the paint of each person covers its visible face (a quick check that nothing was lost)
    cov = []
    for p in range(NP):
        c_ = []
        for t in range(a0, a1):
            v = cv2.resize((VIS[t, p] > 200).astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST) > 0
            if v.sum() > 400:
                c_.append(float(paint[p][t].astype(np.float32)[v].mean()))
        cov.append(float(np.mean(c_)))
    out["paint_on_visible"] = cov
if KW.get("save"):
    np.savez_compressed(KW["save"], **{"p%d" % p: paint[p] for p in range(NP)})
print("RESULT " + json.dumps(out))
