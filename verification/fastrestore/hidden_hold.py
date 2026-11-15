"""v11.2.210 (SDOS-210): the position of a face that is partly hidden behind another - core_pipeline._HiddenHold on its own.
A contact clip (make_kiss.py / make_cross.py; synthetic, public photo, NOT user footage), the REAL detector's 5 points, outline
and box for both people on every frame (det_err.py), handed to _HiddenHold frame by frame exactly as the job does, with person B
in front. The carried 5 points of person A against the clip's true ones:
  detector   what 11.2.209 painted with: the detector's points
  carried    what 11.2.210 paints with
Both less person A's own constant offset on the first frames (how the detector always places that face).
usage: hidden_hold.py PACK CLIP.mp4 GT.npz DETERR.npz [span a:b]"""
import sys, os, types
_pos = [a for a in sys.argv[1:] if "=" not in a]
KW = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
PACK, CLIP, GT, DE = _pos[:4]
a0, a1 = (int(x) for x in (_pos[4] if len(_pos) > 4 else "0:9999").split(":"))
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception
sys.modules["gradio"] = g_
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: ""; sys.modules["huggingface_hub"] = hf
sys.path.insert(0, PACK)
import numpy as np, cv2
import core_pipeline as cp
for _k, _v in KW.items():                      # (v11.2.211: engine switches for A/B runs, e.g. hid_filter=0)
    cp._E._P[_k] = float(_v)
cap = cv2.VideoCapture(CLIP); frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
g = np.load(GT); kps, vis = g["kps"], g["a"]; z = np.load(DE); n = min(len(frames), len(kps), len(z["err"]))
full = [(vis[:, p] > 127).sum(axis=(1, 2)).max() for p in range(2)]
# person B (index 1) is in front whenever the boxes overlap: the job's own decision is not what is tested here
cp._front_relations = lambda records, g, mem=None, stt=None: (
    {(0, 1): False} if (0 in records and 1 in records and cp._bbox_iou(records[0]["bbox"], records[1]["bbox"]) >= 0.06) else {})
H = cp._HiddenHold(); PTS = {}; CAR = {}
last = {}
off0 = np.nanmedian(np.array([z["dkps"][t, 0] - kps[t, 0] for t in range(0, 10)]), axis=0)      # the detector's own offset, apart
rows = []
for t in range(n):
    recs = {}
    for p in range(2):
        if not np.isnan(z["err"][t, p]):
            last[p] = {"kps": z["dkps"][t, p].astype(np.float32), "lmk": z["lmk"][t, p].astype(np.float32),
                       "bbox": z["bbox"][t, p].astype(np.float32), "det": True, "seen_near": True, "xid_near": None}
            recs[p] = dict(last[p])
        elif p in last:                      # not found: the job carries the last position on
            recs[p] = dict(last[p]); recs[p]["det"] = False; recs[p]["seen_near"] = False
    H.step(t, frames[t], recs)
    used = H.apply({k: dict(v) for k, v in recs.items()}, t)
    if 0 in used:
        PTS[t] = used[0]["kps"].copy(); CAR[t] = bool(used[0].get("hid"))
    share = float((vis[t, 0] > 127).sum()) / max(1.0, float(full[0]))
    e_det = float(np.linalg.norm(recs[0]["kps"] - kps[t, 0] - off0, axis=1).mean()) if 0 in recs else np.nan
    e_car = float(np.linalg.norm(used[0]["kps"] - kps[t, 0] - off0, axis=1).mean()) if 0 in used else np.nan
    rows.append((t, share, e_det, e_car, bool(used.get(0, {}).get("hid")), bool(recs.get(0, {}).get("det"))))
R = np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows], float)
print("frame  visible  detector  carried  (carried by the visible part / detected)")
for r in rows[::4]:
    print("  %3d   %3.0f%%   %6.1f   %6.1f   %s %s" % (r[0], 100 * r[1], r[2], r[3], "carried" if r[4] else "       ", "" if r[5] else "not detected"))
sp = (R[:, 0] >= a0) & (R[:, 0] < a1)
hid = sp & (R[:, 4] > 0)
print("\ncarried on %d frames (%s)" % (int(hid.sum()), {k: v for k, v in H.stats.items() if v}))
if hid.any():
    print("on those frames: detector mean %.1f px, worst %.1f px | carried mean %.1f px, worst %.1f px" % (
        np.nanmean(R[hid, 2]), np.nanmax(R[hid, 2]), np.nanmean(R[hid, 3]), np.nanmax(R[hid, 3])))
vis30 = sp & (R[:, 1] >= 0.30)
print("all frames with 30 %% or more of the face visible: detector mean %.1f px, worst %.1f px | carried mean %.1f px, worst %.1f px" % (
    np.nanmean(R[vis30, 2]), np.nanmax(R[vis30, 2]), np.nanmean(R[vis30, 3]), np.nanmax(R[vis30, 3])))
d = np.abs(np.diff(R[sp, 3])); d2 = np.abs(np.diff(R[sp, 2]))
print("largest change of the offset from one frame to the next: detector %.1f px | carried %.1f px" % (np.nanmax(d2), np.nanmax(d)))
# v11.2.211: the wobble of the carried points - each point minus the median of its 5 neighbours, over the carried frames (px)
_w = []
for _t in sorted(PTS):
    if CAR.get(_t) and all((_t + d) in PTS for d in range(-2, 3)):
        _m = np.median(np.stack([PTS[_t + d] for d in range(-2, 3)]), axis=0)
        _w.append(float(np.linalg.norm(PTS[_t] - _m, axis=1).mean()))
print("wobble of the carried points (a frame minus the median of its 5 neighbours): mean %.2f px, 90th %.2f px, worst %.2f px (%d frames)" % (
    np.mean(_w) if _w else float("nan"), np.percentile(_w, 90) if _w else float("nan"), np.max(_w) if _w else float("nan"), len(_w)))
free = sp & (R[:, 1] >= 0.97)
print("apart (97 %% or more visible): carried on %d of %d frames (must be 0)" % (int((free & (R[:, 4] > 0)).sum()), int(free.sum())))
