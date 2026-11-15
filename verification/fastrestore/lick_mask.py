"""v11.2.210: what the face mask does while someone eats an ice cream (make_lick.py clip; synthetic, public photo, NOT user
footage). The real detector and 106 landmarks on every frame (so the lip points are as unsure as in a job, with the scoop on
them), the real AlignedCompositor.build_mask with its memory, frame after frame.
Per frame, in the face crop:
  lips     paint over the footage's own lips where the scoop does not cover them (1 = the swapped lips, 0 = the footage's lips)
  ring     paint over the face within 8 px (of 256) of the scoop, lips left out (low = a band of the footage's face around it)
  object   paint over the scoop where it covers the face (0 = the scoop is shown, as it must be); the face = where the
           occlusion model sees face on the first frame
usage: lick_mask.py PACK CLIP.mp4 XSEG.onnx [SHEET.png] [key=value ...]   (key=value: engine switches, e.g. eat_mouth=0)"""
import sys, os, types, importlib.util
PACK, CLIP, XSEG = sys.argv[1:4]
rest = sys.argv[4:]
SHEET = next((a for a in rest if a.endswith(".png")), None)
KW = dict(a.split("=", 1) for a in rest if "=" in a)
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception
sys.modules["gradio"] = g_
sys.path.insert(0, PACK)
import numpy as np, cv2
spec = importlib.util.spec_from_file_location("se_lick", os.path.join(PACK, "swap_engine.py"))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
assert E.occ_net_load(XSEG, threads=2), "occlusion model did not load"
for k, v in KW.items():
    E._P[k] = float(v)
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "landmark_2d_106"], providers=["CPUExecutionProvider"])
fa.prepare(ctx_id=-1, det_size=(640, 640))
cap = cv2.VideoCapture(CLIP); frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
g = np.load(CLIP[:-4] + "_gt.npz"); A = g["a"]; LIPS = g["lips"]; K0 = g["kps"]
size = 256
comp = E.AlignedCompositor.__new__(E.AlignedCompositor)
tr = E.TrackState(slot=0)
lipm = np.zeros(frames[0].shape[:2], np.uint8); cv2.fillPoly(lipm, [np.round(LIPS).astype(np.int32)], 1)
rows, tiles, last, FACE0, m0 = [], [], None, None, None
for t, fr in enumerate(frames):
    fs = fa.get(fr)
    f = min(fs, key=lambda q: float(np.linalg.norm(q.kps.mean(0) - K0.mean(0)))) if fs else None
    if f is None or float(np.linalg.norm(f.kps.mean(0) - K0.mean(0))) > 80:
        f = last                                  # (the job holds the last geometry when the face is not found)
    if f is None:
        continue
    last = f
    M = E.estimate_norm(f.kps, size)
    at = cv2.warpAffine(fr, M, (size, size), borderMode=cv2.BORDER_REPLICATE)
    m = comp.build_mask(f, M, size, track=tr, aligned_target=at, frame_wh=(fr.shape[1], fr.shape[0]))
    if m0 is None:
        m0 = m.copy()                               # the mask with nothing in front (frame 0)
    a = cv2.warpAffine(A[t].astype(np.float32) / 255.0, M, (size, size))
    lp = cv2.warpAffine(lipm, M, (size, size), flags=cv2.INTER_NEAREST) > 0
    obj = a > 0.5
    free = lp & (a < 0.05)
    if FACE0 is None:                               # the face itself: where the occlusion model sees face before the scoop comes
        FACE0 = cv2.erode((cv2.resize(E.occ_net_raw(at), (size, size)) >= 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    near = cv2.dilate(obj.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool) & (a < 0.05) & ~lp & FACE0
    inner = cv2.erode(obj.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & FACE0
    below = cv2.erode(obj.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~FACE0 & (m0 > 0.3)   # the scoop under the swap's border
    rows.append((t, float(m[free].mean()) if free.sum() > 30 else np.nan, float(m[near].mean()) if near.sum() > 30 else np.nan,
                 float(m[inner].mean()) if inner.sum() > 30 else np.nan, float(getattr(tr, "eat_w", 0.0) or 0.0), int(obj.sum()),
                 float((lp & obj).sum()) / max(1.0, float(lp.sum())), float(m[below].mean()) if below.sum() > 30 else np.nan))
    if SHEET and t in (10, 30, 34, 40, 46, 52, 58, 70, 76):
        raw = E.occ_net_raw(at)
        rawv = cv2.cvtColor((cv2.resize(raw, (size, size)) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) if raw is not None else np.zeros_like(at)
        mv = cv2.cvtColor((m * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        ov = at.copy(); cs, _ = cv2.findContours(lp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov, cs, -1, (0, 255, 0), 1)
        cs, _ = cv2.findContours(obj.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov, cs, -1, (255, 255, 0), 1)
        pts = E._landmarks_to_aligned(f.landmark_2d_106, M, size)
        for q in pts[52:72]:
            cv2.circle(ov, (int(q[0]), int(q[1])), 1, (0, 0, 255), -1)
        shown = (at.astype(np.float32) * (1 - m[..., None]) + np.array([200, 80, 200], np.float32) * m[..., None]).astype(np.uint8)
        col = np.vstack([ov, rawv, mv, shown]); cv2.putText(col, "f%d" % t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1); tiles.append(col)
R = np.array(rows, float)
print("frame  lips  ring  object  eat_w  (object px, share of the lips under the scoop)")
for r in rows[::3]:
    print("  %3d  %4.2f  %4.2f  %4.2f   %4.2f   (%5d, %3.0f%%)" % (r[0], r[1], r[2], r[3], r[4], r[5], 100 * r[6]))
on = R[:, 5] > 200                                  # the scoop is in the crop
at_lip = R[:, 6] > 0.10                             # ...and on the lips
base = np.nanmean(R[:18, 1])
print("\nlips painted with the swap: nothing there %.2f | scoop on the lips: mean %.2f, lowest %.2f | scoop below the lips (between licks): mean %.2f, lowest %.2f" % (
    base, np.nanmean(R[at_lip, 1]), np.nanmin(R[at_lip, 1]), np.nanmean(R[on & ~at_lip, 1]), np.nanmin(R[on & ~at_lip, 1])))
print("face next to the scoop painted: scoop on the lips mean %.2f | below the lips mean %.2f" % (np.nanmean(R[at_lip, 2]), np.nanmean(R[on & ~at_lip, 2])))
print("the scoop painted over (must be near 0): on the lips mean %.2f, worst %.2f | below the lips mean %.2f, worst %.2f" % (
    np.nanmean(R[at_lip, 3]), np.nanmax(R[at_lip, 3]), np.nanmean(R[on & ~at_lip, 3]), np.nanmax(R[on & ~at_lip, 3])))
print("the scoop under the swap's soft border, outside the face: paint mean %.2f" % np.nanmean(R[on, 7]))
d = np.abs(np.diff(R[:, 1])); print("lips paint, change from frame to frame: mean %.3f, largest %.2f" % (np.nanmean(d), np.nanmax(d)))
if SHEET and tiles:
    cv2.imwrite(SHEET, np.hstack(tiles))
