"""v11.2.213 (SDOS-213): at what angle is the pasted face drawn? insightface's 3D-landmark pose model (landmark_3d_68, in the
buffalo_l pack) on the input frame and on the output frame, at the face the detector finds there: the output's yaw against the
input's (and the clip's true yaw, when the clip has a _gt.npz). A replacement drawn less turned than the head reads as a smaller
yaw on the output. Also, from the job's paste masks (bench MASK_DUMP=1): how much paint the face got.
usage: turn_metrics.py CLIP.mp4 OUT.mp4 [MASK_DUMP.npz] [every=1]   (synthetic or public test clips only)"""
import sys, numpy as np, cv2
from insightface.app import FaceAnalysis
CLIP, OUT = sys.argv[1:3]
DUMP = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].endswith(".npz") else None
KW = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
EV = int(KW.get("every", 1))
fa = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "landmark_3d_68"], providers=["CPUExecutionProvider"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
try:
    gt = np.load(CLIP[:-4] + "_gt.npz"); TY = gt["yaw"]
except Exception:
    TY = None
paint = {}
if DUMP:
    d = np.load(DUMP)
    for m, g in zip(d["m"], d["g"]):
        paint[int(g)] = max(paint.get(int(g), 0.0), float(np.mean(m)))
ci, co = cv2.VideoCapture(CLIP), cv2.VideoCapture(OUT)
rows = []; t = 0
while True:
    ok1, a = ci.read(); ok2, b = co.read()
    if not (ok1 and ok2): break
    if t % EV == 0:
        fi = fa.get(a); fo = fa.get(b)
        yi = abs(float(max(fi, key=lambda q: q.det_score).pose[1])) if fi else None
        yo = None; sh = None
        if fo and fi:
            f1 = max(fi, key=lambda q: q.det_score); ref = f1.bbox
            f2 = min(fo, key=lambda q: np.abs(q.bbox - ref).sum()); yo = abs(float(f2.pose[1]))
            # where the detector puts the eyes / nose / mouth on the output against the input, in face heights (the box's)
            hh = float(ref[3] - ref[1]) + 1e-6
            sh = float(np.median(np.linalg.norm(np.asarray(f2.kps) - np.asarray(f1.kps), axis=1))) / hh
        rows.append((t, None if TY is None else abs(float(TY[t])), yi, yo, paint.get(t), sh))
    t += 1
print("frame | true yaw | yaw read on the input | on the output | output - input | paint (mean of the paste mask) | features moved (face heights)")
for r in rows:
    print("%4d | %s | %s | %s | %s | %s | %s" % (r[0], "  -  " if r[1] is None else "%5.1f" % r[1], "  -  " if r[2] is None else "%5.1f" % r[2],
          "  -  " if r[3] is None else "%5.1f" % r[3], "  -  " if (r[2] is None or r[3] is None) else "%+5.1f" % (r[3] - r[2]),
          "  -  " if r[4] is None else "%.2f" % r[4], "  -  " if r[5] is None else "%.3f" % r[5]))
# by band of the true (else the input's) yaw
key = [(r[1] if r[1] is not None else r[2]) for r in rows]
print("\nby yaw band: frames | mean (output - input) yaw | frames drawn 8+ deg less turned | mean paint | features moved: median, frames over 0.06 face heights")
for lo, hi in [(0, 45), (45, 55), (55, 62), (62, 68), (68, 74), (74, 80), (80, 90), (90, 181)]:
    sel = [r for r, k in zip(rows, key) if k is not None and lo <= k < hi and r[2] is not None and r[3] is not None]
    if not sel: continue
    dif = [r[3] - r[2] for r in sel]; pt = [r[4] for r in sel if r[4] is not None]
    shs = [r[5] for r in sel if r[5] is not None]
    print("  %3d-%3d deg: %3d | %+5.1f | %3d | %s | %.3f, %d" % (lo, hi, len(sel), float(np.mean(dif)), sum(1 for x in dif if x <= -8), "%.2f" % np.mean(pt) if pt else "-",
          float(np.median(shs)) if shs else -1, sum(1 for x in shs if x > 0.06)))
