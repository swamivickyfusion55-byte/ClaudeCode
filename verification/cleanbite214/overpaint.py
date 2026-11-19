"""overpaint.py CLIP RESULT [LIPS_RADIUS]: how much of the true bar (ground-truth alpha) the swap changed: the share of bar pixels whose colour
differs from the footage by more than 14 levels, over the whole bar and over the part within R px of the lips box / opening (the bar's tip).
Frames with the bar in the mouth only. 0 = the bar is exactly as filmed."""
import sys, cv2, numpy as np
clip, res = sys.argv[1:3]; R = int(sys.argv[3]) if len(sys.argv) > 3 else 70
g = np.load(clip[:-4] + "_gt.npz"); A = g["a"]
c1, c2 = cv2.VideoCapture(clip), cv2.VideoCapture(res); allr, tipr = [], []; t = 0
if "lips" in g.files and g["lips"].ndim == 2: LB = g["lips"]
while True:
    o1, a = c1.read(); o2, b = c2.read()
    if not (o1 and o2): break
    al = cv2.resize(A[t], (1280, 720), interpolation=cv2.INTER_LINEAR); bar = al > 200
    if bar.sum() > 800:
        d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(2) > 14
        ys, xs = np.nonzero(bar)
        # the tip: the bar pixels nearest the mouth = within R px of the bar's extreme end nearest the lips centre
        if "lips" in g.files and g["lips"].ndim == 2:
            x0, y0, x1, y1 = LB[t]; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        else:
            cx, cy = np.mean(g["L"][t][52:72, 0]), np.mean(g["L"][t][52:72, 1])
        near = np.hypot(xs - cx, ys - cy) < R
        allr.append(float(d[bar].mean())); tipr.append(float(d[ys[near], xs[near]].mean()) if near.sum() > 50 else np.nan)
    t += 1
print("%-34s frames %3d | bar changed by the swap: whole bar %.3f, near the mouth (%d px) %.3f" % (res.split("/")[-3][:34], len(allr), np.mean(allr), R, np.nanmean(tipr)))
