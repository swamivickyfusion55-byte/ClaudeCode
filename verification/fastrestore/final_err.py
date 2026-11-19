"""v11.2.214 (SDOS-214): how close is the job's OUTPUT to the ideal around the mouth, with a bar half in it. The ideal = the same job's output on the bar-free
twin of the clip (NONE_RESULT) with the true bar (ground-truth alpha, the clip's own pixels) laid over it: the replacement's mouth as it looks with no
object, and the object as filmed. Error = mean abs difference (grey levels, BGR) over the lips' window (landmarks 52-71 of the ground truth, padded 45 px),
the bar (dilated 6 px) left out, on the frames where the bar is in the picture (alpha area over 800 px). Also the share of the bar's pixels that the swap
changed (overpaint: > 14 levels), over the bar's part within 70 px of the lips.
usage: final_err.py CLIP.mp4 NONE_RESULT.mp4 LABEL=RESULT.mp4 [LABEL=RESULT.mp4 ...]   (synthetic test clips only)"""
import sys, cv2, numpy as np
CLIP, NONE = sys.argv[1:3]
RES = [a.split("=", 1) for a in sys.argv[3:]]
g = np.load(CLIP[:-4] + "_gt.npz"); A = g["a"]; L = g["L"]


def frames(p):
    c = cv2.VideoCapture(p); out = []
    while True:
        ok, f = c.read()
        if not ok:
            break
        out.append(f)
    return out


IN, NO = frames(CLIP), frames(NONE)
res = {lab: frames(p) for lab, p in RES}
n = min([len(IN), len(NO)] + [len(v) for v in res.values()])
err = {lab: [] for lab in res}; ovp = {lab: [] for lab in res}; fr_used = []
for t in range(n):
    al = cv2.resize(A[t], (IN[t].shape[1], IN[t].shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    bar = al > 0.78
    if int((al > 0.05).sum()) < 800:
        continue
    ideal = NO[t].astype(np.float32) * (1 - al[..., None]) + IN[t].astype(np.float32) * al[..., None]
    lm = L[t][52:72]; x0, y0 = lm.min(0) - 45; x1, y1 = lm.max(0) + 45
    x0, y0, x1, y1 = [int(v) for v in (max(0, x0), max(0, y0), min(IN[t].shape[1], x1), min(IN[t].shape[0], y1))]
    reg = np.zeros(bar.shape, bool); reg[y0:y1, x0:x1] = True
    reg &= ~(cv2.dilate((al > 0.05).astype(np.uint8), np.ones((13, 13), np.uint8)) > 0)
    cx, cy = lm.mean(0)
    ys, xs = np.nonzero(bar); near = np.hypot(xs - cx, ys - cy) < 70
    fr_used.append(t)
    for lab, v in res.items():
        d = np.abs(v[t].astype(np.float32) - ideal).mean(2)
        err[lab].append(float(d[reg].mean()))
        ch = np.abs(v[t].astype(np.int16) - IN[t].astype(np.int16)).max(2) > 14
        ovp[lab].append(float(ch[ys[near], xs[near]].mean()) if near.sum() > 50 else np.nan)
print("%s: %d frames with the bar in the picture" % (CLIP.split("/")[-1], len(fr_used)))
for lab in res:
    e = np.array(err[lab]); o = np.array(ovp[lab])
    print("  %-14s error vs ideal around the mouth: mean %.2f, worst frame %.2f | bar changed near the lips: %.3f" % (lab, e.mean(), e.max(), np.nanmean(o)))
