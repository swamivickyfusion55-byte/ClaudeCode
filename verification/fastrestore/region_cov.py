"""v11.2.211 (SDOS-211): the identity check's "painting" measure on the paint maps of mask_steady.py (save=PAINT.npz): person A's paint over
its visible FACE (the clip's face outline from the public photo, moved by the clip's own similarity, within the visible part, eroded 4 px),
against the same on its fully visible frames (the median) - so a change of the mask can be tuned without a whole job. Synthetic clip
(make_kiss.py / make_cross.py; public photo, NOT user footage). Prints, for each file, the base, the share on frames 60-81, the frames under
40 % of the base (revert) and between 40 and 70 % (weak) among those with at least 30 % of the face visible.
usage: region_cov.py CLIP.mp4 GT.npz PAINT.npz ..."""
import sys, numpy as np, cv2
from insightface.app import FaceAnalysis
CLIP = sys.argv[1]; GT = sys.argv[2]; files = sys.argv[3:]
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
g = np.load(GT); kps, vis = g["kps"], g["a"]; H, W = vis.shape[2:]
_PH = cv2.imread("/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg"); _PF = fa.get(_PH); PHOTO = []
for (x1, y1, x2, y2) in ((466, 268, 573, 415), (745, 338, 845, 479)):
    c = np.array([(x1 + x2) / 2, (y1 + y2) / 2]); f = min(_PF, key=lambda q: float(np.linalg.norm((q.bbox[:2] + q.bbox[2:]) / 2 - c)))
    L = f.landmark_2d_106.astype(np.float32); top = L[:, 1].min(); h = L[:, 1].max() - top
    fh = L[L[:, 1] < top + 0.35 * h].copy(); fh[:, 1] -= 0.30 * h
    PHOTO.append((f.kps.astype(np.float32), np.vstack([L, fh])))
def region(t, p):
    k0, L = PHOTO[p]; M, _ = cv2.estimateAffinePartial2D(k0, np.asarray(kps[t, p], np.float32))
    pts = (L @ M[:, :2].T + M[:, 2]); m = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(m, cv2.convexHull(np.round(pts).astype(np.int32)), 1); return m > 0
full = [(vis[:, p] > 127).sum(axis=(1, 2)).max() for p in range(2)]
def share_series(P):
    sh = []
    for t in range(len(kps)):
        v = vis[t, 0] > 127
        m = cv2.erode((v & region(t, 0)).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        if m.sum() < 400: sh.append(None); continue
        pa = cv2.resize(P[t].astype(np.float32), (W, H)); sh.append(float(pa[m].mean()))
    return sh
for f in files:
    P = np.load(f)["p0"]; sh = share_series(P)
    base = np.median([s for t, s in enumerate(sh) if s is not None and (vis[t, 0] > 127).sum() / full[0] >= 0.9] or [1.0])
    print("%-32s base %.2f | t60-80: %s | revert<0.40: %d weak<0.70: %d (frames with >=30%% visible)" % (f.split("/")[-1], base, " ".join("%.2f" % (sh[t] / base) if sh[t] is not None else "-" for t in range(60, 82, 3)),
          sum(1 for t, s in enumerate(sh) if s is not None and (vis[t, 0] > 127).sum() / full[0] >= 0.3 and s < 0.4 * base),
          sum(1 for t, s in enumerate(sh) if s is not None and (vis[t, 0] > 127).sum() / full[0] >= 0.3 and 0.4 * base <= s < 0.7 * base)))
