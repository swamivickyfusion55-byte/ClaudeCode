import sys, cv2, numpy as np
name = sys.argv[1]
z = np.load("deterr_%s.npz" % name); g = np.load("bench/kiss_%s_gt.npz" % name); kps = g["kps"]; n = len(z["err"])
def hull_mask(pts, shape=(720, 1280)):
    m = np.zeros(shape, np.uint8); cv2.fillConvexPoly(m, cv2.convexHull(np.round(pts).astype(np.int32)), 1); return m > 0
def head_pts(l):
    # the 106 outline plus a forehead: the points of the upper third mirrored up
    top = l[:, 1].min(); h = l[:, 1].max() - top; fh = l[l[:, 1] < top + 0.35 * h].copy(); fh[:, 1] -= 0.30 * h; return np.vstack([l, fh])
ref = {}
for p in range(2):
    t0 = next(t for t in range(n) if not np.isnan(z["err"][t, p]))
    ref[p] = (z["lmk"][t0, p].copy(), kps[t0, p].astype(np.float32))
print("frame | share A | det err A | cover of A by B: detector's A outline, true A outline (B hull) , (B head = hull + forehead) | A 5 pts inside B hull: det, true")
for t in range(0, n, 3):
    if np.isnan(z["err"][t, 1]): continue
    hb = hull_mask(z["lmk"][t, 1]); hbh = hull_mask(head_pts(z["lmk"][t, 1]))
    l0, k0 = ref[0]; M, _ = cv2.estimateAffinePartial2D(k0, kps[t, 0].astype(np.float32)); lt = l0 @ M[:, :2].T + M[:, 2]
    ha_t = hull_mask(lt); ct = (ha_t & hb).sum() / max(1, ha_t.sum()); cth = (ha_t & hbh).sum() / max(1, ha_t.sum())
    inside_t = sum(1 for q in kps[t, 0] if hb[int(np.clip(q[1], 0, 719)), int(np.clip(q[0], 0, 1279))])
    if np.isnan(z["err"][t, 0]):
        print("%3d | %3.0f%% |   nan | det   -  true %.2f  head %.2f | pts det - true %d" % (t, 100 * z["share"][t, 0], ct, cth, inside_t)); continue
    ha_d = hull_mask(z["lmk"][t, 0]); cd = (ha_d & hb).sum() / max(1, ha_d.sum())
    inside_d = sum(1 for q in z["dkps"][t, 0] if hb[int(np.clip(q[1], 0, 719)), int(np.clip(q[0], 0, 1279))])
    print("%3d | %3.0f%% | %5.1f | det %.2f  true %.2f  head %.2f | pts det %d true %d" % (t, 100 * z["share"][t, 0], z["err"][t, 0], cd, ct, cth, inside_d, inside_t))
