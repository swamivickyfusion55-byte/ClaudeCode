"""strip_paint.py GT DUMP [a:b]  - person A's paint over the part of its face that is visible, against the paint the SAME part of the
face gets on a frame with the faces apart (frame 5, moved onto this frame by the true 5 points). 1.00 = painted as when nothing covers it."""
import sys, cv2, numpy as np
g = np.load(sys.argv[1]); vis, kps = g["a"], g["kps"]; z = np.load(sys.argv[2]); sl, gg = z["slot"], z["g"]
a0, a1 = (int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "20:110").split(":"))
byg = {}
for k in range(len(sl)): byg.setdefault(int(sl[k]), {})[int(gg[k])] = k
def centre(k):
    M = z["M"][k]; size = int(z["size"][k]); IM = cv2.invertAffineTransform(M); return IM[:, :2] @ np.array([size / 2, size / 2]) + IM[:, 2]
slot_of = {}
for p in range(2):
    cands = [(s, min(d.items(), key=lambda kv: kv[0])) for s, d in byg.items() if d]
    slot_of[p] = min(cands, key=lambda c: float(np.linalg.norm(centre(c[1][1]) - kps[c[1][0], p].mean(0))))[0]
def pm(t, p):
    k = byg.get(slot_of[p], {}).get(t)
    if k is None: return np.zeros(vis.shape[2:], np.float32)
    size = int(z["size"][k]); m = cv2.resize(z["m"][k].astype(np.float32), (size, size))
    return cv2.warpAffine(m, cv2.invertAffineTransform(z["M"][k]), (vis.shape[3], vis.shape[2]))
full = float((vis[:, 0] > 127).sum(axis=(1, 2)).max()); P5 = pm(5, 0); rows = []
for t in range(a0, min(a1, len(kps))):
    M, _ = cv2.estimateAffinePartial2D(kps[5, 0].astype(np.float32), kps[t, 0].astype(np.float32))
    ref = cv2.warpAffine(P5, M, (vis.shape[3], vis.shape[2]))
    v = cv2.erode(((vis[t, 0] > 200) & (ref > 0.5)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0     # visible AND part of the swap when apart
    share = float((vis[t, 0] > 127).sum()) / full
    if v.sum() < 300: continue
    rows.append((t, share, float(pm(t, 0)[v].mean()) / max(1e-6, float(ref[v].mean()))))
R = np.array(rows)
cov = R[:, 1] < 0.70
print("frames with under 70 %% of A visible: %d | A's visible face painted, against the same part with the faces apart: mean %.2f, lowest frame %.2f, frames under 0.70: %d, under 0.40: %d | largest change in one frame %.2f" % (
    int(cov.sum()), R[cov, 2].mean(), R[cov, 2].min(), int((R[cov, 2] < 0.70).sum()), int((R[cov, 2] < 0.40).sum()), np.abs(np.diff(R[:, 2])).max()))
