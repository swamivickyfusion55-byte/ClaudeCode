"""spill_vis.py GT DUMP [a:b]  - person A's paint that ends up VISIBLE on person B (B is painted afterwards, on top): sum over B's visible
area of paintA x (1 - paintB), in px; and the same as a share of B's area"""
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
v = []
for t in range(a0, min(a1, len(kps))):
    PA, PB = pm(t, 0), pm(t, 1); b = vis[t, 1] > 200
    v.append((t, float((PA * (1 - PB))[b].sum()), float((PA * (1 - PB))[b].sum()) / max(1.0, float(b.sum())), float((PB * (vis[t, 0] > 200)).sum())))
V = np.array(v)
print("A's paint left visible on B: mean %.0f px (%.1f %% of B's area), worst frame %.0f px (frame %d) | B's paint on A's visible area: mean %.0f px, worst %.0f" % (
    V[:, 1].mean(), 100 * V[:, 2].mean(), V[:, 1].max(), int(V[V[:, 1].argmax(), 0]), V[:, 3].mean(), V[:, 3].max()))
