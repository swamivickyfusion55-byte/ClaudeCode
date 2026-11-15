"""Where is the paint lost in contact? Per frame, per person: the engine's own paste mask (MASK_DUMP) inside the person's true
visible face, split by distance from the other person's visible mask.
usage: contact_band.py GT.npz MASK_DUMP.npz [a:b] [step]"""
import sys, cv2, numpy as np
g = np.load(sys.argv[1]); vis, kps = g["a"], g["kps"]; n = len(kps)
z = np.load(sys.argv[2]); sl, gg = z["slot"], z["g"]
a0, a1 = (int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "20:110").split(":")); step = int(sys.argv[4]) if len(sys.argv) > 4 else 2
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
full = [(vis[:, p] > 127).sum(axis=(1, 2)).max() for p in range(2)]
# the face area of each person: the paint on a far-apart frame (frame 5), mapped by the true 5 points
ref = {}
for p in range(2):
    ref[p] = (pm(5, p) > 0.5, kps[5, p].astype(np.float32))
def face_area(t, p):
    m0, k0 = ref[p]; M, _ = cv2.estimateAffinePartial2D(k0, kps[t, p].astype(np.float32))
    return cv2.warpAffine(m0.astype(np.uint8), M, (vis.shape[3], vis.shape[2])) > 0
print("frame | person: visible%  paint in the face that should be painted: all | 0-8px | 8-20 | 20-40 | >40 px from the other | paint ON the other person (px) | change vs prev frame")
prev = {}
for t in range(a0, min(a1, n), step):
    line = "%3d |" % t
    for p in range(2):
        v = (vis[t, p] > 200) & face_area(t, p); o = vis[t, 1 - p] > 127
        share = float((vis[t, p] > 127).sum()) / max(1.0, float(full[p]))
        P = pm(t, p); d = cv2.distanceTransform((~o).astype(np.uint8), cv2.DIST_L2, 3)
        def band(lo, hi):
            m = v & (d >= lo) & (d < hi); return ("%3.0f" % (100 * P[m].mean())) if m.sum() > 60 else "  -"
        spill = float((P * ((vis[t, 1 - p] > 200) if p == 0 else (vis[t, 1 - p] > 200))).sum()) if p == 0 else float((P * (vis[t, 0] > 200)).sum())
        ch = "   -"
        if p in prev:
            m = v & prev[p][1]; ch = "%4.1f" % (100 * np.abs(P - prev[p][0])[m].mean()) if m.sum() > 60 else "   -"
        prev[p] = (P, v)
        line += " %s: vis %3.0f%% all %s | %s %s %s %s | on-other %5.0f | chg %s |" % ("AB"[p], 100 * share, band(0, 1e9), band(0, 8), band(8, 20), band(20, 40), band(40, 1e9), spill, ch)
    print(line)
