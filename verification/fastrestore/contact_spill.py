"""usage: contact_spill.py RUN:CLIP ...   (both faces swapped; mask dump of the run; make_kiss.py clips, synthetic)
Each face's own paint weight (the mask it was composited with, mapped back onto the frame) against the true visible areas:
  spill = paint weight of one face on the OTHER person's visible area (outside its own), % of that person's visible area
  seen  = for A (behind, painted first): A's paint left visible after B is painted over it, A x (1 - B), same base
  cover = paint weight of a face on its own visible area (eroded 6 px), % - how fully it is painted
In contact frames (the two people within 12 px), mean and worst; both directions (A behind, B in front)."""
import sys, cv2, numpy as np
sys.path.insert(0, "/home/user/ClaudeCode/v11.2.95_OcclusionPro"); import swap_engine as E
B = "/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/"
for arg in sys.argv[1:]:
    run, clip = arg.split(":")
    gt = np.load(B + f"kiss_{clip}_gt.npz"); vis, kps = gt["a"], gt["kps"]; n = len(vis)
    z = np.load(f"/tmp/claude-0/bench_work/{run}/mask_dump.npz")
    ent = {}
    for m, M, size, sl in zip(z["m"], z["M"], z["size"], z["slot"]):
        ent.setdefault(int(sl), []).append((m, M, int(size)))
    smap = {}
    for sl, L in ent.items():
        votes = [0, 0]
        for k, (m, M, size) in enumerate(L[:8]):
            IM = cv2.invertAffineTransform(M); c = IM[:, :2] @ np.array([size / 2, size / 2]) + IM[:, 2]
            votes[int(np.argmin([np.linalg.norm(c - kps[k, p].mean(0)) for p in range(2)]))] += 1
        smap[sl] = int(np.argmax(votes))
    res = {p: {"spill": [], "cover": [], "seen": []} for p in range(2)}
    for t in range(n):
        a = vis[t, 0] > 127; b = vis[t, 1] > 127
        if (cv2.dilate(a.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) & b).sum() < 500: continue
        W = {}
        for sl, L in ent.items():
            if t < len(L):
                m, M, size = L[t]; W[smap[sl]] = cv2.warpAffine(cv2.resize(m, (size, size)), cv2.invertAffineTransform(M), (1280, 720))
        for p, w in W.items():
            if p == 0 and 1 in W:
                oth = (vis[t, 1] > 127) & ~(vis[t, 0] > 127)
                if oth.sum() > 2000: res[0]["seen"].append(float((w * (1 - W[1]))[oth].sum()) / float(oth.sum()) * 100)
            own = vis[t, p] > 127; oth = (vis[t, 1 - p] > 127) & ~own
            if oth.sum() > 2000: res[p]["spill"].append(float(w[oth].sum()) / float(oth.sum()) * 100)
            core = cv2.erode(own.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool) & (np.arange(720)[:, None] > 200) & (np.arange(720)[:, None] < 560)
            if core.sum() > 2000: res[p]["cover"].append(float(w[core].mean()) * 100)
    out = []
    for p in range(2):
        sp, cv = np.array(res[p]["spill"]), np.array(res[p]["cover"])
        se = np.array(res[p]["seen"]); sx = f" (left visible {se.mean():.1f}%, worst {se.max():.1f}%)" if p == 0 and len(se) else ""
        out.append(f"{'AB'[p]} ({'behind' if p == 0 else 'in front'}): paint on the other person mean {sp.mean():.1f}% worst {sp.max():.1f}%{sx}, own face painted {cv.mean():.0f}%")
    print(f"{run}: " + "; ".join(out))
