"""usage: kissmetrics.py LABEL... ; label is ks_<tag>_<clip> (clip = cheek | near)
Per person: frames where the swap is missing (reverted to the original face), jitter of the painted face, placement error of the face that was
painted against the true face geometry, and slot/person flips."""
import sys, cv2, numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/pack86')
import swap_engine as E
def frames(path):
    c = cv2.VideoCapture(path); out = []
    while True:
        ok, f = c.read()
        if not ok: break
        out.append(f)
    return out
def analyse(label):
    clip = label.split('_')[-1]; W = f"/tmp/claude-0/bench_work/{label}"
    inp = frames(f"kiss_{clip}.mp4"); out = frames(f"{W}/out/BENCHKS_/final.mp4"); gt = np.load(f"kiss_{clip}_gt.npz"); vis, kps = gt['a'], gt['kps']
    n = min(len(inp), len(out)); res = {}
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    D = np.zeros((n, 2)); ok_ = np.zeros((n, 2), bool)
    for t in range(n):
        for p in range(2):
            reg = cv2.erode((vis[t, p] > 127).astype(np.uint8), ker).astype(bool)
            reg &= (np.arange(720)[:, None] > 200) & (np.arange(720)[:, None] < 560)        # the face part of the person, not hair / shoulders
            if reg.sum() < 3000: continue
            d = np.abs(out[t].astype(np.float32) - inp[t].astype(np.float32)).mean(axis=2)
            D[t, p] = float(d[reg].mean()); ok_[t, p] = True
    for p in range(2):
        ref = np.median(D[:8, p][ok_[:8, p]]) if ok_[:8, p].any() else 1.0
        rev = ok_[:, p] & (D[:, p] < 0.45 * ref)
        c = slice(40, 80); a_ = slice(0, 30)
        res[f"person{p} reverted frames (all / in contact 40-79)"] = f"{int(rev.sum())} / {int(rev[c].sum())}   (painted level {ref:.1f})"
        # jitter of the painted face: mean |change between consecutive frames| of the output inside the region, relative to the input's own change
        jo, ji = [], []
        for t in range(1, n):
            if ok_[t, p] and ok_[t - 1, p]:
                reg = cv2.erode((vis[t, p] > 127).astype(np.uint8), ker).astype(bool) & (np.arange(720)[:, None] > 200) & (np.arange(720)[:, None] < 560)
                reg &= cv2.erode((vis[t - 1, p] > 127).astype(np.uint8), ker).astype(bool)
                if reg.sum() < 3000: continue
                jo.append((t, float(np.abs(out[t].astype(np.float32) - out[t - 1].astype(np.float32)).mean(axis=2)[reg].mean())))
                ji.append((t, float(np.abs(inp[t].astype(np.float32) - inp[t - 1].astype(np.float32)).mean(axis=2)[reg].mean())))
        for nm, lo, hi in (("apart", 0, 30), ("contact", 40, 80)):
            o = [v for t, v in jo if lo <= t < hi]; i_ = [v for t, v in ji if lo <= t < hi]
            if o: res[f"person{p} frame-to-frame change out/in ({nm})"] = f"{np.mean(o):.2f} / {np.mean(i_):.2f}  (x{np.mean(o) / max(np.mean(i_), 1e-3):.2f})"
    # placement of what was painted vs the true face geometry
    try:
        z = np.load(f"{W}/mask_dump.npz"); Ms, sizes, slots = z['M'], z['size'], z['slot']
        cgt = np.stack([[np.mean(kps[t, p], axis=0) for p in range(2)] for t in range(len(kps))])      # (T,2,2) true face centres (mean of the 5 points)
        wgt = np.array([[np.linalg.norm(kps[t, p, 1] - kps[t, p, 0]) for p in range(2)] for t in range(len(kps))])   # inter-eye distance
        rows = []; cnt = {}
        # the k-th placement of a slot belongs to frame k (one per frame while the slot is painted); the slot -> person map comes from the first frames
        ent = []
        for M, size, slot in zip(Ms, sizes, slots):
            IM = cv2.invertAffineTransform(M); dst = E._arcface_dst(int(size)); pts = (dst @ IM[:, :2].T + IM[:, 2])
            k = cnt.get(int(slot), 0); cnt[int(slot)] = k + 1; ent.append((int(slot), k, pts))
        smap = {}
        for sl in cnt:
            votes = [0, 0]
            for (s_, k, pts) in ent:
                if s_ == sl and k < 8:
                    votes[int(np.argmin([np.linalg.norm(cgt[min(k, len(cgt) - 1), p] - pts.mean(axis=0)) for p in range(2)]))] += 1
            smap[sl] = int(np.argmax(votes))
        for (s_, k, pts) in ent:
            t = min(k, len(kps) - 1); p = smap[s_]
            err = np.linalg.norm(pts - kps[t, p], axis=1).mean() / wgt[t, p]
            rows.append((t, p, s_, err, np.linalg.norm(pts[1] - pts[0]) / wgt[t, p]))
        res["slot -> person map"] = str(smap)
        rows = np.array(rows)
        for p in range(2):
            r = rows[rows[:, 1] == p]
            if len(r):
                res[f"person{p} placement error (inter-eye units): median / p95 / frames > 0.25"] = f"{np.median(r[:, 3]):.3f} / {np.percentile(r[:, 3], 95):.3f} / {int((r[:, 3] > 0.25).sum())} of {len(r)}"
                res[f"person{p} slot flips (painting slot changes along the clip)"] = int((np.diff(r[:, 2]) != 0).sum())
                rc = r[(r[:, 0] >= 40) & (r[:, 0] < 80)]
                if len(rc): res[f"person{p} placement error in contact"] = f"median {np.median(rc[:, 3]):.3f} / p95 {np.percentile(rc[:, 3], 95):.3f}   scale x{np.median(rc[:, 4]):.2f}"
    except Exception as e:
        res["placement"] = f"n/a ({e})"
    return res
if __name__ == "__main__":
    for lab in sys.argv[1:]:
        print("==", lab)
        for k, v in analyse(lab).items(): print(f"   {k}: {v}")
