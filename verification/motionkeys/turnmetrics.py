"""usage: turnmetrics.py LABEL...  (label ts_<tag>_<clip>_<Q>). Single painted face vs the true geometry of the synthetic clip (turn_<clip>_gt.npz)."""
import sys, cv2, numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/pack88')
import swap_engine as E
def frames(path):
    c = cv2.VideoCapture(path); out = []
    while True:
        ok, f = c.read()
        if not ok: break
        out.append(f)
    return out
def analyse(label):
    clip = label.split('_')[2]; W = f"/tmp/claude-0/bench_work/{label}"
    inp = frames(f"turn_{clip}.mp4"); out = frames(f"{W}/out/BENCHTS_/final.mp4"); kps = np.load(f"turn_{clip}_gt.npz")['kps']
    n = min(len(inp), len(out)); res = {}
    z = np.load(f"{W}/mask_dump.npz"); Ms, sizes = z['M'], z['size']
    slots = z['slot']; pts_all = []
    for M, size in zip(Ms, sizes):
        IM = cv2.invertAffineTransform(M); pts_all.append(E._arcface_dst(int(size)) @ IM[:, :2].T + IM[:, 2])
    best = None
    for sl in sorted(set(int(x) for x in slots)):
        idx = [i for i, x in enumerate(slots) if int(x) == sl]
        d = np.mean([np.linalg.norm(pts_all[i].mean(axis=0) - kps[k].mean(axis=0)) for k, i in enumerate(idx[:8])])
        if best is None or d < best[0]: best = (d, sl, idx)
    idx = best[2]; errs = []; scal = []
    for t, i in enumerate(idx[:len(kps)]):
        iod = np.linalg.norm(kps[t, 1] - kps[t, 0]); errs.append(np.linalg.norm(pts_all[i] - kps[t], axis=1).mean() / iod); scal.append(np.linalg.norm(pts_all[i][1] - pts_all[i][0]) / iod)
    res["painted placements of the real face (slot %d)" % best[1]] = f"{len(idx)} of {len(kps)} frames"
    errs = np.array(errs); moving = np.arange(len(errs)) >= 20
    res["placement error (inter-eye units) still / moving: median"] = f"{np.median(errs[~moving]):.3f} / {np.median(errs[moving]):.3f}"
    res["placement error moving: p95 / max / frames > 0.25 / > 0.5"] = f"{np.percentile(errs[moving],95):.3f} / {errs[moving].max():.3f} / {int((errs[moving] > .25).sum())} / {int((errs[moving] > .5).sum())} of {int(moving.sum())}"
    # painted region: face box from GT
    D = np.zeros(n); S = np.zeros(n); F = np.zeros(n)
    for t in range(n):
        c = kps[t].mean(axis=0); iod = np.linalg.norm(kps[t, 1] - kps[t, 0]); r = int(1.1 * iod)
        x1, x2, y1, y2 = int(c[0] - r), int(c[0] + r), int(c[1] - r), int(c[1] + r)
        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(1280, x2), min(720, y2)
        if x2 - x1 < 20 or y2 - y1 < 20: continue
        a = out[t][y1:y2, x1:x2].astype(np.float32); b = inp[t][y1:y2, x1:x2].astype(np.float32)
        D[t] = np.abs(a - b).mean()
        g = cv2.cvtColor(out[t][y1:y2, x1:x2], cv2.COLOR_BGR2GRAY); gi = cv2.cvtColor(inp[t][y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        S[t] = cv2.Laplacian(g, cv2.CV_32F).var() / max(1.0, cv2.Laplacian(gi, cv2.CV_32F).var())
    ref = np.median(D[:20]); rev = D[20:n] < 0.45 * ref
    res["reverted frames (swap missing) moving"] = f"{int(rev.sum())} of {n-20}   (painted level {ref:.1f})"
    res["sharpness of the output face region vs the input's (Laplacian var ratio) still / moving"] = f"{np.median(S[:20]):.2f} / {np.median(S[20:n]):.2f}"
    # temporal flicker: frame-to-frame change of the swapped face's brightness inside a face-locked box vs the input's
    ji, jo = [], []
    for t in range(21, n):
        c = kps[t].mean(axis=0); iod = np.linalg.norm(kps[t, 1] - kps[t, 0]); r = int(0.9 * iod)
        x1, x2, y1, y2 = int(c[0] - r), int(c[0] + r), int(c[1] - r), int(c[1] + r)
        if x1 < 0 or y1 < 0 or x2 > 1280 or y2 > 720: continue
        ca = kps[t - 1].mean(axis=0); xa1, xa2, ya1, ya2 = int(ca[0] - r), int(ca[0] + r), int(ca[1] - r), int(ca[1] + r)
        if xa1 < 0 or ya1 < 0 or xa2 > 1280 or ya2 > 720: continue
        # compare the two boxes' mean colour = brightness pumping of the pasted face as it moves
        jo.append(abs(float(out[t][y1:y2, x1:x2].mean()) - float(out[t - 1][ya1:ya2, xa1:xa2].mean())))
        ji.append(abs(float(inp[t][y1:y2, x1:x2].mean()) - float(inp[t - 1][ya1:ya2, xa1:xa2].mean())))
    res["brightness pumping of the face as it moves (mean |change|, out / in)"] = f"{np.mean(jo):.2f} / {np.mean(ji):.2f}"
    return res
if __name__ == "__main__":
    for lab in sys.argv[1:]:
        print("==", lab)
        for k, v in analyse(lab).items(): print(f"   {k}: {v}")
