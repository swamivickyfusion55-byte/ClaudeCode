"""usage: cheekmetrics.py LABEL...   -> roughness of what the swap adds in the cheek band, and of the mask's cheek edge."""
import sys, cv2, numpy as np
FACE = (466, 268, 573, 415)    # f2, y offset 100
x1, y1, x2, y2 = FACE; y1 -= 100; y2 -= 100; fw = x2 - x1; fh = y2 - y1
def frames(path):
    c = cv2.VideoCapture(path); out = []
    while True:
        ok, f = c.read()
        if not ok: break
        out.append(f)
    return out
def band_boxes():
    return [(int(x1 - 0.12 * fw), int(y1 + 0.40 * fh), int(x1 + 0.38 * fw), int(y1 + 0.88 * fh)),
            (int(x1 + 0.62 * fw), int(y1 + 0.40 * fh), int(x2 + 0.12 * fw), int(y1 + 0.88 * fh))]
def metrics(label, clip):
    W = f"/tmp/claude-0/bench_work/{label}"
    inp = frames(clip); out = frames(f"{W}/out/BENCHCK_/final.mp4"); n = min(len(inp), len(out))
    D = []
    for i in range(n):
        d = out[i].astype(np.float32) - inp[i].astype(np.float32)
        D.append(np.concatenate([d[b[1]:b[3], b[0]:b[2]].reshape(-1, 3) for b in band_boxes()]).mean(axis=1))   # grey levels of the swap's contribution
    D = np.array(D)
    d1 = np.abs(np.diff(D, axis=0)).mean(axis=1)             # temporal change of the swap's contribution
    d2 = np.abs(D[2:] - 2 * D[1:-1] + D[:-2]).mean(axis=1)   # temporal 2nd difference (jerk)
    # frames 15-81 = swell active
    sl = slice(15, 81)
    r = dict(label=label, n=n, jerk_active=float(d2[sl].mean()), jerk_all=float(d2.mean()), change_active=float(d1[sl].mean()),
             p95_jerk_active=float(np.percentile(d2[sl], 95)))
    # the mask's cheek edge: rows 0.50..0.70 of the 128 aligned crop, left & right half-width where mask crosses 0.5
    try:
        m = np.load(f"{W}/mask_dump.npz")["m"]
        k = m.shape[0] // n if m.shape[0] % n == 0 and m.shape[0] >= n else 1
        sel = m[0::k] if k > 1 else m
        # pick the entry per frame with the largest mean
        if k > 1:
            sel = np.stack([max(m[j * k:(j + 1) * k], key=lambda a: a.mean()) for j in range(n)])
        rows = slice(int(0.50 * 128), int(0.70 * 128))
        wid = (sel[:, rows, :] > 0.5).sum(axis=2).mean(axis=1)            # mask width (px of 128) in cheek rows
        w1 = np.abs(np.diff(wid)); w2 = np.abs(wid[2:] - 2 * wid[1:-1] + wid[:-2])
        r.update(mask_width_mean=float(wid.mean()), mask_width_jerk_active=float(w2[sl].mean()), mask_width_p95=float(np.percentile(w2[sl], 95)),
                 mask_width_range=float(wid[sl].max() - wid[sl].min()))
    except Exception as e:
        r["mask_err"] = str(e)
    return r
if __name__ == "__main__":
    for lab in sys.argv[1:]:
        clip = "cheek_swell.mp4" if lab.endswith("swell") else "cheek_static.mp4"
        r = metrics(lab, clip); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
