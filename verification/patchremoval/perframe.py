"""usage: perframe.py CLIPDIR name...   per-frame share of the mark left (video_test.py --save output), and pixels changed
farther than 25 px from the mark ('out')."""
import sys, numpy as np, cv2
def frames(p):
    c = cv2.VideoCapture(p); o = []
    while True:
        ok, f = c.read()
        if not ok: return o
        o.append(f)
L = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
for name in sys.argv[2:]:
    g = np.load(f"{sys.argv[1]}/pr_{name}_gt.npz"); mark, occ = g["mark"], g["occ"]
    out, cl, inp = frames(f"/tmp/claude-0/prout_{name}.avi"), frames(f"{sys.argv[1]}/pr_{name}_clean.mp4"), frames(f"{sys.argv[1]}/pr_{name}.mp4")
    rows = []
    for i in range(len(out)):
        mk = mark[i]; lc, lo, li = L(cl[i]), L(out[i]), L(inp[i])
        far = ~cv2.dilate(mk.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))).astype(bool) if mk.any() else np.ones(mk.shape, bool)
        o = 0 if occ[i].any() else int((np.abs(lo - li)[far] > 12).sum())   # (object frames: see video_test 'object')
        if mk.sum() <= 30:
            rows.append(f"{i}:-" + (f"(out {o})" if o else "")); continue
        con = float((lc - li)[mk].mean()); rem = float((lc - lo)[mk].mean())
        r = int(100 * rem / max(con, 1e-3))
        if r > 10 or o: rows.append(f"{i}:{r}%" + (f"(out {o})" if o else ""))
    print(name, " ".join(rows))
