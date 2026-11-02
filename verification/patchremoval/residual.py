"""usage: residual.py CLIPDIR name...   (video_test.py --save outputs) Faint leftover ON the mark: on frames where the mark
was removed (less than half of it left), how much darker than the clean skin the result still is, in lightness L (0-255):
inside the mark (eroded 2 px), on its rim (the mark's outer 2 px), and just outside the mark (2-6 px, should be ~0:
nothing outside the mark should change)."""
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
    out, cl, inp = frames(f"{__import__('os').environ.get('OUTDIR', '/tmp/claude-0')}/prout_{name}.avi"), frames(f"{sys.argv[1]}/pr_{name}_clean.mp4"), frames(f"{sys.argv[1]}/pr_{name}.mp4")
    rin, rrim, rout, cin, n = [], [], [], [], 0
    k = np.ones((5, 5), np.uint8)
    for i in range(min(len(out), len(mark))):
        mk = mark[i]
        if mk.sum() < 60 or occ[i].any(): continue
        lc, lo, li = L(cl[i]), L(out[i]), L(inp[i])
        con = float((lc - li)[mk].mean())
        if con <= 3 or float((lc - lo)[mk].mean()) / con > 0.5: continue        # (missed frames are counted elsewhere)
        inner = cv2.erode(mk.astype(np.uint8), k).astype(bool)
        rim = mk & ~inner
        ring = cv2.dilate(mk.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool) & ~cv2.dilate(mk.astype(np.uint8), k).astype(bool)
        if inner.sum() > 20: rin.append(float((lc - lo)[inner].mean())); cin.append(con)
        rrim.append(float((lc - lo)[rim].mean())); rout.append(float((lc - lo)[ring].mean())); n += 1
    print(f"{name}: {n} frames; darker than clean skin by - inside the mark {np.mean(rin):+.2f} L ({100*np.mean(rin)/np.mean(cin):.1f}% of the mark), "
          f"on its rim {np.mean(rrim):+.2f}, just outside it {np.mean(rout):+.2f}")
