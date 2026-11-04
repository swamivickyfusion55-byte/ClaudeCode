"""v11.2.112: how much of the tattoo is left in a finished job's video (make_bellyreal.py clip).
usage: belly_e2e_eval.py CLIPDIR OUTPUT_MP4 [PACKDIR [length]]  -> share of the tattoo's contrast left per frame (with PACKDIR also
inside the band, from the clip's true shoulders / hips: 8 px or more inside its outline), and the change away from the tattoo."""
import sys, numpy as np, cv2
clip, outp = sys.argv[1], sys.argv[2]
T = np.load(clip + "/truth.npz")
def frames(p):
    c = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(f)
    return fr
src, cln, out = frames(clip + "/clip.mp4"), frames(clip + "/clean.mp4"), frames(outp)
n = min(len(src), len(cln), len(out))
L = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
BC = None
if len(sys.argv) > 3:
    sys.path.insert(0, sys.argv[3]); import belly_clear as BC
LG = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
left, ch_far, left_in = [], [], []
for i in range(n):
    o = out[i] if out[i].shape == src[i].shape else cv2.resize(out[i], (src[i].shape[1], src[i].shape[0]), interpolation=cv2.INTER_AREA)
    m = T["mask"][i]
    con = float((L(cln[i]) - L(src[i]))[m].mean())
    left.append(float((L(cln[i]) - L(o))[m].mean()) / max(1e-6, con))
    if BC is not None:
        tl = T["lm"][i]; lmd = {11: (tl[0][0], tl[0][1], 1.0), 12: (tl[1][0], tl[1][1], 1.0), 23: (tl[2][0], tl[2][1], 1.0), 24: (tl[3][0], tl[3][1], 1.0)}
        bd = BC.region_mask(BC.region_polygon(lmd, LG, 1.0), m.shape)
        mi = m & (cv2.erode(bd.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0)
        if mi.sum() > 20:
            left_in.append(float((L(cln[i]) - L(o))[mi].mean()) / max(1e-6, float((L(cln[i]) - L(src[i]))[mi].mean())))
    far = ~cv2.dilate(m.astype(np.uint8), np.ones((121, 121), np.uint8)).astype(bool)      # well away from the tattoo
    ch_far.append(float(np.abs(L(o) - L(src[i]))[far].mean()))
left = np.array(left)
print("frames %d: tattoo contrast left mean %.1f %%, worst frame %.1f %%, frames over 25 %%: %d; mean |dL| away from the tattoo %.2f (encode noise)"
      % (n, 100 * left.mean(), 100 * left.max(), int((left > 0.25).sum()), float(np.mean(ch_far))))
if left_in:
    li = np.array(left_in)
    print("inside the band: tattoo contrast left mean %.1f %%, worst frame %.1f %%, frames over 25 %%: %d of %d" % (100 * li.mean(), 100 * li.max(), int((li > 0.25).sum()), len(li)))
