"""usage: contact_metrics.py both RUN:CLIP ...   |   contact_metrics.py one RUN:CLIP ...
Two people in contact (make_kiss.py clips: true visible area of each person per frame, B in front of A where they overlap).
both: both faces swapped. Per person, in its own visible face area: frames reverted to the original (output within 45 % of
      the footage compared with its painted level), and frame-to-frame change out/in (flicker), apart vs in contact.
one:  only ONE face swapped (one=1). In the OTHER person's visible area the output must equal the footage: any change
      there is the swapped face's paint over the other person (bleed), % of that person's area changed > 12 grey levels,
      apart vs in contact; and the swapped person's own coverage and reverted frames.  Synthetic, no user footage."""
import sys, glob, cv2, numpy as np
B = "/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/"
def frames(p):
    c = cv2.VideoCapture(p); o = []
    while True:
        ok, f = c.read()
        if not ok: return o
        o.append(f)
ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
band = (np.arange(720)[:, None] > 200) & (np.arange(720)[:, None] < 560)
def region(vis):
    return cv2.erode((vis > 127).astype(np.uint8), ker).astype(bool) & band
mode = sys.argv[1]
for arg in sys.argv[2:]:
    run, clip = arg.split(":")
    inp = frames(B + f"kiss_{clip}.mp4"); out = frames(glob.glob(f"/tmp/claude-0/bench_work/{run}/out/*/final.mp4")[0])
    vis = np.load(B + f"kiss_{clip}_gt.npz")["a"]; n = min(len(inp), len(out), len(vis))
    contact = np.zeros(n, bool)
    for t in range(n):
        a = vis[t, 0] > 127; b = vis[t, 1] > 127
        contact[t] = (cv2.dilate(a.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) & b).sum() > 500
    D = np.zeros((n, 2)); ok = np.zeros((n, 2), bool); CH = np.zeros((n, 2))
    for t in range(n):
        d = cv2.GaussianBlur(np.abs(out[t].astype(np.float32) - inp[t].astype(np.float32)).mean(2), (0, 0), 1.5)
        for p in range(2):
            r = region(vis[t, p])
            if r.sum() < 3000: continue
            ok[t, p] = True; D[t, p] = float(d[r].mean()); CH[t, p] = float((d[r] > 12).mean() * 100)
    if mode == "both":
        msg = []
        for p in range(2):
            ref = np.median(D[:8, p][ok[:8, p]]) if ok[:8, p].any() else 1.0
            rev = ok[:, p] & (D[:, p] < 0.45 * ref)
            jo, ji = [], []
            for t in range(1, n):
                if not (ok[t, p] and ok[t - 1, p] and contact[t]): continue
                r = region(vis[t, p]) & region(vis[t - 1, p])
                if r.sum() < 3000: continue
                jo.append(float(np.abs(out[t].astype(np.float32) - out[t - 1].astype(np.float32)).mean(2)[r].mean()))
                ji.append(float(np.abs(inp[t].astype(np.float32) - inp[t - 1].astype(np.float32)).mean(2)[r].mean()))
            fl = f"x{np.mean(jo) / max(np.mean(ji), 1e-3):.2f}" if jo else "n/a"
            msg.append(f"{'AB'[p]}: reverted {int(rev.sum())} frames ({int((rev & contact).sum())} in contact), flicker in contact {fl}")
        print(f"{run}: contact frames {int(contact.sum())}/{n}; " + "; ".join(msg))
    else:
        sw = int(np.argmax([np.median(CH[ok[:, p], p]) if ok[:, p].any() else 0 for p in range(2)])); ot = 1 - sw
        own = CH[ok[:, sw], sw]; bl = CH[:, ot]
        print(f"{run}: swapped person {'AB'[sw]} ({'behind' if sw == 0 else 'in front'}); its own area changed {np.median(own):.0f}% (frames under 40%: {int((own < 40).sum())}); "
              f"paint over person {'AB'[ot]}: apart {np.mean(bl[ok[:, ot] & ~contact]) if (ok[:, ot] & ~contact).any() else 0:.1f}%, in contact mean {np.mean(bl[ok[:, ot] & contact]) if (ok[:, ot] & contact).any() else 0:.1f}% worst {np.max(bl[ok[:, ot] & contact]) if (ok[:, ot] & contact).any() else 0:.1f}%")
