"""Stand-in measurement on OUR v11.2.118 belly_clear (not the upload): what does a lower dark threshold do on clean skin?
A = 118 defaults (dark_min 12, dark_k 3 x2 noise)   B = the setting Grok's 126 uses (dark_min 6, dark_k 3 x1 noise) written as dark_min 6, dark_k 1.5.
Input: the synthetic belly clip's clean.mp4 (no tattoo), truth landmarks. Variants: plain; soft shading (side light + crease); video noise."""
import sys, os, types
PACK, CLIP = sys.argv[1], sys.argv[2]
sys.path.insert(0, PACK)
import numpy as np, cv2
import belly_clear as BC
T = np.load(os.path.join(CLIP, "truth.npz")); LM = T["lm"]
cap = cv2.VideoCapture(os.path.join(CLIP, "clean.mp4")); fr = []
while True:
    ok, f = cap.read()
    if not ok: break
    fr.append(f)
def lmd(i):
    t = LM[i]; return {11: (t[0][0], t[0][1], 1.0), 12: (t[1][0], t[1][1], 1.0), 23: (t[2][0], t[2][1], 1.0), 24: (t[3][0], t[3][1], 1.0)}
rng = np.random.default_rng(3)
def shade(f, kind, i):
    f = f.astype(np.float32); h, w = f.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    if kind == "plain": g = np.zeros((h, w), np.float32)
    elif kind == "side-light":                       # a soft light falling off sideways, up to -14 grey levels
        g = -14.0 * np.clip((xx - 0.35 * w) / (0.45 * w), 0, 1)
    elif kind == "crease":                           # a soft skin fold: a dark line 9 px wide, -10 levels, across the belly
        L = LM[i]; cy = 0.5 * (L[2][1] + L[3][1]) - 0.30 * abs(0.5 * (L[0][1] + L[1][1]) - 0.5 * (L[2][1] + L[3][1]))
        g = -10.0 * np.exp(-((yy - cy) / 9.0) ** 2)
    elif kind == "noise4":                           # video noise sigma 4 (a phone in low light)
        g = rng.normal(0, 4.0, (h, w)).astype(np.float32)
    out = np.clip(f + g[..., None], 0, 255).astype(np.uint8)
    if kind == "noise4":                             # then a compression pass
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 70]); out = cv2.imdecode(enc, 1)
    return out
cfgs = {"A: 118 (dark_min 12, 6 sigma)": (12.0, 3.0), "B: 126 setting (dark_min 6, 3 sigma)": (6.0, 1.5)}
print("%-34s %-11s %10s %12s" % ("config", "light", "mean px", "frames>200px"))
idx = list(range(5, 85, 8))
for name, (dm, dk) in cfgs.items():
    BC.CFG["dark_min"], BC.CFG["dark_k"] = dm, dk
    for kind in ("plain", "side-light", "crease", "noise4"):
        px = []
        for i in idx:
            img = shade(fr[i], kind, i)
            mk, info = BC.find_marks(img, lmd(i))
            px.append(0 if mk is None else int(mk.sum()))
        print("%-34s %-11s %10.0f %9d/%d" % (name, kind, np.mean(px), sum(p > 200 for p in px), len(px)))
