"""v11.2.142 BellyAnchor unit checks (no models). usage: unit142.py PACKDIR CLIP27   (make_bellytattoo.py clip, navel at 0.27)"""
import sys, os
PACK, CLIP = sys.argv[1], sys.argv[2]
sys.path.insert(0, PACK)
import numpy as np, cv2
import belly_clear as BC
fails = 0
def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else "")); fails += 0 if ok else 1
T = np.load(os.path.join(CLIP, "truth.npz")); c = cv2.VideoCapture(os.path.join(CLIP, "clip.mp4")); fr = []
while True:
    ok, f = c.read()
    if not ok: break
    fr.append(f)
def lmd(i):
    t = T["lm"][i]; return {11: (*t[0], 1.0), 12: (*t[1], 1.0), 23: (*t[2], 1.0), 24: (*t[3], 1.0)}
L0 = lmd(0)
# 1. body coordinates
pts = np.array([[350, 400], [420, 450], [300, 380]], np.float64)
rt = BC.from_body(BC.to_body(pts, L0), L0)
check("body coordinates: image -> (u, t) -> image round trip", float(np.abs(rt - pts).max()) < 1e-6, "%.2e px" % float(np.abs(rt - pts).max()))
ut = BC.to_body(np.array([L0[23][:2], L0[24][:2]]), L0)
check("the hip joints are at u = -0.5 / +0.5, t = 0", np.allclose(ut, [[-0.5, 0.0], [0.5, 0.0]], atol=1e-9), str(ut.round(3).tolist()))
# the same body point follows the body: frame 0 -> frame 45 (the clip moves, zooms and turns)
u_navel = BC.to_body(np.array([T["navel"][0]]), lmd(0)); p45 = BC.from_body(u_navel, lmd(45))
check("a point held in body coordinates follows the body (the navel, frame 0 -> 45)", float(np.linalg.norm(p45[0] - T["navel"][45])) < 2.0,
      "%.2f px from the true navel on frame 45" % float(np.linalg.norm(p45[0] - T["navel"][45])))
# 2. the floor is a hard limit for the pentagon
P = BC.clamp_pentagon_body(np.array([[-0.3, 0.4], [0.3, 0.4], [0.4, -0.2], [0.0, -0.5], [-0.4, 0.05]]))
check("pentagon corners dragged below the floor are put back on it (t >= 0.10)", float(P[:, 1].min()) >= 0.10 - 1e-9, str(P[:, 1].round(3).tolist()))
P2 = BC.clamp_pentagon_body(np.array([[-0.3, 0.9], [0.3, 0.9], [0.4, 0.3], [0.0, 0.2], [-0.4, 0.3]]))
check("...and corners dragged onto the chest come down to the band's top (t <= 0.62)", float(P2[:, 1].max()) <= 0.62 + 1e-9)
# 3. the automatic pentagon
mk, _ = BC.find_marks(fr[0], L0)
pent = BC.auto_pentagon(mk, L0)
pi = np.round(pent).astype(np.int32); ys, xs = np.nonzero(mk)
inside = np.mean([cv2.pointPolygonTest(pi, (float(x), float(y)), False) >= 0 for x, y in zip(xs, ys)])
check("automatic pentagon: 5 corners, every mark pixel inside", pent.shape == (5, 2) and inside == 1.0, "%d corners, %.1f %% inside" % (len(pent), 100 * inside))
d = BC.auto_pentagon(np.zeros_like(mk), L0)
check("no mark: a default pentagon below the navel (5 corners, above the floor)", d.shape == (5, 2) and float(BC.to_body(d, L0)[:, 1].min()) >= 0.10)
# 4. only the marks inside the pentagon
up = T["parts"][0] == 1                                                       # the upper-belly tattoo
ys1, xs1 = np.nonzero(up); box = np.array([[xs1.min() - 12, ys1.min() - 12], [xs1.max() + 12, ys1.min() - 12], [xs1.max() + 18, (ys1.min() + ys1.max()) / 2],
                                           [xs1.max() + 12, ys1.max() + 12], [xs1.min() - 12, ys1.max() + 12]], np.float64)
m2, _ = BC.find_marks(fr[0], L0, region=box)
rec_up = float((m2 & up).sum()) / up.sum()
far = int((m2 & ((T["parts"][0] == 2) | (T["parts"][0] == 3))).sum())        # the tattoo beside the navel, the lower tattoo
# each piece whole or not at all: every piece of the band's marks is either fully taken (its middle inside) or fully left
allm, _ = BC.find_marks(fr[0], L0)
nn, lab, _s, cen = cv2.connectedComponentsWithStats(allm.astype(np.uint8), connectivity=8)
rp = box.astype(np.float32).reshape(-1, 1, 2); wrong = 0
for k in range(1, nn):
    pk = lab == k; ins = cv2.pointPolygonTest(rp, (float(cen[k][0]), float(cen[k][1])), False) >= 0
    wrong += int(((m2 & pk).sum() != pk.sum()) if ins else (m2 & pk).any())
check("a pentagon over the upper tattoo: that tattoo found, the other tattoos not touched; every piece whole or not at all (by its middle)",
      rec_up >= 0.85 and far == 0 and wrong == 0, "%.0f %% of it, %d px of the other tattoos, %d of %d pieces cut" % (100 * rec_up, far, wrong, nn - 1))
# 5. settings parsing
check("pentagon setting: 5 x 2 list or JSON text; junk -> None", BC.parse_pentagon([[0, 1]] * 5).shape == (5, 2) and BC.parse_pentagon("[[0,1],[1,1],[1,0],[0,0],[0.5,-1]]") is not None
      and BC.parse_pentagon("x") is None and BC.parse_pentagon([[0, 1]] * 4) is None and BC.parse_pentagon([[0, float("nan")]] * 5) is None)
# 6. the body path: smoothing and interpolation, every 2nd frame sampled with 3 px of noise
rng = np.random.default_rng(1); samples = {}
for i in range(0, 90, 2):
    t = T["lm"][i] + rng.normal(0, 3.0, (4, 2)); samples[i] = {11: (*t[0], 1.0), 12: (*t[1], 1.0), 23: (*t[2], 1.0), 24: (*t[3], 1.0)}
bp = BC.BodyPath(samples, 90, 30.0)
err_raw = np.mean([np.abs(np.array([samples[i][k][:2] for k in (11, 12, 23, 24)]) - T["lm"][i]).mean() for i in samples])
err_sm = np.mean([np.abs(np.array([bp.at(i)[k][:2] for k in (11, 12, 23, 24)]) - T["lm"][i]).mean() for i in range(2, 88)])
check("body path: the smoothed, interpolated points sit closer to the truth than the noisy samples", err_sm < 0.75 * err_raw,
      "%.2f px (samples %.2f px)" % (err_sm, err_raw))
check("body path: no body beyond the hold", BC.BodyPath({0: samples[0]}, 300, 30.0).at(200) is None)

check("preview marks setting (belly_marks): [u, t, taken, px] rows as a list or JSON (px optional); junk -> None",
      BC.parse_marks([[0.1, 0.4, 1, 300], [0.3, 0.39, 0, 90]]).shape == (2, 4) and BC.parse_marks("[[0.1,0.4,1]]")[0, 3] == 100.0
      and BC.parse_marks("junk") is None and BC.parse_marks([[1, 2]]) is None and BC.parse_marks([]) is None)
# --- the body anchor's rules (swap_engine.anchor_to_body), on a stand-in track: pure translations, body moving 3 px/frame right
import swap_engine as E
def tr_(n=60, img=None, ecc=None):
    class Tr: pass
    tr = Tr(); tr.box = (100, 140, 200, 260); tr.anchor_idx = 0; tr.lo, tr.hi = 0, n - 1; tr.group = "belly"
    tr.A = {i: np.array([[1, 0, img(i)[0]], [0, 1, img(i)[1]]], float) for i in range(n) if img(i) is not None}
    tr.local_A = {i: a.copy() for i, a in tr.A.items()}; tr.conf = {i: 1.0 for i in tr.A}; tr.weak = set()
    tr._ecc = ecc if ecc is not None else (lambda g, A0: (A0, -1.0))
    tr._det = False; tr.tmpl_c = tr.tmpl = np.zeros((40, 60), np.float32)   # (a flat template: the coarse fit gives up, the stand-in ECC decides)
    tr._boxwarp = lambda g, A, border=None: (np.zeros((40, 60), np.float32), np.eye(2, 3))
    return tr
body = lambda i: np.array([[200.0, 0, 3.0 * i + 500], [0, -400.0, 600]])        # (u, t, 1) -> px: torso 400 px, moving 3 px / frame
tx = lambda tr, i: float(tr.A[i][0, 2])
tr = tr_(img=lambda i: (3.0 * i + (2.0 if i % 3 else -2.0), 0.0)); E.anchor_to_body(tr, body)
check("anchor: the image agrees with the body (within the pose's noise) -> every frame kept as the image had it",
      all(abs(tx(tr, i) - (3.0 * i + (2.0 if i % 3 else -2.0))) < 1e-9 for i in range(60)))
tr = tr_(img=lambda i: (3.0 * min(i, 20) if 20 <= i <= 40 else 3.0 * i, 0.0)); E.anchor_to_body(tr, body)
w_ = max(abs(tx(tr, i) - 3.0 * i) for i in range(20, 41)); w2_ = max(abs(tx(tr, i) - 3.0 * i) for i in range(23, 41))
check("anchor: the image hangs on frames 20-40 while the body moves on (60 px by frame 40) -> the cover stays within 8 px of the body",
      w_ <= 8.0, "worst %.1f px on 20-40 (image alone: 60 px), %.1f px on 23-40" % (w_, w2_))
tr = tr_(img=lambda i: None if 20 <= i <= 40 else (3.0 * i + 5.0, 0.0)); E.anchor_to_body(tr, body)
check("anchor: the image lost the patch on 20-40 -> placed by the body with the image's offset carried over, painted (conf >= 0.6)",
      all(i in tr.A and abs(tx(tr, i) - (3.0 * i + 5.0)) < 1e-6 and tr.conf[i] >= 0.6 for i in range(20, 41)))
tr = tr_(img=lambda i: (3.0 * i + (40.0 if 30 <= i <= 32 else 0.0), 0.0)); E.anchor_to_body(tr, body)
check("anchor: a 3-frame disagreement (shorter than the 4-frame run) is not acted on",
      all(abs(tx(tr, i) - (3.0 * i + (40.0 if 30 <= i <= 32 else 0.0))) < 1e-9 for i in range(60)))
def ecc_ok(g, A0):
    A1 = np.array(A0, float).copy(); A1[0, 2] += 4.0; return A1, 0.93
tr = tr_(img=lambda i: None if 20 <= i <= 40 else (3.0 * i, 0.0), ecc=ecc_ok); E.anchor_to_body(tr, body, gray_at=lambda i: np.zeros((8, 8), np.uint8))
check("anchor: a body-placed frame is fitted to the image from the body's place when the fit correlates well and stays near",
      all(abs(tx(tr, i) - (3.0 * i + 4.0)) < 1e-6 for i in range(20, 41)))
def ecc_far(g, A0):
    A1 = np.array(A0, float).copy(); A1[0, 2] += 90.0; return A1, 0.95
tr = tr_(img=lambda i: None if 20 <= i <= 40 else (3.0 * i, 0.0), ecc=ecc_far); E.anchor_to_body(tr, body, gray_at=lambda i: np.zeros((8, 8), np.uint8))
ok_far = all(abs(tx(tr, i) - 3.0 * i) < 1e-6 for i in range(20, 41))
def ecc_low(g, A0):
    A1 = np.array(A0, float).copy(); A1[0, 2] += 4.0; return A1, 0.40
tr = tr_(img=lambda i: None if 20 <= i <= 40 else (3.0 * i, 0.0), ecc=ecc_low); E.anchor_to_body(tr, body, gray_at=lambda i: np.zeros((8, 8), np.uint8))
check("anchor: ...and refused when it jumps away (90 px) or correlates poorly (0.40): the body's place stands",
      ok_far and all(abs(tx(tr, i) - 3.0 * i) < 1e-6 for i in range(20, 41)))
# search + fine fit with the tracker's REAL methods on a textured stand-in clip: the body is 15 px off on the lost frames 20-40
import types
rng = np.random.default_rng(7)
base = sum(cv2.GaussianBlur((rng.random((480, 640)) * 255).astype(np.float32), (0, 0), s_) * w_ for s_, w_ in ((3.0, 1.0), (10.0, 3.0), (30.0, 9.0)))
# (structure at every scale, as skin has: pores and strokes, blotches, shading)
base = cv2.normalize(base, None, 20, 235, cv2.NORM_MINMAX).astype(np.uint8)
grays_ = [cv2.warpAffine(base, np.float32([[1, 0, 3.0 * i], [0, 1, 0]]), (640, 480), borderMode=cv2.BORDER_REFLECT) for i in range(60)]
trr = tr_(img=lambda i: None if 20 <= i <= 40 else (3.0 * i, 0.0)); trr.box = (110, 310, 140, 380); trr._det = True   # as the real tracker:
trr.tmpl = cv2.GaussianBlur(base[110:310, 140:380], (5, 5), 1.0).astype(np.float32); trr.tmpl_c = E._pr_detail(trr.tmpl)  # the lighting-free view
trr._boxwarp = types.MethodType(E.PatchTrack._boxwarp, trr); trr._ecc = types.MethodType(E.PatchTrack._ecc, trr)
body_off = lambda i: np.array([[200.0, 0, 3.0 * i + 500 + (15.0 if 20 <= i <= 40 else 0.0)], [0, -400.0, 600]])
E.anchor_to_body(trr, body_off, gray_at=lambda i: grays_[i])
ws_ = max(abs(tx(trr, i) - 3.0 * i) + abs(float(trr.A[i][1, 2])) for i in range(20, 41))
check("anchor: lost frames, the body 15 px off -> the patch is found in the image from the body's place (coarse-to-fine fit), within 1 px",
      ws_ < 1.0, "worst %.2f px (the body alone: 15 px)" % ws_)
Rr = cv2.getRotationMatrix2D((260.0, 210.0), 3.0, 1.04)                        # the body also 3 deg and 4 % off, not only 28 px
Fo = E._pr_compose(np.vstack([Rr, [0, 0, 1]])[:2], np.array([[1, 0, 90.0 - 22], [0, 1, 0.0 + 17]]))
C2, cc2 = E._body_fit(trr, grays_[30], Fo)
A2, cc3 = trr._ecc(grays_[30], C2); A2b, cc4 = trr._ecc(grays_[30], Fo)
c0_ = np.array([260.0, 210.0, 1.0]); e2 = float(np.linalg.norm(A2 @ c0_ - np.array([350.0, 210.0]))); e0 = float(np.linalg.norm(Fo @ c0_ - np.array([350.0, 210.0])))
check("coarse-to-fine fit: from the body's place 28 px, 3 deg and 4 % off, the patch is found to the pixel (the full-size fit alone fails)",
      e2 < 0.5 and cc3 > 0.9 and (cc4 < 0.7 or float(np.linalg.norm(np.asarray(A2b) @ c0_ - np.array([350.0, 210.0]))) > 2.0),
      "start %.1f px off -> %.2f px (cc %.2f); full-size alone cc %.2f" % (e0, e2, cc3, cc4))
tr = tr_(img=lambda i: (3.0 * i, 0.0)); E.anchor_to_body(tr, lambda i: None if i == 0 else body(i))
check("anchor: no body on the anchor frame -> the track is left exactly as the image had it", all(abs(tx(tr, i) - 3.0 * i) < 1e-9 for i in range(60)))
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
