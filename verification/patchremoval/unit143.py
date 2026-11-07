"""v11.2.143 PatchFocus unit checks (no models: the clip's TRUE shoulders and hips). usage: unit143.py PACKDIR CLIPDIR
CLIPDIR: make_bellytattoo.py ... patch=soft marks=patch (part 6 the soft patch, 4 a mole, 7 a freckle)."""
import sys, os
PACK, CLIP = sys.argv[1], sys.argv[2]
sys.path.insert(0, PACK)
import numpy as np, cv2
import belly_clear as BC
fails = 0
def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else ""), flush=True); fails += 0 if ok else 1
def inside_all(P, pts, tol=0.5):
    c = np.asarray(P, np.float32).reshape(-1, 1, 2)
    return all(cv2.pointPolygonTest(c, (float(x), float(y)), True) >= -tol for x, y in pts)
area = lambda P: float(cv2.contourArea(np.asarray(P, np.float32)))
rng = np.random.default_rng(3)
# 1. the smallest pentagon round points (against the 142 way: corners cut, then grown about the middle until all fit)
def old_way(pts):
    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2).astype(np.float64); c = hull.mean(axis=0); P = BC._five(hull); sc = 1.0
    for _ in range(60):
        Q = c + (P - c) * sc
        if inside_all(Q, hull, 0.0): return Q
        sc *= 1.02
    return Q
shapes = {"ellipse": np.stack([60 * np.cos(a) + 200, 25 * np.sin(a) + 150], 1) for a in [np.linspace(0, 2 * np.pi, 80)]}
shapes["blob"] = rng.normal(0, 1, (300, 2)) * [30, 14] + [300, 200]
shapes["L-shape"] = np.vstack([rng.uniform([0, 0], [80, 15], (200, 2)), rng.uniform([0, 0], [15, 60], (200, 2))]) + 100
ok, notes = True, []
for nm, pts in shapes.items():
    P = BC._min_pentagon(pts); hull_a = area(cv2.convexHull(pts.astype(np.float32)))
    ok &= len(P) == 5 and inside_all(P, pts) and area(P) <= 1.30 * hull_a
    notes.append("%s %.2f x its outline (142: %.2f x)" % (nm, area(P) / hull_a, area(old_way(pts)) / hull_a))
check("smallest pentagon: 5 corners, every point inside, at most 1.3 x the points' own outline", ok, "; ".join(notes))
P = BC._min_pentagon(shapes["blob"]); Q = BC._offset_poly(P, 8.0)
d = [cv2.pointPolygonTest(Q.astype(np.float32).reshape(-1, 1, 2), (float(x), float(y)), True) for x, y in P]
check("offset: the pentagon's corners lie 8 px or more inside the pentagon moved 8 px outwards", len(Q) == 5 and min(d) >= 8.0 - 0.5, "%.1f px" % min(d))
# 2. the main dark patch
T = np.load(os.path.join(CLIP, "truth.npz")); cap = cv2.VideoCapture(os.path.join(CLIP, "clip.mp4")); fr0 = cap.read()[1]
tl = T["lm"][0]; LM = {11: (tl[0][0], tl[0][1], 1.0), 12: (tl[1][0], tl[1][1], 1.0), 23: (tl[2][0], tl[2][1], 1.0), 24: (tl[3][0], tl[3][1], 1.0)}
parts = T["parts"][0]
mk, info = BC.find_marks(fr0, LM)
sel, n_all, n_sel = BC.focus_marks(mk, LM, fr0)
check("main patch: of the marks found (the patch's core, the mole, the freckle) the soft patch is chosen, the mole and the freckle not",
      (sel & (parts == 6)).any() and not (sel & (parts == 4)).any() and not (sel & (parts == 7)).any(), "%d patches found, %d chosen" % (n_all, n_sel))
m2 = np.zeros((400, 500), bool); cv2.circle(m2.view(np.uint8), (150, 200), 20, 1, -1); cv2.circle(m2.view(np.uint8), (205, 200), 14, 1, -1)
cv2.circle(m2.view(np.uint8), (420, 60), 3, 1, -1)
lm2 = {11: (100, 50, 1), 12: (300, 50, 1), 23: (120, 350, 1), 24: (280, 350, 1)}   # hip width 160 px: a neighbour within 48 px joins
s2, _, _ = BC.focus_marks(m2, lm2)
check("main patch: a heavy neighbour close by joins it, a speck far away does not", bool(s2[200, 205]) and not s2[60, 420])
# 3. the dark patch inside the pentagon (true landmarks; the patch's own outline + 8 px as the pentagon)
pk = parts == 6; ys, xs = np.nonzero(pk)
pent = BC._offset_poly(BC._min_pentagon(np.stack([xs, ys], 1)), 8.0)
m3, i3 = BC.find_marks(fr0, LM, region=pent)
nav = T["navel"][0]; navm = np.zeros(pk.shape, np.uint8); cv2.ellipse(navm, (int(round(nav[0])), int(round(nav[1]))), (6, 8), 0, 0, 360, 1, -1)
cov = float((m3 & pk).sum()) / pk.sum()
check("dark patch: inside the pentagon 95 %%+ of the soft patch is taken (the marks finder alone: %.0f %%), the navel not"
      % (100 * float((mk & pk).sum()) / pk.sum()), cov >= 0.95 and not (m3 & (navm > 0)).any(), "%.0f %%, %d px, thr %s" % (100 * cov, int(m3.sum()), i3.get("patch_thr")))
# clean skin: a pentagon of the same size over plain skin, above the patch and away from the navel, the mole and the freckle
band = BC.region_mask(BC.region_polygon(LM), fr0.shape)
inner = cv2.erode(band.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
busy = cv2.dilate(((parts > 0) | (navm > 0)).astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
pc, clean_here, pcm = None, False, None
for dy in np.arange(-300, 160, 6.0):                          # the first spot of the same size on plain skin, well inside the band
    for dx in np.arange(-200, 201, 6.0):
        q = (pent - pent.mean(axis=0)) * 0.65 + pent.mean(axis=0) + np.array([dx, dy]); qm = np.zeros(pk.shape, np.uint8); cv2.fillPoly(qm, [np.round(q).astype(np.int32)], 1); qm = qm > 0
        if (qm & ~inner).any() or (qm & busy).any():
            continue
        pc, clean_here, pcm = q, True, qm
        break
    if pc is not None:
        break
if pc is None:
    pc, pcm = pent, np.zeros(pk.shape, bool)
m4, i4 = BC.find_marks(fr0, LM, region=pc)
check("dark patch: a pentagon over plain skin (pores, mottling, the light across the belly) takes nothing", clean_here and int(m4.sum()) <= 15,
      "%d px (pentagon %d px, on plain skin: %s)" % (int(m4.sum()), int(pcm.sum()), clean_here))
# ...and slid over the whole belly of the clip's CLEAN twin (no mark at all, only the navel): the most it ever takes
cl0 = cv2.VideoCapture(os.path.join(CLIP, "clean.mp4")).read()[1]
small = (pent - pent.mean(axis=0)) * 0.65
ys_b, xs_b = np.nonzero(BC.region_mask(BC.region_polygon(LM), fr0.shape))
took = []
for cy in np.linspace(ys_b.min() + 20, ys_b.max() - 20, 6):
    for cx in np.linspace(xs_b.min() + 25, xs_b.max() - 25, 7):
        mm, _ = BC.find_marks(cl0, LM, region=small + np.array([cx, cy]))
        took.append(0 if mm is None else int(mm.sum()))
check("dark patch: slid over 42 places of the clean belly (the navel among them) it takes at most 30 px anywhere",
      max(took) <= 30, "worst %d px, %d places take anything" % (max(took), sum(1 for v in took if v > 0)))
# a pentagon whose line cuts the mole (its middle outside): the mole stays whole - not half filled by the dark patch
mo = parts == 4; my, mx_ = np.nonzero(mo); c_m = np.array([mx_.mean(), my.mean()])
sq = np.array([[c_m[0] + 3, c_m[1] - 30], [c_m[0] + 50, c_m[1] - 30], [c_m[0] + 60, c_m[1]], [c_m[0] + 50, c_m[1] + 30], [c_m[0] + 3, c_m[1] + 30]])
m5, _ = BC.find_marks(fr0, LM, region=sq)
check("dark patch: a mark the pentagon's line cuts (middle outside) is left whole, not half filled", int((m5 & mo).sum()) == 0, "%d px of the mole" % int((m5 & mo).sum()))
# below the floor nothing: the pentagon dragged down over the hip line
pt_ut = BC.to_body(pent, LM); dt = float(pt_ut[:, 1].min()) - 0.04        # moved down until its lowest corner is at t 0.04 (under the floor)
low = BC.from_body(pt_ut - np.array([0.0, dt]), LM)
m6, _ = BC.find_marks(fr0, LM, region=low)
ut = BC.to_body(np.stack(np.nonzero(m6)[::-1], 1).astype(np.float64), LM) if m6.any() else np.zeros((0, 2))
check("dark patch: with the pentagon dragged over the floor (t 0.10) the patch above it is still taken, nothing below it",
      len(ut) > 0 and float(ut[:, 1].min()) >= 0.10 - 1e-6, "%d px, lowest t %.3f" % (len(ut), float(ut[:, 1].min()) if len(ut) else 1.0))
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
