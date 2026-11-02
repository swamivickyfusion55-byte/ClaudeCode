"""v11.2.101 unit checks (simple patch engine: lighting-free tracking, stroke-margin cover; contact hold in the pipeline).
usage: unit99.py PACKDIR CLIPDIR      Synthetic data only."""
import sys, os, types, numpy as np, cv2
PACK, CLIPS = sys.argv[1], sys.argv[2]
sys.path.insert(0, PACK)
g = types.ModuleType("gradio"); g.Request = type("Request", (), {}); g.update = lambda **k: k; g.Error = Exception
sys.modules["gradio"] = g
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: None; sys.modules["huggingface_hub"] = hf
os.environ.setdefault("PHOENIX_MODEL_DIR", "/tmp/claude-0/occmodels")
import core_pipeline as cp
E = cp._E
fails = 0


def check(name, ok, info=""):
    global fails
    fails += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info else ""))


rng = np.random.default_rng(1)
H = W = 200
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

# --- 1. the lighting-free view: a shadow edge cancels, the mark stays ---------------------------------------------
skin = 170 + 0.1 * xx + rng.normal(0, 2, (H, W)).astype(np.float32)
mole = skin.copy(); mole[((yy - 100) ** 2 + (xx - 100) ** 2) <= 12 ** 2] *= 0.5
shade = (1 - 0.16 / (1 + np.exp(-(xx - 100) / 12.0))).astype(np.float32)
d0, d1 = E._pr_detail(mole), E._pr_detail(mole * shade)
inner = np.zeros((H, W), bool); inner[20:180, 20:180] = True
diff_view = float(np.abs(d0 - d1)[inner].mean())
diff_plain = float(np.abs(mole - mole * shade)[inner].mean() / mole[inner].mean() * 100)
check("lighting-free view: a moving shadow edge (16 %) nearly cancels", diff_view < 0.25 * diff_plain,
      f"{diff_plain:.2f} % in brightness -> {diff_view:.2f} in the view")
check("lighting-free view: the mark itself stays (far above the skin's grain)", float(-d0[((yy - 100) ** 2 + (xx - 100) ** 2) <= 8 ** 2].mean()) > 10.0 * float(np.abs(d0[inner & (((yy - 100) ** 2 + (xx - 100) ** 2) > 40 ** 2)]).mean()),
      f"{float(-d0[((yy - 100) ** 2 + (xx - 100) ** 2) <= 8 ** 2].mean()):.0f} % darker than around")

# --- 2. the stroke-margin cover ------------------------------------------------------------------------------------------
H2 = W2 = 160
y2, x2 = np.mgrid[0:H2, 0:W2].astype(np.float32)
skin2 = np.dstack([150 + 0.1 * x2, 170 + 0.05 * y2, 210 + 0 * x2]).astype(np.float32) + rng.normal(0, 1.0, (H2, W2, 3)).astype(np.float32)
hole2 = ((y2 - 80) ** 2 + (x2 - 80) ** 2) <= 34 ** 2
core2 = ((y2 - 80) ** 2 + (x2 - 80) ** 2) <= 9 ** 2
okm = np.ones((H2, W2), bool)
Lm = lambda a: a.mean(axis=2)
dark = skin2.copy(); dark[hole2] -= 2.5                                   # a fill darker than the frame's skin by 2.5 levels
out = E._pr_cover(skin2, dark, hole2, core2, okm)
inn = cv2.erode(hole2.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
KC = float(E.PATCH_CFG["cover"]); over = (KC - 1.0) * 2.5                  # cover 1.5 = brought up to the frame's level and half the deficit over
res_in = float((Lm(out) - Lm(skin2))[inn].mean())
check("cover: a fill darker than the frame's own skin on the stroke is brought up to it (and a little over)",
      -0.4 < res_in < over + 0.5, f"{-2.5:+.1f} -> {res_in:+.2f} levels (cover {KC})")
check("cover: nothing outside the painted area changes", float(np.abs(out - dark)[~hole2].max()) == 0.0)
edge = hole2 & ~cv2.erode(hole2.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
check("cover: fades toward the painted area's edge (a step of under 1 level there)", float(np.abs(out - dark)[edge].max()) < 1.0, f"{float(np.abs(out - dark)[edge].max()):.2f} at the edge")
light = skin2.copy(); light[hole2] += 3.0
check("cover: a fill lighter than the frame's skin is not darkened (brightening only)", float(np.abs(E._pr_cover(skin2, light, hole2, core2, okm) - light).max()) < 1e-3)
same = skin2.copy()
check("cover: a fill that already matches the frame is left as it is", float(np.abs(E._pr_cover(skin2, same, hole2, core2, okm) - same).max()) < 0.4,
      f"max {float(np.abs(E._pr_cover(skin2, same, hole2, core2, okm) - same).max()):.2f}")
mk2 = skin2.copy(); mk2[core2] -= 60.0                                    # the frame: the mark itself under the core
d2 = skin2.copy(); d2[hole2] -= 2.0
o2 = E._pr_cover(mk2, d2, hole2, core2, okm)
check("cover: the mark's own pixels in the frame are not taken for skin (the level over the core follows the margin)",
      -0.4 < float((Lm(o2) - Lm(skin2))[core2].mean()) < (KC - 1.0) * 2.0 + 0.5, f"core {float((Lm(o2) - Lm(skin2))[core2].mean()):+.2f} levels")
gate = okm.copy(); gate[:, 80:] = False                                  # an object in front over half of it
dg = skin2.copy(); dg[hole2] -= 2.5
og = E._pr_cover(skin2, dg, hole2, core2, gate)
check("cover: pixels held back by the object gate give no evidence, and the object side is not touched wrongly",
      float(np.abs(og - dg)[~hole2].max()) == 0.0)
check("cover is on, and the lighting-free tracker is on", float(E.PATCH_CFG["cover"]) > 0 and bool(E.PATCH_CFG["detail_track"]))

# --- 3. tracking under a moving shadow, on a clip (warp clips: the skin also bends) ---------------------------------
def frames(p):
    c = cv2.VideoCapture(p); o = []
    while True:
        ok, fr = c.read()
        if not ok:
            return o
        o.append(fr)

clip = os.path.join(CLIPS, "pr_warp_mole.mp4")
if os.path.exists(clip):
    fr = frames(clip)
    gt = np.load(os.path.join(CLIPS, "pr_warp_mole_gt.npz"))
    grays = [E._pr_gray(x) for x in fr]
    res = {}
    for det in (False, True):
        E.PATCH_CFG["detail_track"] = det
        tr = E.PatchTrack(20, fr[20], gt["refmask"] > 127, make_fill=False)
        tr.track(lambda i: grays[i], list(range(20, len(fr))))
        res[det] = len(tr.A)
    E.PATCH_CFG["detail_track"] = True
    check("tracking a mole under a moving shadow on bending skin: followed on every frame",
          res[True] >= len(fr) - 20, f"plain view {res[False]} of {len(fr) - 20}, lighting-free {res[True]} of {len(fr) - 20}")
else:
    check("warp clip present (make_patchclip.py OUT warp)", False, clip)

# --- 6. contact hold in the pipeline: _geom_for_frame with a flat hold -----------------------------------------------
def rec(x, det=True):
    return {"kps": np.zeros((5, 2), np.float32) + x, "bbox": np.array([x, 0, x + 100, 120], np.float32), "lmk": None,
            "alpha": 1.0, "guard": 0.0, "src": None, "track": None, "hit_bbox": None, "det": det}
tl = [(0, rec(0.0)), (10, rec(0.0))]
a_plain = [cp._geom_for_frame(tl, g_, 60, max_bracket=16, hold_after=6) for g_ in range(11, 50)]
a_flat = [cp._geom_for_frame(tl, g_, 60, max_bracket=16, hold_after=6, flat=30) for g_ in range(11, 50)]
al = lambda r: 0.0 if r is None else r["alpha"]
check("one-sided hold, no contact: fades as before (flat=0 is the old behaviour)", abs(al(a_plain[29]) - (1 - (30 / 60) ** 2)) < 1e-6,
      f"alpha 30 frames after: {al(a_plain[29]):.2f}")
check("one-sided hold in contact: full strength for the flat part", min(al(r) for r in a_flat[:30]) > 0.999,
      f"lowest over 1 s: {min(al(r) for r in a_flat[:30]):.3f}")
tl2 = [(0, rec(0.0)), (10, rec(0.0)), (50, rec(40.0))]
br_plain = [al(cp._geom_for_frame(tl2, g_, 6, max_bracket=16, hold_after=6)) for g_ in range(11, 50)]
br_cont = [al(cp._geom_for_frame(tl2, g_, 60, max_bracket=60, hold_after=6, flat=30)) for g_ in range(11, 50)]
check("a 40-frame gap in a blind span, no contact: the middle is not painted (as before)", min(br_plain) == 0.0)
check("the same gap while touching another face: bridged at full strength", min(br_cont) > 0.999, f"lowest {min(br_cont):.3f}")
# --- 4. the planner: each position is followed through its own stretch; progress; cancel (the 11 % stall) ----------------------
import time as _t
clip = os.path.join(CLIPS, "pr_proc_mole.mp4")
if os.path.exists(clip):
    sys.argv = [sys.argv[0]] + sys.argv[1:]
    import make_patchclip as MP
    cap = cv2.VideoCapture(clip); fr4 = []
    while len(fr4) < 90:
        ok, f_ = cap.read()
        if not ok: break
        fr4.append(f_)
    gt4 = np.load(os.path.join(CLIPS, "pr_proc_mole_gt.npz")); mark4 = gt4["mark"]
    gr4 = [E._pr_gray(x) for x in fr4]; th4 = [E._pr_thumb(x) for x in gr4]
    sp4 = [(fr4[i], MP.finger_mask(mark4[i], seed=3 + i), i, 45, "mA") for i in (15, 45, 75)]
    E.PATCH_CFG["follow_margin"] = 10
    msgs4, prog4 = [], []
    t0 = _t.time()
    trk = E.plan_tracks(sp4, (fr4[0].shape[1], fr4[0].shape[0]), len(fr4), th4, lambda i: gr4[i], lambda i: fr4[i], log=msgs4.append, progress=prog4.append)
    t_full = _t.time() - t0
    tracks4 = trk[0].tracks if trk else []
    bounds = {15: (0, 55), 45: (5, 85), 75: (35, 89)}               # anchor -> (lo, hi): to the neighbours' anchors + the margin
    ok_b = len(tracks4) == 3 and all(min(t.A) >= bounds[t.anchor_idx][0] and max(t.A) <= bounds[t.anchor_idx][1] for t in tracks4)
    check("planner: 3 positions of a mark are each followed only through their own stretch (neighbour anchors + margin)", ok_b,
          ", ".join(f"{t.anchor_idx}: {min(t.A)}..{max(t.A)}" for t in tracks4))
    cov4 = set()
    for t in tracks4: cov4 |= set(t.A)
    vis4 = [i for i in range(len(fr4)) if mark4[i].sum() > 60]
    check("planner: together they still cover every frame where the mark is in the picture", all(i in cov4 for i in vis4),
          f"{sum(1 for i in vis4 if i in cov4)} of {len(vis4)}")
    check("planner: it reports progress while it works (not silent for minutes)", len(prog4) >= 1 and "following mark" in prog4[-1],
          f"{len(prog4)} messages, last: {prog4[-1] if prog4 else None}")
    t0 = _t.time(); calls4 = [0]
    def stop4():
        calls4[0] += 1
        return (_t.time() - t0) > 1.5
    r4 = E.plan_tracks(sp4, (fr4[0].shape[1], fr4[0].shape[0]), len(fr4), th4, lambda i: gr4[i], lambda i: fr4[i], stop=stop4)
    t_stop = _t.time() - t0
    check("planner: cancelling stops it at once and returns nothing", r4 == [] and t_stop < 0.6 * t_full + 3.0, f"stopped after {t_stop:.1f}s of a {t_full:.1f}s plan")
else:
    check("proc_mole clip present", False, clip)
E.PATCH_CFG["follow_margin"] = 45

# --- 5. the engine's own caps match the configured limits (the engine used to keep only 4 marks / 12 positions) ----------------
import config as _cf5
check("engine caps: marks and positions equal the configured limits, so nothing is silently dropped",
      int(E.PATCH_CFG["max_patches"]) == int(_cf5.PATCH_MAX) and int(E.PATCH_CFG["max_views"]) == int(_cf5.PATCH_MAX_VIEWS),
      f"{E.PATCH_CFG['max_patches']}/{E.PATCH_CFG['max_views']} vs {_cf5.PATCH_MAX}/{_cf5.PATCH_MAX_VIEWS}")
lg5 = []
E.plan_tracks([(np.zeros((40, 40, 3), np.uint8), np.zeros((40, 40), np.uint8), 0, 5, f"m{k}") for k in range(int(E.PATCH_CFG["max_patches"]) + 2)],
              (40, 40), 3, [None] * 3, lambda i: np.zeros((40, 40), np.uint8), lambda i: np.zeros((40, 40, 3), np.uint8), log=lg5.append,
              stop=lambda: True)                                  # (the cap is applied, and logged, before anything is followed)
check("engine caps: marks beyond the cap are reported in the log, never dropped silently", any("WARNING" in m for m in lg5), str(lg5[:1]))

print("ALL PASS" if fails == 0 else f"{fails} FAILED")
