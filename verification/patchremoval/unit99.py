"""v11.2.99 unit checks (patch blend: lighting-proof tracking and gate, skin texture; contact hold in the pipeline).
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

# --- 2. the light field: a shadow edge is followed, an object in the ring is left out --------------------------------
sk3 = np.dstack([150 + 0.1 * xx, 170 + 0.05 * yy, 210 + 0 * xx]).astype(np.float32) + rng.normal(0, 2, (H, W, 3)).astype(np.float32)
hole = ((yy - 100) ** 2 + (xx - 100) ** 2) <= 45 ** 2
orig = sk3.copy(); orig[((yy - 100) ** 2 + (xx - 100) ** 2) <= 30 ** 2] *= 0.45
ins = np.ones((H, W), bool)
f = E._pr_light_field((orig * shade[..., None]).astype(np.float32), orig, hole, ins)
err = float(np.abs(f.mean(axis=2) - shade)[hole].max())
check("light field follows a shadow edge across the painted area (within 2.5 %)", err < 0.025, f"max error {100 * err:.2f} %")
obj = orig.copy(); obj[60:140, 140:200] = (60, 60, 60)
f2 = E._pr_light_field(obj, orig, hole, ins)
check("light field ignores a dark object over a third of the ring", float(np.abs(f2 - 1.0)[hole].max()) < 0.03,
      f"max change {float(np.abs(f2 - 1.0)[hole].max()):.3f}")

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

# --- 4. the object gate is not fooled by a shadow over the mark ------------------------------------------------------
img = np.dstack([150 + 0 * xx, 170 + 0 * xx, 210 + 0 * xx]).astype(np.float32) + rng.normal(0, 1.5, (H, W, 3)).astype(np.float32)
mk = ((yy - 100) ** 2 + ((xx - 100) / 2.0) ** 2) <= 14 ** 2
img[mk] = img[mk] * 0.45
paint = cv2.dilate(mk.astype(np.uint8), E._pr_disk(5)).astype(bool)
E.PATCH_INPAINTER = None
t = E.PatchTrack(0, np.clip(img, 0, 255).astype(np.uint8), paint)
t.A = {1: np.eye(2, 3)}
sh = (1 - 0.40 / (1 + np.exp(-(xx - 100) / 12.0))).astype(np.float32)
f_sh = np.clip(img * sh[..., None], 0, 255).astype(np.uint8)
o = t.apply(f_sh, 1)
L = lambda a: a.astype(np.float32).mean(axis=2)
clean = np.clip(np.dstack([150 + 0 * xx, 170 + 0 * xx, 210 + 0 * xx]) * sh[..., None], 0, 255)
left = float((L(clean) - L(o))[mk].mean()) / float((L(clean) - L(f_sh))[mk].mean())
check("a 40 % shadow edge over the mark: the mark is still removed (not taken for an object)", left < 0.12, f"{100 * left:.1f} % of the mark left")
LIFT = float(E.PATCH_CFG["lift"])
E.PATCH_CFG["light_local"] = False                                            # v11.2.98 lighting
o_old = t.apply(f_sh, 1)
E.PATCH_CFG["light_local"] = True
E.PATCH_CFG["lift"] = 0.0
o_nolift = t.apply(f_sh, 1)
E.PATCH_CFG["lift"] = LIFT
left_old = float((L(clean) - L(o_old))[mk].mean()) / float((L(clean) - L(f_sh))[mk].mean())
pin = cv2.erode(paint.astype(np.uint8), E._pr_disk(2)).astype(bool)
e_new = float(np.abs(L(o) - L(clean))[pin].mean()); e_old = float(np.abs(L(o_old) - L(clean))[pin].mean())
check("...and the fill takes the shadow's shape: it matches the clean shadowed skin over the painted area",
      e_new < 2.5 and e_new < 0.7 * e_old, f"mean error {e_new:.1f} levels (v11.2.98 lighting: {e_old:.1f}; {100 * left_old:.1f} % of the mark left)")
chg = int((np.abs(o.astype(np.int16) - f_sh.astype(np.int16)).max(axis=2)[~cv2.dilate(paint.astype(np.uint8), E._pr_disk(2)).astype(bool)] > 0).sum())
check("...and nothing outside the painted area (+2 px halo) changes", chg == 0, f"{chg} px")

e_nl = float(np.abs(L(o_nolift) - L(clean))[pin].mean())
check("the lift is on, and under a shadow gradient it does not over-brighten (judged with the lighting divided out)",
      LIFT > 0 and abs(e_new - e_nl) < 0.3, f"error with lift {e_new:.2f}, without {e_nl:.2f} levels")

# --- 5. skin texture: a smooth fill gets the grain of the skin around it, inside the painted area only ---------------
tex = np.dstack([150 + 0 * xx, 170 + 0 * xx, 210 + 0 * xx]).astype(np.float32) + cv2.GaussianBlur(rng.normal(0, 6, (H, W)).astype(np.float32), (0, 0), 0.8)[..., None]
h2 = ((yy - 100) ** 2 + (xx - 100) ** 2) <= 30 ** 2
smooth = tex.copy(); smooth[h2] = np.array([150, 170, 210], np.float32)
out = E._pr_texture(tex, smooth, h2)
hp = lambda a: (a.mean(axis=2) - cv2.GaussianBlur(a.mean(axis=2), (0, 0), 1.0))
r_in = E._pr_rstd(hp(out)[cv2.erode(h2.astype(np.uint8), E._pr_disk(3)).astype(bool)])
r_out = E._pr_rstd(hp(tex)[~cv2.dilate(h2.astype(np.uint8), E._pr_disk(4)).astype(bool)])
check("texture: a smooth fill gets the skin's grain (fine band within 25 %)", 0.75 < r_in / r_out < 1.25, f"fill/skin {r_in / r_out:.2f} (was {E._pr_rstd(hp(smooth)[h2]) / r_out:.2f})")
check("texture: nothing outside the painted area changes", float(np.abs(out - smooth)[~h2].max()) == 0.0)
same = E._pr_texture(tex, tex.copy(), h2)
check("texture: a fill that already has the skin's grain gets none added", float(np.abs(same - tex).max()) < 1e-3)
lvl = float((out - smooth)[h2].mean())
check("texture: the grain does not shift the fill's level", abs(lvl) < 0.3, f"{lvl:+.2f}")

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
print("ALL PASS" if fails == 0 else f"{fails} FAILED")
