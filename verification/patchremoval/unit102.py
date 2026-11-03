"""v11.2.102 unit checks (Patch Removal: the mark is found on every frame, opaque skin paste, texture copied from the skin beside it).
usage: unit102.py PACKDIR      Synthetic data only. A stand-in inpainter (cv2.inpaint) is used, so no model is needed."""
import sys, os, types, numpy as np, cv2
PACK = sys.argv[1]
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


E.PATCH_INPAINTER = lambda crop, mk: cv2.inpaint(crop, (mk > 0).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
rng = np.random.default_rng(3)
H, W = 260, 340
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)


def skin_img(seed=3, grain=2.5):
    r = np.random.default_rng(seed)
    base = np.dstack([150 + 0.03 * xx, 170 + 0.02 * yy, 205 + 0 * xx]).astype(np.float32)
    tex = cv2.GaussianBlur(r.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.0) * grain * 2.0
    return base + tex[..., None]


def put_mole(img, cx, cy, rad=7, depth=0.55, blur=0.0):
    m = (((yy - cy) ** 2 + (xx - cx) ** 2) <= rad ** 2).astype(np.float32)
    if blur > 0:
        m = cv2.GaussianBlur(m, (0, 0), blur)
    return img * (1 - depth * m[..., None])


def disc(cx, cy, r):
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2)


# --- 0. the pack ------------------------------------------------------------------------------------------------
check("pack version string is a 1.2.102+ engine", int(E.ENGINE_VERSION.split("-")[1].split(".")[2]) >= 102, E.ENGINE_VERSION)
for k in ("find_mark", "rescue", "opaque", "clone_texture"):
    check(f"PATCH_CFG[{k!r}] is on", bool(E.PATCH_CFG.get(k)), str(E.PATCH_CFG.get(k)))
check("limits: 10 marks / 20 positions", E.PATCH_CFG["max_patches"] == 10 and E.PATCH_CFG["max_views"] == 20)

# --- 1. the mark finder -------------------------------------------------------------------------------------------
skin = skin_img()
a_img = put_mole(skin, 160, 130)
hole0 = disc(160, 130, 13)
core0 = disc(160, 130, 6)
pol, con = 1.0, float((skin.mean(axis=2) - a_img.mean(axis=2))[disc(160, 130, 3)].mean())
area_exp = float(core0.sum())


def find_at(img, cx, cy, **kw):
    gray = img.mean(axis=2).astype(np.float32)
    return E._pr_find_mark(gray, disc(cx, cy, 13), disc(cx, cy, 6), kw.get("pol", pol), kw.get("con", con), kw.get("area", area_exp), kw.get("reach", 25))


moved = put_mole(skin, 168, 124)                    # the mark moved by (8, -6); the pose still says (160, 130)
r = find_at(moved, 160, 130)
check("finder: finds the mark 10 px from where the pose puts it", r is not None and np.hypot(r[1][0] - 168, r[1][1] - 124) < 2.0,
      "none" if r is None else f"centre ({r[1][0]:.1f}, {r[1][1]:.1f}), true (168, 124)")
check("finder: its area is about the mark's (0.7-2.2 x the core)", r is not None and 0.7 * area_exp <= r[2] <= 2.2 * area_exp,
      "none" if r is None else f"{r[2] / area_exp:.2f} x")
check("finder: plain skin (no mark) gives nothing", find_at(skin, 160, 130) is None)
shade = (1 - 0.18 / (1 + np.exp(-(xx - 160) / 10.0)))[..., None]
rs = find_at(moved * shade, 160, 130)
check("finder: still finds the mark across a shadow edge", rs is not None and np.hypot(rs[1][0] - 168, rs[1][1] - 124) < 2.5,
      "none" if rs is None else f"centre ({rs[1][0]:.1f}, {rs[1][1]:.1f})")
blk = skin.copy(); blk[disc(160, 130, 26)] *= 0.55          # a dark blob 3.6 x the mark (shadow in a fold, a button): not the mark
check("finder: a blob far larger than the mark is not taken for it", find_at(blk, 160, 130) is None)
lightm = skin.copy(); lightm[disc(166, 130, 7)] *= 1.35       # the other polarity
check("finder: a mark of the wrong polarity is not taken for it", find_at(lightm, 160, 130) is None)
blurm = put_mole(skin, 168, 124, blur=2.0)
rb = find_at(blurm, 160, 130)
check("finder: a blurred mark (sigma 2) is still found", rb is not None and np.hypot(rb[1][0] - 168, rb[1][1] - 124) < 3.0,
      "none" if rb is None else f"centre ({rb[1][0]:.1f}, {rb[1][1]:.1f})")

# --- 2. the track on a weak pose, and on a strong one -----------------------------------------------------------------
hole_full = disc(160, 130, 13)
tr = E.PatchTrack(0, np.clip(a_img, 0, 255).astype(np.uint8), hole_full)
check("track: knows the mark's polarity and strength", tr.mark_pol == 1.0 and tr.mark_con > 15.0, f"pol {tr.mark_pol:+.0f}, con {tr.mark_con:.0f}")
eye = np.eye(2, 3, dtype=np.float64)
for k in (1, 2, 3, 4):
    tr.A[k] = eye.copy(); tr.conf[k] = 0.9
tr.weak.add(2)
tr.weak.add(4)
u8 = lambda im: np.clip(im, 0, 255).astype(np.uint8)
E.PatchTrack.__init__  # (silence linters)
fr_mark = u8(moved)                                         # mark at (168, 124); pose (160, 130)
out1 = tr.apply(fr_mark, 1)
mk_true = disc(168, 124, 7)
lum = lambda im: im.astype(np.float32).mean(axis=2)
left = float((lum(u8(skin)) - lum(out1))[mk_true].mean() / max(1.0, (lum(u8(skin)) - lum(fr_mark))[mk_true].mean()))
check("strong pose, mark 10 px off the pose: the paint is centred on the mark and covers it (under 12 % left)", abs(left) < 0.12, f"{100 * left:.1f} % left")
out_far = tr.apply(u8(skin), 2)                              # weak pose over plain skin: nothing is painted
check("weak pose, plain skin there (the mark is gone): nothing is painted", int((np.abs(out_far.astype(int) - u8(skin).astype(int)).max(axis=2) > 0).sum()) == 0)
out_w = tr.apply(fr_mark, 4)
left_w = float((lum(u8(skin)) - lum(out_w))[mk_true].mean() / max(1.0, (lum(u8(skin)) - lum(fr_mark))[mk_true].mean()))
check("weak pose, the mark is there: it is found and painted", abs(left_w) < 0.15, f"{100 * left_w:.1f} % left")
blob = u8(skin.copy())
blob[disc(160, 130, 25)] = (blob[disc(160, 130, 25)] * 0.55).astype(np.uint8)
out_blob = tr.apply(blob, 4)
check("weak pose, only a blob 3.5 x the mark's size there: nothing is painted", int((np.abs(out_blob.astype(int) - blob.astype(int)).max(axis=2) > 0).sum()) == 0)
other = skin.copy(); other[disc(166, 130, 7)] = other[disc(166, 130, 7)] * np.array([0.9, 0.55, 0.3], np.float32)   # right size, polarity, wrong colour (blue-ish ink vs brown)
tr_col = E.PatchTrack(0, u8(a_img), hole_full)
out_other = tr_col.apply(u8(other), 1) if 1 in tr_col.A else u8(other)
check("(setup) the colour of the mark is recorded", tr.mark_bgr is not None)

# --- 3. opaque paste: a turned / blurred mark is never taken for an object --------------------------------------------
fr_blur = u8(put_mole(skin, 160, 130, blur=1.6))
o_b = tr.apply(fr_blur, 1)
mk0 = disc(160, 130, 6)
rem = float((lum(u8(skin)) - lum(o_b))[mk0].mean() / max(1.0, (lum(u8(skin)) - lum(fr_blur))[mk0].mean()))
check("opaque: a blurred mark (the anchor's picture no longer lines up) is fully painted over (under 12 % left)", abs(rem) < 0.12, f"{100 * rem:.1f} % left")
obj = u8(skin.copy()); obj[disc(160, 130, 20) & (xx > 150)] = (230, 30, 30)       # a red object over half the mark
o_o = tr.apply(obj, 1)
changed = (np.abs(o_o.astype(int) - obj.astype(int)).max(axis=2) > 12)
check("opaque: an object of another colour in front is not painted over (under 5 % of its pixels changed)",
      float(changed[disc(160, 130, 20) & (xx > 154)].mean()) < 0.05, f"{100 * float(changed[disc(160, 130, 20) & (xx > 154)].mean()):.1f} % changed")
check("opaque: nothing outside the painted area changes", int(changed[~cv2.dilate(hole_full.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)].sum()) == 0)

# --- 4. texture copied from the skin beside the mark -------------------------------------------------------------------
sk = skin_img(seed=5, grain=4.0)
hole_t = disc(160, 130, 14)
smooth = sk.copy(); smooth[hole_t] = cv2.GaussianBlur(sk, (0, 0), 3.0)[hole_t]     # an inpainter's smooth fill
cl = E._pr_texture_clone(smooth, smooth.copy(), hole_t)
hp = lambda im, m: float((im - cv2.GaussianBlur(im, (0, 0), 1.0)).mean(axis=2)[m].std())
ring = cv2.dilate(hole_t.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool) & ~cv2.dilate(hole_t.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
t_skin, t_smooth, t_clone = hp(sk, ring), hp(smooth, hole_t), hp(cl, hole_t)
check("texture: the cloned fill has the skin's own fine texture (0.7-1.4 of the skin's; the smooth fill had less)",
      0.7 * t_skin <= t_clone <= 1.4 * t_skin and t_clone > 1.8 * t_smooth, f"skin {t_skin:.2f}, smooth {t_smooth:.2f}, cloned {t_clone:.2f}")
check("texture: the level of the fill is unchanged (mean within 0.7)", abs(float((cl - smooth)[hole_t].mean())) < 0.7, f"{float((cl - smooth)[hole_t].mean()):+.2f}")
check("texture: nothing outside the painted area changes", float(np.abs(cl - smooth)[~hole_t].max()) == 0.0)

# --- 5. bridges ---------------------------------------------------------------------------------------------------------
tb = E.PatchTrack(0, u8(a_img), hole_full)
tb.A = {10: eye.copy(), 13: eye.copy(), 20: eye.copy(), 24: eye.copy()}; tb.conf = {10: 1.0, 13: 1.0, 20: 0.45, 24: 1.0}
tb.weak = {20}
tb.bridge()
check("bridge: a short gap between two followed frames is a normal pose", 11 in tb.A and 11 not in tb.weak and 12 not in tb.weak)
check("bridge: a gap that ends on a rescued (weak) pose is weak", all(k in tb.A and k in tb.weak for k in (21, 22, 23)), str(sorted(tb.weak)))
check("bridge: poses are held a few frames past the ends and are weak", 9 in tb.A and 9 in tb.weak and 25 in tb.A and 25 in tb.weak)

print("\n%d check(s) failed" % fails if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
