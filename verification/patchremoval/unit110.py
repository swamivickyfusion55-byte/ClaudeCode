"""v11.2.110 unit checks: lower-belly clear skin (belly_clear.py + its use in core_pipeline._patch_prepare).
usage: unit110.py PACKDIR      Synthetic skin only; the body's landmarks are given (a stand-in for MediaPipe Pose), the
inpainter is cv2.inpaint (no model). Writes a small test clip to the scratch folder."""
import sys, os, types, math, tempfile, numpy as np, cv2
PACK = sys.argv[1]
sys.path.insert(0, PACK)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
g = types.ModuleType("gradio"); g.Request = type("Request", (), {}); g.update = lambda **k: k; g.Error = Exception
sys.modules["gradio"] = g
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: None; sys.modules["huggingface_hub"] = hf
os.environ.setdefault("PHOENIX_MODEL_DIR", "/tmp/claude-0/occmodels")
import core_pipeline as cp
import belly_clear as BC
import make_patchclip as MP
E = cp._E
fails = 0


def check(name, ok, info=""):
    global fails
    fails += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info else ""))


E.PATCH_INPAINTER = lambda crop, mk: cv2.inpaint(crop, (mk > 0).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
W, H = 960, 720
skin_big = MP.proc_skin()                                     # procedural skin, 1800 x 1200
base = cv2.resize(skin_big[200:200 + 720 * 1, 300:300 + 960], (W, H)).astype(np.float32)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
# the body: shoulders and hips (a front view), image coordinates
LM = {11: (650.0, 130.0, 0.99), 12: (310.0, 130.0, 0.99), 23: (590.0, 610.0, 0.98), 24: (370.0, 610.0, 0.98)}
poly = BC.region_polygon(LM)
band = BC.region_mask(poly, (H, W))


def P(t, side):                                               # a point t of the way from the hip to the shoulder
    h = np.array(LM[23 if side == "l" else 24][:2]); s = np.array(LM[11 if side == "l" else 12][:2])
    return h + t * (s - h)


def draw(with_marks=True, navel_in_band=False):
    img = base.copy()
    nav = 0.5 * (P(0.27, "l") + P(0.27, "r"))                 # the navel: a small dark round hollow
    img[((yy - nav[1]) ** 2 + (xx - nav[0]) ** 2) <= 9 ** 2] *= 0.45
    if navel_in_band:                                         # (a pose that puts the navel just inside the band's top)
        nav2 = 0.5 * (P(0.19, "l") + P(0.19, "r"))
        img[((yy - nav2[1]) ** 2 + (xx - nav2[0]) ** 2) <= 9 ** 2] *= 0.45
    marks = np.zeros((H, W), bool)
    if with_marks:
        c = 0.5 * (P(0.10, "l") + P(0.10, "r"))              # a tattoo: blue-black ink strokes below the navel
        ink = np.zeros((H, W), np.uint8)
        for k in range(5):
            a = k * 0.6
            p1 = (int(c[0] - 70 + 30 * k), int(c[1] - 20 + 10 * math.sin(a)))
            p2 = (int(c[0] - 40 + 30 * k), int(c[1] + 25 - 12 * math.cos(a)))
            cv2.line(ink, p1, p2, 255, 5)
        cv2.circle(ink, (int(c[0] + 95), int(c[1])), 14, 255, 4)
        inkm = ink > 0
        img[inkm] = img[inkm] * 0.35 + np.array([90.0, 45.0, 35.0]) * 0.65
        mole_c = 0.5 * (P(0.05, "l") + P(0.05, "r")) + np.array([-120.0, 0.0])
        molem = ((yy - mole_c[1]) ** 2 + (xx - mole_c[0]) ** 2) <= 7 ** 2
        img[molem] *= 0.55
        marks = inkm | molem
    # clothing: a waistband of dark denim across the band's lower part, reaching its bottom edge
    wb = (yy > P(0.02, "l")[1] - 4) & (xx > 200) & (xx < 760)   # (cloth runs on past the band's sides, as real clothing does)
    img[wb] = np.array([110.0, 70.0, 45.0]) + 4.0 * np.sin(xx[wb] / 3.0)[:, None]
    # a mole OUTSIDE the band (above the navel): must stay
    up = 0.5 * (P(0.55, "l") + P(0.55, "r"))
    upm = ((yy - up[1]) ** 2 + (xx - up[0]) ** 2) <= 7 ** 2
    img[upm] *= 0.55
    return np.clip(img, 0, 255).astype(np.uint8), marks & ~wb, wb, upm


# --- 1. the band ----------------------------------------------------------------------------------------------------------------
check("band: between just below the navel and the hip line, wider than the hip joints", poly is not None
      and abs(float(poly[:, 1].min()) - float(P(BC.CFG["t_top"], "l")[1])) < 2.0 and float(poly[:, 1].max()) > LM[23][1]
      and float(poly[:, 0].max()) > LM[23][0] + 30 and float(poly[:, 0].min()) < LM[24][0] - 30, str(np.round(poly).astype(int).tolist()))
nav_y = float(0.5 * (P(0.27, "l") + P(0.27, "r"))[1])
check("band: the navel's level is above the band", float(poly[:, 1].min()) > nav_y + 5, f"band top {poly[:, 1].min():.0f}, navel {nav_y:.0f}")
lo = dict(LM); lo[23] = (590.0, 610.0, 0.2)
check("band: no band when a hip is not seen (visibility 0.2)", BC.region_polygon(lo) is None)
side = dict(LM); side[23] = (481.0, 610.0, 0.98); side[24] = (479.0, 610.0, 0.98)
check("band: no band for a side view (hips on top of each other)", BC.region_polygon(side) is None)

# --- 2. finding the marks -------------------------------------------------------------------------------------------------------------
img, truth, wb, upm = draw()
mk, info = BC.find_marks(img, LM)
rec = float((mk & truth).sum()) / max(1, truth.sum())
check("marks: the tattoo and the mole below the navel are found (95 % of their pixels)", mk is not None and rec >= 0.95, f"{100 * rec:.1f} %, {info}")
check("marks: the waistband (clothing) is left alone (under 1 % of it)", float((mk & wb).sum()) / max(1, wb.sum()) < 0.01, f"{100 * float((mk & wb).sum()) / max(1, wb.sum()):.2f} %")
check("marks: nothing outside the band (the mole above the navel stays)", int((mk & ~band).sum()) == 0 and int((mk & upm).sum()) == 0)
extra = float((mk & ~cv2.dilate(truth.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)).sum()) / max(1, band.sum())
check("marks: plain skin is not taken for a mark (under 1 % of the band)", extra < 0.01, f"{100 * extra:.2f} % of the band")
img2, truth2, wb2, _ = draw(navel_in_band=True)
mk2, info2 = BC.find_marks(img2, LM)
nav2 = 0.5 * (P(0.19, "l") + P(0.19, "r"))
navm = ((yy - nav2[1]) ** 2 + (xx - nav2[0]) ** 2) <= 9 ** 2
check("marks: a navel that falls just inside the band's top is left alone", mk2 is not None and float((mk2 & navm).sum()) / navm.sum() < 0.1 and info2["navel"], f"{100 * float((mk2 & navm).sum()) / navm.sum():.0f} % covered")
clean, _, _, _ = draw(with_marks=False)
mk0, info0 = BC.find_marks(clean, LM)
check("marks: clean skin gives (almost) nothing", mk0 is None or float(mk0.sum()) < 0.005 * band.sum(), f"{0 if mk0 is None else int(mk0.sum())} px")
check("marks: no landmarks, no marks", BC.find_marks(img, None)[0] is None)

# --- 3. even skin -------------------------------------------------------------------------------------------------------------------
bt = BC.BellyTrack({0: LM}, 10, 30.0)
out = bt.apply(clean, 0)
hp = lambda im, m: float((im.astype(np.float32) - cv2.GaussianBlur(im.astype(np.float32), (0, 0), 1.5)).mean(axis=2)[m].std())
inner = cv2.erode(band.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool) & ~cv2.dilate(wb.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
t0, t1 = hp(clean, inner), hp(out, inner)
check("even skin: the band's fine texture is softened (by 25 % or more at strength 0.5) but not wiped (kept 30 % or more)", t1 <= 0.75 * t0 and t1 >= 0.3 * t0, f"{t0:.2f} -> {t1:.2f}")
far = ~cv2.dilate(band.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
check("even skin: nothing outside the band changes", int(np.abs(out.astype(int) - clean.astype(int))[far].max()) == 0)
wbi = cv2.erode(wb.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
check("even skin: clothing is not smoothed (mean change under 1 level)", float(np.abs(out.astype(np.float32) - clean.astype(np.float32))[wbi].mean()) < 1.0,
      f"{float(np.abs(out.astype(np.float32) - clean.astype(np.float32))[wbi].mean()):.2f}")
check("even skin: level unchanged (mean within 1 level)", abs(float(out.astype(np.float32)[inner].mean() - clean.astype(np.float32)[inner].mean())) < 1.0)
check("even skin: no body found on a frame -> the frame is untouched", BC.BellyTrack({}, 10, 30.0).apply(clean, 0) is clean)

# --- 4. landmarks between samples; key frames -----------------------------------------------------------------------------------------
LM2 = {k: (v[0] + 30.0, v[1] + 10.0, v[2]) for k, v in LM.items()}
bt = BC.BellyTrack({0: LM, 6: LM2}, 60, 30.0)
mid = bt.landmarks_at(3)
check("landmarks: halfway between two samples", mid is not None and abs(mid[23][0] - (LM[23][0] + 15.0)) < 1e-6 and abs(mid[23][1] - (LM[23][1] + 5.0)) < 1e-6)
check("landmarks: held a little past the last sample, then none", bt.landmarks_at(6 + 20) is not None and bt.landmarks_at(6 + 40) is None)
samples = {i: LM for i in range(0, 300, 3)}
keys = BC.choose_keys(samples, 300, 30.0, keys_max=8)
check("key frames: spread over the video, at least 1.5 s apart, at most 8", 1 <= len(keys) <= 8 and all(b - a >= 45 for a, b in zip(keys, keys[1:])) and keys[-1] > 200, str(keys))

# --- 5. through the planning pass of the pipeline, on a moving clip -------------------------------------------------------------------
N = 45
frames, cleanf, Ms = [], [], []
imgm, truthm, wbm, upmm = draw()
clean0, _, _, _ = draw(with_marks=False)
for i in range(N):
    a = 4.0 * math.sin(i / 9.0); dx, dy = 25.0 * math.sin(i / 7.0), 10.0 * math.cos(i / 8.0)
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), a, 1.0 + 0.03 * math.sin(i / 11.0)); M[0, 2] += dx; M[1, 2] += dy
    frames.append(cv2.warpAffine(imgm, M, (W, H), borderMode=cv2.BORDER_REFLECT))
    cleanf.append(cv2.warpAffine(clean0, M, (W, H), borderMode=cv2.BORDER_REFLECT))
    Ms.append(M)
tmpd = tempfile.mkdtemp(prefix="u110_", dir=os.environ.get("TMPDIR", "/tmp/claude-0"))
vp = os.path.join(tmpd, "belly.avi")
vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (W, H))
for f in frames:
    vw.write(f)
vw.release()
dec = []
c_ = cv2.VideoCapture(vp)
while True:
    ok, f = c_.read()
    if not ok:
        break
    dec.append(f)


class StubPose:                                                # the body's true landmarks on each frame (a stand-in for MediaPipe)
    error = None
    def __init__(self):
        self.k = 0
    def detect(self, bgr):
        best, bi = 1e18, 0
        for j, f in enumerate(dec):                             # (which frame is this? the decoded clip, nearest)
            d = float(np.abs(f[::16, ::16].astype(np.int16) - bgr[::16, ::16].astype(np.int16)).mean())
            if d < best:
                best, bi = d, j
        M = Ms[bi]
        return {k: (float(M[0, 0] * v[0] + M[0, 1] * v[1] + M[0, 2]), float(M[1, 0] * v[0] + M[1, 1] * v[1] + M[1, 2]), v[2]) for k, v in LM.items()}
    def close(self):
        pass


BC.MediaPipePose = StubPose
msgs = []
import logging
class _H(logging.Handler):
    def emit(self, r):
        msgs.append(r.getMessage())
logging.getLogger().addHandler(_H()); logging.getLogger().setLevel(logging.INFO)
tracks = cp._patch_prepare(vp, [], 0, N - 1, 1, N, W, H, lambda f: f, cv2.INTER_AREA, 30.0, 30.0, lambda *a: None, lambda: False, belly=True)
check("pipeline: the planning pass returns the belly's mark track(s) and the even-skin pass", len(tracks) >= 2 and isinstance(tracks[-1], BC.BellyTrack), str([type(t).__name__ for t in tracks]))
check("pipeline: the log says where marks were found", any("BELLY-CLEAR key frame" in m for m in msgs) and any("marks found on" in m for m in msgs),
      "; ".join(m for m in msgs if "BELLY" in m)[:300])
left, wbc, outc = [], [], []
Lf = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
for i in range(N):
    f = dec[i]
    for t in tracks:
        f = t.apply(f, i)
    Mi = Ms[i]
    tm = cv2.warpAffine(truthm.astype(np.uint8), Mi, (W, H), flags=cv2.INTER_NEAREST) > 0
    wbi_ = cv2.warpAffine(cv2.erode(wbm.astype(np.uint8), np.ones((7, 7), np.uint8)), Mi, (W, H), flags=cv2.INTER_NEAREST) > 0
    bandi = cv2.warpAffine(band.astype(np.uint8), Mi, (W, H), flags=cv2.INTER_NEAREST) > 0
    con = float((Lf(cleanf[i]) - Lf(dec[i]))[tm].mean())
    rem = float((Lf(cleanf[i]) - Lf(f))[tm].mean())
    left.append(rem / max(1.0, con))
    wbc.append(float((np.abs(f.astype(np.int16) - dec[i].astype(np.int16)).max(axis=2) > 12)[wbi_].mean()))
    farm = ~cv2.dilate(bandi.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    outc.append(int((np.abs(f.astype(np.int16) - dec[i].astype(np.int16)).max(axis=2) > 0)[farm].sum()))
left = np.array(left)
check("pipeline: on the moving clip the tattoo and the mole are removed (under 10 % of their contrast left on average, no frame over 35 %)",
      float(left.mean()) < 0.10 and float(left.max()) < 0.35, f"mean {100 * left.mean():.1f} %, worst {100 * left.max():.1f} %")
check("pipeline: the waistband is not painted over (under 2 % of it changed by more than 12 levels)", max(wbc) < 0.02, f"worst {100 * max(wbc):.2f} %")
check("pipeline: nothing outside the band changes", max(outc) == 0, f"worst {max(outc)} px")
check("pipeline: the even-skin pass ran on every frame", BC.STATS["frames"] >= N - 1, str(BC.STATS["frames"]))

# --- 6. without the pose model the job goes on --------------------------------------------------------------------------------------------
class NoPose(StubPose):
    error = "ModuleNotFoundError: No module named 'mediapipe'"
    def detect(self, bgr):
        return None
BC.MediaPipePose = NoPose
msgs.clear()
tr2 = cp._patch_prepare(vp, [], 0, N - 1, 1, N, W, H, lambda f: f, cv2.INTER_AREA, 30.0, 30.0, lambda *a: None, lambda: False, belly=True)
check("no pose model: nothing is applied, the log says so, the job continues", tr2 == [] and any("could not be loaded" in m for m in msgs), "; ".join(m for m in msgs if "BELLY" in m)[:200])

print("\n%d check(s) failed" % fails if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
