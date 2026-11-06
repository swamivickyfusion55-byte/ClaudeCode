"""v11.2.140 BellyTattoo checks: belly clean over the whole belly (tattoos and marks above, beside and below the navel).
usage: unit140.py PACKDIR CLIP27 CLIP36 [CLIPW] [--detect-only]   (CLIPW: make_bellytattoo.py ... 0.27 waistband)   (clips from make_bellytattoo.py, navel at 0.27 / 0.36;
PHOENIX_MODEL_DIR must hold pose_landmarker_full.task (and the LaMa model for the preview))
--detect-only: section A only (the mark finder on the true landmarks) - used to compare with the v11.2.118 pack."""
import sys, os, json, types
PACK, C27, C36 = sys.argv[1], sys.argv[2], sys.argv[3]
DETECT_ONLY = "--detect-only" in sys.argv
CW = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else None
sys.path.insert(0, PACK)
g = types.ModuleType("gradio"); g.Request = type("Request", (), {}); g.update = lambda **k: k; g.Error = Exception
sys.modules.setdefault("gradio", g)
import logging; logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
import numpy as np, cv2
import belly_clear as BC
import clean_belly as CB
fails = 0
def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else ""), flush=True)
    fails += 0 if ok else 1
def frames(p):
    c = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(f)
    return fr
NAMES = {1: "upper-belly tattoo", 2: "tattoo beside the navel", 3: "lower-belly tattoo", 4: "brown mole"}
def lmd(T, i):
    t = T["lm"][i]; return {11: (t[0][0], t[0][1], 1.0), 12: (t[1][0], t[1][1], 1.0), 23: (t[2][0], t[2][1], 1.0), 24: (t[3][0], t[3][1], 1.0)}
def tmap(T, i, shape):
    """t (0 at the hip joints, 1 at the shoulders) of every pixel, and the torso length in px."""
    t = T["lm"][i]; sh = 0.5 * (t[0] + t[1]); hp = 0.5 * (t[2] + t[3]); up = sh - hp; tl = float(np.linalg.norm(up)); u = up / tl
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return ((xx - hp[0]) * u[0] + (yy - hp[1]) * u[1]) / tl, tl
def floor_px(L, shape):
    """Signed distance (px) above the line 0.10 of the way from the hip joints to the shoulders, parallel to the hip line (the
    code's floor; the bikini line tilts with the pelvis): + above, - below. L: landmark dict."""
    lh, rh, ls, rs = (np.array(L[k][:2], np.float64) for k in (23, 24, 11, 12))
    pl, pr = lh + 0.10 * (ls - lh), rh + 0.10 * (rs - rh)
    v = pr - pl; v = v / np.linalg.norm(v); nrm = np.array([v[1], -v[0]])
    if np.dot(nrm, 0.5 * (ls + rs) - 0.5 * (lh + rh)) < 0: nrm = -nrm
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (xx - pl[0]) * nrm[0] + (yy - pl[1]) * nrm[1]
POSE_ERR = 8.0      # px: the detected pose's hips vs the true ones on this clip (measured 2-6 px) - for checks against the TRUE floor
def navel_disc(T, i, shape, r=None):
    """The navel itself (the generator's 5 x 7 px ellipse, 1 px of its soft edge added), less any tattoo pixel touching it."""
    n = T["navel"][i]; yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    e = ((xx - n[0]) / 6.0) ** 2 + ((yy - n[1]) / 8.0) ** 2 < 1.0
    return e & (T["parts"][i] == 0)
def shorts_mask(img, T, i):
    Y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[..., 0]; tt, _ = tmap(T, i, img.shape)
    return (Y < 70) & (tt < 0.12) & (tt > -0.6) & ~cv2.dilate((T["parts"][i] > 0).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)

# ---- A. the mark finder on the true landmarks -------------------------------------------------------------------------------
for tag, CL in (("navel 0.27", C27), ("navel 0.36", C36)):
    T = np.load(os.path.join(CL, "truth.npz")); src = frames(os.path.join(CL, "clip.mp4")); cln = frames(os.path.join(CL, "clean.mp4"))
    rec3a = []
    rec = {k: [] for k in NAMES}; nav_px, sh_px, sh_all, below, above, fp = [], 0, 0, 0, 0, []
    for i in (0, 22, 45, 67):
        L = lmd(T, i); mk, info = BC.find_marks(src[i], L)
        mk = np.zeros(src[i].shape[:2], bool) if mk is None else mk
        for k in NAMES:
            pk = T["parts"][i] == k
            rec[k].append(float((mk & pk).sum()) / max(1, pk.sum()))
        nav_px.append(int((mk & navel_disc(T, i, mk.shape)).sum()))
        s = shorts_mask(src[i], T, i); sh_px += int((mk & s).sum()); sh_all += int(s.sum())
        tt, tl = tmap(T, i, mk.shape); fl = floor_px(L, mk.shape)
        below += int((mk & (fl < 0)).sum()); above += int((mk & (tt > 0.62 + 6.0 / tl)).sum())
        p3 = (T["parts"][i] == 3) & (fl >= 4 + 4)       # (the lower tattoo's pixels the floor rule may cover: 4 px margin + 4 px mark growth)
        rec3a.append(float((mk & p3).sum()) / max(1, p3.sum()))
        m0, _ = BC.find_marks(cln[i], L); fp.append(0 if m0 is None else int(m0.sum()))
    for k, nm in NAMES.items():
        need = 0.70 if k == 4 else 0.85
        if k == 3:
            check("A %s: the %s found above the floor (85 %% or more), 4 frames" % (tag, nm), min(rec3a) >= need,
                  "worst %.1f %%, mean %.1f %%; the whole tattoo incl. its strip at the floor: worst %.1f %%" % (100 * min(rec3a), 100 * np.mean(rec3a), 100 * min(rec[k])))
            continue
        check("A %s: the %s found (%d %% or more), 4 frames" % (tag, nm, int(need * 100)), min(rec[k]) >= need,
              "worst %.1f %%, mean %.1f %%" % (100 * min(rec[k]), 100 * np.mean(rec[k])))
    check("A %s: the navel left alone" % tag, max(nav_px) == 0, "%s px of the navel marked" % nav_px)
    check("A %s: black shorts not marked (under 3 %%)" % tag, sh_px < 0.03 * max(1, sh_all), "%d of %d px" % (sh_px, sh_all))
    check("A %s: nothing marked below the floor (0.10 above the hip line, parallel to it)" % tag, below == 0, "%d px" % below)
    check("A %s: nothing marked on the chest (above 0.62)" % tag, above == 0, "%d px" % above)
    check("A %s: clean skin (the same frames without marks): under 60 px marked" % tag, max(fp) < 60, "%s px" % fp)
# ---- A2. a dark waistband across the belly above the floor, a tattoo stroke running into it (the thin-stroke rescue) -------
if CW:
    T = np.load(os.path.join(CW, "truth.npz")); src = frames(os.path.join(CW, "clip.mp4")); cln = frames(os.path.join(CW, "clean.mp4"))
    r5, r1, r2, r4, wbm, wba, nav_px, fp = [], [], [], [], 0, 0, [], []
    for i in (0, 22, 45, 67):
        L = lmd(T, i); mk, info = BC.find_marks(src[i], L)
        mk = np.zeros(src[i].shape[:2], bool) if mk is None else mk
        for k, lst in ((5, r5), (1, r1), (2, r2), (4, r4)):
            pk = T["parts"][i] == k; lst.append(float((mk & pk).sum()) / max(1, pk.sum()))
        wb = T["waist"][i] & ~cv2.dilate((T["parts"][i] > 0).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        wbm += int((mk & wb).sum()); wba += int(wb.sum())
        nav_px.append(int((mk & navel_disc(T, i, mk.shape)).sum()))
        m0, _ = BC.find_marks(cln[i], L); fp.append(0 if m0 is None else int(m0.sum()))
    check("A2 waistband: the stroke running into it found (85 % or more), 4 frames", min(r5) >= 0.85, "worst %.1f %%" % (100 * min(r5)))
    check("A2 waistband: the waistband itself not marked (under 3 %)", wbm < 0.03 * max(1, wba), "%d of %d px" % (wbm, wba))
    check("A2 waistband: the other tattoos and the mole still found", min(r1) >= 0.85 and min(r2) >= 0.85 and min(r4) >= 0.70,
          "worst %.0f / %.0f / %.0f %%" % (100 * min(r1), 100 * min(r2), 100 * min(r4)))
    check("A2 waistband: the navel left alone", max(nav_px) == 0, "%s px" % nav_px)
    check("A2 waistband: the same frames without marks: under 60 px marked", max(fp) < 60, "%s px" % fp)
if DETECT_ONLY:
    print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails)); os._exit(0)

# ---- B. the job path, with the real pose model ------------------------------------------------------------------------------
T = np.load(os.path.join(C27, "truth.npz")); src = frames(os.path.join(C27, "clip.mp4")); cln = frames(os.path.join(C27, "clean.mp4"))
pos = CB.build_clean_belly(os.path.join(C27, "clip.mp4"))
check("B job: positions found on the moving clip", len(pos) >= 1, "%d position(s)" % len(pos))
for p in pos:
    i = int(round(p["t"] * 30.0)); m = cv2.imread(p["mask"], cv2.IMREAD_GRAYSCALE) > 127
    r = {k: float((m & (T["parts"][i] == k)).sum()) / max(1, (T["parts"][i] == k).sum()) for k in NAMES}
    Lt = lmd(T, i); flt = floor_px(Lt, m.shape)
    p3 = (T["parts"][i] == 3) & (flt >= 4 + 4 + POSE_ERR)
    r3 = float((m & p3).sum()) / max(1, p3.sum())
    check("B job, frame %d: every tattoo found (85 %% or more; the lower one above the floor), the mole (70 %%)" % i,
          all(r[k] >= (0.70 if k == 4 else 0.85) for k in (1, 2, 4)) and r3 >= 0.85,
          ", ".join("%s %.1f %%" % (NAMES[k].split()[0], 100 * r[k]) for k in (1, 2, 4)) + ", lower %.1f %% above the floor (%.1f %% of all of it)" % (100 * r3, 100 * r[3]))
    check("B job, frame %d: the navel left alone" % i, int((m & navel_disc(T, i, m.shape)).sum()) == 0, "%d px" % int((m & navel_disc(T, i, m.shape)).sum()))
    check("B job, frame %d: nothing marked below the true floor (the pose's error allowed: %d px)" % (i, POSE_ERR),
          int((m & (flt < -POSE_ERR)).sum()) == 0, "%d px" % int((m & (flt < -POSE_ERR)).sum()))

# ---- C. the preview route (the adapter's function, as the app calls it) ----------------------------------------------------
import phoenix_api_adapter as A
import core_pipeline as cp
os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradio_tmp"); os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)
fp = os.path.join(os.environ["GRADIO_TEMP_DIR"], "frame140.png"); cv2.imwrite(fp, src[0])
o1, o2, js = A.api_clean_belly_preview(fp, json.dumps({"belly_length": 1.0, "belly_width": 1.0}))
r = json.loads(js)
check("C preview: ok, body found", r.get("ok") and r.get("body_found"), "marks %s px, %s ms, %s" % (r.get("marks_px"), r.get("ms"), r.get("version")))
img = cv2.imread(fp)
raw_out, _, _ = CB.preview_frame(img, BC.still_pose().detect(img), 1.0, 1.0, remove_fn=cp._E.remove_patch_image)
Lf = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
for k, nm in NAMES.items():
    pk = T["parts"][0] == k
    con = float((Lf(cln[0]) - Lf(src[0]))[pk].mean()); rem = float((Lf(cln[0]) - Lf(raw_out))[pk].mean())
    check("C preview: the %s removed (under 15 %% of its contrast left)" % nm, rem / con < 0.15, "%.1f %% left" % (100 * rem / con))
check("C preview: the navel not changed", float(np.abs(Lf(raw_out) - Lf(img))[navel_disc(T, 0, img.shape)].max()) < 6.0,
      "max %.1f L" % float(np.abs(Lf(raw_out) - Lf(img))[navel_disc(T, 0, img.shape)].max()))
d = np.abs(raw_out.astype(int) - img.astype(int)).max(axis=2)
fd = floor_px(BC.still_pose().detect(img), img.shape); ft = floor_px(lmd(T, 0), img.shape)
check("C preview: nothing changed below the floor of the detected pose (the line the code guards)", int(((d > 0) & (fd < 0)).sum()) == 0,
      "%d px" % int(((d > 0) & (fd < 0)).sum()))
check("C preview: nothing changed below the true floor (the pose's error allowed: %d px)" % POSE_ERR, int(((d > 0) & (ft < -POSE_ERR)).sum()) == 0,
      "%d px; lowest change %.1f px above the true floor" % (int(((d > 0) & (ft < -POSE_ERR)).sum()), float(ft[d > 0].min()) if (d > 0).any() else 0.0))
band0 = BC.region_mask(BC.region_polygon(BC.still_pose().detect(img)), img.shape[:2])
outside = ~cv2.dilate(band0.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
check("C preview: nothing changed outside the band (beyond 4 px)", int(((d > 0) & outside).sum()) == 0, "%d px" % int(((d > 0) & outside).sum()))
check("C preview: the outlined picture is written", o2 is not None and cv2.imread(o2) is not None)
_, _, js3 = A.api_clean_belly_preview(fp, json.dumps({"belly_length": 9, "belly_width": "x"}))
r3 = json.loads(js3)
check("C preview: out-of-range length / junk breadth brought into range", r3.get("length") == 1.5 and r3.get("width") == 1.0, "length %s, width %s" % (r3.get("length"), r3.get("width")))
cv2.imwrite(os.path.join(os.environ["GRADIO_TEMP_DIR"], "preview140.png"), np.hstack([img, raw_out]))
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails), flush=True)
os._exit(0)
