"""v11.2.112 unit checks: belly clean's pose model on MediaPipe 0.10.30+ (the Tasks API; the Space resolved mediapipe 0.10.35, which
has no legacy mp.solutions - SDOS-112).
usage: unit112.py PACKDIR POSE_JPG CLIPDIR CLIPDIR_PIXELATED       (run with MediaPipe 0.10.35 importable)
POSE_JPG = MediaPipe's own pose test picture (storage.googleapis.com/mediapipe-assets/pose.jpg); CLIPDIR = make_bellyreal.py output
with the head kept (head=keep), CLIPDIR_PIXELATED with it pixelated."""
import sys, os, json, types, shutil, tempfile, logging, numpy as np, cv2
PACK, POSE_JPG, CLIP, CLIPX = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, PACK)
logging.basicConfig(level=logging.WARNING)
fails = 0
def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + (("  [" + extra + "]") if extra else ""))
    fails += 0 if ok else 1

MD = tempfile.mkdtemp(prefix="u112models_"); os.environ["PHOENIX_MODEL_DIR"] = MD
import belly_clear as BC
BC._model_dirs = lambda: [MD]          # (only this check's own folder: the machine may hold a copy elsewhere)
import mediapipe as mp
check("MediaPipe here is a version without the legacy mp.solutions (as on the Space)", not hasattr(mp, "solutions"), mp.__version__)

# --- 1. the model file: downloaded once, checked, reused ------------------------------------------------------------------------
p1 = BC.MediaPipePose(still=True)
img = cv2.imread(POSE_JPG)
lm = p1.detect(img)
mf = os.path.join(MD, "pose_landmarker_full.task")
check("Tasks backend loads (the pose model downloaded into the model folder, size + sha256 checked)",
      p1.error is None and p1.backend and "tasks" in p1.backend and os.path.isfile(mf) and BC._file_ok(mf, BC.POSE_MODEL_SHA256, BC.POSE_MODEL_BYTES),
      str(p1.backend or p1.error))
import urllib.request as _ur
_real_open = _ur.urlopen
def _no_net(*a, **k):
    raise OSError("no network in this check")
_ur.urlopen = _no_net
p2 = BC.MediaPipePose(still=True); p2.detect(img)
check("a second load reuses the file (no download)", p2.error is None, str(p2.error))
good = open(mf, "rb").read()
open(mf, "wb").write(b"broken" * 1000)
class _Resp:
    def __init__(self, b): self.b, self.i = b, 0
    def read(self, n=-1):
        c = self.b[self.i:] if n < 0 else self.b[self.i:self.i + n]; self.i += len(c); return c
    def __enter__(self): return self
    def __exit__(self, *a): return False
_ur.urlopen = lambda *a, **k: _Resp(good)
p3 = BC.MediaPipePose(still=True); p3.detect(img)
check("a broken model file is downloaded again", p3.error is None and BC._file_ok(mf, BC.POSE_MODEL_SHA256, BC.POSE_MODEL_BYTES), str(p3.error))
_ur.urlopen = lambda *a, **k: _Resp(b"<html>not the model</html>")
os.remove(mf)
p4 = BC.MediaPipePose(still=True); r4 = p4.detect(img)
check("a bad download is refused (size / sha256): no pose, the reason kept, no file left", r4 is None and p4.error and "sha256" in p4.error
      and not os.path.exists(mf) and not [f for f in os.listdir(MD) if ".part" in f], str(p4.error)[:120])
open(mf, "wb").write(good)
_ur.urlopen = _real_open

# --- 2. the landmarks ---------------------------------------------------------------------------------------------------------------
ref = {11: (546, 320), 12: (449, 324), 23: (522, 472), 24: (464, 466)}
err = max(float(np.hypot(lm[k][0] - ref[k][0], lm[k][1] - ref[k][1])) for k in ref) if lm else 1e9
check("still image: shoulders and hips where they are (within 12 px), visible", lm is not None and err < 12 and min(v[2] for v in lm.values()) > 0.9, "max error %.1f px" % err)
check("still image: the band is found on the full picture", BC.region_polygon(lm) is not None)
check("an empty picture: no body", p1.detect(np.full((480, 640, 3), 128, np.uint8)) is None)
pv = BC.MediaPipePose()
cap = cv2.VideoCapture(os.path.join(CLIP, "clip.mp4")); fr = []
while True:
    ok, f = cap.read()
    if not ok: break
    fr.append(f)
got = [pv.detect(f, t_ms=int(3 * i * 1000 / 30)) for i, f in enumerate(fr[::3])]
again = pv.detect(fr[0], t_ms=0)            # (a timestamp that does not rise must not raise)
check("video mode: the body found on every sampled frame of the moving clip; a repeated timestamp is accepted",
      all(g is not None for g in got) and again is not None, "%d of %d" % (sum(g is not None for g in got), len(got)))
pv.close()
TL = np.load(os.path.join(CLIP, "truth.npz"))["lm"]
def _err(lm, t):                            # (left / right may come swapped - a back view - the band does not care)
    a = np.array([lm[k][:2] for k in (11, 12, 23, 24)]); return float(min(np.abs(a - t).max(), np.abs(a[[1, 0, 3, 2]] - t).max()))
for mode in ("video", "still"):
    pm = BC.MediaPipePose(still=(mode == "still")); E_ = []; nb = 0
    for i in range(0, len(fr), 3):
        l_ = pm.detect(fr[i], t_ms=int(i * 1000 / 30))
        if l_ is not None:
            E_.append(_err(l_, TL[i])); nb += BC.region_polygon(l_) is not None
    pm.close()
    hw = float(np.linalg.norm(TL[0][2] - TL[0][3]))
    check("%s mode, the moving clip: shoulders and hips within 15 %% of the hip width (median), the band on every sampled frame" % mode,
          len(E_) == len(fr[::3]) and np.median(E_) < 0.15 * hw and nb == len(fr[::3]),
          "median %.0f px, worst %.0f px, hip width %.0f px; band on %d of %d" % (np.median(E_), np.max(E_), hw, nb, len(fr[::3])))
capx = cv2.VideoCapture(os.path.join(CLIPX, "clip.mp4")); frx = []
while True:
    ok, f = capx.read()
    if not ok: break
    frx.append(f)
TX = np.load(os.path.join(CLIPX, "truth.npz"))["lm"]
pm = BC.MediaPipePose(); S_ = {}
for i in range(0, len(frx), 3):                # (the job's path: video mode, only usable poses kept, then the jump check)
    l_ = pm.detect(frx[i], t_ms=int(i * 1000 / 30))
    if l_ is not None and BC.region_polygon(l_) is not None:
        S_[i] = l_
pm.close()
S2, nd = BC.clean_samples(S_)
far = lambda S: sum(_err(l_, TX[i]) > 0.5 * float(np.linalg.norm(TX[i][2] - TX[i][3])) for i, l_ in S.items())
check("head pixelated (MediaPipe unsure), the job's path: no band far off the body", far(S2) == 0,
      "band on %d of %d sampled frames (%d dropped as jumps), %d far off" % (len(S2), len(frx[::3]), nd, far(S2)))
ps = BC.MediaPipePose(still=True); nst = sum(1 for i in range(0, len(frx), 3) if (lambda q: q is not None and BC.region_polygon(q) is not None)(ps.detect(frx[i]))); ps.close()
print("   (info) head pixelated, still image: a band on %d of %d frames - the preview shows its outline" % (nst, len(frx[::3])))

# --- 2b. the side order and the jump check ------------------------------------------------------------------------------------
A_ = {11: (500.0, 200.0, 1.0), 12: (300.0, 200.0, 1.0), 23: (460.0, 500.0, 1.0), 24: (340.0, 500.0, 1.0)}
B_ = {11: A_[12], 12: A_[11], 23: A_[24], 24: A_[23]}          # the same body, left and right swapped (taken for a back view)
check("left / right swapped: put back in the same order", BC.canonical(B_) == A_ and BC.canonical(A_) == A_)
bt_ = BC.BellyTrack({0: BC.canonical(A_), 6: BC.canonical(B_)}, 7, 30.0)
pm_ = BC.region_polygon(bt_.landmarks_at(3))
check("a blend between a sample and a swapped one keeps the band's breadth", pm_ is not None and abs(float(np.linalg.norm(pm_[0] - pm_[1])) - float(np.linalg.norm(BC.region_polygon(A_)[0] - BC.region_polygon(A_)[1]))) < 1.0)
seq = {3 * i: {k: (v[0] + 6.0 * i, v[1] + 2.0 * i, 1.0) for k, v in A_.items()} for i in range(20)}
_, d0 = BC.clean_samples(seq)
seq2 = dict(seq); seq2[30] = {k: (v[0] - 260.0, v[1] + 120.0, 1.0) for k, v in seq[30].items()}
k2, d1 = BC.clean_samples(seq2)
check("jump check: a body moving fast and steadily keeps every sample; one sample far off is dropped", d0 == 0 and d1 == 1 and 30 not in k2, "%d / %d dropped" % (d0, d1))
h = img.shape[0]
sq = p1.detect(img[300:660, 220:780])           # the head out of the frame: MediaPipe squeezes the shoulders / hips together
check("head out of the frame: the squeezed pose MediaPipe returns gives no band (rather than a wrong one)",
      sq is None or BC.region_polygon(sq) is None,
      "hip width / torso %.2f" % (np.hypot(sq[23][0] - sq[24][0], sq[23][1] - sq[24][1]) / np.hypot(0.5 * (sq[11][0] + sq[12][0] - sq[23][0] - sq[24][0]), 0.5 * (sq[11][1] + sq[12][1] - sq[23][1] - sq[24][1]))) if sq else "none")
p1.close(); p2.close(); p3.close()

# --- 2c. black shorts at the band's bottom, a tattoo running into them (found by testing on 0.10.35's real pose) -------------
f9 = fr[9]; l9 = BC.MediaPipePose(still=True).detect(f9); T9 = np.load(os.path.join(CLIP, "truth.npz"))["mask"][9]
mk9, in9 = BC.find_marks(f9, l9)
b9 = BC.region_mask(BC.region_polygon(l9), f9.shape)
yv = cv2.cvtColor(f9, cv2.COLOR_BGR2YCrCb)[..., 0]
shorts = b9 & (yv < 70) & ~cv2.dilate(T9.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
inner = b9 & (cv2.erode(b9.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0)
rec9 = float((mk9 & T9 & inner).sum()) / max(1.0, float((T9 & inner).sum()))
check("black shorts at the band's bottom are clothing, not skin with dark marks (under 3 % of them marked)",
      mk9 is not None and float((mk9 & shorts).sum()) < 0.03 * float(shorts.sum()), "%d of %d px marked; clothing %d px" % (int((mk9 & shorts).sum()), int(shorts.sum()), in9["clothing_px"]))
check("the tattoo strokes running into the shorts are marks (85 %% or more of the tattoo inside the band)", rec9 >= 0.85, "%.1f %%" % (100 * rec9))
check("a V of tattoo strokes at the band's top middle is not taken for the navel", not in9["navel"])

# --- 2d. the per-frame clean-up: a mark still showing in the band is filled on that frame ---------------------------------------
TT = np.load(os.path.join(CLIP, "truth.npz")); cln_ = cv2.VideoCapture(os.path.join(CLIP, "clean.mp4"))
cfr = [cln_.read()[1] for _ in range(31)]
i_ = 30; tl_ = TT["lm"][i_]; lmT = {11: (tl_[0][0], tl_[0][1], 1.0), 12: (tl_[1][0], tl_[1][1], 1.0), 23: (tl_[2][0], tl_[2][1], 1.0), 24: (tl_[3][0], tl_[3][1], 1.0)}
btc = BC.BellyTrack({i_: lmT}, 31, 30.0, 1.0, 1.0, 0.0)          # (strength 0: the clean-up alone)
o_ = btc.apply(fr[i_], i_)
bT = BC.region_mask(BC.region_polygon(lmT), fr[i_].shape); mT = TT["mask"][i_] & (cv2.erode(bT.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0)
Lq = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
lq = float((Lq(cfr[i_]) - Lq(o_))[mT].mean()) / max(1e-6, float((Lq(cfr[i_]) - Lq(fr[i_]))[mT].mean()))
chg = np.abs(o_.astype(int) - fr[i_].astype(int)).max(axis=2) > 0
check("clean-up alone (no fill carried from the key frames): the tattoo inside the band filled on that frame (under 25 % left)", lq < 0.25 and BC.STATS["cleaned"] == 1, "%.1f %% left" % (100 * lq))
check("clean-up: nothing changed outside the band", int((chg & ~bT).sum()) == 0)
o2_ = BC.BellyTrack({i_: lmT}, 31, 30.0, 1.0, 1.0, 0.0).apply(cfr[i_], i_)
check("clean-up on clean skin: at most a few small dark spots filled (under 1 % of the band)", int((np.abs(o2_.astype(int) - cfr[i_].astype(int)).max(axis=2) > 0).sum()) < 0.01 * int(bT.sum()),
      "%d px" % int((np.abs(o2_.astype(int) - cfr[i_].astype(int)).max(axis=2) > 0).sum()))

# --- 3. loading failures ---------------------------------------------------------------------------------------------------------------
from mediapipe.tasks.python import vision as _v
_orig = _v.PoseLandmarker.create_from_options
def _nolib(*a, **k):
    raise OSError("libGLESv2.so.2: cannot open shared object file: No such file or directory")
_v.PoseLandmarker.create_from_options = _nolib
p5 = BC.MediaPipePose(); r5 = p5.detect(img, 0)
check("missing system library: no pose, and the error names the packages to add", r5 is None and p5.error and "libgles2" in p5.error and "packages.txt" in p5.error, str(p5.error)[:140])
_v.PoseLandmarker.create_from_options = _orig
real_mp = sys.modules["mediapipe"]
fake = types.ModuleType("mediapipe"); fake.__version__ = "0.10.14-fake"
class _Lm:
    def __init__(self, x, y): self.x, self.y, self.visibility = x, y, 0.99
class _Pose:
    def __init__(self, **k): self.k = k
    def process(self, rgb):
        L = [_Lm(0.5, 0.5)] * 33; L[11], L[12], L[23], L[24] = _Lm(0.6, 0.3), _Lm(0.4, 0.3), _Lm(0.57, 0.7), _Lm(0.43, 0.7)
        return types.SimpleNamespace(pose_landmarks=types.SimpleNamespace(landmark=L))
    def close(self): pass
fake.solutions = types.SimpleNamespace(pose=types.SimpleNamespace(Pose=_Pose))
sys.modules["mediapipe"] = fake
p6 = BC.MediaPipePose(); r6 = p6.detect(np.zeros((100, 200, 3), np.uint8))
check("an older MediaPipe with mp.solutions: the legacy pose is used, as before", p6.backend and "solutions" in p6.backend and r6 and abs(r6[11][0] - 120) < 1e-6, str(p6.backend))
sys.modules["mediapipe"] = None
p7 = BC.MediaPipePose(); r7 = p7.detect(img)
check("no MediaPipe at all: no pose, the reason kept (the job goes on)", r7 is None and p7.error and "Error" in p7.error, str(p7.error)[:100])
sys.modules["mediapipe"] = real_mp

# --- 4. the preview route with the real pose model ------------------------------------------------------------------------------
g = types.ModuleType("gradio"); g.Request = type("Request", (), {}); g.update = lambda **k: k; g.Error = Exception
sys.modules.setdefault("gradio", g)
import phoenix_api_adapter as A
BC._STILL["pose"] = None
tmp = tempfile.mkdtemp(prefix="u112_"); os.environ["GRADIO_TEMP_DIR"] = tmp
fp = os.path.join(tmp, "frame.jpg"); cv2.imwrite(fp, fr[0], [cv2.IMWRITE_JPEG_QUALITY, 96])
truth = np.load(os.path.join(CLIP, "truth.npz"))
o1, o2, js = A.api_belly_preview(fp, json.dumps({"belly_length": 1.0, "belly_width": 1.0}))
r = json.loads(js)
check("preview (real pose model): ok, body found, the pose backend reported", r.get("ok") and r.get("body_found") and "tasks" in str(r.get("pose")),
      "pose %s, marks %s px, %s ms" % (r.get("pose"), r.get("marks_px"), r.get("ms")))
if o1:
    cl = cv2.VideoCapture(os.path.join(CLIP, "clean.mp4")).read()[1]
    after = cv2.imread(o1); tm = truth["mask"][0]
    Lf = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    con = float((Lf(cl) - Lf(fr[0]))[tm].mean()); rem = float((Lf(cl) - Lf(after))[tm].mean())
    band = BC.region_mask(np.array(r["band"], np.float32) if r.get("band") else None, fr[0].shape)
    inner = band & (cv2.erode(band.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0)
    rem_in = float((Lf(cl) - Lf(after))[tm & inner].mean()) / max(1e-6, float((Lf(cl) - Lf(fr[0]))[tm & inner].mean()))
    check("preview: the tattoo 8 px or more inside the band removed (under 15 % of its contrast left)", rem_in < 0.15,
          "%.1f %% left there; %.1f %% over the whole tattoo, whose peaks the band's top edge cuts" % (100 * rem_in, 100 * rem / con))
    o1l, _, jsl = A.api_belly_preview(fp, json.dumps({"belly_length": 1.3}))
    afl = cv2.imread(o1l) if o1l else fr[0]
    reml = float((Lf(cl) - Lf(afl))[tm].mean()) / con
    check("preview, Length 130 % (the whole tattoo inside the band): the tattoo removed (under 15 % left over all of it)", reml < 0.15,
          "%.1f %% left" % (100 * reml))
    f0 = cv2.imread(fp); lm0 = BC.still_pose().detect(f0)
    a0, _, i0 = BC.preview(f0, lm0, 1.0, 1.0, 0.5, A.cp._E.remove_patch_image)
    b0 = BC.region_mask(BC.region_polygon(lm0), f0.shape)
    ch_ = (np.abs(a0.astype(int) - f0.astype(int)).max(axis=2) > 0) & ~b0
    dd_ = cv2.distanceTransform((~b0).astype(np.uint8), cv2.DIST_L2, 5)
    check("preview: nothing changed beyond 4 px outside the band (the fill's soft edge where a mark crosses the outline)",
          (not ch_.any()) or float(dd_[ch_].max()) <= 4.0, "%d px outside, furthest %.1f px" % (int(ch_.sum()), float(dd_[ch_].max()) if ch_.any() else 0.0))
    t1 = __import__("time").time(); A.api_belly_preview(fp, "{}"); ms2 = (__import__("time").time() - t1) * 1000
    print("   (info) a second preview took %.0f ms (pose model and fill loaded)" % ms2)
_, _, jsq = A.api_belly_preview(os.path.join(tmp, "sq.jpg") if cv2.imwrite(os.path.join(tmp, "sq.jpg"), img[300:660, 220:780]) else "", "{}")
rq = json.loads(jsq)
check("preview, head out of the frame: ok, no band, the reason says why", rq.get("ok") and rq.get("body_found") is False and "head" in str(rq.get("reason")), jsq[:200])

# --- 5. the Space's files -----------------------------------------------------------------------------------------------------
pk = open(os.path.join(PACK, "packages.txt")).read().split()
rq_ = open(os.path.join(PACK, "requirements.txt")).read()
check("packages.txt has libgles2 and libegl1 (MediaPipe's native library needs them)", "libgles2" in pk and "libegl1" in pk and "ffmpeg" in pk and "libgl1" in pk, str(pk))
check("requirements pin mediapipe 0.10.35 (the tested version)", "\nmediapipe==0.10.35\n" in rq_)
import config as C
check("version v11.2.112", C.VERSION == "v11.2.112")
shutil.rmtree(MD, ignore_errors=True)
print("\n%d check(s) failed" % fails if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
