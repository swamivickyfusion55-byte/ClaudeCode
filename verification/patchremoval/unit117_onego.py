"""v11.2.117 one-go belly clean: job path (build_clean_belly) and the preview route, on the synthetic belly clip.
usage: onego_test.py PACKDIR CLIPDIR   (PHOENIX_MODEL_DIR must hold pose_landmarker_full.task)"""
import sys, os, json, types
PACK, CLIP = sys.argv[1], sys.argv[2]
sys.path.insert(0, PACK)
g = types.ModuleType("gradio"); g.Request = type("Request", (), {}); g.update = lambda **k: k; g.Error = Exception
sys.modules.setdefault("gradio", g)
import logging; logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
import numpy as np, cv2
import belly_clear as BC
import clean_belly as CB
fails = 0
def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else ""))
    fails += 0 if ok else 1
T = np.load(os.path.join(CLIP, "truth.npz")); LM = T["lm"]
def frames(p):
    c = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(f)
    return fr
src = frames(os.path.join(CLIP, "clip.mp4")); cln = frames(os.path.join(CLIP, "clean.mp4"))
def lmd(i):
    t = LM[i]; return {11: (t[0][0], t[0][1], 1.0), 12: (t[1][0], t[1][1], 1.0), 23: (t[2][0], t[2][1], 1.0), 24: (t[3][0], t[3][1], 1.0)}

# 1. the job path
pos = CB.build_clean_belly(os.path.join(CLIP, "clip.mp4"))
check("job: positions found on the moving clip", len(pos) >= 1, "%d position(s)" % len(pos))
for p in pos:
    i = int(round(p["t"] * 30.0))
    m = cv2.imread(p["mask"], cv2.IMREAD_GRAYSCALE) > 127
    L = lmd(i); band = BC.region_mask(BC.region_polygon(L), m.shape)
    tat = T["mask"][i]; tin = tat & (cv2.erode(band.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
    rec = float((m & tin).sum()) / max(1, tin.sum())
    sh = 0.5 * (L[11][1] + L[12][1]); hp = 0.5 * (L[23][1] + L[24][1]); torso = float(np.hypot(0.5 * (L[11][0] + L[12][0]) - 0.5 * (L[23][0] + L[24][0]), hp - sh))
    nav = (0.5 * (L[11][0] + L[12][0] + L[23][0] + L[24][0]) / 2, sh + 0.73 * (hp - sh))
    navd = np.hypot(*np.meshgrid(np.arange(m.shape[1]) - nav[0], np.arange(m.shape[0]) - nav[1])) < 0.07 * torso
    # 85 %: the job's band comes from the detected pose, which is off by up to ~10 px (the hip line). The engine's video result
    # (belly clip, 90 frames, tattoo left 1.5 %) is the real check - see belly_engine_eval.
    check("job, frame %d: the tattoo inside the band found (85 %% or more)" % i, rec >= 0.85, "%.1f %% of %d px" % (100 * rec, tin.sum()))
    check("job, frame %d: no mark outside the band" % i, int((m & ~band).sum()) == 0, "%d px outside" % int((m & ~band).sum()))
    check("job, frame %d: the navel left alone" % i, int((m & navd).sum()) < 0.05 * max(1, navd.sum()), "%d px of the navel marked" % int((m & navd).sum()))
    Y = cv2.cvtColor(src[i], cv2.COLOR_BGR2YCrCb)[..., 0]
    shorts = band & (Y < 70) & ~cv2.dilate(tat.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    check("job, frame %d: black shorts not marked (under 3 %%)" % i, int((m & shorts).sum()) < 0.03 * max(1, shorts.sum()), "%d of %d px" % (int((m & shorts).sum()), shorts.sum()))

# 2. the preview route (the adapter's function, as the app calls it)
import phoenix_api_adapter as A
import core_pipeline as cp
os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradio_tmp"); os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)
fp = os.path.join(os.environ["GRADIO_TEMP_DIR"], "frame0.png"); cv2.imwrite(fp, src[0])   # PNG: what the app sends (JPEG blurs the mark colour)
o1, o2, js = A.api_clean_belly_preview(fp, json.dumps({"belly_length": 1.0, "belly_width": 1.0}))
r = json.loads(js)
check("preview: ok, body found, the pose backend reported", r.get("ok") and r.get("body_found") and "tasks" in str(r.get("pose")), "marks %s px, pose %s, %s ms" % (r.get("marks_px"), r.get("pose"), r.get("ms")))
if o1:
    after = cv2.imread(o1); outl = cv2.imread(o2)
    Lf = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    tm = T["mask"][0]; L0 = lmd(0); band0 = BC.region_mask(BC.region_polygon(L0), src[0].shape[:2])
    tin = tm & (cv2.erode(band0.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
    con = float((Lf(cln[0]) - Lf(src[0]))[tin].mean()); rem = float((Lf(cln[0]) - Lf(after))[tin].mean())
    check("preview: the tattoo inside the band removed (under 15 % of its contrast left)", rem / con < 0.15, "%.1f %% left" % (100 * rem / con))
    # nothing outside the band changes (checked before the JPEG: the route's output is a JPEG at quality 93, whose ringing at the
    # sharp edges alone gives ~100 px above 16 levels outside the band)
    raw_out, _, _ = CB.preview_frame(cv2.imread(fp), BC.still_pose().detect(cv2.imread(fp)), 1.0, 1.0, remove_fn=cp._E.remove_patch_image)
    d = np.abs(raw_out.astype(int) - cv2.imread(fp).astype(int)).max(axis=2)
    outside = ~cv2.dilate(band0.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    check("preview: nothing changed outside the band (before the JPEG, beyond 4 px)", int(((d > 0) & outside).sum()) == 0, "%d px" % int(((d > 0) & outside).sum()))
    check("preview: the outlined picture is written", outl is not None and outl.shape == after.shape)
o3, _, js3 = A.api_clean_belly_preview(fp, json.dumps({"belly_length": 9, "belly_width": "x"}))
r3 = json.loads(js3)
check("preview: out-of-range length / junk breadth brought into range", r3.get("length") == 1.5 and r3.get("width") == 1.0, "length %s, width %s" % (r3.get("length"), r3.get("width")))
_, _, js4 = A.api_clean_belly_preview(os.path.join(os.environ["GRADIO_TEMP_DIR"], "missing.jpg"), "{}")
check("preview: an unreadable frame: ok false with a reason", json.loads(js4).get("ok") is False)
blank = os.path.join(os.environ["GRADIO_TEMP_DIR"], "blank.jpg"); cv2.imwrite(blank, np.full((480, 640, 3), 128, np.uint8))
_, _, js5 = A.api_clean_belly_preview(blank, "{}")
r5 = json.loads(js5)
check("preview: no body on the frame: ok, body_found false, reason given", r5.get("ok") and r5.get("body_found") is False and r5.get("reason") == "no body found", js5[:140])
print("\n%d check(s) failed" % fails if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
