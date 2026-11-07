"""v11.2.143 PatchFocus: the automatic pentagon goes tight round the main dark patch, and the patch is cleaned as a whole (its faint
edge too) - in the preview and on the job's key frames. Real pose model (MediaPipe 0.10.35), LaMa fill, the adapter's preview route.
usage: preview143.py PACKDIR CLIPDIR   (make_bellytattoo.py ... patch=soft marks=patch: part 6 the soft patch, 4 a mole, 7 a freckle)
Prints the measurements for any pack (run it on v11.2.142 too, for the comparison); the checks are v11.2.143's."""
import sys, os, json, types, tempfile
PACK, CLIP = sys.argv[1], sys.argv[2]
sys.path.insert(0, PACK)
g = types.ModuleType("gradio"); g.Request = type("Request", (), {}); g.update = lambda **k: k; g.Error = Exception
sys.modules.setdefault("gradio", g)
import logging; logging.basicConfig(level=logging.WARNING)
import numpy as np, cv2
import belly_clear as BC, clean_belly as CB, phoenix_api_adapter as A, core_pipeline as cp
TMP = tempfile.mkdtemp(prefix="p143_"); os.environ["GRADIO_TEMP_DIR"] = TMP
fails = 0
def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else ""), flush=True); fails += 0 if ok else 1
T = np.load(os.path.join(CLIP, "truth.npz"))
def frames(p):
    c = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(f)
    return fr
src, cln = frames(os.path.join(CLIP, "clip.mp4")), frames(os.path.join(CLIP, "clean.mp4"))
Lf = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
def poly_mask(P, shape):
    m = np.zeros(shape[:2], np.uint8); cv2.fillPoly(m, [np.round(np.asarray(P, np.float32)).astype(np.int32)], 1); return m > 0
i = 0
fp = os.path.join(TMP, "f143_0.png"); cv2.imwrite(fp, src[i])
o1, o2, js = A.api_clean_belly_preview(fp, json.dumps({})); r = json.loads(js)
pent = np.array(r.get("pentagon") or [], np.float64)
# the measures on the raw cleaned frame (the adapter's file is a JPEG: its compression changes every frame a little)
out, _, _ = CB.preview_frame(src[i], BC.still_pose().detect(src[i]), remove_fn=cp._E.remove_patch_image)
parts = T["parts"][i]; patch = parts == 6; mole = parts == 4; freckle = parts == 7
con = Lf(cln[i]) - Lf(src[i])
left = lambda m: float((Lf(cln[i]) - Lf(out))[m].mean()) / max(1e-6, float(con[m].mean()))
lev = lambda m: float((Lf(cln[i]) - Lf(out))[m].mean())            # how much darker than the clean skin it still is (grey levels)
pm = poly_mask(pent, src[i].shape) if len(pent) == 5 else np.zeros(patch.shape, bool)
hull = cv2.convexHull(np.stack(np.nonzero(patch)[::-1], 1).astype(np.float32))
ratio = float(pm.sum()) / max(1.0, float(cv2.contourArea(hull)))
fringe = patch & (con < 12.0)
nav = T["navel"][i]; navm = np.zeros(patch.shape, np.uint8); cv2.ellipse(navm, (int(round(nav[0])), int(round(nav[1]))), (7, 9), 0, 0, 360, 1, -1); navm = navm > 0
d_nav = float(np.abs(Lf(out) - Lf(src[i]))[navm].mean())
changed = np.abs(out.astype(int) - src[i].astype(int)).max(axis=2) > 0
outside_change = int((changed & ~(cv2.dilate(pm.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)).sum())
print("MEASURE pentagon: source %s, %.0f px (%.1f x the patch's own outline); patch inside %.0f %%; mole inside %d px, freckle %d px; "
      "patch left %.1f %% (its faint edge, under 12 levels: %.1f %%, %.1f levels darker than clean skin, was %.1f); navel changed %.1f levels; mole left %.0f %%, freckle %.0f %%; "
      "changed outside the pentagon %d px; patch_px %s, marks_outside %s"
      % (r.get("pentagon_source"), pm.sum(), ratio, 100 * (pm & patch).sum() / patch.sum(), (pm & mole).sum(), (pm & freckle).sum(),
         100 * left(patch), 100 * left(fringe), lev(fringe), float(con[fringe].mean()), d_nav, 100 * left(mole), 100 * left(freckle), outside_change, r.get("patch_px"), r.get("marks_outside")), flush=True)
check("auto pentagon round the main dark patch: the whole patch inside, the mole and the freckle outside",
      r.get("pentagon_source") == "auto" and (pm & patch).sum() >= 0.98 * patch.sum() and (pm & mole).sum() == 0 and (pm & freckle).sum() == 0)
check("...and tight: at most 2 x the patch's own outline (was the whole belly's marks, grown about the middle)", ratio <= 2.0, "%.2f x" % ratio)
check("preview: the patch cleaned as a whole - under 15 % of its darkness left; its faint edge under 2.5 levels darker than clean skin",
      left(patch) < 0.15 and lev(fringe) < 2.5, "%.1f %%; edge %.1f levels (was %.1f)" % (100 * left(patch), lev(fringe), float(con[fringe].mean())))
check("preview: the navel, the mole and the freckle untouched; nothing changed away from the pentagon",
      d_nav < 2.0 and left(mole) > 0.9 and left(freckle) > 0.9 and outside_change == 0,
      "navel %.1f levels, mole %.0f %%, freckle %.0f %%, %d px" % (d_nav, 100 * left(mole), 100 * left(freckle), outside_change))
# the job with what the preview returned (the app sends pentagon_body and marks_body back as belly_pentagon / belly_marks)
pos = CB.build_clean_belly(os.path.join(CLIP, "clip.mp4"), pentagon_body=BC.parse_pentagon(r.get("pentagon_body")), marks_body=BC.parse_marks(r.get("marks_body")))
notes, ok = [], len(pos) >= 1
for p in pos:
    k = int(round(p["t"] * 30.0)); m = cv2.imread(p["mask"], 0) > 127; pk = T["parts"][k]
    cov = float((m & (pk == 6)).sum()) / max(1, int((pk == 6).sum())); stray = int((m & ((pk == 4) | (pk == 7))).sum())
    notes.append("frame %d: patch %.0f %%, mole/freckle %d px, mask %d px" % (k, 100 * cov, stray, int(m.sum()))); ok &= cov >= 0.95 and stray == 0
print("MEASURE job: " + "; ".join(notes), flush=True)
check("job with the preview's pentagon: on every key frame the whole patch (95 %+ of it) and no stray mark", ok, "%d positions" % len(pos))
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails), flush=True)
sys.stdout.flush(); os._exit(0)
