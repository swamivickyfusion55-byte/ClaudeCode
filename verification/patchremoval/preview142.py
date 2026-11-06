"""v11.2.142: the belly preview and job with the pentagon - real pose model (MediaPipe 0.10.35), LaMa fill, the adapter's route.
usage: preview142.py PACKDIR CLIP27   (make_bellytattoo.py clip; PHOENIX_MODEL_DIR holds the pose model and LaMa)"""
import sys, os, json, types
PACK, CLIP = sys.argv[1], sys.argv[2]
sys.path.insert(0, PACK)
g = types.ModuleType("gradio"); g.Request = type("Request", (), {}); g.update = lambda **k: k; g.Error = Exception
sys.modules.setdefault("gradio", g)
import logging; logging.basicConfig(level=logging.WARNING)
import numpy as np, cv2
import belly_clear as BC, clean_belly as CB, phoenix_api_adapter as A, core_pipeline as cp
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
import tempfile; TMP = tempfile.mkdtemp(prefix="p142_"); os.environ["GRADIO_TEMP_DIR"] = TMP
Lf = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
NAMES = {1: "upper tattoo", 2: "tattoo beside the navel", 3: "lower tattoo", 4: "mole"}
def left(i, out, k):
    pk = T["parts"][i] == k; con = float((Lf(cln[i]) - Lf(src[i]))[pk].mean()); return float((Lf(cln[i]) - Lf(out))[pk].mean()) / con
def preview(i, st):
    fp = os.path.join(TMP, "f142_%d.png" % i); cv2.imwrite(fp, src[i])
    o1, o2, js = A.api_clean_belly_preview(fp, json.dumps(st)); r = json.loads(js)
    lm = BC.still_pose().detect(src[i])
    raw, _, info = CB.preview_frame(src[i], lm, 1.0, 1.0, remove_fn=cp._E.remove_patch_image,
                                    pentagon_px=BC.parse_pentagon(st.get("belly_pentagon_px")), pentagon_body=BC.parse_pentagon(st.get("belly_pentagon")))
    return r, raw, lm, o2
# 1. automatic
r, raw, lm0, o2 = preview(0, {})
ok5 = r.get("pentagon") is not None and len(r["pentagon"]) == 5 and r.get("pentagon_body") is not None and len(r["pentagon_body"]) == 5
check("auto: 5-corner pentagon in px and in body coordinates, source auto", ok5 and r.get("pentagon_source") == "auto", str(r.get("pentagon_source")))
lf = {k: left(0, raw, k) for k in NAMES}
check("auto: every mark inside the pentagon removed (under 15 % left)", all(v < 0.15 for v in lf.values()),
      ", ".join("%s %.1f %%" % (NAMES[k], 100 * v) for k, v in lf.items()))
check("auto: the outlined picture (pentagon drawn) is written", o2 is not None and cv2.imread(o2) is not None)
auto_body = r["pentagon_body"]
# 2. the user's pentagon over the upper tattoo only
up = T["parts"][0] == 1; ys, xs = np.nonzero(up)
pp = [[float(xs.min() - 10), float(ys.min() - 10)], [float(xs.max() + 10), float(ys.min() - 10)], [float(xs.max() + 16), float((ys.min() + ys.max()) / 2)],
      [float(xs.max() + 10), float(ys.max() + 6)], [float(xs.min() - 10), float(ys.max() + 6)]]
r2, raw2, _, _ = preview(0, {"belly_pentagon_px": pp})
l2 = {k: left(0, raw2, k) for k in NAMES}
check("user pentagon over the upper tattoo: it is removed, the others stay (beside / lower tattoo over 85 % left)",
      l2[1] < 0.15 and l2[2] > 0.85 and l2[3] > 0.85, ", ".join("%s %.1f %%" % (NAMES[k], 100 * v) for k, v in l2.items()))
check("user pentagon: source app, returned as given (inside the band)", r2.get("pentagon_source") == "app" and
      float(np.abs(np.array(r2["pentagon"]) - np.array(pp)).max()) < 0.6, "max move %.2f px" % float(np.abs(np.array(r2["pentagon"]) - np.array(pp)).max()))
user_body = r2["pentagon_body"]
# 3. a pentagon dragged below the floor
low = [[300, 420], [480, 420], [500, 520], [390, 600], [280, 520]]
r3, raw3, _, _ = preview(0, {"belly_pentagon_px": low})
tb = np.array(r3["pentagon_body"])[:, 1]
d = np.abs(raw3.astype(int) - src[0].astype(int)).max(axis=2)
fd = BC.to_body(np.stack(np.nonzero(d > 0)[::-1], 1).astype(np.float64), lm0) if (d > 0).any() else np.zeros((0, 2))
check("a pentagon dragged below the floor: its corners put back on the floor (t >= 0.10)", float(tb.min()) >= 0.10 - 1e-6, str(tb.round(3).tolist()))
check("...and nothing changes below the floor (body t < 0.10 minus 4 px)", int((fd[:, 1] < 0.10 - 4.0 / float(np.linalg.norm(BC.body_affine(lm0)[:, 1]))).sum()) == 0 if len(fd) else True,
      "%d px changed in all" % int((d > 0).sum()))
# 4. the body pentagon on another frame follows the body
r4, raw4, lm45, _ = preview(45, {"belly_pentagon": user_body})
true45 = BC.from_body(np.array(user_body), {11: (*T["lm"][45][0], 1), 12: (*T["lm"][45][1], 1), 23: (*T["lm"][45][2], 1), 24: (*T["lm"][45][3], 1)})
err = float(np.linalg.norm(np.array(r4["pentagon"]) - true45, axis=1).max())
check("the user's pentagon (body coordinates) laid on frame 45 follows the body (within the pose's error, 12 px)", err < 12.0, "worst corner %.1f px" % err)
l4 = {k: left(45, raw4, k) for k in NAMES}
check("...and on frame 45 it removes the upper tattoo again, leaves the others", l4[1] < 0.15 and l4[2] > 0.85 and l4[3] > 0.85,
      ", ".join("%s %.1f %%" % (NAMES[k], 100 * v) for k, v in l4.items()))
# 5. the job with the user's pentagon. The mole sits ON the pentagon's line (its middle on the line's corner, inside on about half
#    the frames by the pose's few px): the preview left it (frame 0), so the job must leave it on every key frame
mb = BC.parse_marks(r2.get("marks_body"))
def mole_flag(rows, lm, i):
    pk = T["parts"][i] == 4; ys_, xs_ = np.nonzero(pk); c = BC.to_body(np.array([[xs_.mean(), ys_.mean()]]), lm)[0]
    d = [np.hypot(r[0] - c[0], r[1] - c[1]) for r in rows]; return rows[int(np.argmin(d))]
mf = mole_flag(mb, lm0, 0) if mb is not None else None
check("preview: each mark returned with its place on the body and whether it was taken (the mole: left)",
      mb is not None and len(mb) >= 3 and mf is not None and mf[2] == 0, "%s marks; %s" % (None if mb is None else len(mb), None if mb is None else mb.round(3).tolist()))
def job(marks):
    pos = CB.build_clean_belly(os.path.join(CLIP, "clip.mp4"), pentagon_body=np.array(user_body), marks_body=marks)
    ok_job, notes, moles = len(pos) >= 1, [], []
    for p in pos:
        i = int(round(p["t"] * 30.0)); m = cv2.imread(p["mask"], 0) > 127
        a = float((m & (T["parts"][i] == 1)).sum()) / max(1, (T["parts"][i] == 1).sum()); o_ = int((m & ((T["parts"][i] == 2) | (T["parts"][i] == 3))).sum())
        mo = float((m & (T["parts"][i] == 4)).sum()) / max(1, (T["parts"][i] == 4).sum()); moles.append(mo)
        notes.append("frame %d: upper %.0f %%, other tattoos %d px, mole %.0f %%" % (i, 100 * a, o_, 100 * mo)); ok_job &= a >= 0.85 and o_ == 0
    return ok_job, notes, moles
ok_job, notes, moles = job(mb)
check("job with the user's pentagon and the preview's marks: the upper tattoo taken on every key frame, the other tattoos on none", ok_job, "; ".join(notes))
check("...the mole on the line: left on every key frame, as the preview left it", len(moles) >= 1 and all(v <= 0.05 for v in moles),
      " ".join("%.0f %%" % (100 * v) for v in moles))
ok2, notes2, moles2 = job(None)
check("job with the pentagon only (no preview marks): the upper tattoo taken, the other tattoos not; the mole on the line gets ONE decision "
      "for every key frame", ok2 and len(moles2) >= 1 and (all(v <= 0.05 for v in moles2) or all(v >= 0.85 for v in moles2)),
      "; ".join(notes2))
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails), flush=True)
os._exit(0)
