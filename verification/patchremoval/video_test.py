"""usage: video_test.py ENGINE_DIR CLIPDIR name[,name...] [--nomodel] [--save] [--start=N]
--views=2: also the second painted position (make_patchclip.py cut clips); --start=N: only frames N.. (a later part of a split job: the painted frame is not in it).
PATCH_MODULE=swap_engine with ENGINE_DIR = a pack folder tests the pack's own copy.
Runs Patch Removal (the engine's own planner + per-frame removal) on make_patchclip.py clips and scores it against the
clean twin (synthetic + public photos, no user footage). Per clip, over the frames where the mark is in the picture:
  left       how much of the mark is still visible: mean (clean - output) lightness on the mark, % of the mark's contrast
  worst      the worst frame of that
  missed     frames where the mark is in the picture but was not removed (> 50 % of its contrast left)
  error      mean |output - clean| on the mark (L, 0-255) - texture/lighting mismatch, includes codec noise
  shimmer    frame-to-frame change on the mark beyond the clean clip's own, (L) - fixed content should not boil
  object     frames 40-59 (something passes over the mark): share of the object's pixels changed > 12 (must stay ~0)
  outside    pixels changed > 12 farther than 25 px from the mark, worst frame (must stay ~0; frames without an object)"""
import sys, os, time, numpy as np, cv2
ENGINE, CLIPS, NAMES = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
NOMODEL = "--nomodel" in sys.argv
SAVE = "--save" in sys.argv
START = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--start=")), "0"))
VIEWS = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--views=")), "1"))
sys.path.insert(0, ENGINE)
import importlib
E = importlib.import_module(os.environ.get("PATCH_MODULE", "patchfix"))
if os.environ.get("PATCH_CFG"):                 # v11.2.99: e.g. PATCH_CFG='{"lift": 0}' - try a setting
    import json
    E.PATCH_CFG.update(json.loads(os.environ["PATCH_CFG"]))
OUTDIR = os.environ.get("OUTDIR", "/tmp/claude-0")
if not NOMODEL:
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 2; so.log_severity_level = 3
    sess = ort.InferenceSession(os.environ.get("LAMA", "/tmp/claude-0/occmodels/inpainting_lama_2025jan.onnx"), so,
                                providers=["CPUExecutionProvider"])

    def lama(crop, mk):
        h, w = crop.shape[:2]
        x = cv2.resize(crop[..., ::-1], (512, 512), interpolation=cv2.INTER_AREA if max(h, w) > 512 else cv2.INTER_CUBIC)
        x = x.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        m = (cv2.resize(mk.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)[None, None]
        o = sess.run(None, {"image": x, "mask": m})[0][0].transpose(1, 2, 0)
        o = np.clip(o, 0, 255).astype(np.uint8)[..., ::-1]
        return cv2.resize(o, (w, h), interpolation=cv2.INTER_AREA if max(h, w) < 512 else cv2.INTER_CUBIC)
    E.PATCH_INPAINTER = lama


def frames(p):
    c = cv2.VideoCapture(p); o = []
    while True:
        ok, f = c.read()
        if not ok:
            return o
        o.append(f)


L = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
for name in NAMES:
    inp = frames(os.path.join(CLIPS, f"pr_{name}.mp4"))[START:]
    cln = frames(os.path.join(CLIPS, f"pr_{name}_clean.mp4"))[START:]
    g = np.load(os.path.join(CLIPS, f"pr_{name}_gt.npz"))
    mark, occ = g["mark"][START:], g["occ"][START:]
    n = len(inp)
    for k in E.PATCH_STATS:
        E.PATCH_STATS[k] = 0 if isinstance(E.PATCH_STATS[k], int) else 0.0
    grays = [E._pr_gray(f) for f in inp]
    thumbs = [E._pr_thumb(x) for x in grays]
    t0 = time.time()
    msgs = []
    specs = [(g["ref"], g["refmask"], int(round(float(g["ref_t"]) * 30)) - START, 45, "m1")]
    if VIEWS >= 2 and "ref2" in g:          # v11.2.97: the same mark painted at a second position
        specs.append((g["ref2"], g["refmask2"], int(round(float(g["ref_t2"]) * 30)) - START, 45, "m1"))
    tracks = E.plan_tracks(specs, (inp[0].shape[1], inp[0].shape[0]),
                           n, thumbs, lambda i: grays[i], lambda i: inp[i], log=msgs.append)
    t_plan = time.time() - t0
    t0 = time.time()
    out = []
    for i, f in enumerate(inp):
        for tr in tracks:
            f = tr.apply(f, i)
        out.append(f)
    t_apply = (time.time() - t0) / n
    left, err, shim, objc, outside, missed = [], [], [], [], [], 0
    prev = None
    for i in range(n):
        mk = mark[i]
        reg = cv2.dilate(mk.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        lo, lc, li = L(out[i]), L(cln[i]), L(inp[i])
        if mk.sum() > 30:
            con = float((lc - li)[mk].mean())
            rem = float((lc - lo)[mk].mean())
            frac = rem / con if con > 3 else 0.0
            left.append(frac)
            if frac > 0.5:
                missed += 1
            err.append(float(np.abs(lc - lo)[reg].mean()))
            if prev is not None and prev[2].sum() > 30:
                both = reg & prev[2]
                if both.sum() > 30:
                    shim.append(float(np.abs((lo - prev[0]) - (lc - prev[1]))[both].mean()))
        if occ[i].any():
            core = cv2.erode(occ[i].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            objc.append(float((np.abs(lo - li)[core] > 12).mean()))
        if not occ[i].any():                     # (with the mark covered, its own area cannot be told apart)
            far = ~cv2.dilate(mk.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))).astype(bool) if mk.any() else np.ones(mk.shape, bool)
            outside.append(int((np.abs(lo - li)[far] > 12).sum()))
            if outside[-1] > 50 and os.environ.get("SHOW_OUT"):
                print(f"    frame {i + START}: {outside[-1]} px changed away from the mark")
        prev = (lo, lc, reg)
    # v11.2.98: on the painted frame, nothing outside the painted area (+1 px) may change
    ai = int(round(float(g["ref_t"]) * 30)) - START
    if 0 <= ai < n:
        pa = cv2.dilate((g["refmask"] > 127).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        dpx = np.abs(out[ai].astype(np.int16) - inp[ai].astype(np.int16)).max(axis=2)
        outside_paint = int((dpx[~pa] > 3).sum())
    else:
        outside_paint = -1
    left = np.array(left)
    print(f"{name}: planned in {t_plan:.1f}s, {t_apply*1000:.0f} ms/frame | left {100*left.mean():.1f}% of the mark, worst {100*left.max():.1f}%, "
          f"missed {missed}/{len(left)} frames | error {np.mean(err):.1f} L | shimmer {np.mean(shim):.2f} L | "
          f"object changed {100*np.mean(objc) if objc else 0:.1f}% (worst {100*np.max(objc) if objc else 0:.1f}%) | outside {max(outside)} px"
          f" | changed outside the painted area on the painted frame {outside_paint} px")
    for m_ in msgs:
        print("   ", m_)
    print("    stats:", {k: (round(v) if isinstance(v, float) else v) for k, v in E.PATCH_STATS.items()})
    if SAVE:
        vw = cv2.VideoWriter(f"{OUTDIR}/prout_{name}.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (inp[0].shape[1], inp[0].shape[0]))
        for f in out:
            vw.write(f)
        vw.release()
        np.save(f"{OUTDIR}/prout_{name}_left.npy", left)
