"""usage: stick_test.py ENGINE_DIR CLIPDIR name[,name...]     (PATCH_MODULE=swap_engine for a pack folder; env PATCH_CFG as video_test.py)
Does the paint STICK to the body?  Per clip, over the frames where the mark is in the picture and the engine painted:
  offset   distance between the centre of what the engine changed (output vs input, > 6 levels, near the mark) and the mark's true
           centre (ground truth), mean and 95th percentile, px
  slide    frame-to-frame change of that offset (the paint moving against the body), mean and 95th percentile, px
  size     the mark's size in px (equivalent diameter), for scale
Synthetic clips only (make_patchclip.py)."""
import sys, os, numpy as np, cv2
ENGINE, CLIPS, NAMES = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
sys.path.insert(0, ENGINE)
import importlib
E = importlib.import_module(os.environ.get("PATCH_MODULE", "patchfix"))
if os.environ.get("PATCH_CFG"):
    import json
    E.PATCH_CFG.update(json.loads(os.environ["PATCH_CFG"]))
import onnxruntime as ort
so = ort.SessionOptions(); so.intra_op_num_threads = 2; so.log_severity_level = 3
sess = ort.InferenceSession(os.environ.get("LAMA", "/tmp/claude-0/occmodels/inpainting_lama_2025jan.onnx"), so, providers=["CPUExecutionProvider"])


def lama(crop, mk):
    h, w = crop.shape[:2]
    x = cv2.resize(crop[..., ::-1], (512, 512), interpolation=cv2.INTER_AREA if max(h, w) > 512 else cv2.INTER_CUBIC)
    x = x.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    m = (cv2.resize(mk.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)[None, None]
    o = sess.run(None, {"image": np.ascontiguousarray(x), "mask": m})[0][0].transpose(1, 2, 0)
    o = np.clip(o, 0, 255).astype(np.uint8)[..., ::-1]
    return cv2.resize(np.ascontiguousarray(o), (w, h), interpolation=cv2.INTER_AREA if max(h, w) < 512 else cv2.INTER_CUBIC)


E.PATCH_INPAINTER = lama


def frames(p):
    c = cv2.VideoCapture(p); o = []
    while True:
        ok, f = c.read()
        if not ok:
            return o
        o.append(f)


for name in NAMES:
    inp = frames(os.path.join(CLIPS, f"pr_{name}.mp4"))
    g = np.load(os.path.join(CLIPS, f"pr_{name}_gt.npz"))
    mark = g["mark"]
    n = len(inp)
    grays = [E._pr_gray(f) for f in inp]
    thumbs = [E._pr_thumb(x) for x in grays]
    specs = [(g["ref"], g["refmask"], int(round(float(g["ref_t"]) * 30)), 45, "m1")]
    tracks = E.plan_tracks(specs, (inp[0].shape[1], inp[0].shape[0]), n, thumbs, lambda i: grays[i], lambda i: inp[i])
    offs, dxy, painted, tot = [], [], 0, 0
    cps, cgs, areas = [], [], []
    prev = None
    sizes = []
    for i in range(n):
        f = inp[i]
        for tr in tracks:
            f = tr.apply(f, i)
        mk = mark[i]
        if mk.sum() <= 30:
            prev = None
            continue
        tot += 1
        ys, xs = np.nonzero(mk)
        cg = np.array([xs.mean(), ys.mean()])
        sizes.append(2.0 * np.sqrt(mk.sum() / np.pi))
        d = (np.abs(f.astype(np.int16) - inp[i].astype(np.int16)).max(axis=2) > 6)
        near = cv2.dilate(mk.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))).astype(bool)
        d &= near
        if d.sum() < 8:
            prev = None
            continue
        painted += 1
        yy, xx = np.nonzero(d)
        cp = np.array([xx.mean(), yy.mean()])
        o = cp - cg
        cps.append(cp); cgs.append(cg); areas.append(float(d.sum()))
        offs.append(float(np.hypot(*o)))
        if prev is not None:
            dxy.append(float(np.hypot(*(o - prev))))
        prev = o
    offs, dxy = np.array(offs), np.array(dxy)
    # wobble: second difference of the painted area's centre, beyond the body's own (mark ground truth); area flicker: how much the
    # size of the painted area jumps from frame to frame (a fixed hole carried by the body changes only smoothly)
    cps, cgs, areas = np.array(cps), np.array(cgs), np.array(areas)
    sd2 = lambda c: np.linalg.norm(c[:-2] - 0.5 * (c[:-2] + c[2:]) * 0 - 0.5 * c[1:-1] * 2 + 0, axis=1) if False else np.linalg.norm(0.5 * (c[:-2] + c[2:]) - c[1:-1], axis=1)
    wob = max(0.0, float(sd2(cps).mean() - sd2(cgs).mean())) if len(cps) > 4 else float("nan")
    flick = float(np.abs(areas[1:-1] - 0.5 * (areas[:-2] + areas[2:])).mean() / max(1.0, areas.mean())) if len(areas) > 4 else float("nan")
    print(f"{name}: painted {painted}/{tot} frames | offset mean {offs.mean():.2f} p95 {np.percentile(offs, 95):.2f} px | "
          f"slide mean {dxy.mean():.2f} p95 {np.percentile(dxy, 95):.2f} px | wobble {wob:.2f} px | area flicker {100 * flick:.1f} % | mark size {np.mean(sizes):.0f} px", flush=True)
