"""probe3.py PACK CLIP NONE_CLIP FIRST LAST STEP [OUT.png]: how close is inswapper_128's output (outside the bar) to the ideal - its output on the bar-free
frame - when its input is: the frame as it is | the pipeline's own erase (the pack's _erase_objects) | the whole true bar inpainted (Telea, ground truth)
| the pipeline's mask filled with a smooth skin colour (normalised convolution over the face pixels around it) | the pipeline's mask grown along the bar
by colour, then inpainted. Error = mean abs difference over the lower face of the 128 crop, bar excluded. Ground-truth bar for the mask of 'gt'.
env ERASE_YAW lifts the pack's yaw gate."""
import sys, os, types, cv2, numpy as np
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception; sys.modules["gradio"] = g_
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: ""; sys.modules["huggingface_hub"] = hf
PACK = sys.argv[1]; sys.path.insert(0, PACK)
import logging; logging.disable(logging.CRITICAL)
import swap_engine as E
MD = "/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/face/models/"
assert E.occ_net_load(MD + "xseg_2.onnx", threads=2)
if os.environ.get("ERASE_YAW"): E._P["obj_erase_yaw"] = float(os.environ["ERASE_YAW"])
import json
if os.environ.get("PJ"): E._P.update(json.loads(os.environ["PJ"]))
FULL = os.environ.get("FULL") == "1"
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
clip, none, a0, a1, st = sys.argv[2:7]; OUT = sys.argv[7] if len(sys.argv) > 7 else None
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106", "recognition"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
sw = model_zoo.get_model(MD + "inswapper_128.onnx", providers=["CPUExecutionProvider"])
src = max(fa.get(cv2.imread("/home/user/ClaudeCode/verification/fastrestore/src_a.jpg")), key=lambda f: f.det_score)
g = np.load(clip[:-4] + "_gt.npz"); A = g["a"]
comp = E.AlignedCompositor(None)
cc, cn = cv2.VideoCapture(clip), cv2.VideoCapture(none)
def frame(c, i): c.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = c.read(); return f
def mk(x, i): return cv2.resize(x[i], (1280, 720), interpolation=cv2.INTER_LINEAR)
NAMES = ("frame", "pipe", "gt", "skin", "grow") if FULL else (("frame", "pipe", "gt", "gtin", "gtout") if os.environ.get("INOUT") == "1" else ("frame", "pipe", "gt"))
errs = {k: [] for k in NAMES}; rows = []
def skinfill(f, m, face):
    hull = np.zeros(f.shape[:2], np.uint8); cv2.fillConvexPoly(hull, cv2.convexHull(face.landmark_2d_106.astype(np.int32)), 1)
    md = cv2.dilate(m, np.ones((9, 9), np.uint8))
    valid = ((hull > 0) & (md == 0)).astype(np.float32)
    k = (0, 0)
    num = cv2.GaussianBlur(f.astype(np.float32) * valid[..., None], k, 14); den = cv2.GaussianBlur(valid, k, 14)[..., None] + 1e-4
    fill = num / den
    out = f.copy(); out[m > 0] = np.clip(fill[m > 0], 0, 255).astype(np.uint8)
    return out
def growfill(f, m, face):
    ys, xs = np.nonzero(m)
    if len(xs) == 0: return f
    hull = np.zeros(f.shape[:2], np.uint8); cv2.fillConvexPoly(hull, cv2.convexHull(face.landmark_2d_106.astype(np.int32)), 1)
    fw = float(face.bbox[2] - face.bbox[0])
    # seeds: the mask's pixels on the hull's edge side (outermost); grow with floodFill (floating range) from the mask's outer pixels, outside the hull only
    big = np.zeros((f.shape[0] + 2, f.shape[1] + 2), np.uint8)
    out = m.copy()
    ring = cv2.dilate(m, np.ones((5, 5), np.uint8)) & ~m
    seeds = np.argwhere((ring > 0) & (hull == 0))
    tol = 9
    f2 = cv2.GaussianBlur(f, (0, 0), 1.2)
    for (y, x) in seeds[:: max(1, len(seeds) // 12)]:
        fm = np.zeros((f.shape[0] + 2, f.shape[1] + 2), np.uint8)
        cv2.floodFill(f2.copy(), fm, (int(x), int(y)), 0, (tol,) * 3, (tol,) * 3, 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
        big |= fm
    gr = big[1:-1, 1:-1] > 0
    R = int(0.7 * fw)
    near = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * R + 1, 2 * R + 1))) > 0
    gr &= near & (hull == 0)
    out = ((out > 0) | gr).astype(np.uint8)
    out = cv2.dilate(out, np.ones((5, 5), np.uint8))
    return cv2.inpaint(f, out * 255, 6, cv2.INPAINT_TELEA), int(out.sum())
for i in range(int(a0), int(a1) + 1, int(st)):
    f, fn = frame(cc, i), frame(cn, i)
    bar = (mk(A, i) > 20).astype(np.uint8)
    bd = cv2.dilate(bar, np.ones((7, 7), np.uint8))
    face = max(fa.get(fn), key=lambda q: q.det_score)
    ideal, M = sw.get(fn, face, src, paste_back=False); ideal = np.asarray(ideal, np.uint8)
    pipe = comp._erase_objects(f, f, face, None, None, True)
    m = (np.abs(pipe.astype(np.int16) - f.astype(np.int16)).max(2) > 0).astype(np.uint8)
    gt = cv2.inpaint(f, bd * 255, 6, cv2.INPAINT_TELEA)
    if os.environ.get("INOUT") == "1":
        hl = np.zeros(f.shape[:2], np.uint8); cv2.fillConvexPoly(hl, cv2.convexHull(face.landmark_2d_106.astype(np.int32)), 1)
        hl = cv2.dilate(hl, np.ones((9, 9), np.uint8))
        gin = cv2.inpaint(f, (bd * hl) * 255, 6, cv2.INPAINT_TELEA)               # the true bar only inside the face outline
        gout = cv2.inpaint(f, (bd * (1 - hl)) * 255, 6, cv2.INPAINT_TELEA)       # the true bar only outside it
    sk = skinfill(f, m, face) if (FULL and m.any()) else f
    gr = growfill(f, m, face) if (FULL and m.any()) else (f, 0)
    barc = cv2.warpAffine(bar * 255, M, (128, 128)); excl = cv2.dilate((barc > 20).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    tiles = []
    for name, img in zip(NAMES, ((f, pipe, gt, gin, gout) if os.environ.get('INOUT') == '1' else (f, pipe, gt, sk, gr[0] if isinstance(gr, tuple) else gr))[:len(NAMES)]):
        fake, _ = sw.get(img, face, src, paste_back=False); fake = np.asarray(fake, np.uint8)
        d = np.abs(fake.astype(np.float32) - ideal.astype(np.float32)).mean(2); reg = np.zeros((128, 128), bool); reg[56:124, 10:118] = True; reg &= ~excl
        errs[name].append(float(d[reg].mean()))
        tiles.append(cv2.resize(fake[56:124, 10:118], None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC))
    tiles.append(cv2.resize(ideal[56:124, 10:118], None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC))
    for im, l in zip(tiles, ("%d frame" % i,) + NAMES[1:] + ("ideal",)): cv2.putText(im, l, (4, 14), 0, 0.45, (0, 255, 255), 1)
    rows.append(np.hstack(tiles))
print("%s frames %s-%s: mean abs error vs ideal outside the bar: " % (clip.split("/")[-1], a0, a1) + " | ".join("%s %.2f" % (k, np.mean(errs[k])) for k in NAMES))
if OUT: cv2.imwrite(OUT, np.vstack(rows[:: max(1, len(rows) // 4)]))
