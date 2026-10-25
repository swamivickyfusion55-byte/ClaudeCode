"""usage: [FAST=1] profvideo_metrics.py TAG NAME...   (runs tp_<TAG>_<NAME>; clips of make_profclip.py)
From the videos themselves: every frame, the pixels the swap changed (output vs footage, > 10 grey levels after a light
blur) mapped back onto the still photo through the known camera motion. Paint off the face / on the face as % of the
face area (BiSeNet face parse of the photo); wobble = share of the face's surroundings whose changed/unchanged state
flips from one frame to the next (a steady paste: low); frames where the face lost more than a third of its paint
against the clip's median (a face that drops out or deforms for a moment)."""
import sys, os, glob, cv2, numpy as np, onnxruntime as ort
sys.path.insert(0, "/home/user/ClaudeCode/verification/motionkeys"); import flicker as FL
from insightface.app import FaceAnalysis
B = "/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/"
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection"]); fa.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.28)
bis = ort.InferenceSession("/tmp/claude-0/occmodels/bisenet_resnet_34.onnx", providers=["CPUExecutionProvider"])
FAST = os.environ.get("FAST") == "1"
def DX(g): return 160 * np.sin(g / 4.0) if FAST else 40 * np.sin(g / 11.0)
def parse_face(img, bbox):
    x1, y1, x2, y2 = bbox; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; s = max(x2 - x1, y2 - y1) * 1.7
    A = np.float32([[512 / s, 0, 256 - cx * 512 / s], [0, 512 / s, 256 - cy * 512 / s]])
    x = (cv2.warpAffine(img, A, (512, 512))[:, :, ::-1].astype(np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    o = bis.run(None, {bis.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0].argmax(0)
    return cv2.warpAffine(np.isin(o, [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]).astype(np.uint8), cv2.invertAffineTransform(A), (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
tag = sys.argv[1]
for name in sys.argv[2:]:
    im = cv2.resize(cv2.imread(f"/tmp/claude-0/prnet/TestImages/AFLW2000/{name}.jpg"), (700, 700), interpolation=cv2.INTER_CUBIC)
    face = max(fa.get(im), key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    fm = parse_face(im, face.bbox); fw = max(8, int((face.bbox[2] - face.bbox[0]) * 0.02))
    fmd = cv2.dilate(fm, np.ones((2 * fw + 1, 2 * fw + 1), np.uint8)); area = float(fm.sum())
    near = cv2.dilate(fm, np.ones((61, 61), np.uint8)).astype(bool)
    inp = FL.frames(B + ("profast_" if FAST else "prof_") + name + ".mp4"); out = FL.frames(glob.glob(f"/tmp/claude-0/bench_work/tp_{tag}_{name}/out/*/final.mp4")[0])
    sp, cv, ch = [], [], []
    for g in range(5, min(len(inp), len(out))):
        s = 1.0 + 0.06 * np.sin(g / 19.0)
        A = np.float32([[s, 0, 640 - 350 * s + DX(g)], [0, s, 360 - 350 * s + 12 * np.cos(g / 13.0)]])
        d = cv2.GaussianBlur(np.abs(out[g].astype(np.float32) - inp[g].astype(np.float32)).max(2), (0, 0), 1.5)
        c = cv2.warpAffine((d > 10).astype(np.float32), cv2.invertAffineTransform(A), (700, 700))
        sp.append(float((c * (1 - fmd)).sum()) / area * 100); cv.append(float((c * fm).sum()) / area * 100); ch.append(c)
    wob = [float(np.abs(ch[i] - ch[i - 1])[near].mean()) * 100 for i in range(1, len(ch))]
    med = np.median(cv); drops = [5 + i for i, v in enumerate(cv) if v < med * 0.67]
    print(f"{tag} {name}: paint off the face {np.mean(sp):.1f}%  on the face {np.mean(cv):.1f}%  wobble {np.mean(wob):.2f}%  "
          f"frames where the face lost >1/3 of its paint: {len(drops)} {drops[:12]}")
