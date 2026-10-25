"""usage: python3 profile_test.py PACK OUTDIR [OVR k=v,...]
Extreme side poses on real photographs (AFLW2000 samples from the PRNet repo, public benchmark; no user footage).
Real detector (SCRFD 640, thresh 0.28), real swapper, the pack's own AlignedCompositor. Reference face area: BiSeNet face
parser on the ORIGINAL photo (skin, brows, eyes, glasses, nose, mouth, lips; ears excluded). Reported per photo, from the
paint weight actually used (the aligned mask warped back to the photo):
  spill    = paint weight outside the face (dilated 2 % of face width), as % of the face area  -> swap painted on hair/background
  cover    = paint weight inside the face, as % of the face area                                 -> how much of the face is swapped
  edge     = mean |step| across the paint boundary on the output, grey levels                    -> visible seam"""
import sys, os, glob, cv2, numpy as np, scipy.io as sio, types
PACK, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, PACK)
import onnxruntime as ort, insightface
from insightface.app import FaceAnalysis
import swap_engine as E
for kv in filter(None, (sys.argv[3] if len(sys.argv) > 3 else "").split(",")):
    k, v = kv.split("="); E._P[k] = float(v)
E.occ_net_load("/tmp/claude-0/occmodels/xseg_2.onnx", threads=2)
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"]); fa.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.28)
sw = insightface.model_zoo.get_model("/tmp/claude-0/swapmodel/inswapper_128.onnx", providers=["CPUExecutionProvider"])
SRC = cv2.imread("/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/src_a.jpg")
src = max(fa.get(SRC), key=lambda f: (f.bbox[2] - f.bbox[0]))
bis = ort.InferenceSession("/tmp/claude-0/occmodels/bisenet_resnet_34.onnx", providers=["CPUExecutionProvider"])
FACE_CLS = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]

def parse_face(img, bbox):
    x1, y1, x2, y2 = bbox; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; s = max(x2 - x1, y2 - y1) * 1.7
    A = np.float32([[512 / s, 0, 256 - cx * 512 / s], [0, 512 / s, 256 - cy * 512 / s]])
    crop = cv2.warpAffine(img, A, (512, 512), borderMode=cv2.BORDER_CONSTANT)
    x = (crop[:, :, ::-1].astype(np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    o = bis.run(None, {bis.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0].argmax(0)
    m = np.isin(o, FACE_CLS).astype(np.uint8)
    return cv2.warpAffine(m, cv2.invertAffineTransform(A), (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)

cap = {}
_pb = E.AlignedCompositor.paste_back
def pb(img, fake, mask, M, **kw):
    cap["mask"], cap["M"] = mask.copy(), np.asarray(M, np.float32).copy()
    return _pb(img, fake, mask, M, **kw)
E.AlignedCompositor.paste_back = staticmethod(pb)
comp = E.AlignedCompositor(sw)
rows = []
files = sorted(glob.glob("/tmp/claude-0/prnet/TestImages/AFLW2000/*.jpg"))
for f in files:
    yaw = float(sio.loadmat(f[:-4] + ".mat")["Pose_Para"][0][1] * 180 / np.pi)
    for flip in (0, 1):
        img = cv2.imread(f)
        if flip: img = img[:, ::-1].copy()
        name = os.path.basename(f)[:-4] + ("m" if flip else "")
        faces = fa.get(img)
        if not faces:
            rows.append((name, yaw, None)); continue
        face = max(faces, key=lambda q: (q.bbox[2] - q.bbox[0]) * (q.bbox[3] - q.bbox[1]))
        face.landmark_2d_106 = getattr(face, "landmark_2d_106", None)
        out, ok = comp.run(img.copy(), img, face, src, track=None, occlusion_guard=0.0)
        if not ok:
            rows.append((name, yaw, None)); continue
        H, W = img.shape[:2]
        w = cv2.warpAffine(cap["mask"], cv2.invertAffineTransform(cap["M"]), (W, H), flags=cv2.INTER_LINEAR)
        fm = parse_face(img, face.bbox)
        fw = max(8, int((face.bbox[2] - face.bbox[0]) * 0.02))
        fmd = cv2.dilate(fm, np.ones((2 * fw + 1, 2 * fw + 1), np.uint8))
        area = max(1.0, float(fm.sum()))
        spill = float((w * (1 - fmd)).sum()) / area * 100
        cover = float((w * fm).sum()) / area * 100
        edge_band = (w > 0.25) & (w < 0.75)
        gx = cv2.Sobel(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F, 0, 1)
        gi = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F, 1, 0)
        gj = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F, 0, 1)
        edge = float((np.hypot(gx, gy) - np.hypot(gi, gj))[edge_band].mean()) / 8 if edge_band.any() else 0.0
        rows.append((name, yaw * (-1 if flip else 1), (spill, cover, edge)))
        vis = np.concatenate([img, out, cv2.cvtColor((np.clip(w, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) // 2 + (fm[:, :, None] * np.array([0, 90, 0], np.uint8))], 1)
        cv2.imwrite(f"{OUT}/{name}.jpg", vis)
print("%-12s %5s %7s %7s %7s" % ("photo", "yaw", "spill%", "cover%", "edge"))
for n, y, r in rows:
    print("%-12s %5.0f " % (n, y) + ("%7.1f %7.1f %7.2f" % r if r else "  (no face found)"))
