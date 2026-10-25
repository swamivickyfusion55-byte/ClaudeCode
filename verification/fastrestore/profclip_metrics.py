"""usage: profclip_metrics.py TAG NAME...  (runs tp_<TAG>_<NAME> of make_profclip.py clips)
The paint weight of every frame (mask dump of the run) is mapped back onto the still photo through the known camera
motion. Reported over frames 5-59: paint outside the face (BiSeNet face area of the photo, dilated 2 % of face width)
and inside it, as % of the face area; and mask wobble: mean frame-to-frame change of the paint weight on the photo,
inside the face's surroundings, in % (0 = a perfectly steady mask)."""
import sys, cv2, numpy as np, onnxruntime as ort
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection"]); fa.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.28)
bis = ort.InferenceSession("/tmp/claude-0/occmodels/bisenet_resnet_34.onnx", providers=["CPUExecutionProvider"])
def parse_face(img, bbox):
    x1, y1, x2, y2 = bbox; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; s = max(x2 - x1, y2 - y1) * 1.7
    A = np.float32([[512 / s, 0, 256 - cx * 512 / s], [0, 512 / s, 256 - cy * 512 / s]])
    crop = cv2.warpAffine(img, A, (512, 512))
    x = (crop[:, :, ::-1].astype(np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    o = bis.run(None, {bis.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0].argmax(0)
    return cv2.warpAffine(np.isin(o, [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]).astype(np.uint8), cv2.invertAffineTransform(A), (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)
tag = sys.argv[1]
for name in sys.argv[2:]:
    im = cv2.resize(cv2.imread(f"/tmp/claude-0/prnet/TestImages/AFLW2000/{name}.jpg"), (700, 700), interpolation=cv2.INTER_CUBIC)
    face = max(fa.get(im), key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    fm = parse_face(im, face.bbox); fw = max(8, int((face.bbox[2] - face.bbox[0]) * 0.02))
    fmd = cv2.dilate(fm, np.ones((2 * fw + 1, 2 * fw + 1), np.uint8)); area = float(fm.sum())
    near = cv2.dilate(fm, np.ones((61, 61), np.uint8)).astype(bool)
    z = np.load(f"/tmp/claude-0/bench_work/tp_{tag}_{name}/mask_dump.npz")
    # the dump's frame field is not filled by the bench; each slot's masks come one per frame, in order
    per = {}
    cand = {}
    for sl in sorted(set(int(x) for x in z["slot"])):
        idx = [i for i, x in enumerate(z["slot"]) if int(x) == sl]
        cand[sl] = idx
    scores = {}
    for sl, idx in cand.items():
        i = idx[len(idx) // 2]; size = int(z["size"][i]); g = len(idx) // 2
        s_ = 1.0 + 0.06 * np.sin(g / 19.0); A = np.float32([[s_, 0, 640 - 350 * s_ + 40 * np.sin(g / 11.0)], [0, s_, 360 - 350 * s_ + 12 * np.cos(g / 13.0)]])
        w = cv2.warpAffine(cv2.warpAffine(cv2.resize(z["m"][i], (size, size)), cv2.invertAffineTransform(z["M"][i]), (1280, 720)), cv2.invertAffineTransform(A), (700, 700))
        scores[sl] = float((w * fm).sum())
    use = max(scores, key=scores.get)
    for g, i in enumerate(cand[use]):
        m, M, size = z["m"][i], z["M"][i], int(z["size"][i])
        s = 1.0 + 0.06 * np.sin(g / 19.0); dx = 40 * np.sin(g / 11.0); dy = 12 * np.cos(g / 13.0)
        A = np.float32([[s, 0, 640 - 350 * s + dx], [0, s, 360 - 350 * s + dy]])
        w = cv2.warpAffine(cv2.resize(m, (size, size)), cv2.invertAffineTransform(M), (1280, 720))
        wp = cv2.warpAffine(w, cv2.invertAffineTransform(A), (700, 700))
        per[g] = wp
    gs = [g for g in sorted(per) if g >= 5]
    spill = [float((per[g] * (1 - fmd)).sum()) / area * 100 for g in gs]
    cover = [float((per[g] * fm).sum()) / area * 100 for g in gs]
    wob = [float(np.abs(per[gs[i]] - per[gs[i - 1]])[near].mean()) * 100 for i in range(1, len(gs))]
    print(f"{tag} {name}: frames {len(gs)}  paint off the face {np.mean(spill):.1f}%  on the face {np.mean(cover):.1f}%  "
          f"mask wobble mean {np.mean(wob):.2f}% worst {np.max(wob):.2f}%")
