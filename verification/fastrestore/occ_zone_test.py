"""usage: occ_zone_test.py PACK   The occlusion net's cut under three zone settings, offline (no swap):
  a = mouth/chin only (v11.2.91-94), b = anywhere inside the outline shrunk 8 %, c = anywhere inside the outline grown 10 %.
Objects: make_objects.py clips at frame 40 (object fully in): share of the object (inside the paint area) cut.
Clean faces: AFLW2000 samples + mirrors + insightface sample photo, as is / blurred / dark + noise: share of the face's
own visible area (BiSeNet skin, brows, eyes, nose, lips; hair excluded) cut - should be ~0."""
import sys, glob, cv2, numpy as np, onnxruntime as ort
sys.path.insert(0, sys.argv[1]); import swap_engine as E
from insightface.app import FaceAnalysis
E.occ_net_load("/tmp/claude-0/occmodels/xseg_2.onnx", threads=2)
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.28)
bis = ort.InferenceSession("/tmp/claude-0/occmodels/bisenet_resnet_34.onnx", providers=["CPUExecutionProvider"])
CFG = {"a": dict(occ_net_zone=0.0, occ_net_hull_erode=0.08), "b": dict(occ_net_zone=1.0, occ_net_hull_erode=0.08), "c": dict(occ_net_zone=1.0, occ_net_hull_erode=-0.10)}
S = 256
def keepmap(img, face, cfg):
    for k, v in cfg.items(): E._P[k] = v
    M = E.estimate_norm(face.kps, S); al = cv2.warpAffine(img, M, (S, S), borderMode=cv2.BORDER_REPLICATE)
    pts = E._landmarks_to_aligned(face.landmark_2d_106, M, S)
    E._OCCNET_RAW["key"] = None
    return E.occ_net_keep(al, pts, S, M=M, frame_wh=(img.shape[1], img.shape[0])), M, al
def facemask(al):
    x = (cv2.resize(al, (512, 512))[:, :, ::-1].astype(np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    o = bis.run(None, {bis.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0].argmax(0)
    return cv2.resize(np.isin(o, [1, 2, 3, 4, 5, 10, 11, 12, 13]).astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST).astype(bool)
B = "/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/"
print("OBJECTS (share of the object cut, inside the paint area):")
for spec in ("hand_eye", "hand_cheek", "cup_forehead", "phone_nose"):
    c = cv2.VideoCapture(B + f"obj_{spec}.mp4"); [c.read() for _ in range(40)]; ok, fr = c.read()
    c0 = cv2.VideoCapture(B + f"obj_{spec}.mp4"); ok, clean = c0.read()
    A = np.load(B + f"obj_{spec}_gt.npz")["a"][40]
    face = min(fa.get(clean), key=lambda f: abs((f.bbox[0] + f.bbox[2]) / 2 - 640))   # geometry from the clean frame (the object can hide landmarks)
    row = []
    for nm, cfg in CFG.items():
        km, M, al = keepmap(fr, face, cfg)
        a = cv2.warpAffine(A, M, (S, S)) > 127
        paint = E.canonical_template(S) > 0.05
        obj = a & paint
        row.append(f"{nm} {100 * float((km[obj] < 0.5).mean()) if obj.any() else float('nan'):5.1f}%")
    print(f"  {spec:13s} " + "  ".join(row))
print("CLEAN FACES (share of the visible face cut by mistake: mean / worst / faces over 1 %):")
imgs = []
for f in sorted(glob.glob("/tmp/claude-0/prnet/TestImages/AFLW2000/*.jpg")):
    im = cv2.imread(f); imgs += [im, im[:, ::-1].copy()]
imgs.append(cv2.imread("/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg"))
res = {k: {"normal": [], "blur": [], "dark": []} for k in CFG}
rng = np.random.default_rng(3)
for im in imgs:
    for face in fa.get(im):
        if face.bbox[2] - face.bbox[0] < 60: continue
        M = E.estimate_norm(face.kps, S); al0 = cv2.warpAffine(im, M, (S, S), borderMode=cv2.BORDER_REPLICATE); fm = facemask(al0)
        if fm.sum() < 2000: continue
        for var in ("normal", "blur", "dark"):
            img = im if var == "normal" else (cv2.GaussianBlur(im, (0, 0), 2.0) if var == "blur" else np.clip(im.astype(np.float32) * 0.35 + rng.normal(0, 4, im.shape), 0, 255).astype(np.uint8))
            for nm, cfg in CFG.items():
                km, _, _ = keepmap(img, face, cfg)
                res[nm][var].append(100 * float((km[fm] < 0.5).mean()))
for nm in CFG:
    print("  " + nm + ": " + "   ".join(f"{v} {np.mean(x):.2f} / {np.max(x):.1f} / {int((np.array(x) > 1).sum())} of {len(x)}" for v, x in res[nm].items()))
