"""usage: mask_equiv.py PACK OUT.npz   Build the paint mask of every face in the AFLW2000 samples (+ mirrors) and the
insightface sample photo with PACK's own build_mask (no track, occlusion net on). Compare two packs' outputs with
mask_equiv.py --cmp A.npz B.npz."""
import sys, glob, cv2, numpy as np
if sys.argv[1] == "--cmp":
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    same = diff = 0
    for k in a.files:
        d = float(np.abs(a[k] - b[k]).max())
        if d == 0.0: same += 1
        else: diff += 1; print(f"  differs: {k} (max {d:.3f})")
    print(f"identical masks: {same}, different: {diff}"); sys.exit()
sys.path.insert(0, sys.argv[1]); import swap_engine as E
from insightface.app import FaceAnalysis
E.occ_net_load("/tmp/claude-0/occmodels/xseg_2.onnx", threads=2)
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.28)
imgs = []
for f in sorted(glob.glob("/tmp/claude-0/prnet/TestImages/AFLW2000/*.jpg")):
    im = cv2.imread(f); imgs += [(f[-13:-4], im), (f[-13:-4] + "m", im[:, ::-1].copy())]
imgs.append(("t1", cv2.imread("/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg")))
comp = E.AlignedCompositor(None); out = {}
for name, im in imgs:
    for i, face in enumerate(fa.get(im)):
        M = E.estimate_norm(face.kps, 128)
        al = cv2.warpAffine(im, M, (128, 128), borderMode=cv2.BORDER_REPLICATE)
        m = comp.build_mask(face, M, 128, track=None, aligned_target=al, frame_wh=(im.shape[1], im.shape[0]))
        out[f"{name}_{i}_pw{E.profile_weight(face.kps) if hasattr(E, 'profile_weight') else 0:.2f}"] = m
np.savez_compressed(sys.argv[2], **out); print(len(out), "masks")
