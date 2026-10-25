"""usage: mouth_false.py PACK   Does the one-mouth-while-eating rule fire WITHOUT food? Real photos (AFLW2000 samples + mirrors,
insightface sample photo), each as is and with a synthetic open mouth (dark interior + a band of teeth inside the lips).
Prints the share of the crop cut outside the lips (the rule's evidence) and whether the rule would start (> eat_lo)."""
import sys, glob, cv2, numpy as np
sys.path.insert(0, sys.argv[1]); import swap_engine as E
from insightface.app import FaceAnalysis
E.occ_net_load("/tmp/claude-0/occmodels/xseg_2.onnx", threads=2)
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.28)
imgs = []
for f in sorted(glob.glob("/tmp/claude-0/prnet/TestImages/AFLW2000/*.jpg")):
    im = cv2.imread(f); imgs += [(f[-13:-4], im), (f[-13:-4] + "m", im[:, ::-1].copy())]
t1 = cv2.imread("/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg"); imgs.append(("t1", t1))
S = 256; fired = 0; n = 0; worst = 0.0
class T: eat_w = 0.0
for name, im in imgs:
    for face in fa.get(im):
        if face.bbox[2] - face.bbox[0] < 40: continue
        M = E.estimate_norm(face.kps, S)
        for opened in (0, 1):
            img = im.copy()
            if opened:
                L = face.landmark_2d_106[52:72]; c = L.mean(0); w = L[:, 0].max() - L[:, 0].min(); h = L[:, 1].max() - L[:, 1].min()
                cv2.ellipse(img, (int(c[0]), int(c[1])), (int(w * 0.36), int(max(3, h * 0.33))), 0, 0, 360, (25, 20, 35), -1, cv2.LINE_AA)
                cv2.ellipse(img, (int(c[0]), int(c[1] - h * 0.18)), (int(w * 0.30), int(max(2, h * 0.08))), 0, 0, 360, (225, 230, 235), -1, cv2.LINE_AA)
            al = cv2.warpAffine(img, M, (S, S), borderMode=cv2.BORDER_REPLICATE)
            pts = E._landmarks_to_aligned(face.landmark_2d_106, M, S)
            keep = E.occ_net_keep(al, pts, S, M=M, frame_wh=(img.shape[1], img.shape[0]))
            if keep is None: continue
            t = T(); t.eat_w = 0.0
            E.EAT_STATS["frames"] = 0
            m2 = E._eating_mouth(np.ones((S, S), np.float32), keep, pts, S, None)
            lips = pts[52:72]; lm = np.zeros((S, S), np.uint8); cv2.fillConvexPoly(lm, cv2.convexHull(np.round(lips).astype(np.int32)), 1)
            pad = int(round(S * E._P["eat_lip_pad"])); lp = cv2.dilate(lm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1)))
            zr = int(round((lips[:, 0].max() - lips[:, 0].min()) * 0.6)); zone = cv2.dilate(lm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * zr + 1, 2 * zr + 1)))
            frac = float(((keep < 0.5) & (zone > 0) & (lp == 0)).sum()) / (S * S)
            n += 1; f_ = float(m2.min()) < 0.999; fired += f_; worst = max(worst, frac)
            if f_ or frac > 0.0005: print("%-12s %s evidence %.4f %s" % (name, "open" if opened else "    ", frac, "FIRES" if f_ else ""))
print("cases %d, rule fired on %d, largest evidence %.4f (starts at %.3f)" % (n, fired, worst, E._P["eat_lo"]))
