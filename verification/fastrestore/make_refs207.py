"""v11.2.207: Detect-frame pictures for the bench (synthetic clips; insightface's public sample photo; NOT user media).
Writes, next to the clip: refA.jpg / refB.jpg (each person cropped from frame 0, as the app's Detect step does) and
refX.jpg (a different person of the sample photo, the one least like A and B: a picture that matches nobody in the clip).
usage: make_refs207.py CLIP.mp4 SAMPLE_PHOTO.jpg"""
import sys, os, cv2, numpy as np
CLIP, PHOTO = sys.argv[1:3]
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition", "genderage"], providers=["CPUExecutionProvider"])
fa.prepare(ctx_id=-1, det_size=(640, 640))
cap = cv2.VideoCapture(CLIP); ok, fr = cap.read()
gt = np.load(CLIP[:-4] + "_gt.npz")["kps"][0]
out = os.path.dirname(os.path.abspath(CLIP)); base = os.path.basename(CLIP)[:-4]
embs = []
for k, name in ((0, "A"), (1, "B")):
    c = gt[k].mean(0); w = float(np.ptp(gt[k][:, 0])) * 3.2
    x0, y0 = int(max(0, c[0] - w)), int(max(0, c[1] - w * 1.1)); x1, y1 = int(c[0] + w), int(c[1] + w * 1.1)
    crop = fr[y0:y1, x0:x1]
    p = os.path.join(out, "%s_ref%s.jpg" % (base, name)); cv2.imwrite(p, crop)
    f = fa.get(crop); f = max(f, key=lambda z: (z.bbox[2] - z.bbox[0])) if f else None
    embs.append(f.normed_embedding); print(name, p, "gender", f.gender if f is not None else None)
ph = cv2.imread(PHOTO); faces = fa.get(ph)
best = min(faces, key=lambda z: max(float(np.dot(z.normed_embedding, e)) for e in embs))
x0, y0, x1, y1 = [int(v) for v in best.bbox]; pad = int((x1 - x0) * 0.6)
crop = ph[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
p = os.path.join(out, "%s_refX.jpg" % base); cv2.imwrite(p, crop)
print("X", p, "gender", best.gender, "sim to A %.2f B %.2f" % tuple(float(np.dot(best.normed_embedding, e)) for e in embs))
