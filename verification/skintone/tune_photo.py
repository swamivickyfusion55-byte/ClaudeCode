"""SDOS-205 tuning on insightface's public sample photo: two arms whose owner is known by looking (A: person 2's arm hanging
behind person 3; B: person 5's arm passing person 4's head). usage: tune_photo.py PACK key=value ..."""
import sys, os
sys.path.insert(0, sys.argv[1])
import numpy as np, cv2, skin_tone as ST
from insightface.app import FaceAnalysis
for kv in sys.argv[2:]:
    k, v = kv.split("="); setattr(ST, k, float(v))
img = cv2.imread("/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg")
fa = FaceAnalysis(name="buffalo_l", root=os.environ["IFROOT"], allowed_modules=["detection"], providers=["CPUExecutionProvider"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
faces = sorted(fa.get(img), key=lambda f: f.bbox[0]); people = {i: dict(kps=f.kps, bbox=f.bbox) for i, f in enumerate(faces)}
R = {"A": (285, 450, 345, 600, 1, 2), "B": (895, 380, 935, 600, 4, 3)}   # region, true owner (0-based), wrong one
side, sk = max(img.shape[:2]), None
sq = cv2.copyMakeBorder(img, 0, side - img.shape[0], 0, side - img.shape[1], cv2.BORDER_CONSTANT)
P = cv2.resize(ST._skin_prob(cv2.cvtColor(cv2.resize(sq, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))[0], (side, side))[:img.shape[0], :img.shape[1]]
res = {}
for ch in range(6):
    _, ms, _ = ST.preview(img, people, {ch: 0.6}); res[ch] = ms.get(ch, np.zeros(img.shape[:2], np.float32))
line = []
for nm, (x0, y0, x1, y1, good, bad) in R.items():
    sk = P[y0:y1, x0:x1] > 0.5
    g_ = float(res[good][y0:y1, x0:x1][sk].mean()); b_ = float(res[bad][y0:y1, x0:x1][sk].mean())
    line.append("%s: owner %.2f, neighbour %.2f" % (nm, g_, b_))
# spill: each chosen person's mask on the other faces (core) and total
print(" | ".join(line), "|", " ".join(sys.argv[2:]))
