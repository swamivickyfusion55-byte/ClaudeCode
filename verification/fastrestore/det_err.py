"""det_err.py CLIP GT OUT.npz  - the detector's own 5 points / box against the truth, per frame and person, with the visible share"""
import sys, cv2, numpy as np
from insightface.app import FaceAnalysis
CLIP, GT, OUT = sys.argv[1:4]
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
c = cv2.VideoCapture(CLIP); fr = []
while True:
    ok, f = c.read()
    if not ok: break
    fr.append(f)
g = np.load(GT); kps, vis = g["kps"], g["a"]; n = min(len(fr), len(kps))
full = [(vis[:, p] > 127).sum(axis=(1, 2)).max() for p in range(2)]
res = {"err": np.full((n, 2), np.nan), "share": np.zeros((n, 2)), "w": np.full((n, 2), np.nan), "dkps": np.full((n, 2, 5, 2), np.nan), "lmk": np.full((n, 2, 106, 2), np.nan), "bbox": np.full((n, 2, 4), np.nan), "score": np.full((n, 2), np.nan)}
for t in range(n):
    faces = fa.get(fr[t]); used = set()
    for p in range(2):
        res["share"][t, p] = float((vis[t, p] > 127).sum()) / max(1.0, float(full[p]))
    cand = sorted(((float(np.linalg.norm(f.kps.mean(0) - kps[t, p].mean(0))), i, p) for i, f in enumerate(faces) for p in range(2)))
    donep = set()
    for d, i, p in cand:
        if i in used or p in donep or d > 150: continue
        used.add(i); donep.add(p); f = faces[i]
        res["err"][t, p] = float(np.linalg.norm(f.kps - kps[t, p], axis=1).mean()); res["w"][t, p] = float(f.bbox[2] - f.bbox[0])
        res["dkps"][t, p] = f.kps; res["lmk"][t, p] = f.landmark_2d_106; res["bbox"][t, p] = f.bbox; res["score"][t, p] = f.det_score
np.savez_compressed(OUT, **res)
for t in range(0, n, 4):
    print("%3d | A share %3.0f%% err %5.1f w %4.0f sc %.2f | B share %3.0f%% err %5.1f w %4.0f sc %.2f" % (t, 100 * res["share"][t, 0], res["err"][t, 0], res["w"][t, 0], res["score"][t, 0], 100 * res["share"][t, 1], res["err"][t, 1], res["w"][t, 1], res["score"][t, 1]))
