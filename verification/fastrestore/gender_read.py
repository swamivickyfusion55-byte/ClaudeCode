"""v11.2.212 (SDOS-212): how sure and how right is the gender model (insightface genderage, as the job runs it) on each person of a
synthetic clip, frame by frame? The real detector on every frame, each detection matched to its person by the true 5 points (both
people jointly), the model's two gender scores turned into P(male). Grouped by the face's view (front: nose offset under 0.34 eye
distances, as the tracker's front-view tests) and by contact (boxes touching). Synthetic clips of the public sample photo, NOT user
footage. The true genders are given on the command line.
usage: gender_read.py CLIP.mp4 GT.npz GENDER_A GENDER_B   (0 = woman, 1 = man; 'x' = skip that person)  [one=1: a one-person clip]"""
import sys, itertools, numpy as np, cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo import attribute
from insightface.utils import face_align
CLIP, GT, GA, GB = sys.argv[1:5]
TRUE = [None if g == "x" else int(g) for g in (GA, GB)]
_orig = attribute.Attribute.get
def _get(self, img, face):
    r = _orig(self, img, face)
    b = face.bbox; w, h = b[2] - b[0], b[3] - b[1]; c = (b[2] + b[0]) / 2, (b[3] + b[1]) / 2
    aimg, _ = face_align.transform(img, c, self.input_size[0], self.input_size[0] / (max(w, h) * 1.5), 0)
    blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.input_std, tuple(aimg.shape[0:2][::-1]), (self.input_mean,) * 3, swapRB=True)
    face["glogit"] = float(self.session.run(self.output_names, {self.input_name: blob})[0][0][1])   # = -score[0]; P(male) = 1/(1+e^(-2l))
    return r
attribute.Attribute.get = _get
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "genderage"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
cap = cv2.VideoCapture(CLIP); g = np.load(GT); kps = g["kps"]
def yaw(k):
    k = np.asarray(k, np.float32); eye = (k[0] + k[1]) / 2; d = np.linalg.norm(k[1] - k[0]) + 1e-6
    return abs(float(k[2][0] - eye[0])) / d
rows = []; t = 0
while True:
    ok, fr = cap.read()
    if not ok or t >= len(kps): break
    fs = fa.get(fr); best, bc = None, 1e18
    for combo in itertools.product(list(range(len(fs))) + [None], repeat=2):
        u = [c for c in combo if c is not None]
        if len(set(u)) != len(u): continue
        cost = sum(150.0 if c is None else float(np.linalg.norm(fs[c].kps - kps[t, p], axis=1).mean()) for p, c in enumerate(combo))
        if cost < bc: best, bc = combo, cost
    for p, c in enumerate(best):
        if c is None or TRUE[p] is None: continue
        f = fs[c]; other = [q for q in fs if q is not f]
        touch = any((min(f.bbox[2], q.bbox[2]) > max(f.bbox[0], q.bbox[0]) and min(f.bbox[3], q.bbox[3]) > max(f.bbox[1], q.bbox[1])) for q in other)
        pm = 1.0 / (1.0 + np.exp(-2.0 * f["glogit"]))
        rows.append((t, p, pm, yaw(f.kps), touch, float(f.det_score), float(f.bbox[2] - f.bbox[0])))
    t += 1
R = rows
def summ(sel, name):
    if not sel: print("  %-34s no frames" % name); return
    right = [(r[2] if TRUE[r[1]] == 1 else 1 - r[2]) for r in sel]          # P of the true gender
    wrong88 = sum(1 for x in right if x <= 0.12); wrong70 = sum(1 for x in right if x < 0.30); unsure = sum(1 for x in right if 0.30 <= x < 0.70)
    print("  %-34s %4d face-frames | P(true gender) median %.3f, lowest %.3f | confidently WRONG (P(wrong) >= 0.88): %d | wrong at >= 0.70: %d | unsure: %d"
          % (name, len(sel), float(np.median(right)), float(min(right)), wrong88, wrong70, unsure))
print(CLIP.split("/")[-1])
for p in (0, 1):
    if TRUE[p] is None: continue
    sel = [r for r in R if r[1] == p]
    print(" person %s (%s)" % ("AB"[p], "man" if TRUE[p] else "woman"))
    summ([r for r in sel if r[3] < 0.34 and not r[4]], "front view, not touching")
    summ([r for r in sel if r[3] >= 0.34 and not r[4]], "turned (0.34+), not touching")
    summ([r for r in sel if r[4]], "touching the other face")
