"""Occlusion-net used only INSIDE the face's own landmark hull (106-point outline, eroded 4 %): the outline of the swap stays as it is
(v11.2.89), only holes in the face are taken. Clean faces (false cut) and a realistic scoop (hit / face next to it wrongly cut)."""
import sys, numpy as np, cv2, onnxruntime as ort, time
sys.path.insert(0, 'lip'); sys.path.insert(0, 'lip86')
from realset import load_faces, jitter, SIZE
from food import food
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name='buffalo_l', root='/root/.insightface', allowed_modules=['detection', 'landmark_2d_106'], providers=['CPUExecutionProvider']); fa.prepare(ctx_id=-1, det_size=(256, 256))
faces = load_faces(); faces = faces + [f[:, ::-1].copy() for f in faces]
HULL = []
for f in faces:
    d = max(fa.get(f), key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    h = np.zeros((SIZE, SIZE), np.uint8); cv2.fillConvexPoly(h, cv2.convexHull(d.landmark_2d_106.astype(np.int32)), 1)
    HULL.append(cv2.erode(h, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.08 * SIZE) | 1,) * 2)).astype(bool))
models = sys.argv[1:] or ['xseg_1', 'xseg_2', 'xseg_3']
def run(sess, img):
    x = cv2.resize(img, (256, 256)).astype(np.float32)[None] / 255.0
    m = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, :, :, 0]
    m = cv2.resize(np.clip(m, 0, 1), (SIZE, SIZE))
    return (cv2.GaussianBlur(m, (0, 0), 5).clip(0.5, 1) - 0.5) * 2
COLS = {"vanilla": (235, 240, 248), "chocolate": (40, 55, 85), "strawberry": (150, 150, 235), "mango": (60, 150, 240), "caramel": (70, 130, 200), "mint": (150, 215, 170)}
POS = {"mouth": (0.50, 0.80, 0.17), "lips": (0.50, 0.74, 0.12), "cheekL": (0.30, 0.62, 0.14)}
for mn in models:
    s = ort.InferenceSession(f'/tmp/claude-0/occmodels/{mn}.onnx', providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(1); fc = []; kinds = []
    for i, f in enumerate(faces):
        for k in range(5):
            img = np.clip(jitter(f, rng), 0, 255).astype(np.float32)
            if k == 1: img = img * 0.45                                                        # dim
            if k == 2: img = img * 0.18 + rng.standard_normal(img.shape).astype(np.float32) * 4  # dark + noise
            if k == 3: img = np.clip(img * np.linspace(1.3, 0.5, SIZE)[None, :, None], 0, 255)     # side light
            if k == 4: img = cv2.GaussianBlur(img, (0, 0), 2.5)                                   # motion/defocus blur
            img = np.clip(img, 0, 255).astype(np.uint8)
            m = run(s, img); fc.append(100.0 * float((HULL[i] & (m < 0.5)).sum()) / HULL[i].sum()); kinds.append(k)
    fc = np.array(fc); kinds = np.array(kinds)
    print(f"== {mn}  CLEAN faces, face interior called occluded: " + " | ".join(f"{nm} mean {fc[kinds == k].mean():.2f}% worst {fc[kinds == k].max():.2f}%" for k, nm in enumerate(("normal", "dim", "dark+noise", "side light", "blur"))))
    for pn, (cx, cy, r) in POS.items():
        hits = []; ring = []
        for cn, col in COLS.items():
            for i, f in enumerate(faces[:12]):
                base = np.clip(jitter(f, rng), 0, 255).astype(np.float32)
                img, a = food(base, cx, cy, r, col, rng)
                img = np.clip(img, 0, 255).astype(np.uint8); m = run(s, img)
                obj = (a > 0.5) & HULL[i]
                if obj.sum() < 50: continue
                hits.append(float((m[obj] < 0.5).mean()))
                near = cv2.dilate(obj.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~cv2.dilate((a > 0.05).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & HULL[i]
                if near.sum() > 20: ring.append(float((m[near] < 0.5).mean()))
        print(f"   object at {pn:7s}: object inside the face called occluded {100*np.mean(hits):5.1f}% (worst {100*np.min(hits):5.1f}%) | face 2-7 px around it wrongly cut {100*np.mean(ring):5.1f}%")
