"""Rule under test: a region the net calls occluded is cut only if (a) it lies inside the eroded landmark hull and (b) the connected
piece touches the mouth/chin core (x 0.28-0.72, y >= 0.56 of the aligned crop). Clean faces + scoop at mouth/lips/cheek; xseg_2."""
import sys, numpy as np, cv2, onnxruntime as ort, time
exec(open('lip91/xseg_test2.py').read().split("models = sys.argv")[0])
COLS = {"vanilla": (235, 240, 248), "chocolate": (40, 55, 85), "strawberry": (150, 150, 235), "mango": (60, 150, 240), "caramel": (70, 130, 200), "mint": (150, 215, 170)}
POS = {"mouth": (0.50, 0.80, 0.17), "lips": (0.50, 0.74, 0.12), "cheekL": (0.30, 0.62, 0.14)}
yy, xx = np.mgrid[0:SIZE, 0:SIZE] / float(SIZE)
CORE = (xx > 0.33) & (xx < 0.67) & (yy >= 0.58)
SIG = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
def net(sess, img):
    x = cv2.resize(img, (256, 256)).astype(np.float32)[None] / 255.0
    m = np.clip(sess.run(None, {'input': x})[0][0, :, :, 0], 0, 1)
    m = (cv2.GaussianBlur(m, (0, 0), SIG).clip(0.5, 1) - 0.5) * 2
    return cv2.resize(m, (SIZE, SIZE))
def keep_from(m, hull):
    occ = (m < 0.5) & hull
    n, lab = cv2.connectedComponents(occ.astype(np.uint8), connectivity=8)
    k = np.ones_like(m)
    for c in range(1, n):
        comp = lab == c
        if (comp & CORE).sum() >= 0.002 * SIZE * SIZE:
            k[comp] = m[comp]
    return k
s = ort.InferenceSession('/tmp/claude-0/occmodels/xseg_2.onnx', providers=['CPUExecutionProvider'])
rng = np.random.default_rng(1); fc = []; kinds = []
for i, f in enumerate(faces):
    for k in range(5):
        img = np.clip(jitter(f, rng), 0, 255).astype(np.float32)
        if k == 1: img = img * 0.45
        if k == 2: img = img * 0.18 + rng.standard_normal(img.shape).astype(np.float32) * 4
        if k == 3: img = np.clip(img * np.linspace(1.3, 0.5, SIZE)[None, :, None], 0, 255)
        if k == 4: img = cv2.GaussianBlur(img, (0, 0), 2.5)
        img = np.clip(img, 0, 255).astype(np.uint8)
        kk = keep_from(net(s, img), HULL[i])
        if k == 1 and (HULL[i] & (kk < 0.5)).sum() > 0.02 * HULL[i].sum():
            v = img.copy(); v[HULL[i] & (kk < 0.5)] = (0, 0, 255); cv2.imwrite(f'lip91/zone_fc_{i}.png', v)
        fc.append(100.0 * float((HULL[i] & (kk < 0.5)).sum()) / HULL[i].sum()); kinds.append(k)
fc = np.array(fc); kinds = np.array(kinds)
print(f"blur sigma {SIG}: CLEAN faces, face cut: " + " | ".join(f"{nm} mean {fc[kinds == k].mean():.2f}% worst {fc[kinds == k].max():.2f}%" for k, nm in enumerate(("normal", "dim", "dark+noise", "side light", "blur"))))
for pn, (cx, cy, r) in POS.items():
    hits = []; ring = []; lipcov = []
    for cn, col in COLS.items():
        for i, f in enumerate(faces):
            base = np.clip(jitter(f, rng), 0, 255).astype(np.float32)
            img, a = food(base, cx, cy, r, col, rng)
            img = np.clip(img, 0, 255).astype(np.uint8); kk = keep_from(net(s, img), HULL[i])
            obj = (a > 0.5) & HULL[i]
            if obj.sum() < 50: continue
            hits.append(float((kk[obj] < 0.5).mean()))
            near = cv2.dilate(obj.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~cv2.dilate((a > 0.05).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & HULL[i]
            if near.sum() > 20: ring.append(float((kk[near] < 0.5).mean()))
    print(f"   object at {pn:7s}: object cut {100*np.mean(hits):5.1f}% (worst case {100*np.min(hits):5.1f}%) | face 2-7 px around it wrongly cut {100*np.mean(ring):5.1f}%")
