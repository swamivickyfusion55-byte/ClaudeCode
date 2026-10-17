"""Does the detector (as configured in the pipeline: SCRFD, det_size 640, thresh 0.28) lose faces in low light, and does an analysis-only
enhanced copy of the frame (same size, photometric only: gamma + CLAHE on luma) recover them? Synthetic low light from the real-photo clip."""
import sys, cv2, numpy as np
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name='buffalo_l', root='/root/.insightface', allowed_modules=['detection'], providers=['CPUExecutionProvider'])
fa.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.28)
def frames(p, lim=70):
    c = cv2.VideoCapture(p); o = []
    while len(o) < lim:
        ok, f = c.read()
        if not ok: break
        o.append(f)
    return o
fr = frames('turn_calm.mp4'); kps = np.load('turn_calm_gt.npz')['kps'][:len(fr)]
rng = np.random.default_rng(1)
def lowlight(img, gain, noise, face_gain=None, c=None, r=None):
    f = img.astype(np.float32) / 255.0
    f = f ** 1.0
    if face_gain is not None:           # backlit: only the face area is dark, the background stays bright
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
        d = np.sqrt(((xx - c[0]) / (1.5 * r)) ** 2 + ((yy - c[1]) / (1.8 * r)) ** 2); w = np.clip((1.25 - d) / 0.4, 0, 1)[:, :, None]
        f = f * (1 - w) + f * face_gain * w
    else: f = f * gain
    f = f + rng.standard_normal(f.shape).astype(np.float32) * noise / 255.0 * (0.4 + 3.0 * np.sqrt(np.clip(f, 0, 1)))   # signal-dependent sensor noise
    return np.clip(f * 255.0 + 0.5, 0, 255).astype(np.uint8)
def enhance(img, clip=2.0, grid=8, gamma=0.6):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); L = lab[:, :, 0]
    lut = (np.power(np.arange(256) / 255.0, gamma) * 255.0).astype(np.uint8)
    L = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(cv2.LUT(L, lut)); lab[:, :, 0] = L
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
def score(imgs):
    hit = 0; err = []; n = 0; conf = []
    for t, im in enumerate(imgs):
        gt = kps[t]; iod = np.linalg.norm(gt[1] - gt[0]); fs = fa.get(im); n += 1
        best = None
        for f in fs:
            d = np.linalg.norm(f.kps.mean(axis=0) - gt.mean(axis=0)) / iod
            if d < 0.6 and (best is None or d < best[0]): best = (d, f)
        if best is not None:
            hit += 1; err.append(np.linalg.norm(best[1].kps - gt, axis=1).mean() / iod); conf.append(float(best[1].det_score))
    return hit / n, (np.median(err) if err else float('nan')), (np.mean(conf) if conf else float('nan'))
base = score(fr[:40]); print("normal light: recall %.2f  kps err %.3f  det score %.2f" % base)
for name, kw in (("dim (gain 0.30, noise 4)", dict(gain=0.30, noise=4)), ("dark (gain 0.12, noise 4)", dict(gain=0.12, noise=4)), ("very dark (gain 0.06, noise 5)", dict(gain=0.06, noise=5)), ("backlit (face x0.12, bright bg)", dict(face_gain=0.12, noise=3))):
    ll = []
    for t in range(40):
        c = kps[t].mean(axis=0); r = np.linalg.norm(kps[t, 1] - kps[t, 0])
        ll.append(lowlight(fr[t], kw.get('gain'), kw['noise'], kw.get('face_gain'), c, r))
    a = score(ll); b = score([enhance(x) for x in ll]); c2 = score([enhance(x, clip=3.0, gamma=0.5) for x in ll])
    print("%-34s mean luma %5.1f | as is: recall %.2f err %.3f score %.2f | CLAHE g0.6: recall %.2f err %.3f score %.2f | CLAHE g0.5 c3: recall %.2f err %.3f score %.2f" % (name, np.mean([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).mean() for x in ll]), *a, *b, *c2))
