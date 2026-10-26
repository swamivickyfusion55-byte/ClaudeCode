"""Which 5 key points are closer to the truth at large head turns: the detector's (SCRFD) or the ones taken from the
106-point landmarks? AFLW2000 samples (public, with true 3-D landmarks pt3d_68 and yaw) + mirrors. Error per point in % of
the true face size (sqrt of the 68-point box area); eye = centre of the eye contour."""
import glob, sys, cv2, numpy as np, scipy.io as sio
# optional: PACK -> also the points the pack uses (swap_engine.side_eye_kps, v11.2.94)
E = None
if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1]); import swap_engine as E
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.28)
def from106(L):
    e1, e2 = L[33:43].mean(0), L[87:97].mean(0)
    eyes = sorted([e1, e2], key=lambda p: p[0])
    m = sorted([L[52], L[61]], key=lambda p: p[0])
    return np.array([eyes[0], eyes[1], L[86], m[0], m[1]], np.float32)
rows = []
for f in sorted(glob.glob("/tmp/claude-0/prnet/TestImages/AFLW2000/*.jpg")):
    m = sio.loadmat(f[:-4] + ".mat"); yaw = float(m["Pose_Para"][0][1] * 180 / np.pi); P = m["pt3d_68"][:2].T.astype(np.float32)
    for flip in (0, 1):
        img = cv2.imread(f); W = img.shape[1]; G = P.copy()
        if flip: img = img[:, ::-1].copy(); G[:, 0] = W - 1 - G[:, 0]
        e = sorted([G[36:42].mean(0), G[42:48].mean(0)], key=lambda p: p[0]); mo = sorted([G[48], G[54]], key=lambda p: p[0])
        gt = np.array([e[0], e[1], G[30], mo[0], mo[1]], np.float32)
        size = float(np.sqrt((G[:, 0].max() - G[:, 0].min()) * (G[:, 1].max() - G[:, 1].min())))
        fs = fa.get(img)
        if not fs: continue
        q = min(fs, key=lambda q: np.linalg.norm(q.kps.mean(0) - gt.mean(0)))
        a = np.linalg.norm(q.kps - gt, axis=1) / size * 100; b = np.linalg.norm(from106(q.landmark_2d_106) - gt, axis=1) / size * 100
        c = np.linalg.norm(E.side_eye_kps(q.kps, q.landmark_2d_106) - gt, axis=1) / size * 100 if E is not None else a
        rows.append((abs(yaw), a, b, c))
for lo, hi in ((0, 30), (30, 50), (50, 90)):
    sel = [r for r in rows if lo <= r[0] < hi]
    if not sel: continue
    A = np.array([r[1] for r in sel]); Bm = np.array([r[2] for r in sel]); C = np.array([r[3] for r in sel])
    print(f"yaw {lo:2d}-{hi:2d} deg ({len(sel):2d} faces): detector  eyes {A[:, :2].mean():4.1f}%  nose {A[:, 2].mean():4.1f}%  mouth {A[:, 3:].mean():4.1f}%   |   "
          f"106-point  eyes {Bm[:, :2].mean():4.1f}%  nose {Bm[:, 2].mean():4.1f}%  mouth {Bm[:, 3:].mean():4.1f}%   |   pack eyes {C[:, :2].mean():4.1f}%")
    for r in sel:
        if lo >= 30: print(f"      yaw {r[0]:3.0f}: detector eyes {r[1][0]:4.1f} {r[1][1]:4.1f}  | 106 eyes {r[2][0]:4.1f} {r[2][1]:4.1f}  | pack eyes {r[3][0]:4.1f} {r[3][1]:4.1f}")

# Where the swapped face's eyes are DRAWN: the swap template's eye points mapped back into the photo through the
# similarity fit of the 5 points (estimate_norm), against the true eyes. Visible eye = the one farther from the nose.
if E is not None:
    print("drawn eye position (template eye through the 5-point fit) vs the true eye, % of face size:")
    for f in sorted(glob.glob("/tmp/claude-0/prnet/TestImages/AFLW2000/*.jpg")):
        m = sio.loadmat(f[:-4] + ".mat"); yaw = float(m["Pose_Para"][0][1] * 180 / np.pi); P = m["pt3d_68"][:2].T.astype(np.float32)
        if abs(yaw) < 50: continue
        for flip in (0, 1):
            img = cv2.imread(f); W = img.shape[1]; G = P.copy()
            if flip: img = img[:, ::-1].copy(); G[:, 0] = W - 1 - G[:, 0]
            e = sorted([G[36:42].mean(0), G[42:48].mean(0)], key=lambda p: p[0])
            size = float(np.sqrt((G[:, 0].max() - G[:, 0].min()) * (G[:, 1].max() - G[:, 1].min())))
            q = min(fa.get(img), key=lambda q: np.linalg.norm(q.kps[:2].mean(0) - np.mean(e, 0)))
            res = []
            for kp in (q.kps, E.side_eye_kps(q.kps, q.landmark_2d_106)):
                IM = cv2.invertAffineTransform(E.estimate_norm(np.asarray(kp, np.float32), 128))
                T = E._arcface_dst(128)[:2] @ IM[:, :2].T + IM[:, 2]
                near = 0 if np.linalg.norm(kp[0] - kp[2]) > np.linalg.norm(kp[1] - kp[2]) else 1
                res.append(np.linalg.norm(T[near] - e[near]) / size * 100)
            print(f"   yaw {abs(yaw):3.0f}{'m' if flip else ' '}: visible eye drawn {res[0]:4.1f}% off with the detector's points, {res[1]:4.1f}% with v11.2.94's")
