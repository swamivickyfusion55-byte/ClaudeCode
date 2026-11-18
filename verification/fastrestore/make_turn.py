"""v11.2.213 (SDOS-213): a head turning away from the camera, as test footage with a known pose on every frame (head3d.py: the public
"Lee Perry-Smith" head scan, CC BY 3.0; no user media). The face goes from the camera to MAXYAW deg away and back - at 80-110 deg
only a sliver of the face (10-20 %) is visible - with a slow nod and tilt, a slow camera drift and sensor noise.
usage: make_turn.py OUT.mp4 [maxyaw=110] [frames=150] [side=1|-1] [roll=4] [deg=1]
Ground truth OUT_gt.npz: yaw / pitch / roll per frame, the true 5 points (eyes, nose tip, mouth corners; also when hidden),
'vis' = the share of the face visible (against the frontal view), 'a' = the visible face mask (half resolution), 'up' = the
head's up direction in the picture (unit vector, at the face).  deg=1: also a phone-quality copy OUT with 'deg' (noise, blur, crf 30)."""
import sys, os, math, subprocess
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from head3d import Head, rot
from insightface.app import FaceAnalysis

OUT = sys.argv[1]
KW = dict(a.split("=", 1) for a in sys.argv[2:])
MAXY = float(KW.get("maxyaw", 110)); N = int(KW.get("frames", 150)); SIDE = float(KW.get("side", 1)); ROLL = float(KW.get("roll", 4))
W, H = 1280, 720
h = Head("/tmp/claude-0/head3d/LeePerrySmith.glb", "/tmp/claude-0/head3d/Map-COL.jpg")
t1 = cv2.imread("/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg")
bg = cv2.GaussianBlur(cv2.resize(t1, (W, H)), (0, 0), 18)
fa = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "landmark_2d_106"], providers=["CPUExecutionProvider"])
fa.prepare(ctx_id=-1, det_size=(640, 640))

# --- the face on the head: from a frontal render, the detector's 5 points and the 106-point outline, carried back onto the mesh --
C0, HP = (640, 420), 760
im0, cov0, id0 = h.render(0, 0, 0, W, H, C0, HP, bg)
f0 = max(fa.get(im0), key=lambda q: q.det_score)
R0 = rot(0, 0, 0)
P0 = h.P @ R0.T
D = 4.0 * h.h; fo = HP * D / h.h


def backproject(pt):
    """the 3D point of the mesh under image point pt in the frontal render"""
    x, y = int(round(pt[0])), int(round(pt[1]))
    ti = int(id0[y, x])
    a, b, c = h.F[ti]
    # the triangle's plane: intersect the camera ray
    o = np.array([0, 0, D]); dvec = np.array([(pt[0] - C0[0]) / fo, -(pt[1] - C0[1]) / fo, -1.0])
    A_, B_, C_ = P0[a], P0[b], P0[c]
    n = np.cross(B_ - A_, C_ - A_)
    t = float(np.dot(A_ - o, n) / np.dot(dvec, n))
    return o + t * dvec


KP3 = np.array([backproject(p) for p in f0.kps])            # the 5 points on the head (head coordinates, frontal = identity)
outline = cv2.convexHull(f0.landmark_2d_106[:33].astype(np.float32)).reshape(-1, 2)
# the face's triangles: those the frontal view shows inside the face outline (jaw line + up to the brows' height, as the engine's)
top = float(f0.landmark_2d_106[:, 1].min()) - 0.10 * float(np.ptp(f0.landmark_2d_106[:, 1]))
hull = np.vstack([outline, [[outline[:, 0].min(), top], [outline[:, 0].max(), top]]]).astype(np.float32)
hull = cv2.convexHull(hull)
fm = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(fm, hull.astype(np.int32), 1)
FACE_T = np.zeros(len(h.F), bool)
ids = id0[(fm > 0) & (id0 >= 0)]
FACE_T[np.unique(ids)] = True
area0 = float(((fm > 0) & (id0 >= 0)).sum())

rng = np.random.default_rng(13)
tmp = OUT + ".avi"; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (W, H))
Y, Pp, Rr, K, VIS, A, UP = [], [], [], [], [], [], []
n_out = int(N * 0.46); n_hold = int(N * 0.14)
for i in range(N):
    if i < n_out:
        s = 0.5 - 0.5 * math.cos(math.pi * i / n_out)
    elif i < n_out + n_hold:
        s = 1.0
    else:
        s = 0.5 + 0.5 * math.cos(math.pi * (i - n_out - n_hold) / max(1, N - n_out - n_hold - 1))
    yaw = SIDE * MAXY * s
    pitch = 5.0 * math.sin(i / 17.0)
    roll = ROLL * math.sin(i / 23.0 + 0.7)
    cx = 640 + 30 * math.sin(i / 29.0); cy = 420 + 10 * math.cos(i / 31.0); hp = HP * (1.0 + 0.04 * math.sin(i / 37.0))
    im, cov, idb = h.render(yaw, pitch, roll, W, H, (cx, cy), hp, bg)
    im = np.clip(im.astype(np.int16) + rng.integers(-3, 4, im.shape, dtype=np.int16), 0, 255).astype(np.uint8)
    vw.write(im)
    kp = h.project(KP3, yaw, pitch, roll, W, H, (cx, cy), hp)
    vm = (idb >= 0) & FACE_T[np.clip(idb, 0, None)]
    R = rot(yaw, pitch, roll)
    c3 = KP3.mean(0) @ R.T; u3 = (KP3.mean(0) + np.array([0, 0.1 * h.h, 0])) @ R.T
    p_ = h.project(np.stack([KP3.mean(0), KP3.mean(0) + np.array([0, 0.1 * h.h, 0])]), yaw, pitch, roll, W, H, (cx, cy), hp)
    up = p_[1] - p_[0]; up = up / (np.linalg.norm(up) + 1e-9)
    Y.append(yaw); Pp.append(pitch); Rr.append(roll); K.append(kp); VIS.append(float(vm.sum()) / area0)
    A.append(cv2.resize(vm.astype(np.uint8) * 255, (W // 2, H // 2), interpolation=cv2.INTER_AREA)); UP.append(up)
vw.release()
subprocess.run(["/tmp/claude-0/bin/ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", OUT], check=True)
os.remove(tmp)
np.savez_compressed(OUT[:-4] + "_gt.npz", yaw=np.array(Y), pitch=np.array(Pp), roll=np.array(Rr), kps=np.array(K, np.float32),
                    vis=np.array(VIS), a=np.array(A), up=np.array(UP), kp3=KP3)
if KW.get("deg") == "1":
    d = OUT[:-4] + "deg.mp4"
    subprocess.run(["/tmp/claude-0/bin/ffmpeg", "-y", "-loglevel", "error", "-i", OUT, "-vf", "noise=alls=8:allf=t,gblur=sigma=0.9",
                    "-c:v", "libx264", "-crf", "30", "-pix_fmt", "yuv420p", d], check=True)
    import shutil; shutil.copy(OUT[:-4] + "_gt.npz", d[:-4] + "_gt.npz")
print("frames %d, yaw max %.0f, visible share at the turn's peak %.2f, lowest %.2f" % (N, max(np.abs(Y)), VIS[n_out], min(VIS)))
