"""v11.2.214 (SDOS-214): an ice cream bar held to the mouth of a head seen from the SIDE. A real profile photo (AFLW2000 sample,
public benchmark photo, NOT user footage; 2x enlarged) on a 1280x720 canvas over a blurred copy of itself. A bar (a rounded
rectangle, shaded like a cylinder, a streak of highlight) comes in from the front of the face along the line of the lips, its tip
a few px inside the lips' front contour (the lips lie over the tip: the bar goes between them), rests there, moves a little, and
leaves. A slow camera drift, zoom and sensor noise as the other clips. 150 frames at 30 fps. The photo's mouth is closed: the bar
is held against the lips.
usage: make_bar_side.py PHOTO COLOUR [TILT_DEG] [PREFIX]   PHOTO: image00475 (man, 65 deg, faces left), image02420 (man, 71 deg,
  faces left), image00251 (woman, -55 deg, faces right); COLOUR: choc, cream, strawberry, mint, none; TILT: the bar's angle from the
  horizontal (positive = its far end lower)
  -> PREFIX_PHOTO_COLOUR.mp4 + _gt.npz ('a' bar alpha at half resolution, 'L' the 106 landmarks and 'kps' per frame as drawn, 'lips'
     the lips' box per frame, 'tip' the bar's tip x per frame, 0 = no bar)"""
import cv2, numpy as np, subprocess, os, sys
from insightface.app import FaceAnalysis
PHOTO, COLOUR = sys.argv[1:3]
TILT = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
PFX = sys.argv[4] if len(sys.argv) > 4 else "side"
W, H, N, SC = 1280, 720, 150, 2.0
src = cv2.imread("/tmp/claude-0/prnet/TestImages/AFLW2000/%s.jpg" % PHOTO)
photo = cv2.resize(src, None, fx=SC, fy=SC, interpolation=cv2.INTER_CUBIC)
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
f0 = max(fa.get(src), key=lambda q: q.det_score)
L0 = f0.landmark_2d_106 * SC; K0 = f0.kps * SC
OFF = np.array([640 - K0[:, 0].mean(), 390 - K0[:, 1].mean()], np.float32)
L0 = L0 + OFF; K0 = K0 + OFF
bgi = cv2.GaussianBlur(cv2.resize(photo, (W, H)), (0, 0), 30).astype(np.float32)
M0 = np.float32([[1, 0, OFF[0]], [0, 1, OFF[1]]])
pw = cv2.warpAffine(photo, M0, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
mk = np.zeros(photo.shape[:2], np.float32); mk[6:-6, 6:-6] = 1; mk = cv2.GaussianBlur(mk, (0, 0), 5)
mw = cv2.warpAffine(mk, M0, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[..., None]
CANVAS = np.clip(bgi * (1 - mw) + pw * mw, 0, 255)
lips = L0[52:72]
# which way the face looks: the nose tip against the eyes' middle
facing = -1.0 if (K0[2, 0] < (K0[0, 0] + K0[1, 0]) / 2) else 1.0
front = lips[:, 0].min() if facing < 0 else lips[:, 0].max()                   # the lips' front-most x
lip_c_y = float(np.median(lips[:, 1])); lip_h = float(lips[:, 1].max() - lips[:, 1].min())
T_BAR = max(0.9 * lip_h, 34.0)
COLS = {"choc": (38, 52, 88), "cream": (232, 238, 244), "strawberry": (140, 130, 235), "mint": (170, 215, 150)}
col = COLS.get(COLOUR)
ang = np.radians(TILT)
axis = np.array([facing * np.cos(ang), np.sin(ang)], np.float32)               # from the tip outwards, away from the face
nrm = np.array([axis[1], -axis[0]], np.float32)


def draw_bar(tip, rng):
    S = 3
    p0 = np.array(tip, np.float32); p1 = p0 + axis * 700.0
    x0 = int(max(0, min(p0[0], p1[0]) - T_BAR - 10)); x1 = int(min(W, max(p0[0], p1[0]) + T_BAR + 10))
    y0 = int(max(0, min(p0[1], p1[1]) - T_BAR - 10)); y1 = int(min(H, max(p0[1], p1[1]) + T_BAR + 10))
    h, w = y1 - y0, x1 - x0
    a = np.zeros((h * S, w * S), np.uint8)
    q0 = ((p0 - [x0, y0]) * S).astype(np.int32); q1 = ((p1 - [x0, y0]) * S).astype(np.int32)
    cv2.line(a, tuple(int(v) for v in q0), tuple(int(v) for v in q1), 255, int(T_BAR * S))
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    q = (np.stack([xx - p0[0], yy - p0[1]], -1) @ nrm) / (T_BAR / 2)
    sh = 0.62 + 0.50 * np.cos((q + 0.35) * 1.15) ** 2 - 0.12 * (np.abs(q) > 0.88)
    sh += 0.28 * np.exp(-((q + 0.50) / 0.09) ** 2)
    tex = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 1.2) * 4.0
    c = np.clip(np.array(col, np.float32)[None, None, :] * sh[..., None] + tex[..., None], 0, 255)
    fc = np.zeros((H, W, 3), np.float32); fa_ = np.zeros((H, W), np.float32)
    fc[y0:y1, x0:x1] = c; fa_[y0:y1, x0:x1] = a
    return fc, fa_


def tip_path(i):
    t = i - 20
    if col is None or t < 0 or i >= 135:
        return None
    x_in = front - facing * 9.0                                                  # the tip a little inside the lips' front contour
    y_in = lip_c_y
    out = 300.0
    if t < 30: u = t / 30.0; k = u * u * (3 - 2 * u); sg = out * (1 - k)
    elif t < 95: sg = 5.0 * np.sin((t - 30) / 5.0) + 3.0 * np.sin((t - 30) / 13.0)
    else: u = min(1.0, (t - 95) / 20.0); k = u * u * (3 - 2 * u); sg = out * k
    return (x_in + axis[0] * sg, y_in + axis[1] * sg)


tmp = f"{PFX}_{PHOTO}_{COLOUR}.avi"
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (W, H))
nrng = np.random.default_rng(3)
GA, GLM, GK, GT, GLB = [], [], [], [], []
lbox = np.array([lips[:, 0].min(), lips[:, 1].min(), lips[:, 0].max(), lips[:, 1].max()], np.float32)
for i in range(N):
    base = CANVAS.copy(); a_bar = np.zeros((H, W), np.float32)
    p = tip_path(i)
    if p is not None:
        c, a = draw_bar(p, np.random.default_rng(7))
        base = base * (1 - a[..., None]) + c * a[..., None]; a_bar = a
    s = 1.0 + 0.03 * np.sin(i / 19.0); dx = 10 * np.sin(i / 11.0); dy = 4 * np.cos(i / 13.0)
    A = np.float32([[s, 0, 640 - 640 * s + dx], [0, s, 360 - 360 * s + dy]])
    fr = cv2.warpAffine(base, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    fr = np.clip(fr + nrng.normal(0, 1.5, fr.shape), 0, 255).astype(np.uint8)
    vw.write(fr)
    GA.append((cv2.resize(cv2.warpAffine(a_bar, A, (W, H), flags=cv2.INTER_LINEAR), (W // 2, H // 2), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8))
    GLM.append(L0 @ A[:, :2].T + A[:, 2]); GK.append(K0 @ A[:, :2].T + A[:, 2])
    GLB.append(np.array([lbox[:2] @ A[:, :2].T + A[:, 2], lbox[2:] @ A[:, :2].T + A[:, 2]]).reshape(-1))
    GT.append(0.0 if p is None else float(A[0, :2] @ np.array(p) + A[0, 2]))
vw.release()
subprocess.run(["/tmp/claude-0/bin/ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", f"{PFX}_{PHOTO}_{COLOUR}.mp4"], check=True)
os.remove(tmp)
np.savez_compressed(f"{PFX}_{PHOTO}_{COLOUR}_gt.npz", a=np.array(GA), L=np.array(GLM, np.float32), kps=np.array(GK, np.float32), lips=np.array(GLB, np.float32), tip=np.array(GT, np.float32))
print("ok", PHOTO, COLOUR, "facing", "left" if facing < 0 else "right", "lips front x %.0f, y %.0f, height %.0f, bar %.0f px" % (front, lip_c_y, lip_h, T_BAR))
