"""v11.2.214 (SDOS-214): an ice cream bar half in the open mouth. A real face with the mouth open (AFLW2000 sample image03324,
public benchmark photo, NOT user footage; 2x enlarged, face ~310 px) on a 1280x720 canvas over a blurred copy of itself. A bar
(a rounded rectangle, one colour, shaded like a cylinder, a streak of highlight) comes up from below the chin, in front of the
lower lip, and goes UP INTO the opening, where the UPPER lip lies over its top: the bar is visible only below the upper inner lip
contour (and inside the outer lip outline only within the opening or below it) - exactly the occlusion of a bar held to the mouth
from below. It rests there, moves a little (chewing), goes down and out. A slow camera drift, zoom and sensor noise as the
profile clips. 150 frames at 30 fps.
usage: [PHOTO=image03324] [FLIP=1] make_bar.py COLOUR [TILT_DEG] [OUT_PREFIX]   COLOUR: choc, cream, strawberry, mint, none (no bar: the reference run)
  -> bar_<COLOUR>.mp4 + bar_<COLOUR>_gt.npz: 'a' bar alpha (half resolution), 'open' the opening between the lips, 'lips' the lip
     band (outer outline minus the opening), 'L' the 106 landmarks and 'kps' the 5 points per frame (as drawn in the frame),
     'tip' the bar's tip y (frame coordinates, 0 = no bar)."""
import cv2, numpy as np, subprocess, os, sys
from insightface.app import FaceAnalysis
COLOUR = sys.argv[1]
TILT = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
PFX = sys.argv[3] if len(sys.argv) > 3 else "bar"
W, H, N = 1280, 720, 150
SC = 2.0
PHOTO = os.environ.get("PHOTO", "image03324")                         # another AFLW2000 sample photo (an open mouth, a turned head)
src = cv2.imread("/tmp/claude-0/prnet/TestImages/AFLW2000/%s.jpg" % PHOTO)
if os.environ.get("FLIP") == "1":
    src = cv2.flip(src, 1)                                               # the head turned the other way (the bar comes from the right)
photo = cv2.resize(src, None, fx=SC, fy=SC, interpolation=cv2.INTER_CUBIC)
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
f0 = max(fa.get(src), key=lambda q: q.det_score)
L0 = f0.landmark_2d_106 * SC; K0 = f0.kps * SC
OFF = np.array([640 - K0[:, 0].mean(), 390 - K0[:, 1].mean()], np.float32)     # the face's 5 points centred on (640, 390)
L0 = L0 + OFF; K0 = K0 + OFF
# the canvas: the photo over a blurred copy of itself (feathered), so there is no second face at the frame's edge
bgi = cv2.GaussianBlur(cv2.resize(photo, (W, H)), (0, 0), 30).astype(np.float32)
M0 = np.float32([[1, 0, OFF[0]], [0, 1, OFF[1]]])
pw = cv2.warpAffine(photo, M0, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
mk = np.zeros(photo.shape[:2], np.float32); mk[6:-6, 6:-6] = 1; mk = cv2.GaussianBlur(mk, (0, 0), 5)
mw = cv2.warpAffine(mk, M0, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[..., None]
CANVAS = np.clip(bgi * (1 - mw) + pw * mw, 0, 255)
OUTER = [52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55]
INNER = [65, 66, 62, 70, 69, 57, 60, 54]
UPPER_INNER = [65, 66, 62, 70, 69]
op_poly = L0[INNER]; out_poly = L0[OUTER]
mouth_c = op_poly.mean(0)
op_top = float(L0[UPPER_INNER, 1].min()); op_bot = float(L0[[57, 60, 54], 1].max())
chin_y = float(L0[53, 1]) + 40
COLS = {"choc": (38, 52, 88), "cream": (232, 238, 244), "strawberry": (140, 130, 235), "mint": (170, 215, 150)}
col = COLS.get(COLOUR)
ang = np.radians(TILT)
axis = np.array([np.sin(ang), np.cos(ang)], np.float32)                 # from the tip outwards: down (and to the right); TILT 90 = to the right
nrm = np.array([axis[1], -axis[0]], np.float32)
TH = float(os.environ.get("TH", "0.55"))                                 # the bar's thickness as a share of the opening's width
T_BAR = max(TH * float(np.linalg.norm(L0[69] - L0[65])), 30.0)


def poly_mask(pts, shape, ss=3):
    m = np.zeros((shape[0] * ss, shape[1] * ss), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(pts) * ss).astype(np.int32)], 255)
    return cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def upper_clip_mask(shape):
    """1 where the bar is visible: below the upper inner lip contour inside the mouth's width (the upper lip lies over the bar),
    and everywhere beyond the mouth's corners (a bar that leaves the mouth sideways is in front of the cheek)"""
    pts = L0[UPPER_INNER]; pts = pts[np.argsort(pts[:, 0])]
    xl, xr = float(pts[0, 0]), float(pts[-1, 0])
    poly = np.vstack([[xl - 400, pts[0, 1]], pts, [xr + 400, pts[-1, 1]], [xr + 400, shape[0] + 50], [xl - 400, shape[0] + 50]])
    m = poly_mask(poly, shape)
    if os.environ.get("SIDE", "1") == "1":
        m[:, int(xr):] = 1.0; m[:, :int(xl)] = 1.0
        m = cv2.GaussianBlur(m, (0, 0), 0.8)
    return m


def draw_bar(tip, rng):
    """tip: (x, y) of the bar's top end. Returns colour (HxWx3) and alpha (HxW) for the whole canvas, drawn 3x on an ROI."""
    S = 3
    p0 = np.array(tip, np.float32); p1 = p0 + axis * 900.0
    x0 = int(max(0, min(p0[0], p1[0]) - T_BAR - 10)); x1 = int(min(W, max(p0[0], p1[0]) + T_BAR + 10))
    y0 = int(max(0, min(p0[1], p1[1]) - 10)); y1 = H
    h, w = y1 - y0, x1 - x0
    a = np.zeros((h * S, w * S), np.uint8)
    q0 = ((p0 - [x0, y0]) * S).astype(np.int32); q1 = ((p1 - [x0, y0]) * S).astype(np.int32)
    cv2.line(a, tuple(int(v) for v in q0), tuple(int(v) for v in q1), 255, int(T_BAR * S))
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    rel = np.stack([xx - p0[0], yy - p0[1]], -1)
    q = (rel @ nrm) / (T_BAR / 2)                                          # -1 .. 1 across the bar
    sh = 0.62 + 0.50 * np.cos((q + 0.35) * 1.15) ** 2 - 0.12 * (np.abs(q) > 0.88)
    sh += 0.28 * np.exp(-((q + 0.50) / 0.09) ** 2)                         # a streak of highlight
    tex = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 1.2) * 4.0
    c = np.clip(np.array(col, np.float32)[None, None, :] * sh[..., None] + tex[..., None], 0, 255)
    full_c = np.zeros((H, W, 3), np.float32); full_a = np.zeros((H, W), np.float32)
    full_c[y0:y1, x0:x1] = c; full_a[y0:y1, x0:x1] = a
    return full_c, full_a


UP = upper_clip_mask((H, W))
OPEN = poly_mask(op_poly, (H, W)); OUTM = poly_mask(out_poly, (H, W)); LIPS = np.clip(OUTM - OPEN, 0, 1)


HORIZ = TILT >= 55.0
def tip_path(i):
    """the bar's tip (x, y): out of the mouth along the bar's own axis, into the opening, a little chewing, and back out"""
    if HORIZ:
        y_in = 0.5 * (op_top + op_bot) - 2.0 - (T_BAR / 2.0) * (axis[0] / max(axis[1], 1e-3)) * 0.0         # the bar's middle on the opening's middle
        x_in = mouth_c[0] - 0.30 * float(np.linalg.norm(L0[69] - L0[65]))
        out = 330.0
    else:
        y_in = op_top + 0.30 * (op_bot - op_top)                           # the tip under the upper lip, inside the opening
        x_in = mouth_c[0] - 6
        out = (chin_y + 60 - y_in) / max(axis[1], 1e-3)
    t = i - 20
    if col is None or t < 0 or i >= 135:
        return None
    if t < 30: u = t / 30.0; k = u * u * (3 - 2 * u); sgm = out * (1 - k); xo = 0.0
    elif t < 95: sgm = 5.0 * np.sin((t - 30) / 5.0) + 3.0 * np.sin((t - 30) / 13.0); xo = 0.0
    else: u = min(1.0, (t - 95) / 20.0); k = u * u * (3 - 2 * u); sgm = out * k; xo = 0.0
    return (x_in + axis[0] * sgm + xo, y_in + axis[1] * sgm)


tmp = f"{PFX}_{COLOUR}.avi"
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (W, H))
rng = np.random.default_rng(11); nrng = np.random.default_rng(3)
GA, GO, GL, GLM, GK, GT = [], [], [], [], [], []
for i in range(N):
    base = CANVAS.copy(); a_bar = np.zeros((H, W), np.float32)
    p = tip_path(i)
    if p is not None:
        c, a = draw_bar(p, np.random.default_rng(7))
        # visible: below the upper lip's inner contour; and beyond the outer lip outline everywhere
        vis = np.clip(UP + (1 - OUTM) * 0, 0, 1)
        a = a * UP
        base = base * (1 - a[..., None]) + c * a[..., None]; a_bar = a
    s = 1.0 + 0.03 * np.sin(i / 19.0); dx = 10 * np.sin(i / 11.0); dy = 4 * np.cos(i / 13.0)
    A = np.float32([[s, 0, 640 - 640 * s + dx], [0, s, 360 - 360 * s + dy]])
    fr = cv2.warpAffine(base, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    fr = np.clip(fr + nrng.normal(0, 1.5, fr.shape), 0, 255).astype(np.uint8)
    vw.write(fr)
    def wp(m): return cv2.resize(cv2.warpAffine(m, A, (W, H), flags=cv2.INTER_LINEAR), (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    GA.append((wp(a_bar) * 255).astype(np.uint8)); GO.append((wp(OPEN) * 255).astype(np.uint8)); GL.append((wp(LIPS) * 255).astype(np.uint8))
    GLM.append(L0 @ A[:, :2].T + A[:, 2]); GK.append(K0 @ A[:, :2].T + A[:, 2])
    GT.append(0.0 if p is None else float(p[0] * 0 + (A[1, :2] @ np.array(p) + A[1, 2])))
vw.release()
subprocess.run(["/tmp/claude-0/bin/ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", f"{PFX}_{COLOUR}.mp4"], check=True)
os.remove(tmp)
np.savez_compressed(f"{PFX}_{COLOUR}_gt.npz", a=np.array(GA), open=np.array(GO), lips=np.array(GL), L=np.array(GLM, np.float32), kps=np.array(GK, np.float32), tip=np.array(GT, np.float32))
print("ok", COLOUR, "mouth centre %.0f %.0f, opening %.0f..%.0f, bar %.0f px thick" % (mouth_c[0], mouth_c[1], op_top, op_bot, T_BAR))
