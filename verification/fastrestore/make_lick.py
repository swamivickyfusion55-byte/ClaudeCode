"""v11.2.210: eating an ice cream. The centre face of insightface's sample photo (public, 3x enlarged, face ~320 px); a scoop on a
cone comes up from below the chin until it covers the middle of the LOWER lip (the upper lip and the mouth corners stay
visible), holds, goes down to the chin, and comes back - three licks in 150 frames. NOT user footage.
usage: make_lick.py COLOUR [RADIUS_PX]  -> lick_<COLOUR>.mp4 + lick_<COLOUR>_gt.npz (object alpha per frame, lips outline, the
face's 5 points) ; COLOUR: vanilla, strawberry, caramel, choc, none (no object: the reference run)"""
import cv2, numpy as np, subprocess, os, sys
from insightface.app import FaceAnalysis
src = cv2.imread("/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg")
BASE = cv2.resize(src[220:460, 307:733], (1280, 720), interpolation=cv2.INTER_CUBIC)
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
f0 = min(fa.get(BASE), key=lambda f: abs((f.bbox[0] + f.bbox[2]) / 2 - 640))
L = f0.landmark_2d_106; lips = L[52:72]
lx0, ly0 = lips.min(0); lx1, ly1 = lips.max(0)
COLS = {"vanilla": (235, 240, 248), "strawberry": (150, 150, 235), "caramel": (70, 150, 215), "choc": (40, 55, 85)}


def draw(cx, cy, R, col, rng):
    """A scoop (radius R, centre cx, cy) on a cone below it; returns colour image and alpha (720p), drawn 3x and reduced."""
    S = 3; H, W = 720 * S, 1280 * S; a = np.zeros((H, W), np.uint8); c = np.zeros((H, W, 3), np.float32)
    cone = np.array([[cx - 0.80 * R, cy + 0.45 * R], [cx + 0.80 * R, cy + 0.45 * R], [cx, cy + 3.6 * R]]) * S
    cv2.fillConvexPoly(a, np.round(cone).astype(np.int32), 255); c[a > 0] = (95, 150, 205)             # wafer
    for j in range(-8, 9):                                                                                # cross-hatch
        x = int((cx + j * 0.22 * R) * S)
        cv2.line(c, (x, int((cy + 0.45 * R) * S)), (x + int(1.2 * R * S), int((cy + 3.6 * R) * S)), (70, 120, 175), S)
        cv2.line(c, (x, int((cy + 0.45 * R) * S)), (x - int(1.2 * R * S), int((cy + 3.6 * R) * S)), (70, 120, 175), S)
    sc = np.zeros((H, W), np.uint8); cv2.circle(sc, (int(cx * S), int(cy * S)), int(R * S), 255, -1)
    cv2.ellipse(sc, (int(cx * S), int((cy + 0.45 * R) * S)), (int(1.05 * R * S), int(0.45 * R * S)), 0, 0, 360, 255, -1)
    a = np.maximum(a, sc); c[sc > 0] = col
    a = cv2.resize(a, (1280, 720), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    c = cv2.resize(c, (1280, 720), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0:720, 0:1280].astype(np.float32)
    shade = 0.80 + 0.30 * np.clip(1.0 - np.hypot(xx - (cx - 0.3 * R), yy - (cy - 0.3 * R)) / (2.2 * R), 0, 1)   # a soft highlight
    tex = cv2.GaussianBlur(rng.standard_normal((720, 1280)).astype(np.float32), (0, 0), 1.5) * 5
    return np.clip(c * shade[:, :, None] + tex[:, :, None], 0, 255), a


def build(name, R, n=150):
    col = COLS.get(name); tmp = f"lick_{name}.avi"; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720))
    gt = []; nr = np.random.default_rng(5)
    cx0 = (lx0 + lx1) / 2
    y_top = ly0 + 0.30 * (ly1 - ly0) + R          # the scoop's top edge 30 % down the lip outline: over the lower lip
    y_low = ly1 + 0.9 * R + R                     # the scoop's top edge under the lower lip (on the chin)
    y_out = 720 + 1.2 * R                         # out of the picture
    for i in range(n):
        f = np.clip(BASE.astype(np.int16) + nr.integers(-1, 2, BASE.shape, dtype=np.int16), 0, 255).astype(np.uint8).astype(np.float32)
        t = i - 20
        if col is None or t < 0 or i >= n - 8: p = None
        elif t < 10: p = (cx0, y_out + (y_top - y_out) * (t + 1) / 10.0)                 # in, to the lip
        else:
            ph = (t - 10) % 36                                                          # one lick: 14 at the lip, 6 down, 10 low, 6 up
            if ph < 14: p = (cx0 + 2.0 * np.sin(ph / 2.0), y_top + 1.5 * np.cos(ph / 3.0))
            elif ph < 20: p = (cx0, y_top + (y_low - y_top) * (ph - 13) / 6.0)
            elif ph < 30: p = (cx0 + 2.0 * np.sin(ph / 2.0), y_low + 1.5 * np.cos(ph / 3.0))
            else: p = (cx0, y_low + (y_top - y_low) * (ph - 29) / 6.0)
        A = np.zeros((720, 1280), np.float32)
        if p is not None:
            c, A = draw(p[0], p[1], R, col, np.random.default_rng(7)); f = f * (1 - A[:, :, None]) + c * A[:, :, None]
        vw.write(np.clip(f, 0, 255).astype(np.uint8)); gt.append((A * 255).astype(np.uint8))
    vw.release()
    subprocess.run(["/tmp/claude-0/bin/ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", f"lick_{name}.mp4"], check=True); os.remove(tmp)
    np.savez_compressed(f"lick_{name}_gt.npz", a=np.array(gt), lips=L[52:64].astype(np.float32), lmk=L.astype(np.float32), kps=f0.kps.astype(np.float32),
                        ytop=np.float32(y_top), ylow=np.float32(y_low))
    print(name, "R", R, "lips box", [round(float(v), 1) for v in (lx0, ly0, lx1, ly1)])


R = float(sys.argv[2]) if len(sys.argv) > 2 else 0.14 * float(f0.bbox[2] - f0.bbox[0])
build(sys.argv[1], R)
