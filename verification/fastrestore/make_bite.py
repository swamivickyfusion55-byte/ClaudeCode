"""Partial bite: a scoop pressed against the LOWER lip of the centre face of insightface's sample photo (both lips still
visible), plus the same clip with no food. 100 frames, 1280x720, face ~320 px. NOT user footage.
usage: make_bite.py COLOUR [RADIUS]   -> bite_<COLOUR>.mp4 + bite_<COLOUR>_gt.npz (food alpha, lips box) and bite_none.mp4"""
import cv2, numpy as np, subprocess, os, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/lip86"); from food import food
from insightface.app import FaceAnalysis
src = cv2.imread("/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg")
win = src[220:460, 307:733]; BASE = cv2.resize(win, (1280, 720), interpolation=cv2.INTER_CUBIC)
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
f0 = min(fa.get(BASE), key=lambda f: abs((f.bbox[0] + f.bbox[2]) / 2 - 640))
lips = f0.landmark_2d_106[52:72]
lx0, ly0 = lips.min(0); lx1, ly1 = lips.max(0)
COLS = {"vanilla": (235, 240, 248), "strawberry": (150, 150, 235), "caramel": (70, 150, 215), "choc": (40, 55, 85)}
def build(name, R):
    col = COLS.get(name); tmp = f"bite_{name}.avi"; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720))
    gt = []; nrng = np.random.default_rng(5)
    cx = ((lx0 + lx1) / 2 - 280) / 720.0
    # scoop top edge at 20 % down the lip outline: covers the lower lip, leaves the upper lip and the corners
    cy = (ly0 + 0.20 * (ly1 - ly0)) / 720.0 + R
    for i in range(100):
        f = BASE.copy()
        f = np.clip(f.astype(np.int16) + nrng.integers(-1, 2, f.shape, dtype=np.int16), 0, 255).astype(np.uint8)
        if col is None or i < 25: p = None
        elif i < 35: u = (i - 24) / 10.0; p = (cx, cy + 0.15 * (1 - u))
        elif i < 85: p = (cx + 0.004 * np.sin((i - 35) / 3.0), cy + 0.004 * np.cos((i - 35) / 4.0))
        else: u = (i - 84) / 10.0; p = (cx, cy + 0.15 * u)
        a_full = np.zeros((720, 1280), np.uint8)
        if p is not None:
            w_ = f[:, 280:1000].astype(np.float32)
            out, a = food(w_, p[0], p[1], R, col, np.random.default_rng(7), texture=1.0, ss=2, ang=90)
            f[:, 280:1000] = np.clip(out, 0, 255).astype(np.uint8); a_full[:, 280:1000] = (a * 255).astype(np.uint8)
        gt.append(a_full); vw.write(f)
    vw.release()
    subprocess.run(["/tmp/claude-0/bin/ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", f"bite_{name}.mp4"], check=True); os.remove(tmp)
    np.savez_compressed(f"bite_{name}_gt.npz", a=np.array(gt), lips=np.array([lx0, ly0, lx1, ly1]))
R = float(sys.argv[2]) if len(sys.argv) > 2 else 0.055
build(sys.argv[1], R); build("none", R)
print("lips box", lx0, ly0, lx1, ly1)
