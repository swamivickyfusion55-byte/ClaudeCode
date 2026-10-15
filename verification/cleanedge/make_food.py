"""Realistic-ish ice cream (shaded, textured, specular, cone) held at the mouth of the centre face of insightface's sample photo,
3x enlarged so the face is ~320 px wide. Saves GT alpha. NOT user footage."""
import cv2, numpy as np, subprocess, os, sys
sys.path.insert(0, '../lip85'); from food import food
src = cv2.imread("/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg")
win = src[220:460, 307:733]; BASE = cv2.resize(win, (1280, 720), interpolation=cv2.INTER_CUBIC)
COLS = {"vanilla": (235, 240, 248), "choc": (40, 55, 85), "mint": (150, 215, 170)}
def build(name, n=100):
    col = COLS[name]; tmp = f"food_{name}.avi"; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720))
    gt = []; nrng = np.random.default_rng(5)
    for i in range(n):
        f = BASE.copy()
        f = np.clip(f.astype(np.int16) + nrng.integers(-1, 2, f.shape, dtype=np.int16), 0, 255).astype(np.uint8)
        # position of the scoop centre in the 720 window (x offset 280): rest at the mouth
        if i < 25: p = None
        elif i < 35: u = (i - 24) / 10.0; p = (0.50 + 0.28 * (1 - u), 0.667 + 0.20 * (1 - u))
        elif i < 85: p = (0.50 + 0.008 * np.sin((i - 35) / 3.0), 0.667 + 0.006 * np.cos((i - 35) / 4.0))
        else: u = (i - 84) / 10.0; p = (0.50 + 0.28 * u, 0.667 + 0.20 * u)
        a_full = np.zeros((720, 1280), np.uint8)
        if p is not None:
            win_ = f[:, 280:1000].astype(np.float32)
            out, a = food(win_, p[0], p[1], 0.134, col, np.random.default_rng(7), texture=1.0, ss=2)
            f[:, 280:1000] = np.clip(out, 0, 255).astype(np.uint8); a_full[:, 280:1000] = (a * 255).astype(np.uint8)
        gt.append(a_full); vw.write(f)
    vw.release()
    subprocess.run(["/tmp/claude-0/bin/ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", f"food_{name}.mp4"], check=True); os.remove(tmp)
    np.savez_compressed(f"food_{name}_gt.npz", a=np.array(gt)); print(name, os.path.getsize(f"food_{name}.mp4"))
for nm in sys.argv[1:] or ("vanilla", "choc"): build(nm)
