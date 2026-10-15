"""Synthetic clips: the centre face of insightface's sample photo with the cheeks puffing out and chewing. NOT user footage."""
import cv2, numpy as np, subprocess, os
exec(open('make_scene.py').read().split('def build')[0])
def bulge(img, cx, cy, R, s):
    h, w = img.shape[:2]; yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy; r2 = (dx * dx + dy * dy) / (R * R); g = np.exp(-r2)
    f = 1.0 - s * g
    return cv2.remap(img, (cx + dx * f).astype(np.float32), (cy + dy * f).astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
def S_of(i, amp=0.30):
    if i < 15: return 0.0
    if i < 27: return amp * (i - 15) / 12.0
    if i < 69: return amp * (1.0 + 0.30 * np.sin(2 * np.pi * 2.5 * (i - 27) / 30.0)) * 0.85
    if i < 81: return amp * 0.85 * (81 - i) / 12.0
    return 0.0
def build(kind, out, n=90):
    rng = np.random.default_rng(3); tmp = out + ".avi"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720))
    x1, y1, x2, y2 = FACE["f2"]; y1 -= 100; y2 -= 100; fw = x2 - x1; fh = y2 - y1
    cl = (x1 + 0.20 * fw, y1 + 0.64 * fh); cr = (x1 + 0.80 * fw, y1 + 0.64 * fh); R = 0.30 * fw
    for i in range(n):
        f = src.copy()
        for nm in ("f0", "f1", "f3", "f4", "f5"): hide(f, nm)
        if kind == "swell":
            s = S_of(i)
            if s > 0: f = bulge(bulge(f, cl[0], cl[1], R, s), cr[0], cr[1], R, s)
        f = np.clip(f.astype(np.int16) + rng.integers(-1, 2, f.shape, dtype=np.int16), 0, 255).astype(np.uint8); vw.write(f)
    vw.release()
    subprocess.run(["/tmp/claude-0/bin/ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", out], check=True); os.remove(tmp)
    print(out, os.path.getsize(out))
for k in ("static", "swell"): build(k, f"cheek_{k}.mp4")
