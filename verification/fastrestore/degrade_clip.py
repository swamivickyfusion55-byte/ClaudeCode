"""degrade_clip.py IN.mp4 OUT.mp4 [sigma=6] [blur=0.8] [crf=30]  - a synthetic clip made to look like phone footage: camera noise, a little blur,
per-frame exposure change, then lossy H.264. The ground truth (_gt.npz) is copied unchanged: nothing moves, only the picture is worse.
(public sample photo clips only; NOT user footage)"""
import sys, os, shutil, subprocess, cv2, numpy as np
src, dst = sys.argv[1:3]; kw = dict(a.split("=") for a in sys.argv[3:])
sig, blur, crf = float(kw.get("sigma", 6)), float(kw.get("blur", 0.8)), int(kw.get("crf", 30))
cap = cv2.VideoCapture(src); rng = np.random.default_rng(21); tmp = dst + ".tmp.avi"; vw = None
while True:
    ok, f = cap.read()
    if not ok: break
    if vw is None: vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (f.shape[1], f.shape[0]))
    x = cv2.GaussianBlur(f.astype(np.float32), (0, 0), blur) if blur > 0 else f.astype(np.float32)
    x = x * (1.0 + rng.normal(0, 0.012)) + rng.normal(0, sig, x.shape)          # exposure wobble + camera noise
    vw.write(np.clip(x, 0, 255).astype(np.uint8))
vw.release()
subprocess.run(["/tmp/claude-0/bin/ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", dst], check=True); os.remove(tmp)
g = os.path.splitext(src)[0] + "_gt.npz"
if os.path.exists(g): shutil.copy(g, os.path.splitext(dst)[0] + "_gt.npz")
print(dst, os.path.getsize(dst))
