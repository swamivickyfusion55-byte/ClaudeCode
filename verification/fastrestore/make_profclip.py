"""(FAST=1 in the environment: fast sweep +-160 px with matching motion blur -> profast_<NAME>.mp4)
Side-view clips from AFLW2000 sample photos (public benchmark, no user footage): the photo on a 1280x720 canvas with a
slow camera drift (+-40 px), a slow zoom (+-6 %) and sensor noise, 60 frames at 30 fps.  usage: make_profclip.py NAME..."""
import cv2, numpy as np, subprocess, os, sys
for name in sys.argv[1:]:
    im = cv2.imread(f"/tmp/claude-0/prnet/TestImages/AFLW2000/{name}.jpg")
    big = cv2.resize(im, (700, 700), interpolation=cv2.INTER_CUBIC)
    fast = os.environ.get("FAST") == "1"; stem = ("profast_" if fast else "prof_") + name
    tmp = f"{stem}.avi"; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720)); rng = np.random.default_rng(3)
    for i in range(60):
        s = 1.0 + 0.06 * np.sin(i / 19.0); dx = (160 * np.sin(i / 4.0)) if fast else 40 * np.sin(i / 11.0); dy = 12 * np.cos(i / 13.0)
        A = np.float32([[s, 0, 640 - 350 * s + dx], [0, s, 360 - 350 * s + dy]])
        f = cv2.warpAffine(big, A, (1280, 720), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if fast:   # motion blur along the sweep, as long as the move in this frame
            L = int(abs(40 * np.cos(i / 4.0))) | 1
            if L > 1: f = cv2.filter2D(f, -1, np.ones((1, L), np.float32) / L)
        f = np.clip(f.astype(np.int16) + rng.integers(-2, 3, f.shape, dtype=np.int16), 0, 255).astype(np.uint8)
        vw.write(f)
    vw.release()
    subprocess.run(["/tmp/claude-0/bin/ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", f"{stem}.mp4"], check=True); os.remove(tmp)
    print(name, "ok")
