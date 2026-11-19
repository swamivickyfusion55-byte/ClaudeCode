"""ideal.py CLIP(with bar) NONE_RESULT OUT.mp4 : the ideal for a bar clip - the swap of the bar-free frame (the no-bar run's output) with
the true bar (ground-truth alpha, the clip's own pixels) pasted on top. Only valid for clips made with the same camera path as the no-bar clip."""
import sys, cv2, numpy as np
clip, none, out = sys.argv[1:4]
g = np.load(clip[:-4] + "_gt.npz"); A = g["a"]
c1, c2 = cv2.VideoCapture(clip), cv2.VideoCapture(none); vw = cv2.VideoWriter(out + ".avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720)); t = 0
while True:
    o1, a = c1.read(); o2, b = c2.read()
    if not (o1 and o2): break
    al = cv2.resize(A[t], (1280, 720), interpolation=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255.0
    al = np.clip(al * 1.0, 0, 1)
    vw.write(np.clip(b.astype(np.float32) * (1 - al) + a.astype(np.float32) * al, 0, 255).astype(np.uint8)); t += 1
vw.release()
import subprocess; subprocess.run(["/tmp/claude-0/bin/ffmpeg", "-y", "-loglevel", "error", "-i", out + ".avi", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", out], check=True)
import os; os.remove(out + ".avi")
