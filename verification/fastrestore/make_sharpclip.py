"""Sharp footage for the restorer test: the 3DDFA repo's sample photo (emma_input.jpg, public test image, face ~490 px at
its native resolution - nothing upscaled), a 1280x720 window around the largest face, whole-pixel camera drift (no
resampling blur) and light sensor noise, 60 frames. Saves the true 5 key points per frame (turn_sharp_gt.npz) for the
metrics.  NOT user footage."""
import cv2, numpy as np, subprocess, os
from insightface.app import FaceAnalysis
im = cv2.imread("/tmp/claude-0/ddfa/samples/emma_input.jpg")
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
f = max(fa.get(im), key=lambda q: q.bbox[2] - q.bbox[0]); cx, cy = (f.bbox[0] + f.bbox[2]) / 2, (f.bbox[1] + f.bbox[3]) / 2
x0, y0 = int(cx - 640), int(cy - 360)
tmp = "turn_sharp.avi"; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720)); rng = np.random.default_rng(4); kps = []
for i in range(60):
    dx, dy = int(round(30 * np.sin(i / 9.0))), int(round(10 * np.cos(i / 11.0)))
    fr = im[y0 + dy:y0 + dy + 720, x0 + dx:x0 + dx + 1280].copy()
    fr = np.clip(fr.astype(np.int16) + rng.integers(-2, 3, fr.shape, dtype=np.int16), 0, 255).astype(np.uint8)
    vw.write(fr); kps.append(f.kps - np.float32([x0 + dx, y0 + dy]))
vw.release()
subprocess.run(["/tmp/claude-0/bin/ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "turn_sharp.mp4"], check=True); os.remove(tmp)
np.savez_compressed("turn_sharp_gt.npz", kps=np.array(kps)); print("face", int(f.bbox[2] - f.bbox[0]), "px")
