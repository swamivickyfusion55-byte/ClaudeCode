"""Shot change from public photos (AFLW2000 samples, no user footage): frames 0-29 photo A, frames 30-59 photo B, both
on a 1280x720 canvas with the same slow drift; plus B alone for all 60 frames (the reference: what frames 30-59
should look like). Saves the true 5 key points per frame.  -> cut_AB.mp4, cut_Bref.mp4, cut_AB_gt.npz"""
import cv2, numpy as np, subprocess, os
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
def load(n):
    im = cv2.resize(cv2.imread(f"/tmp/claude-0/prnet/TestImages/AFLW2000/{n}.jpg"), (700, 700), interpolation=cv2.INTER_CUBIC)
    f = max(fa.get(im), key=lambda q: q.bbox[2] - q.bbox[0]); return im, f.kps
A, kA = load("image03614"); B, kB = load("image00050")
def frame(im, k, i, ox):
    dx = ox + 25 * np.sin(i / 9.0); dy = 8 * np.cos(i / 11.0)
    M = np.float32([[1, 0, 640 - 350 + dx], [0, 1, 360 - 350 + dy]])
    return cv2.warpAffine(im, M, (1280, 720), borderMode=cv2.BORDER_REFLECT), (k @ M[:, :2].T + M[:, 2])
rng = np.random.default_rng(6)
def write(name, pick):
    tmp = name + ".avi"; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720)); ks = []
    for i in range(60):
        im, k, ox = pick(i)
        f, kk = frame(im, k, i, ox)
        vw.write(np.clip(f.astype(np.int16) + rng.integers(-2, 3, f.shape, dtype=np.int16), 0, 255).astype(np.uint8)); ks.append(kk)
    vw.release()
    subprocess.run(["/tmp/claude-0/bin/ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", name + ".mp4"], check=True); os.remove(tmp)
    return np.array(ks)
kab = write("cut_AB", lambda i: (A, kA, -60) if i < 30 else (B, kB, 60))
kb = write("cut_Bref", lambda i: (B, kB, 60))
np.savez_compressed("cut_AB_gt.npz", kps=kab, kps_ref=kb); print("ok")
