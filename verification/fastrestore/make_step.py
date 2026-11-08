"""v11.2.176: a fast step down stairs - the case the user reported (swap lost for a few frames in one rapid step). Two cut-out faces of
insightface's sample photo (as make_kiss.py, 1.8x), NOT user footage. A walks down: three steps, each a 90 px drop in 3 frames
(30 px a frame) with vertical motion blur, between them a slow drift and a small bob; B stands to the right with a slow sway.
Writes kiss_<name>.mp4 and kiss_<name>_gt.npz (a: each face's visible mask per frame, kps: true 5 points).
usage: make_step.py [name speed_px_per_frame scale steps]   (default: step 30 1.8 25,55,85; the hard one: step2 60 2.3 30,75)"""
import cv2, numpy as np, subprocess, os, sys
import make_kiss as K
def vblur(img, L):
    if L < 2:
        return img
    L = int(L) | 1; k = np.zeros((L, L), np.float32); k[:, L // 2] = 1.0 / L
    return cv2.filter2D(img, -1, k)
NAME = sys.argv[1] if len(sys.argv) > 1 else "step"; SPEED = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
SC = float(sys.argv[3]) if len(sys.argv) > 3 else 1.8
PK = K.photo_kps(); ca, ma, cna, ia = K.cutout("f2", scale=SC); cb, mb, cnb, ib = K.cutout("f3", scale=SC)
bg = cv2.resize(cv2.GaussianBlur(K.src[100:560, 300:900], (0, 0), 25), (1280, 720))
n = 120; steps = tuple(int(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 else (25, 55, 85)
ya = 210.0 if SPEED <= 30 else 170.0; gt, gtk = [], []
tmp = f"kiss_{NAME}.avi"; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720)); rng = np.random.default_rng(5)
def kp_frame(Kp, info, cn, px, py):
    bx1, by1, sc = info; kc = (Kp - np.array([bx1, by1])) * sc
    return kc - np.array(cn) + np.array([px, py])
for i in range(n):
    v = SPEED if any(s <= i < s + 3 for s in steps) else 0.0
    ya += v
    xa = 420.0 + 0.8 * i; yb = 330.0 + 6.0 * np.sin(i / 10.0); xb = 900.0 + 10.0 * np.sin(i / 14.0)
    bob = 4.0 * np.sin(i * 2 * np.pi / 15.0) if v == 0 else 0.0
    canvas = bg.astype(np.float32).copy()
    wa = K.place(canvas, vblur(ca, v), vblur(ma, v), cna, xa, ya + bob, 0.0)
    wb = K.place(canvas, cb, mb, cnb, xb, yb, 0.0)
    gtk.append(np.stack([kp_frame(PK["f2"], ia, cna, xa, ya + bob), kp_frame(PK["f3"], ib, cnb, xb, yb)]))
    vw.write(np.clip(canvas + rng.integers(-1, 2, canvas.shape).astype(np.float32), 0, 255).astype(np.uint8))
    gt.append(np.stack([(wa * (1 - wb) * 255).astype(np.uint8), (wb * 255).astype(np.uint8)]))
vw.release()
subprocess.run(["/tmp/claude-0/bin/ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", f"kiss_{NAME}.mp4"], check=True); os.remove(tmp)
np.savez_compressed(f"kiss_{NAME}_gt.npz", a=np.array(gt), kps=np.array(gtk)); print(NAME, os.path.getsize(f"kiss_{NAME}.mp4"), "A y end", ya)
