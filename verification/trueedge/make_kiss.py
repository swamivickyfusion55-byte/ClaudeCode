"""Two people whose faces come together (cheek to cheek, then a kiss-like hold with a small rhythmic push) and part again. Cut-outs of two
faces of insightface's sample photo, 3x enlarged, soft elliptical masks, B in front of A where they overlap. NOT user footage.
Writes kiss_<name>.mp4 and the ground truth (each face's visible mask per frame)."""
import cv2, numpy as np, subprocess, os, sys
FACE = {"f2": (466, 268, 573, 415), "f3": (745, 338, 845, 479), "f4": (903, 62, 1013, 205), "f1": (268, 145, 372, 267)}
src = cv2.imread("/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg")
def photo_kps():
    from insightface.app import FaceAnalysis
    fa = FaceAnalysis(name='buffalo_l', root='/root/.insightface', allowed_modules=['detection'], providers=['CPUExecutionProvider']); fa.prepare(ctx_id=-1, det_size=(640, 640))
    fs = fa.get(src); out = {}
    for nm, (x1, y1, x2, y2) in FACE.items():
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        best = min(fs, key=lambda f: (((f.bbox[0] + f.bbox[2]) / 2 - cx) ** 2 + ((f.bbox[1] + f.bbox[3]) / 2 - cy) ** 2))
        out[nm] = best.kps.copy()
    return out
def cutout(name, pad=0.55, scale=3.0):
    x1, y1, x2, y2 = FACE[name]; w, h = x2 - x1, y2 - y1
    bx1, by1, bx2, by2 = max(0, int(x1 - pad * w)), max(0, int(y1 - pad * h)), min(src.shape[1], int(x2 + pad * w)), min(src.shape[0], int(y2 + pad * h * 0.6))
    crop = cv2.resize(src[by1:by2, bx1:bx2], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    H, W = crop.shape[:2]; yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = ((x1 + x2) / 2 - bx1) * scale, ((y1 + y2) / 2 - by1) * scale
    rx, ry = 0.78 * w * scale, 0.85 * h * scale
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    m = np.clip((1.0 - d) / 0.16, 0, 1).astype(np.float32)
    return crop, m, (cx, cy), (bx1, by1, scale)
def place(canvas, crop, m, c, px, py, ang, sc=1.0):
    H, W = crop.shape[:2]; M = cv2.getRotationMatrix2D(c, ang, sc); M[0, 2] += px - c[0]; M[1, 2] += py - c[1]
    wc = cv2.warpAffine(crop, M, (1280, 720), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    wm = cv2.warpAffine(m, M, (1280, 720), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    canvas[:] = canvas * (1 - wm[:, :, None]) + wc.astype(np.float32) * wm[:, :, None]
    return wm
def build(name, a="f2", b="f3", n=120, gap_end=200.0, rot=10.0, pulse=0.0, bscale=1.0):
    PK = photo_kps(); ca, ma, cna, ia = cutout(a); cb, mb, cnb, ib = cutout(b); gtk = []
    bg = cv2.resize(cv2.GaussianBlur(src[100:560, 300:900], (0, 0), 25), (1280, 720))
    tmp = f"kiss_{name}.avi"; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720)); gt = []
    rng = np.random.default_rng(9)
    for i in range(n):
        # approach 0-40, contact hold 40-80 (rhythmic push), part 80-120
        if i < 40: t = i / 40.0
        elif i < 80: t = 1.0
        else: t = 1.0 - (i - 80) / 40.0
        push = 7.0 * np.sin((i - 40) * 2 * np.pi / 15.0) if 40 <= i < 80 else 0.0
        sep = 520.0 + (gap_end - 520.0) * t - push                # distance between the centres (px), 520 -> gap_end
        ang = rot * t
        canvas = bg.astype(np.float32).copy()
        pa = 1.0 + pulse * np.sin(i * 2 * np.pi / 6.0); pb = bscale * (1.0 - pulse * np.sin(i * 2 * np.pi / 6.0))
        wa = place(canvas, ca, ma, cna, 640 - sep / 2, 360, ang, pa)
        wb = place(canvas, cb, mb, cnb, 640 + sep / 2, 360, -ang, pb)
        def kp_frame(K, info, cn, px, py, an):
            bx1, by1, sc = info; kc = (K - np.array([bx1, by1])) * sc
            R = cv2.getRotationMatrix2D(cn, an, 1.0)[:, :2]; return (kc - np.array(cn)) @ R.T + np.array([px, py])
        gtk.append(np.stack([kp_frame(PK[a], ia, cna, 640 - sep / 2, 360, ang), kp_frame(PK[b], ib, cnb, 640 + sep / 2, 360, -ang)]))
        f = np.clip(canvas + rng.integers(-1, 2, canvas.shape).astype(np.float32), 0, 255).astype(np.uint8); vw.write(f)
        gt.append(np.stack([(wa * (1 - wb) * 255).astype(np.uint8), (wb * 255).astype(np.uint8)]))
    vw.release()
    subprocess.run(["/tmp/claude-0/bin/ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", f"kiss_{name}.mp4"], check=True); os.remove(tmp)
    np.savez_compressed(f"kiss_{name}_gt.npz", a=np.array(gt), kps=np.array(gtk)); print(name, os.path.getsize(f"kiss_{name}.mp4"))
if __name__ == "__main__":
    build("cheek", gap_end=250.0)      # 330-px faces, centres 250 px apart = about 25 % overlap
    build("near", gap_end=300.0)       # centres 300 px apart: faces touching, no overlap
    build("deep", gap_end=165.0, rot=24.0)
    build("flip", gap_end=210.0, rot=14.0, pulse=0.045, bscale=1.11)   # equal-sized faces whose apparent sizes swap order every few frames (detector size jitter), B truly in front
    build("hide", gap_end=105.0, rot=32.0)   # centres 105 px apart: most of A is behind B, A is only a sliver and a tilted eye for 40 frames   # centres 165 px apart (about half of A is behind B), heads tilted toward each other like a kiss
