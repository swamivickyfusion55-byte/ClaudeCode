"""v11.2.141: identity mix-up clips - two people whose faces meet and who END ON THE OTHER SIDE, so a tracker that swaps them is caught.
Same faces and build as make_kiss.py (insightface's sample photo, cut-outs, B in front of A where they overlap). NOT user footage.
  cross   A walks left->right, B right->left, B in front: A is hidden behind B for about 12 frames, then both carry on
  hug     cheek to cheek, the pair turns half a circle about its middle (B stays in front), ends with the sides swapped
Writes kiss_<name>.mp4 and kiss_<name>_gt.npz (a: each face's visible mask per frame, kps: each face's 5 points per frame).
usage: make_cross.py [cross|hug ...]   (run from the output folder)"""
import os, sys, subprocess, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_kiss as K

def build_path(name, pos, n=120, a="f2", b="f3"):
    """pos(i) -> (xa, ya, ang_a, xb, yb, ang_b): each face's centre and tilt on frame i."""
    PK = K.photo_kps(); ca, ma, cna, ia = K.cutout(a); cb, mb, cnb, ib = K.cutout(b)
    bg = cv2.resize(cv2.GaussianBlur(K.src[100:560, 300:900], (0, 0), 25), (1280, 720))
    tmp = f"kiss_{name}.avi"; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720)); gt, gtk = [], []
    rng = np.random.default_rng(11)
    def kp_frame(Kp, info, cn, px, py, an):
        bx1, by1, sc = info; kc = (Kp - np.array([bx1, by1])) * sc
        R = cv2.getRotationMatrix2D(cn, an, 1.0)[:, :2]; return (kc - np.array(cn)) @ R.T + np.array([px, py])
    for i in range(n):
        xa, ya, aa, xb, yb, ab = pos(i)
        canvas = bg.astype(np.float32).copy()
        wa = K.place(canvas, ca, ma, cna, xa, ya, aa)
        wb = K.place(canvas, cb, mb, cnb, xb, yb, ab)
        gtk.append(np.stack([kp_frame(PK[a], ia, cna, xa, ya, aa), kp_frame(PK[b], ib, cnb, xb, yb, ab)]))
        f = np.clip(canvas + rng.integers(-1, 2, canvas.shape).astype(np.float32), 0, 255).astype(np.uint8); vw.write(f)
        gt.append(np.stack([(wa * (1 - wb) * 255).astype(np.uint8), (wb * 255).astype(np.uint8)]))
    vw.release()
    subprocess.run(["/tmp/claude-0/bin/ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                    f"kiss_{name}.mp4"], check=True); os.remove(tmp)
    np.savez_compressed(f"kiss_{name}_gt.npz", a=np.array(gt), kps=np.array(gtk)); print(name, os.path.getsize(f"kiss_{name}.mp4"))

def cross(i, n=120):
    t = i / (n - 1.0)                                   # A: 300 -> 980, B: 980 -> 300 (they pass at frame ~60)
    return 300 + 680 * t, 370, 0.0, 980 - 680 * t, 350, 0.0

def hug(i, n=120):
    # 20 frames coming together, 80 frames turning half a circle about the pair's middle (radius 125 px: cheek to cheek), 20 apart
    if i < 20:
        r, th = 125 + 140 * (1 - i / 20.0), 0.0
    elif i < 100:
        r, th = 125.0, np.pi * (i - 20) / 80.0
    else:
        r, th = 125 + 140 * ((i - 100) / 20.0), np.pi
    cx, cy = 640.0, 360.0
    xa, ya = cx - r * np.cos(th), cy + 0.35 * r * np.sin(th)
    xb, yb = cx + r * np.cos(th), cy - 0.35 * r * np.sin(th)
    return xa, ya, 8.0 * np.cos(th), xb, yb, -8.0 * np.cos(th)

if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["cross", "hug"]):
        build_path(nm, {"cross": cross, "hug": hug}[nm])
