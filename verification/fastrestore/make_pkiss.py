"""v11.2.144: a kiss IN PROFILE - the case the frontal contact clips (make_kiss.py, make_cross.py) never exercise. Two public AFLW2000
sample photos (from the PRNet repository's TestImages; no user footage): A = image00251 (turned 55 deg, facing right) on the left,
B = image02420 (turned 71 deg, facing left) on the right, B in front where the heads overlap. Each head is cut out along its own silhouette
(1.5 x, soft edge) on a blurred background. 120 frames at 30 fps:
  0-25    the heads come together until the lips meet (B's lips over A's by a few px)
  25-100  the kiss: lips pressed together with a small rhythmic push, the couple sways together (up to 45 px sideways, 15 px up and
          down) and both heads tilt together (up to 4 deg)
  100-120 they part
Writes kiss_<name>.mp4 and kiss_<name>_gt.npz: a (each person's visible mask per frame), kps (each person's true 5 points per frame,
from the photo's 68 landmarks: eye centres, nose tip, mouth corners) and region (each person's face outline per frame: the 68 points
and a forehead estimate - identity_check.py uses it in place of its frontal photo's outline).
usage: make_pkiss.py AFLW2000_DIR [name]   (run from the output folder)"""
import os, sys, subprocess, cv2, numpy as np, scipy.io as sio
D = sys.argv[1]; NAME = sys.argv[2] if len(sys.argv) > 2 else "pkiss"
SC = 1.5
def load(name):
    im = cv2.imread(os.path.join(D, name + ".jpg")); pt = sio.loadmat(os.path.join(D, name + ".mat"))["pt3d_68"][:2].T.astype(np.float64)
    k5 = np.array([pt[36:42].mean(0), pt[42:48].mean(0), pt[30], pt[48], pt[54]])
    mouth = pt[48:68].mean(0)
    x0, y0 = pt.min(0); x1, y1 = pt.max(0); w, h = x1 - x0, y1 - y0
    top = pt[17:27].copy(); top[:, 1] -= 0.30 * h                          # a forehead estimate above the brows
    region = np.vstack([pt, top])
    crop = cv2.resize(im, None, fx=SC, fy=SC, interpolation=cv2.INTER_CUBIC)
    H, W = crop.shape[:2]
    # the head's own silhouette, not an ellipse (an ellipse keeps the photo's background in front of a profile, which then covers
    # the other face): the 68 points' hull - its front edge runs along the nose, lips and chin - with the forehead and the back of
    # the head (the side away from the nose, 0.45 of the face width further, and up to the crown) added
    back = -1.0 if pt[30, 0] > (x0 + x1) / 2 else 1.0
    crown = np.array([[(x0 + x1) / 2 + back * 0.15 * w, y0 - 0.55 * h], [(x0 + x1) / 2 + back * 0.55 * w, y0 - 0.25 * h],
                      [(x0 + x1) / 2 + back * 0.75 * w, y0 + 0.35 * h], [(x0 + x1) / 2 + back * 0.55 * w, y1 + 0.05 * h]])
    hull = cv2.convexHull(np.round(SC * np.vstack([pt, top, crown])).astype(np.int32))
    m = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(m, hull, 255)
    m = cv2.GaussianBlur(cv2.erode(m, np.ones((5, 5), np.uint8)).astype(np.float32) / 255.0, (0, 0), 3.0)
    return crop, m, SC * mouth, SC * k5, SC * region
def place(canvas, crop, m, c, px, py, ang):
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), ang, 1.0); M[0, 2] += px - c[0]; M[1, 2] += py - c[1]
    wc = cv2.warpAffine(crop, M, (1280, 720), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    wm = cv2.warpAffine(m, M, (1280, 720), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    canvas[:] = canvas * (1 - wm[:, :, None]) + wc.astype(np.float32) * wm[:, :, None]
    return wm, M
def path(i, n):
    if i < 25:
        gap, u = 320.0 * (1.0 - i / 25.0) - 12.0 * (i / 25.0), 0
    elif i < 100:
        u = i - 25; gap = -12.0 - 7.0 * np.sin(u * 2 * np.pi / 18.0)
    else:
        gap, u = -12.0 + 332.0 * ((i - 100) / 20.0), 75
    cx = 640.0 + 45.0 * np.sin(u * 2 * np.pi / 55.0)
    cy = 400.0 + 15.0 * np.sin(u * 2 * np.pi / 40.0)
    tilt = 4.0 * np.sin(u * 2 * np.pi / 60.0)
    return gap, cx, cy, tilt
if __name__ == "__main__":
    ca, ma, mca, ka, ra = load("image00251"); cb, mb, mcb, kb, rb = load("image02420")
    bgsrc = cv2.imread("/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg")
    bg = cv2.resize(cv2.GaussianBlur(bgsrc[100:560, 300:900], (0, 0), 25), (1280, 720))
    tmp = f"kiss_{NAME}.avi"; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720))
    rng = np.random.default_rng(13); n = 120; gt, gtk, gtr = [], [], []
    T = lambda M, P: P @ M[:, :2].T + M[:, 2]
    for i in range(n):
        gap, cx, cy, tilt = path(i, n)
        canvas = bg.astype(np.float32).copy()
        wa, Ma = place(canvas, ca, ma, mca, cx - gap / 2.0, cy, tilt)
        wb, Mb = place(canvas, cb, mb, mcb, cx + gap / 2.0, cy, tilt)
        gtk.append(np.stack([T(Ma, ka), T(Mb, kb)])); gtr.append(np.stack([T(Ma, ra), T(Mb, rb)]))
        f = np.clip(canvas + rng.integers(-1, 2, canvas.shape).astype(np.float32), 0, 255).astype(np.uint8); vw.write(f)
        gt.append(np.stack([(wa * (1 - wb) * 255).astype(np.uint8), (wb * 255).astype(np.uint8)]))
    vw.release()
    subprocess.run(["/tmp/claude-0/bin/ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                    f"kiss_{NAME}.mp4"], check=True); os.remove(tmp)
    np.savez_compressed(f"kiss_{NAME}_gt.npz", a=np.array(gt), kps=np.array(gtk), region=np.array(gtr))
    print(NAME, os.path.getsize(f"kiss_{NAME}.mp4"))
