"""One face of insightface's sample photo (3x cut-out, soft elliptical mask) that sweeps, rolls and turns FAST over a blurred background.
NOT user footage. Writes turn_<name>.mp4 + turn_<name>_gt.npz (true 5-point key points per frame, inter-eye unit)."""
import cv2, numpy as np, subprocess, os, sys
exec(open('make_kiss.py').read().split('def build')[0])
def build(name, face="f2", n=120, sweep=220.0, period=20.0, roll=0.0, rollp=24.0, yaw=0.0, yawp=36.0, motionblur=0):
    PK = photo_kps()[face]; crop, m, cn, info = cutout(face); bx1, by1, sc = info
    bg = cv2.resize(cv2.GaussianBlur(src[100:560, 300:900], (0, 0), 25), (1280, 720))
    tmp = f"turn_{name}.avi"; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720)); gtk = []; rng = np.random.default_rng(3)
    kc0 = (PK - np.array([bx1, by1])) * sc
    for i in range(n):
        act = 1.0 if i >= 20 else 0.0                                   # 20 still frames (the tracker warms up), then the motion
        ph = (i - 20)
        px = 640 + act * sweep * np.sin(2 * np.pi * ph / period)
        an = act * roll * np.sin(2 * np.pi * ph / rollp)
        sx = 1.0 - act * yaw * abs(np.sin(2 * np.pi * ph / yawp))
        R = cv2.getRotationMatrix2D((0, 0), an, 1.0)[:, :2] @ np.diag([sx, 1.0])
        M = np.zeros((2, 3), np.float32); M[:, :2] = R; M[:, 2] = np.array([px, 360]) - R @ np.array(cn)
        wc = cv2.warpAffine(crop, M, (1280, 720), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        wm = cv2.warpAffine(m, M, (1280, 720), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        canvas = bg.astype(np.float32) * (1 - wm[:, :, None]) + wc.astype(np.float32) * wm[:, :, None]
        gtk.append((kc0 @ R.T) + M[:, 2])
        if motionblur and act:
            vx = abs(2 * np.pi * sweep / period * np.cos(2 * np.pi * ph / period)); k = int(min(motionblur, max(1, vx * 0.5))) | 1
            if k > 1:
                ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0 / k; canvas = cv2.filter2D(canvas, -1, ker)
        f = np.clip(canvas + rng.integers(-1, 2, canvas.shape).astype(np.float32), 0, 255).astype(np.uint8); vw.write(f)
    vw.release()
    subprocess.run(["/tmp/claude-0/bin/ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", f"turn_{name}.mp4"], check=True); os.remove(tmp)
    np.savez_compressed(f"turn_{name}_gt.npz", kps=np.array(gtk)); print(name, os.path.getsize(f"turn_{name}.mp4"))
if __name__ == "__main__":
    build("sweep", sweep=220.0, period=24.0, motionblur=15)                       # peak ~58 px/frame
    build("mix", sweep=180.0, period=26.0, roll=22.0, rollp=18.0, yaw=0.35, yawp=30.0, motionblur=11)
    build("calm", sweep=40.0, period=60.0, roll=4.0, rollp=50.0)                   # reference: slow
