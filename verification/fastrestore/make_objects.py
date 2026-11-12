"""Objects passing in front of a face anywhere (not only the mouth): the centre face of insightface's sample photo (public,
3x enlarged, face ~320 px), 80 frames: the object slides in over 4 frames, holds 34 frames with a small jitter, leaves over
4 frames. Objects: 'hand' (skin-coloured, sampled from the face itself, with finger strips and shading - the hardest case),
'cup' (white, shaded), 'phone' (dark rounded rectangle). Places: eye, forehead, cheek, nose, sweep (v11.2.206: moving steadily across the face). Ground truth: object alpha per
frame. NOT user footage.  usage: make_objects.py OBJ_PLACE...   -> obj_<OBJ>_<PLACE>.mp4 + _gt.npz"""
import cv2, numpy as np, subprocess, os, sys
from insightface.app import FaceAnalysis
src = cv2.imread("/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg")
BASE = cv2.resize(src[220:460, 307:733], (1280, 720), interpolation=cv2.INTER_CUBIC)
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
f0 = min(fa.get(BASE), key=lambda f: abs((f.bbox[0] + f.bbox[2]) / 2 - 640)); k = f0.kps; ed = float(np.linalg.norm(k[1] - k[0]))
PLACE = {"eye": (k[0], 0.55), "forehead": ((k[0] + k[1]) / 2 + np.array([0, -0.75 * ed]), 0.6), "cheek": (k[4] + np.array([0.45 * ed, -0.35 * ed]), 0.55), "nose": (k[2], 0.5)}
skin = np.median(BASE[int(k[2][1]) - 10:int(k[2][1]) + 10, int(k[2][0]) + int(0.4 * ed):int(k[2][0]) + int(0.6 * ed)].reshape(-1, 3), axis=0)
def draw(obj, cx, cy, s, rng):
    S = 3; H, W = 720 * S, 1280 * S; a = np.zeros((H, W), np.uint8); col = np.zeros((H, W, 3), np.float32)
    cx, cy, s = cx * S, cy * S, s * S
    if obj == "hand":
        cv2.ellipse(a, (int(cx), int(cy)), (int(s * 0.55), int(s * 0.42)), 15, 0, 360, 255, -1)
        for j in range(4):
            x = cx - s * 0.45 + j * s * 0.3; cv2.ellipse(a, (int(x), int(cy - s * 0.55)), (int(s * 0.11), int(s * 0.38)), 8, 0, 360, 255, -1)
        base = np.array(skin, np.float32) * 0.97
        col[:] = base
    elif obj == "cup":
        cv2.rectangle(a, (int(cx - s * 0.4), int(cy - s * 0.55)), (int(cx + s * 0.4), int(cy + s * 0.55)), 255, -1)
        col[:] = (235, 238, 240)
    else:
        cv2.rectangle(a, (int(cx - s * 0.3), int(cy - s * 0.6)), (int(cx + s * 0.3), int(cy + s * 0.6)), 255, -1)
        col[:] = (40, 38, 45)
    a = cv2.resize(a, (1280, 720), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    col = cv2.resize(col, (1280, 720))
    xx = np.arange(1280, dtype=np.float32)[None, :]; shade = 0.85 + 0.25 * np.clip((xx - cx / S + s / S) / (2 * s / S), 0, 1)
    tex = cv2.GaussianBlur(rng.standard_normal((720, 1280)).astype(np.float32), (0, 0), 1.5) * 6
    col = np.clip(col * shade[:, :, None] + tex[:, :, None], 0, 255)
    return col, a
# v11.2.206: 'sweep' - the object moves steadily across the face at cheek / nose height (frames 20-58), for trails behind it
PLACE["sweep"] = ((k[2] + k[4]) / 2, 0.55)
def build(spec, n=80):
    obj, place = spec.split("_"); (cx0, cy0), sc = PLACE[place]; s = sc * ed * 1.6
    tmp = f"obj_{spec}.avi"; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1280, 720)); gt = []; nr = np.random.default_rng(5)
    for i in range(n):
        f = np.clip(BASE.astype(np.int16) + nr.integers(-1, 2, BASE.shape, dtype=np.int16), 0, 255).astype(np.uint8).astype(np.float32)
        if place == "sweep":
            p = None if (i < 20 or i >= 58) else (cx0 - 1.6 * s + 3.2 * s * (i - 20) / 37.0, cy0)
        elif i < 20 or i >= 62: p = None
        elif i < 24: u = (i - 19) / 4.0; p = (cx0 + 2.2 * s * (1 - u), cy0)
        elif i < 58: p = (cx0 + 0.03 * s * np.sin(i / 3.0), cy0 + 0.02 * s * np.cos(i / 4.0))
        else: u = (i - 57) / 4.0; p = (cx0 + 2.2 * s * u, cy0)
        A = np.zeros((720, 1280), np.float32)
        if p is not None:
            col, A = draw(obj, p[0], p[1], s, np.random.default_rng(7)); f = f * (1 - A[:, :, None]) + col * A[:, :, None]
        vw.write(np.clip(f, 0, 255).astype(np.uint8)); gt.append((A * 255).astype(np.uint8))
    vw.release()
    subprocess.run(["/tmp/claude-0/bin/ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", f"obj_{spec}.mp4"], check=True); os.remove(tmp)
    np.savez_compressed(f"obj_{spec}_gt.npz", a=np.array(gt)); print(spec)
for sp in sys.argv[1:]: build(sp)
