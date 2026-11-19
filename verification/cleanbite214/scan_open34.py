"""scan_open34.py: AFLW2000 sample photos with a head turned 30-65 deg (annotated pose) and the mouth open (insightface 106 landmarks: inner mouth height / width)"""
import os, sys, glob, numpy as np, cv2, scipy.io as sio
from insightface.app import FaceAnalysis
D = "/tmp/claude-0/prnet/TestImages/AFLW2000"
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
INNER = [65, 66, 62, 70, 69, 57, 60, 54]
out = []
for mp in sorted(glob.glob(D + "/*.mat")):
    try: pose = sio.loadmat(mp)["Pose_Para"][0][:3]
    except Exception: continue
    yaw = abs(np.degrees(pose[1]))
    if not (28 <= yaw <= 68): continue
    img = cv2.imread(mp[:-4] + ".jpg")
    if img is None: continue
    fs = fa.get(img)
    if not fs: continue
    f = max(fs, key=lambda q: q.det_score)
    L = f.landmark_2d_106
    ip = L[INNER]; h = ip[:, 1].max() - ip[:, 1].min(); w = ip[:, 0].max() - ip[:, 0].min()
    fw = f.bbox[2] - f.bbox[0]
    out.append((os.path.basename(mp)[:-4], round(float(yaw)), round(float(h / max(w, 1)), 2), int(fw), round(float(f.det_score), 2)))
out.sort(key=lambda t: -t[2])
for t in out[:25]: print(t)
