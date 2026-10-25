"""Calibrate a yaw proxy from the detector's 5 key points against AFLW2000's true yaw (public photos, mirrored too)."""
import glob, cv2, numpy as np, scipy.io as sio
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection"]); fa.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.28)
def proxy(k):
    le, re, nose = k[0], k[1], k[2]; ev = re - le; ed = np.linalg.norm(ev) + 1e-6
    roll = np.arctan2(ev[1], ev[0]); c, s = np.cos(-roll), np.sin(-roll); rel = nose - (le + re) / 2
    h = (rel[0] * c - rel[1] * s) / ed                      # nose sideways from the eye midpoint, in eye distances
    ml, mr = k[3], k[4]; md = np.linalg.norm(mr - ml) + 1e-6
    return h, ed / md
for f in sorted(glob.glob("/tmp/claude-0/prnet/TestImages/AFLW2000/*.jpg")):
    yaw = float(sio.loadmat(f[:-4] + ".mat")["Pose_Para"][0][1] * 180 / np.pi)
    for flip in (0, 1):
        img = cv2.imread(f); img = img[:, ::-1].copy() if flip else img
        fs = fa.get(img)
        if not fs: print(f[-13:-4], "none"); continue
        q = max(fs, key=lambda q: (q.bbox[2] - q.bbox[0]) * (q.bbox[3] - q.bbox[1]))
        h, r = proxy(q.kps)
        print("%s%s yaw %4.0f  h %+.2f  eye/mouth %.2f" % (f[-13:-4], "m" if flip else " ", -yaw if flip else yaw, h, r))
