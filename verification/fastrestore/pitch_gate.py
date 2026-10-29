"""usage: pitch_gate.py PACK   The pose gate (_pose_confidence) on public photos with known head pitch (AFLW2000 samples,
mirrored too): its three signals (nose below the eyes 'vert', eye line down the box 't', landmark fit) and the final
confidence. Below 0.5 a fresh detection is not accepted; a track already painting keeps it down to 0.10."""
import sys, glob, types, cv2, numpy as np, scipy.io as sio
PACK = sys.argv[1]; sys.path.insert(0, PACK)
g = types.ModuleType("gradio"); g.Request = type("Request", (), {}); g.update = lambda **k: k; g.Error = Exception; sys.modules["gradio"] = g
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: None; sys.modules["huggingface_hub"] = hf
import core_pipeline as cp, swap_engine as E
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection"]); fa.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.28)
def parts(f):
    k = np.asarray(f.kps, np.float32); le, re, nose = k[0], k[1], k[2]; ev = re - le; ed = float(np.linalg.norm(ev)) + 1e-6
    roll = float(np.arctan2(ev[1], ev[0])); c, sn = np.cos(-roll), np.sin(-roll); mid = (le + re) / 2; rel = nose - mid
    vert = (rel[0] * sn + rel[1] * c) / ed
    x1, y1, x2, y2 = [float(v) for v in f.bbox]; down = np.array([sn, c]); down /= np.linalg.norm(down)
    pr = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]]) @ down; t = (float(mid @ down) - pr.min()) / (pr.max() - pr.min() + 1e-6)
    return vert, t, E.landmark_fit_error(k, 128), ed / min(x2 - x1, y2 - y1)
rows = []
for fn in sorted(glob.glob("/tmp/claude-0/prnet/TestImages/AFLW2000/*.jpg")):
    m = sio.loadmat(fn[:-4] + ".mat"); pitch, yaw = [float(v) * 180 / np.pi for v in m["Pose_Para"][0][:2]]
    for flip in (0, 1):
        img = cv2.imread(fn); img = img[:, ::-1].copy() if flip else img
        fs = fa.get(img)
        if not fs: rows.append((pitch, yaw, None)); continue
        f = max(fs, key=lambda q: (q.bbox[2] - q.bbox[0]) * (q.bbox[3] - q.bbox[1]))
        rows.append((pitch, yaw, parts(f) + (cp._pose_confidence(f),)))
print("pitch  yaw | vert (ramp 0.06-0.18)  t (0.58-0.72)  fit (0.20-0.44)  eye/box | confidence")
for p, y, r in sorted(rows, key=lambda r: r[0]):
    print(f"{p:5.0f} {y:5.0f} | " + ("no face" if r is None else f"{r[0]:5.2f}  {r[1]:5.2f}  {r[2] if r[2] is None else round(r[2], 2)}  {r[3]:5.2f} | {r[4]:.2f}"))
