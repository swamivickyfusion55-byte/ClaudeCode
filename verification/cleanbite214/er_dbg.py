"""er_dbg.py PACK OUT.png CLIP:frames : the erase's pieces on a (profile) frame: aligned crop, XSeg face probability, eroded hull, the 'object' pixels, core box."""
import sys, types, numpy as np, cv2
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception; sys.modules["gradio"] = g_
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: ""; sys.modules["huggingface_hub"] = hf
PACK, OUT = sys.argv[1:3]; sys.path.insert(0, PACK)
import logging; logging.disable(logging.CRITICAL)
import swap_engine as E
assert E.occ_net_load("/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/face/models/xseg_2.onnx", threads=2)
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
rows = []
for spec in sys.argv[3:]:
    clip, _, frs = spec.partition(":"); frs = [int(v) for v in frs.split(",")]
    cap = cv2.VideoCapture(clip)
    for i in frs:
        cap.set(1, i); ok, f = cap.read()
        face = max(fa.get(f), key=lambda q: q.det_score)
        M = E.estimate_norm(face.kps, 128)
        al = cv2.warpAffine(f, M, (128, 128), borderMode=cv2.BORDER_REPLICATE)
        W = 256
        m = E.occ_net_raw(al); mb = cv2.GaussianBlur(m, (0, 0), 3.0)
        pts = E._landmarks_to_aligned(face.landmark_2d_106, M, 128) * (W / 128.0)
        hull = np.zeros((W, W), np.uint8); cv2.fillConvexPoly(hull, cv2.convexHull(np.round(pts).astype(np.int32)), 1)
        e = int(round(0.08 * W)) | 1
        hull_e = cv2.erode(hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e)))
        obj = (mb < 0.05) & (hull_e > 0)
        big = cv2.resize(al, (W, W), interpolation=cv2.INTER_CUBIC)
        pm = cv2.cvtColor((np.clip(mb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        ov = big.copy(); ov[obj] = (0.4 * ov[obj] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        cnts, _ = cv2.findContours(hull_e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(ov, cnts, -1, (0, 255, 0), 1)
        cv2.rectangle(ov, (int(W * (0.5 - 0.17)), int(W * 0.58)), (int(W * (0.5 + 0.17)), W - 1), (255, 255, 0), 1)
        for p in pts: cv2.circle(big, (int(p[0]), int(p[1])), 1, (0, 255, 0), -1)
        print(clip.split("/")[-1][:24], i, "yaw offset %.2f" % E._yaw_offset(face.kps), "hull_e px %d obj px %d" % (int(hull_e.sum()), int(obj.sum())),
              "p inside hull_e: median %.2f min %.2f" % (float(np.median(mb[hull_e > 0])), float(mb[hull_e > 0].min())))
        rows.append(np.hstack([big, pm, ov]))
cv2.imwrite(OUT, np.vstack(rows))
