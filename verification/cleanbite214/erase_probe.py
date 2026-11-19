"""erase_probe.py PACK OUT.png CLIP[:frames] ... : the 11.2.214 erase on real frames - which pixels the swap network is NOT shown (a frame-space mask from the pack's
own _erase_objects: pixels that changed), against the ground-truth bar and opening where the clip has them."""
import sys, types, numpy as np, cv2
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception; sys.modules["gradio"] = g_
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: ""; sys.modules["huggingface_hub"] = hf
PACK, OUT = sys.argv[1:3]; sys.path.insert(0, PACK)
import logging; logging.disable(logging.CRITICAL)
import swap_engine as E
assert E.occ_net_load("/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/face/models/xseg_2.onnx", threads=2)
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
import os
if os.environ.get('ERASE_YAW'): E._P['obj_erase_yaw'] = float(os.environ['ERASE_YAW'])
comp = E.AlignedCompositor(None)
tiles = []
for spec in sys.argv[3:]:
    clip, _, frs = spec.partition(":"); frs = [int(v) for v in frs.split(",")] if frs else [60]
    gt = None
    try: gt = np.load(clip[:-4] + "_gt.npz")
    except Exception: pass
    cap = cv2.VideoCapture(clip)
    for i in frs:
        cap.set(1, i); ok, f = cap.read()
        face = max(fa.get(f), key=lambda q: q.det_score)
        out = comp._erase_objects(f, f, face, None, None, True)
        chg = (np.abs(out.astype(np.int16) - f.astype(np.int16)).max(2) > 0)
        line = "%s f%d: erased %d px" % (clip.split("/")[-1][:20], i, int(chg.sum()))
        if gt is not None and "a" in gt.files:
            bar = cv2.resize(gt["a"][i], (1280, 720), interpolation=cv2.INTER_LINEAR) > 200
            hl = None
            if "open" in gt.files: opn = cv2.resize(gt["open"][i], (1280, 720), interpolation=cv2.INTER_LINEAR) > 100
            else: opn = None
            M = E.estimate_norm(face.kps, 128); IM = cv2.invertAffineTransform(M)
            # the erase can only reach inside the face outline: judge the bar inside the landmark hull dilated a little
            hull = np.zeros((720, 1280), np.uint8); cv2.fillConvexPoly(hull, cv2.convexHull(face.landmark_2d_106.astype(np.int32)), 1)
            inh = bar & (hull > 0)
            cov = float((chg & inh).sum()) / max(1, int(inh.sum()))
            line += " | bar inside the face outline %d px: covered %.2f" % (int(inh.sum()), cov)
            if opn is not None:
                cav = opn & ~(cv2.dilate(bar.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
                line += " | cavity not under the bar %d px: erased %.3f" % (int(cav.sum()), float((chg & cav).sum()) / max(1, int(cav.sum())))
        print(line)
        v = f.copy(); v[chg] = (0.35 * v[chg] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
        x0, y0, x1, y1 = [int(a) for a in (face.bbox[0] - 30, face.bbox[1] + 100, face.bbox[2] + 130, face.bbox[3] + 60)]
        tiles.append(cv2.resize(np.hstack([f[max(0, y0):y1, max(0, x0):x1], v[max(0, y0):y1, max(0, x0):x1], out[max(0, y0):y1, max(0, x0):x1]]), None, fx=0.9, fy=0.9))
Wm = max(t.shape[1] for t in tiles[:6]); tiles = [np.pad(t, ((0, 0), (0, Wm - t.shape[1]), (0, 0))) for t in tiles[:6]]
cv2.imwrite(OUT, np.vstack(tiles))
