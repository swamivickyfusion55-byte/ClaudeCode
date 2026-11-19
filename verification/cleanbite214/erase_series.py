"""erase_series.py PACK CLIP [ERASE_YAW env]: the 214 erase on EVERY frame of a clip: pixels taken per frame, and (clips with ground truth) the share of the bar inside the
face outline that is covered and the share of the open mouth's cavity (away from the bar) that is touched. Steadiness: frames with the bar in the mouth where nothing
was taken, and the largest frame-to-frame change of the taken area."""
import sys, os, types, numpy as np, cv2
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception; sys.modules["gradio"] = g_
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: ""; sys.modules["huggingface_hub"] = hf
PACK, CLIP = sys.argv[1:3]; sys.path.insert(0, PACK)
import logging; logging.disable(logging.CRITICAL)
import swap_engine as E
assert E.occ_net_load("/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/face/models/xseg_2.onnx", threads=2)
if os.environ.get("ERASE_YAW"): E._P["obj_erase_yaw"] = float(os.environ["ERASE_YAW"])
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
comp = E.AlignedCompositor(None)
gt = np.load(CLIP[:-4] + "_gt.npz") if os.path.exists(CLIP[:-4] + "_gt.npz") else None
cap = cv2.VideoCapture(CLIP); i = 0; px, cov, cav, inbar = [], [], [], []
while True:
    ok, f = cap.read()
    if not ok: break
    fs = fa.get(f)
    if not fs: px.append(np.nan); cov.append(np.nan); cav.append(np.nan); inbar.append(0); i += 1; continue
    face = max(fs, key=lambda q: q.det_score)
    out = comp._erase_objects(f, f, face, None, None, True)
    chg = np.abs(out.astype(np.int16) - f.astype(np.int16)).max(2) > 0
    px.append(int(chg.sum()))
    c = np.nan; v = np.nan; ib = 0
    if gt is not None and "a" in gt.files:
        bar = cv2.resize(gt["a"][i], (1280, 720), interpolation=cv2.INTER_LINEAR) > 200
        hl = np.zeros((720, 1280), np.uint8); cv2.fillConvexPoly(hl, cv2.convexHull(face.landmark_2d_106.astype(np.int32)), 1)
        inb = bar & (hl > 0)
        if int(inb.sum()) > 300:
            ib = 1; c = float((chg & inb).sum()) / int(inb.sum())
            if "open" in gt.files:
                opn = cv2.resize(gt["open"][i], (1280, 720), interpolation=cv2.INTER_LINEAR) > 100
                cm = opn & ~(cv2.dilate(bar.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0)
                v = float((chg & cm).sum()) / max(1, int(cm.sum()))
    cov.append(c); cav.append(v); inbar.append(ib); i += 1
px = np.array(px, float); cov = np.array(cov, float); cav = np.array(cav, float); inbar = np.array(inbar)
n = len(px); on = inbar == 1
print("%s: %d frames; erased frames %d; bar in the mouth on %d frames" % (os.path.basename(CLIP), n, int((px > 0).sum()), int(on.sum())))
if on.any():
    print("  on the bar frames: taken %d..%d px (mean %d); bar inside the outline covered: min %.2f mean %.2f; cavity touched: max %.3f mean %.3f; frames with nothing taken %d" % (
        np.nanmin(px[on]), np.nanmax(px[on]), np.nanmean(px[on]), np.nanmin(cov[on]), np.nanmean(cov[on]), np.nanmax(cav[on]) if np.isfinite(cav[on]).any() else -1,
        np.nanmean(cav[on]) if np.isfinite(cav[on]).any() else -1, int((px[on] == 0).sum())))
    print("  missed frames (bar inside the outline, nothing taken): %s" % [int(k) for k in np.nonzero(on & (px == 0))[0]])
    print("  first/last frames with the bar inside the outline: %d / %d; first/last frames with something taken: %d / %d" % (np.nonzero(on)[0][0], np.nonzero(on)[0][-1], np.nonzero(px > 0)[0][0] if (px > 0).any() else -1, np.nonzero(px > 0)[0][-1] if (px > 0).any() else -1))
    d = np.abs(np.diff(px)); d = d[np.isfinite(d)]
    print("  largest frame-to-frame change of the area taken %d px (mean taken %d); off-bar frames with something taken: %d" % (int(d.max()), int(np.nanmean(px[on])), int(((px > 0) & ~on).sum())))
else:
    print("  frames with something taken: %d (max %d px)" % (int((px > 0).sum()), int(np.nanmax(px)) if n else 0))
