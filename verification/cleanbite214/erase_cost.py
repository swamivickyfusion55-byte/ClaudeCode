"""erase_cost.py PACK CLIP FRAME: time of the 214 erase per face-frame at 720p, 1080p and 4K (the clip's frame scaled up), split into the net+mask and the inpainting."""
import sys, os, time, types, numpy as np, cv2
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception; sys.modules["gradio"] = g_
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: ""; sys.modules["huggingface_hub"] = hf
PACK, CLIP, FR = sys.argv[1], sys.argv[2], int(sys.argv[3]); sys.path.insert(0, PACK)
import logging; logging.disable(logging.CRITICAL)
import swap_engine as E
assert E.occ_net_load("/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/face/models/xseg_2.onnx", threads=2)
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
comp = E.AlignedCompositor(None)
cap = cv2.VideoCapture(CLIP); cap.set(1, FR); ok, f0 = cap.read()
for name, s in (("720p", 1.0), ("1080p", 1.5), ("4K", 3.0)):
    f = cv2.resize(f0, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s != 1.0 else f0
    face = max(fa.get(f), key=lambda q: q.det_score)
    comp._erase_objects(f, f, face, None, None, True)            # warm up
    t = []
    for _ in range(4):
        t0 = time.perf_counter(); out = comp._erase_objects(f, f, face, None, None, True); t.append((time.perf_counter() - t0) * 1000)
    # the pieces: the net and mask alone
    M = E.estimate_norm(face.kps, 128); al = cv2.warpAffine(f, M, (128, 128), borderMode=cv2.BORDER_REPLICATE)
    t2 = []
    for _ in range(4):
        t0 = time.perf_counter(); E.object_erase_mask(al, E._landmarks_to_aligned(face.landmark_2d_106, M, 128), 128, M=M, frame_wh=(f.shape[1], f.shape[0])); t2.append((time.perf_counter() - t0) * 1000)
    print("%-6s frame %dx%d face %d px wide: erase %.0f ms (net + mask %.0f ms, the rest %.0f ms)" % (name, f.shape[1], f.shape[0], int(face.bbox[2] - face.bbox[0]), np.median(t), np.median(t2), np.median(t) - np.median(t2)))
