"""v11.2.206 (SDOS-206): what the new occlusion steps cost, per face and frame. The mask step (build_mask) of v11.2.205 and of
v11.2.206 timed on the same synthetic object clip (make_objects.py; insightface's public sample photo, NOT user media), one
after the other, on the same cores, 3 passes each, the best pass kept.
usage: time206.py PACK205 PACK206 OBJCLIP.mp4 XSEG.onnx"""
import sys, os, time, types, importlib.util
P205, P206, CLIP, XSEG = sys.argv[1:5]
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception
sys.modules["gradio"] = g_
sys.path.insert(0, P206)
import numpy as np, cv2


def load(name, pack):
    spec = importlib.util.spec_from_file_location(name, os.path.join(pack, "swap_engine.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


E5, E6 = load("se205", P205), load("se206", P206)
for E in (E5, E6):
    assert E.occ_net_load(XSEG, threads=2)
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "landmark_2d_106"], providers=["CPUExecutionProvider"])
fa.prepare(ctx_id=-1, det_size=(640, 640))
cap = cv2.VideoCapture(CLIP); frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
f0 = min(fa.get(frames[0]), key=lambda f: abs((f.bbox[0] + f.bbox[2]) / 2 - 640))


def one_pass(E, size):
    comp = E.AlignedCompositor.__new__(E.AlignedCompositor)
    tr = E.TrackState(slot=0)
    M = E.estimate_norm(f0.kps, size)
    ats = [cv2.warpAffine(fr, M, (size, size), borderMode=cv2.BORDER_REPLICATE) for fr in frames[:70]]
    t0 = time.perf_counter()
    for fr, at in zip(frames, ats):
        comp.build_mask(f0, M, size, track=tr, aligned_target=at, frame_wh=(fr.shape[1], fr.shape[0]))
    return (time.perf_counter() - t0) * 1000.0 / len(ats)


for size in (256, 512):
    r = {}
    for name, E in (("v11.2.205", E5), ("v11.2.206", E6)):
        one_pass(E, size)
        r[name] = min(one_pass(E, size) for _ in range(3))
    print("face crop %d px: mask step %.1f ms (v11.2.205) -> %.1f ms (v11.2.206), +%.1f ms per face and frame"
          % (size, r["v11.2.205"], r["v11.2.206"], r["v11.2.206"] - r["v11.2.205"]), flush=True)
