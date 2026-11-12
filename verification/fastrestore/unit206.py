"""v11.2.206 (SDOS-206): the occlusion mask, frame after frame, on a synthetic object clip (make_objects.py: a hand or cup
over the face of insightface's public sample photo; NOT user media).
1. With the new switches off, v11.2.206's mask equals v11.2.205's on every frame (exactly the old behaviour).
2. With them on: a valid mask; over the object, where it covers the face, clearly less of the swapped face is painted;
   the rest of the face is painted exactly as by v11.2.205 (no false cut, no shrinking while the object is there, no ghost
   after it moves on).
usage: unit206.py PACK205 PACK206 OBJCLIP.mp4 XSEG.onnx"""
import sys, os, types, importlib.util
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
    assert E.occ_net_load(XSEG, threads=2), "occlusion model did not load"
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "landmark_2d_106"], providers=["CPUExecutionProvider"])
fa.prepare(ctx_id=-1, det_size=(640, 640))
cap = cv2.VideoCapture(CLIP); frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
gt = np.load(CLIP[:-4] + "_gt.npz")["a"]
f0 = min(fa.get(frames[0]), key=lambda f: abs((f.bbox[0] + f.bbox[2]) / 2 - 640))       # the face, found once (the object covers it later)
fails = 0


def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else ""), flush=True); fails += 0 if ok else 1


def run(E, switches, n=None):
    for k, v in switches.items():
        E._P[k] = v
    comp = E.AlignedCompositor.__new__(E.AlignedCompositor)
    tr = E.TrackState(0) if "slot" not in E.TrackState.__init__.__code__.co_varnames else E.TrackState(slot=0)
    size = 256
    M = E.estimate_norm(f0.kps, size)
    out = []
    for t, fr in enumerate(frames[:n]):
        at = cv2.warpAffine(fr, M, (size, size), borderMode=cv2.BORDER_REPLICATE)
        out.append(comp.build_mask(f0, M, size, track=tr, aligned_target=at, frame_wh=(fr.shape[1], fr.shape[0])))
    return out, M


off = {"occ_flow": 0.0, "occ_instant": 0.0, "occ_snap": 0.0, "occ_net_wide": 0.0}
ON = {"occ_flow": 1.0, "occ_instant": 1.0, "occ_snap": 1.0, "occ_net_wide": 1.0}
m5, M = run(E5, {}, 70)
m6, _ = run(E6, off, 70)
d = max(float(np.abs(a - b).max()) for a, b in zip(m5, m6))
check("switches off: v11.2.206's mask equals v11.2.205's on all 70 frames", d < 1e-6, "largest difference %.2e" % d)
m6on, _ = run(E6, ON, 70)
ok = all(np.isfinite(m).all() and m.min() >= 0 and m.max() <= 1 for m in m6on)
check("switches on: a valid mask (0-1) on every frame", ok)


def obj_mask(t, size=256):
    a = cv2.warpAffine(gt[t].astype(np.float32) / 255.0, M, (size, size)) > 0.5
    return cv2.erode(a.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


# the face: where the occlusion model sees face on the first frame (before the object comes)
size = m5[0].shape[0]
face0 = cv2.resize((E6.occ_net_raw(cv2.warpAffine(frames[0], M, (size, size), borderMode=cv2.BORDER_REPLICATE)) >= 0.5)
                   .astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
rows, rest = [], []
for t in range(18, 66, 1):
    o = obj_mask(t)
    near = cv2.dilate((cv2.warpAffine(gt[t].astype(np.float32) / 255.0, M, (size, size)) > 0.5).astype(np.uint8),
                      np.ones((15, 15), np.uint8)).astype(bool)
    r = (m5[t] > 0.05) & ~near
    rest.append(float(((m6on[t] < m5[t] - 0.1) & r).sum()) / max(1.0, float(r.sum())))
    of = o & face0
    if of.sum() < 200:
        continue
    rows.append((t, float(m5[t][of].mean()), float(m6on[t][of].mean()), float(m5[t][o].mean()), float(m6on[t][o].mean())))
print("frame  face paint over the object where it covers the face (all of the object): v11.2.205 -> v11.2.206 (0 = shown)")
for t, a, b, c, d in rows[:6] + rows[-4:]:
    print("  %2d   %.3f -> %.3f   (%.3f -> %.3f)" % (t, a, b, c, d))
mo5, mo6 = np.mean([r[1] for r in rows]), np.mean([r[2] for r in rows])
check("over the object where it covers the face (mean of the frames it is in): v11.2.206 paints at least 20 % less than v11.2.205",
      mo6 <= 0.8 * mo5 or mo6 <= 0.05, "%.3f vs %.3f; the whole object %.3f vs %.3f" % (
          mo6, mo5, np.mean([r[4] for r in rows]), np.mean([r[3] for r in rows])))
check("the rest of the face is painted as by v11.2.205: under 1 % of it painted less on average, under 6 % on any frame",
      np.mean(rest) <= 0.01 and np.max(rest) <= 0.06, "mean %.2f %%, worst frame %.2f %%" % (100 * np.mean(rest), 100 * np.max(rest)))
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
