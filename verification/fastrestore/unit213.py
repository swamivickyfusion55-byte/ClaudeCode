"""v11.2.213 (SDOS-213) unit checks: a head turned beyond what the swap can draw gives way to its own face. Stand-in faces and a
stand-in pose model (no models loaded, no user media), plus the real 3D-landmark model on the rendered test head when
HEAD=1 (head3d.py, the public Lee Perry-Smith scan).
usage: unit213.py PACK212 PACK213"""
import sys, os, types, math
P212, P213 = sys.argv[1:3]
import numpy as np
import logging
logging.disable(logging.CRITICAL)
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception
sys.modules["gradio"] = g_
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: ""; sys.modules["huggingface_hub"] = hf
sys.path.insert(0, P213)
import core_pipeline as cp
E = cp._E
fails = 0


def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else ""), flush=True); fails += 0 if ok else 1


class F:
    """a detection: box and 5 points; side = the nose's offset from the eye midpoint in eye distances (the 5-point turn)"""
    def __init__(s, side=0.0, ed=60.0, cx=400.0, cy=300.0):
        s.bbox = np.array([cx - 100, cy - 120, cx + 100, cy + 120], np.float32)
        le, re = np.array([cx - ed / 2, cy - 30]), np.array([cx + ed / 2, cy - 30])
        s.kps = np.array([le, re, [cx + side * ed, cy + 10], [cx - ed / 2 + 5, cy + 50], [cx + ed / 2 - 5, cy + 50]], np.float32)
        s.det_score = 0.8


# 1. the weight
w = [E.turn_weight(y) for y in (None, 0, 60, 76, 78, 80, 82, 84, 86, 89)]
check("turn weight: 1 up to 78 deg (as the 3D-landmark model reads), smoothly to 0 at 86, 0 beyond",
      w[0] == 1 and w[1] == 1 and w[2] == 1 and w[3] == 1 and w[4] == 1 and 0.5 < w[5] < 1 and abs(w[6] - 0.5) < 1e-6 and 0 < w[7] < 0.5
      and w[8] == 0 and w[9] == 0 and all(a >= b for a, b in zip(w[1:], w[2:])), str([round(x, 2) for x in w]))
E._P["turn_fade"] = 0
check("turn_fade = 0: weight 1 at any turn (11.2.212)", E.turn_weight(89) == 1.0)
E._P["turn_fade"] = 1

# 2. the reading: the model only for faces that look turned, smoothed on the track
calls = []
YAW = {"v": 0.0, "nod": 5.0, "tilt": 5.0}
E.turn_pose = lambda img, f: (calls.append(1), (YAW["v"], YAW["nod"], YAW["tilt"]))[1]
img = np.zeros((600, 800, 3), np.uint8)
tr = E.TrackState(0)
cp._turn_read(img, F(side=0.1), tr)
check("a front face: the pose model is not asked (no cost), the track reads 0", len(calls) == 0 and tr.turn_yaw == 0.0, "calls %d, yaw %s" % (len(calls), tr.turn_yaw))
YAW["v"] = 84.0
cp._turn_read(img, F(side=1.2), tr)
c1 = tr.turn_yaw
cp._turn_read(img, F(side=1.2), tr)
cp._turn_read(img, F(side=1.2), tr)
c2 = tr.turn_yaw
check("a turned face: asked, and the track follows a head turning away within 3 detections (0.7 a step)",
      len(calls) == 3 and abs(c1 - 58.8) < 0.01 and c2 > 80 and E.turn_weight(c2) < 0.7, "%.1f, %.1f, weight %.2f" % (c1, c2, E.turn_weight(c2)))
cp._turn_read(img, F(side=0.2), tr)       # the 5 points read front for a moment (a bad read) while the track was turned
check("after a strong turn the model is still asked when the 5 points look front (they are not trusted there)", len(calls) == 4)
YAW["v"] = 10.0
for _ in range(4):
    cp._turn_read(img, F(side=1.0), tr)
check("turning back: the weight is back to 1 within 4 detections", E.turn_weight(tr.turn_yaw) == 1.0, "yaw %.1f" % tr.turn_yaw)

# 2b. past profile: the head turns on, away from the camera; the model then reads a 62-75 deg turn with a wild nod AND tilt
trp = E.TrackState(5); trp.turn_yaw = 84.0
YAW.update(v=70.0, nod=62.0, tilt=55.0)
for _ in range(3):
    cp._turn_read(img, F(side=3.0), trp)
check("past profile (a head already turned 84 deg reads 70 with nod 62 / tilt 55): kept as turned away, weight 0",
      trp.turn_yaw > 86 and E.turn_weight(trp.turn_yaw) == 0.0, "yaw %.1f" % trp.turn_yaw)
trf = E.TrackState(6)
cp._turn_read(img, F(side=3.0), trf)
check("...but a head seen for the first time with such a reading is read as it shows (a real nodded 3/4 face): weight 1",
      abs(trf.turn_yaw - 70.0) < 1e-6 and E.turn_weight(trf.turn_yaw) == 1.0, "yaw %.1f" % trf.turn_yaw)
YAW.update(v=70.0, nod=20.0, tilt=18.0)
cp._turn_read(img, F(side=1.5), trp)
cp._turn_read(img, F(side=1.5), trp)
check("...and turning back (a plausible 70 deg reading) brings the replacement back within 2 detections", E.turn_weight(trp.turn_yaw) == 1.0,
      "yaw %.1f" % trp.turn_yaw)
YAW.update(v=0.0, nod=5.0, tilt=5.0)

# 3. a detection with no track yet keeps its own reading; the record takes it
YAW["v"] = 88.0
f = F(side=2.0)
cp._turn_read(img, f, None)
rec = cp._geom_record(f, None, 1.0, 0.0, None)
check("a new face turned away: its record allows none of the replacement", rec is not None and rec["turn_w"] == 0.0, str(rec and rec["turn_w"]))
tr2 = E.TrackState(1); tr2.kps = f.kps; tr2.bbox = f.bbox; tr2.turn_yaw = 82.0
rec2 = cp._geom_record(F(side=0.0), tr2, 1.0, 0.0, None)
check("a carried / predicted face takes its track's turn (82 deg: half)", abs(rec2["turn_w"] - 0.5) < 1e-6, "%.3f" % rec2["turn_w"])
rec3 = cp._geom_record(F(side=0.0), None, 1.0, 0.0, None)
check("a face never read: weight 1 (as 11.2.212)", rec3["turn_w"] == 1.0)
mid = cp._geom_lerp(dict(rec2, turn_w=1.0), dict(rec2, turn_w=0.0), 0.25)
check("between two key frames the weight is interpolated like the paint's alpha", abs(mid["turn_w"] - 0.75) < 1e-6, "%.3f" % mid["turn_w"])
old = {k: v for k, v in rec2.items() if k != "turn_w"}
mid2 = cp._geom_lerp(old, old, 0.5)
check("records without a weight (made elsewhere) count as 1", mid2["turn_w"] == 1.0)

# 4. failure safety: a pose model that fails changes nothing
E.turn_pose = lambda img, f: None
tr3 = E.TrackState(2)
cp._turn_read(img, F(side=1.5), tr3)
check("no reading (model missing or failed): the track has none, the weight stays 1", getattr(tr3, "turn_yaw", None) is None and cp._turn_w_of(F(), tr3) == 1.0)
def boom(img, f):
    raise RuntimeError("x")
E.turn_pose = boom
try:
    cp._turn_read(img, F(side=1.5), tr3); ok = True
except Exception:
    ok = False
check("an error in the reading never reaches the job", ok)

# 5. the real model on the rendered head (optional): a turn reading and a weight per true turn
if os.environ.get("HEAD") == "1":
    import importlib
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    E2 = importlib.reload(E) if False else E
    import cv2
    from head3d import Head
    from insightface.app import FaceAnalysis
    fa = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=["CPUExecutionProvider"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
    spec_m = importlib.util.spec_from_file_location("se213b", os.path.join(P213, "swap_engine.py"))
    Eb = importlib.util.module_from_spec(spec_m); spec_m.loader.exec_module(Eb)
    ok_load = Eb.turn_model_load(fa)
    check("the 3D-landmark model loads from the face pack (1k3d68.onnx)", ok_load)
    h = Head("/tmp/claude-0/head3d/LeePerrySmith.glb", "/tmp/claude-0/head3d/Map-COL.jpg")
    t1 = cv2.imread("/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg"); bg = cv2.GaussianBlur(cv2.resize(t1, (1280, 720)), (0, 0), 18)
    res = []
    for yaw in (0, 40, 60, 70, 75, 85, 90, 100):
        im, _, _ = h.render(yaw, 3, -2, 1280, 720, (640, 420), 760, bg)
        fs = fa.get(im)
        y = Eb.turn_yaw(im, max(fs, key=lambda q: q.det_score)) if fs else None
        res.append((yaw, y, Eb.turn_weight(y)))
    print("   true turn -> read, weight: " + ", ".join("%d -> %s, %.2f" % (a, "-" if b is None else "%.0f" % b, c) for a, b, c in res))
    check("rendered head: full replacement up to a 75 deg turn, none from 90 deg",
          all(c == 1.0 for a, b, c in res if a <= 75) and all(c <= 0.05 for a, b, c in res if a >= 90))
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
