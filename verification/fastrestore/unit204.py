"""v11.2.204 (SDOS-204): the Optimized smart keys - contact and side-pose tests. Stand-in records, no models. usage: unit204.py PACKDIR"""
import sys, types
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception
sys.modules["gradio"] = g_
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: ""; sys.modules["huggingface_hub"] = hf
sys.path.insert(0, sys.argv[1])
import numpy as np, core_pipeline as cp, swap_engine as E
fails = 0
def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else ""), flush=True); fails += 0 if ok else 1
def rec(x, w=200, h_nose=0.0, roll=0.0):
    """5 points of a face whose nose sits h_nose eye distances sideways of the eye midpoint (the engine's yaw measure)."""
    ed = 0.4 * w; cx, cy = x + w / 2, 300 + w / 2
    pts = np.array([[-ed / 2, -0.1 * w], [ed / 2, -0.1 * w], [h_nose * ed, 0.08 * w], [-0.3 * ed, 0.25 * w], [0.3 * ed, 0.25 * w]])
    c, s = np.cos(roll), np.sin(roll); pts = pts @ np.array([[c, s], [-s, c]])
    return {"bbox": np.array([x, 300, x + w, 300 + w], np.float32), "kps": (pts + [cx, cy]).astype(np.float32)}
check("two faces apart (centres 1.7 face sizes): not contact", not cp._faces_touch({0: rec(100), 1: rec(440)}))
check("two faces 20 px apart, or overlapping: contact", cp._faces_touch({0: rec(100), 1: rec(320)}) and cp._faces_touch({0: rec(100), 1: rec(250)}))
check("frontal (offset 0.10) and 36 deg (0.34): not a side pose", not cp._side_pose({0: rec(100, h_nose=0.10)}) and not cp._side_pose({0: rec(100, h_nose=0.34)}))
check("about 45 deg (0.51) and 60 deg (0.9): side pose", cp._side_pose({0: rec(100, h_nose=0.51)}) and cp._side_pose({0: rec(100, h_nose=-0.9)}))
check("the same 45 deg face rolled 30 deg (head tilted): still a side pose; a tilted frontal face is not",
      cp._side_pose({0: rec(100, h_nose=0.51, roll=0.5)}) and not cp._side_pose({0: rec(100, h_nose=0.1, roll=0.5)}))
check("one of two faces in a side pose is enough", cp._side_pose({0: rec(100), 1: rec(600, h_nose=0.7)}))
check("switches: smart_keys and side_key_h exist (0 / a higher value restore v11.2.203)",
      int(E._P.get("smart_keys", -1)) == 1 and abs(float(E._P.get("side_key_h", 0)) - 0.40) < 1e-9)
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
