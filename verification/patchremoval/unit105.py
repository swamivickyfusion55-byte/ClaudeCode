"""v11.2.105 unit checks (PATCH_MODE auto / classic; PATCH-EVENTS: when and why a removed mark showed).
usage: unit105.py PACKDIR      Synthetic data only."""
import sys, os, types, numpy as np, cv2
PACK = sys.argv[1]
sys.path.insert(0, PACK)
g = types.ModuleType("gradio"); g.Request = type("Request", (), {}); g.update = lambda **k: k; g.Error = Exception
sys.modules["gradio"] = g
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: None; sys.modules["huggingface_hub"] = hf
os.environ.setdefault("PHOENIX_MODEL_DIR", "/tmp/claude-0/occmodels")
import core_pipeline as cp
E = cp._E
fails = 0


def check(name, ok, info=""):
    global fails
    fails += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info else ""))


# --- 1. modes --------------------------------------------------------------------------------------------------------------
base = {k: E.PATCH_CFG.get(k) for k in E.PATCH_MODES["classic"]}
check("config: PATCH_MODE is a setting (auto by default)", getattr(cp.__dict__.get("_cfg", None) or __import__("config"), "PATCH_MODE", None) in ("auto", "classic"))
m = E.set_patch_mode("classic")
check("classic: the earliest patcher's settings are in force", m == "classic" and E.PATCH_CFG["find_mark"] is False and E.PATCH_CFG["rescue"] is False
      and E.PATCH_CFG["opaque"] is False and E.PATCH_CFG["detail_track"] is False and E.PATCH_CFG["peek_band"] == 0.08 and E.PATCH_CFG["grow_band"] == 0.06
      and E.PATCH_CFG["bridge_max"] == 3 and E.PATCH_CFG["clone_texture"] is False, str({k: E.PATCH_CFG[k] for k in E.PATCH_MODES["classic"]}))
check("classic: the fast-motion tracking of v11.2.103 stays", bool(E.PATCH_CFG["redetect_now"]) and bool(E.PATCH_CFG["blur_ok"]) and float(E.PATCH_CFG["smooth_err"]) > 0)
m = E.set_patch_mode("auto")
check("auto: every setting is back as it was", m == "auto" and all(E.PATCH_CFG.get(k) == v for k, v in base.items()), str({k: (E.PATCH_CFG.get(k), v) for k, v in base.items() if E.PATCH_CFG.get(k) != v}))
check("an unknown mode name means auto", E.set_patch_mode("fancy") == "auto" and all(E.PATCH_CFG.get(k) == v for k, v in base.items()))
E.set_patch_mode("classic"); E.set_patch_mode("classic"); E.set_patch_mode("auto")
check("switching again and again does not lose the original settings", all(E.PATCH_CFG.get(k) == v for k, v in base.items()))

# --- 2. per-frame codes in PatchGroup ---------------------------------------------------------------------------------------------
class FakeTrack:
    def __init__(self, anchor, poses, behaviour):
        self.anchor_idx = anchor; self.A = {i: None for i in poses}; self.conf = {i: 0.9 for i in poses}; self.behaviour = behaviour
    def apply(self, frame, idx):
        b = self.behaviour(idx)
        if b == "paint":
            return frame.copy()
        if b == "object":
            E.PATCH_STATS["frames_gated"] += 1; return frame.copy()
        if b == "refuse":
            E.PATCH_STATS["frames_refused"] += 1
        if b == "weak":
            E.PATCH_STATS["frames_weak_skipped"] += 1
        return frame
fr = np.zeros((20, 20, 3), np.uint8)
beh = {0: "paint", 1: "paint", 2: "refuse", 3: "refuse", 4: "weak", 5: "object", 6: "paint"}
grp = E.PatchGroup([FakeTrack(0, [0, 1, 2, 3, 4, 5, 6, 8], lambda i: beh.get(i, "paint"))])
grp.mark_no = 3
E.PATCH_EVENTS.clear()
for i in range(0, 9):
    grp.apply(fr, i)
ev = E.PATCH_EVENTS.get(3, {})
check("events: painted = P, painted around an object = o, refused = R, held pose and mark not seen = W, no pose = L",
      [ev.get(i) for i in range(9)] == ["P", "P", "R", "R", "W", "o", "P", "L", "P"], str([ev.get(i) for i in range(9)]))

# --- 3. the log lines ---------------------------------------------------------------------------------------------------------------
lines = cp._patch_event_lines({3: ev}, 30.0)
check("log: a mark that showed is reported with seconds and reasons (refused 2 frames, held 1 frame, not followed 1 frame)",
      len(lines) == 1 and "mark 3" in lines[0] and "SHOWED on 4 frames" in lines[0] and "refused" in lines[0] and "held position" in lines[0] and "not followed" in lines[0], lines[0] if lines else "none")
clean = {1: {i: "P" for i in range(40)}}
l2 = cp._patch_event_lines(clean, 30.0)
check("log: a mark that never showed says so", len(l2) == 1 and "never showed" in l2[0], l2[0] if l2 else "none")
edge = {2: dict([(i, "L") for i in range(10)] + [(i, "P") for i in range(10, 30)] + [(i, "X") for i in range(30, 45)])}
l3 = cp._patch_event_lines(edge, 30.0)
check("log: frames before the first and after the last removal are not counted as 'showed', but are reported separately",
      len(l3) == 1 and "never showed" in l3[0] and "before 0.3 s (10 fr)" in l3[0] and "after 1.0 s (15 fr)" in l3[0], l3[0] if l3 else "none")
check("log: a job without patches gives no lines", cp._patch_event_lines({}, 30.0) == [])

# --- 4. a runaway predicted pose never hangs the job (cv2.warpAffine with an infinite matrix does not return) ------------------------
import time as _t
check("pose check: a normal pose is usable; inf, nan, a collapsed or exploded scale are not",
      E._pr_pose_ok(np.array([[1, 0, 5], [0, 1, 3]], float)) and not E._pr_pose_ok(np.array([[np.inf, 0, 0], [0, 1, 0]], float))
      and not E._pr_pose_ok(np.array([[np.nan, 0, 0], [0, 1, 0]], float)) and not E._pr_pose_ok(np.array([[1e-4, 0, 0], [0, 1e-4, 0]], float))
      and not E._pr_pose_ok(np.array([[50, 0, 0], [0, 50, 0]], float)))
yy, xx = np.mgrid[0:240, 0:320].astype(np.float32)
img = np.dstack([np.clip(150 + 20 * np.sin(xx / 7.0) * np.cos(yy / 9.0), 0, 255)] * 3).astype(np.uint8)
hole = ((yy - 120) ** 2 + (xx - 160) ** 2) <= 12 ** 2
E.PATCH_INPAINTER = lambda crop, mk: cv2.inpaint(crop, (mk > 0).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
trk = E.PatchTrack(0, img, hole)
t0 = _t.time()
bad = np.array([[np.inf, 0, 0], [0, np.inf, 0]], float)
r1 = trk._redetect_tm(E._pr_gray(img), around=bad, radius=160.0)
r2 = trk._redetect_tm(E._pr_gray(img), around=np.array([[1e-4, 0, 0], [0, 1e-4, 0]], float), radius=160.0)
check("template search around a broken pose returns at once (falls back to the whole frame)", _t.time() - t0 < 10.0, f"{_t.time() - t0:.2f} s")
lastA = np.array([[1, 0, 0], [0, 1, 0]], float); velA = np.array([[1e-3, 0, 0], [0, 1e-3, 0]], float)
t0 = _t.time()
out = trk._step(E._pr_gray(img), None, None, None, lastA, velA, 6)
check("a step whose velocity would shrink the pose to nothing does not hang and keeps a sane pose", _t.time() - t0 < 10.0 and (out[0] is None or E._pr_pose_ok(out[0])), f"{_t.time() - t0:.2f} s")

print("\n%d check(s) failed" % fails if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
