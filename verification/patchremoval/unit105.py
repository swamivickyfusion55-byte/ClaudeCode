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
check("log: frames before the first and after the last removal (the mark not in the video yet / any more) are not counted", len(l3) == 1 and "never showed" in l3[0], l3[0] if l3 else "none")
check("log: a job without patches gives no lines", cp._patch_event_lines({}, 30.0) == [])

print("\n%d check(s) failed" % fails if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
