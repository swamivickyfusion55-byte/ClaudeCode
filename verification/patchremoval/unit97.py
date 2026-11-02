"""v11.2.97 unit checks (several painted positions per mark, gap bridging, contact calm).  usage: unit97.py PACKDIR CLIPDIR
Public photo (insightface t1.jpg) and synthetic data only."""
import sys, os, types, base64, numpy as np, cv2
PACK, CLIPS = sys.argv[1], sys.argv[2]
sys.path.insert(0, PACK)
g = types.ModuleType("gradio"); g.Request = type("Request", (), {}); g.update = lambda **k: k; g.Error = Exception
sys.modules["gradio"] = g
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: None; sys.modules["huggingface_hub"] = hf
os.environ.setdefault("PHOENIX_MODEL_DIR", "/tmp/claude-0/occmodels")
import core_pipeline as cp
import phoenix_api_adapter as ad
E = cp._E
fails = 0


def check(name, ok, info=""):
    global fails
    fails += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info else ""))


# --- several positions of one mark: the nearest view is used, the next one if it refuses ------------------------------
class FakeTrack:
    def __init__(self, anchor, frames, conf=0.9, refuse=()):
        self.anchor_idx, self.A, self.conf, self.refuse = anchor, {f: np.eye(2, 3) for f in frames}, {f: conf for f in frames}, set(refuse)

    def apply(self, frame, idx):
        if idx in self.refuse:
            return frame
        out = frame.copy(); out[0, 0] = self.anchor_idx; return out


f0 = np.zeros((4, 4), np.int32)
t1, t2 = FakeTrack(10, range(0, 100)), FakeTrack(80, range(40, 100))
grp = E.PatchGroup([t1, t2])
check("positions: a frame near the 1st painted position uses it", grp.apply(f0, 20)[0, 0] == 10)
check("positions: a frame near the 2nd painted position uses it", grp.apply(f0, 70)[0, 0] == 80)
check("positions: where only one position is followed, that one is used", grp.apply(f0, 5)[0, 0] == 10)
t3 = FakeTrack(80, range(40, 100), refuse={75})
check("positions: if the nearest refuses the frame, the next one is tried", E.PatchGroup([t1, t3]).apply(f0, 75)[0, 0] == 10)
t4 = FakeTrack(80, range(40, 100), conf=0.5)
check("positions: a nearer view with clearly lower confidence gives way", E.PatchGroup([t1, t4]).apply(f0, 70)[0, 0] == 10)

# --- gap bridging -------------------------------------------------------------------------------------------------------
tr = E.PatchTrack.__new__(E.PatchTrack)
tr.box, tr.size = (100, 200, 100, 200), 40
A0 = np.array([[1, 0, 0], [0, 1, 0]], float); A3 = np.array([[1, 0, 6.0], [0, 1, 3.0]], float)
tr.A, tr.conf = {10: A0, 13: A3}, {10: 0.9, 13: 0.9}
tr.bridge()
check("bridging: a 2-frame gap between agreeing poses is filled, in between", 11 in tr.A and 12 in tr.A and abs(tr.A[11][0, 2] - 2.0) < 1e-6)
tr.A, tr.conf = {10: A0, 20: A3}, {10: 0.9, 20: 0.9}
tr.bridge()
check("bridging: a long gap (9 frames) is not filled", 15 not in tr.A)
tr.A, tr.conf = {10: A0, 12: np.array([[1, 0, 400.0], [0, 1, 0]], float)}, {10: 0.9, 12: 0.9}
tr.bridge()
check("bridging: poses that disagree (a jump) are not bridged", 11 not in tr.A)

# --- the job settings: groups --------------------------------------------------------------------------------------------
gt = np.load(os.path.join(CLIPS, "pr_proc_mole_gt.npz"))
b = lambda im, ext: base64.b64encode(cv2.imencode(ext, im)[1].tobytes()).decode()
v = lambda grp_, t: {"group": grp_, "image_b64": b(gt["ref"], ".jpg"), "mask_b64": b(gt["refmask"], ".png"), "t": t}
pl, err = ad._load_patches({"patches": [v("mA", 0.6), v("mA", 3.1), v("mB", 1.0)]})
check("settings: 3 positions of 2 marks are accepted, group kept", err is None and [p["group"] for p in pl] == ["mA", "mA", "mB"], str(err))
_, e5 = ad._load_patches({"patches": [v(f"m{k}", 0.1) for k in range(5)]})
_, e13 = ad._load_patches({"patches": [v("mA", 0.1)] * 13})
check("settings: 5 marks, or 13 positions, are refused with a reason", e5 is not None and e13 is not None,
      f"{(e5 or {}).get('error')} | {(e13 or {}).get('error')}")
pl1, _ = ad._load_patches({"patches": [{"image_b64": b(gt["ref"], ".jpg"), "mask_b64": b(gt["refmask"], ".png"), "t": 1}]})
check("settings: an entry without a group (app 1.2.25) is its own mark", pl1 and pl1[0]["group"] is None)
sp = cp._patch_specs({"patches": pl})
check("job: the group reaches the engine", [x[3] for x in sp] == ["mA", "mA", "mB"])

# --- contact calm: the occluder net does not cut next to the other face ------------------------------------------------
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
img = cv2.imread("/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg")
f = max(fa.get(img), key=lambda q: q.bbox[2] - q.bbox[0]); S = 128; M = E.estimate_norm(f.kps, S)
al = cv2.warpAffine(img, M, (S, S)); comp = E.AlignedCompositor(None)
IM = cv2.invertAffineTransform(M); other = f.landmark_2d_106 + (IM[:, :2] @ np.array([0.40 * S, 0.0]))
op = E._landmarks_to_aligned(E.with_forehead(other), M, S)
near = np.zeros((S, S), np.uint8); cv2.fillConvexPoly(near, cv2.convexHull(np.round(op).astype(np.int32)), 1)
near = cv2.dilate(near, np.ones((5, 5), np.uint8)).astype(bool)
_orig = E.occ_net_keep
E.occ_net_keep = lambda *a, **k: np.ones((S, S), np.float32)
E._P["occ_novel"] = 0.0
m_free = comp.build_mask(f, M, S, track=None, aligned_target=al, rivals=[{"lmk": other, "front": False}])
cand = near & (m_free > 0.6)                                           # next to the other face, painted
ys, xs = np.nonzero(cand)
k = int(np.argmin(xs)) if len(xs) else 0
cy, cx = (int(ys[k]), int(xs[k]) + 2) if len(xs) else (S // 2, S // 2)
cut = np.ones((S, S), np.float32)
cv2.circle(cut, (cx, cy), 6, 0.0, -1)                                  # a cut there
cv2.circle(cut, (int(0.30 * S), int(0.40 * S)), 6, 0.0, -1)          # and one far from it (this face's left eye area)
E.occ_net_keep = lambda *a, **k: cut.copy()
E._P["occ_novel"] = 0.0
riv = [{"lmk": other, "front": False}]
E._P["contact_calm"] = 0.0
m_off = comp.build_mask(f, M, S, track=None, aligned_target=al, rivals=riv)
E._P["contact_calm"] = 0.08
E._P["contact_full"] = 1.01                      # light contact: only the zone next to the other face is calmed
m_on = comp.build_mask(f, M, S, track=None, aligned_target=al, rivals=riv)
E._P["contact_full"] = 0.15                      # heavy contact (this overlap): the net is off for the whole face
m_full = comp.build_mask(f, M, S, track=None, aligned_target=al, rivals=riv)
m_none = comp.build_mask(f, M, S, track=None, aligned_target=al, rivals=None)
E.occ_net_keep = _orig
pn, pf = (cy, cx), (int(0.40 * S), int(0.30 * S))
check("light contact: next to the other face the occluder net's cut is ignored",
      m_off[pn] < 0.3 and m_on[pn] > m_off[pn] + 0.3, f"{m_off[pn]:.2f} -> {m_on[pn]:.2f}")
check("light contact: away from the other face the net's cut still applies at once", m_on[pf] < 0.3, f"{m_on[pf]:.2f}")
check("contact calm: a face with no other face nearby is unchanged", m_none[pf] < 0.3)
check("full contact (the other face covers much of this crop): the net cuts nothing on this face",
      abs(m_full[pn] - m_free[pn]) < 0.05 and abs(m_full[pf] - m_free[pf]) < 0.05,
      f"near {m_full[pn]:.2f} (uncut {m_free[pn]:.2f}), far {m_full[pf]:.2f} (uncut {m_free[pf]:.2f})")
print("ALL PASS" if fails == 0 else f"{fails} FAILED")
