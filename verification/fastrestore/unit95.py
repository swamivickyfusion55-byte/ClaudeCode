"""usage: unit95.py PACK   Unit checks of the v11.2.95 occlusion work (synthetic geometry + insightface's sample photo)."""
import sys, types, cv2, numpy as np
PACK = sys.argv[1]; sys.path.insert(0, PACK)
g = types.ModuleType("gradio"); g.Request = type("Request", (), {}); g.update = lambda **k: k; g.Error = Exception; sys.modules["gradio"] = g
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: None; sys.modules["huggingface_hub"] = hf
import swap_engine as E, core_pipeline as cp
from insightface.app import FaceAnalysis
fails = 0
def check(name, ok, info=""):
    global fails
    fails += 0 if ok else 1; print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info else ""))
# --- front / behind with hysteresis ---------------------------------------------------------------------------------
cp._ORDER_MEM.clear(); cp._ORDER_CUE.clear(); seq = []
for f, r in enumerate([1.02] + [0.97, 1.01] * 4 + [0.85] * 5):     # outline-edge evidence of face 0 over face 1
    cp._ORDER_NOW[0] = f; cp._ORDER_CUE[(0, 1)] = r; seq.append(cp._in_front(0, 1, 100, 100))
check("order: near-equal evidence never flips (within 10 %)", all(seq[:9]), str(seq[:9]))
check("order: flips after the evidence clearly favours the other face for 4 frames, not before", seq[9:12] == [True, True, True] and seq[12] is False, str(seq[9:]))
cp._ORDER_CUE.clear(); held = []
for f in range(20, 30):                                               # outlines apart: no evidence, other face larger
    cp._ORDER_NOW[0] = f; held.append(cp._in_front(0, 1, 100, 160))
check("order: without outline evidence the decision is held (size alone never flips it)", not any(held), str(held))
cp._ORDER_MEM.clear()
cp._ORDER_NOW[0] = 100; x = [cp._in_front(0, 1, 100, 130) for _ in range(6)]
check("order: several calls in one frame count once", len(set(x)) == 1)
check("order: answer is symmetric", cp._in_front(1, 0, 130, 100) == (not cp._in_front(0, 1, 100, 130)))
# --- one face per pixel ----------------------------------------------------------------------------------------------
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
img = cv2.imread("/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg")
f = max(fa.get(img), key=lambda q: q.bbox[2] - q.bbox[0]); S = 128; M = E.estimate_norm(f.kps, S)
al = cv2.warpAffine(img, M, (S, S)); comp = E.AlignedCompositor(None)
# a fake second face: this face's landmarks shifted right by 45 % of the crop (overlapping the right cheek)
IM = cv2.invertAffineTransform(M); shift = (IM[:, :2] @ np.array([0.45 * S, 0.0]))
other = f.landmark_2d_106 + shift
base = comp.build_mask(f, M, S, track=None, aligned_target=al)
mf = comp.build_mask(f, M, S, track=None, aligned_target=al, rivals=[{"lmk": other, "front": True}])
mb = comp.build_mask(f, M, S, track=None, aligned_target=al, rivals=[{"lmk": other, "front": False}])
op = E._landmarks_to_aligned(other, M, S); om = np.zeros((S, S), np.uint8); cv2.fillConvexPoly(om, cv2.convexHull(np.round(op).astype(np.int32)), 1)
mp = E._landmarks_to_aligned(f.landmark_2d_106, M, S); mm = np.zeros((S, S), np.uint8); cv2.fillConvexPoly(mm, cv2.convexHull(np.round(mp).astype(np.int32)), 1)
inner_other = cv2.erode(om, np.ones((7, 7), np.uint8)).astype(bool)
check("face in front: this face paints nothing inside the other face's outline", float(mf[inner_other].max()) < 0.02, f"max {float(mf[inner_other].max()):.3f}")
far = cv2.erode(mm, np.ones((9, 9), np.uint8)).astype(bool) & ~cv2.dilate(om, np.ones((25, 25), np.uint8)).astype(bool)
check("face in front: the rest of this face is untouched", np.allclose(mf[far], base[far], atol=1e-4))
spill = inner_other & ~cv2.dilate(mm, np.ones((9, 9), np.uint8)).astype(bool)
check("face behind: this face no longer paints onto it outside its own outline", (not spill.any()) or float(mb[spill].max()) < 0.02, f"before {float(base[spill].max()) if spill.any() else 0:.2f}")
mine = cv2.erode(mm, np.ones((9, 9), np.uint8)).astype(bool)
check("face behind: this face's own area is untouched", np.allclose(mb[mine], base[mine], atol=1e-4))
fh = E._landmarks_to_aligned(E.with_forehead(f.landmark_2d_106), M, S); fm = np.zeros((S, S), np.uint8); cv2.fillConvexPoly(fm, cv2.convexHull(np.round(fh).astype(np.int32)), 1)
dist = cv2.distanceTransform((1 - om).astype(np.uint8), cv2.DIST_L2, 5)
beside = (dist >= 1) & (dist < 14) & ~cv2.dilate(fm, np.ones((9, 9), np.uint8)).astype(bool) & (base > 0.2)
check("other face in front: next to it this face keeps only its own outline (no oval past the jaw onto the other's neck/hair)",
      beside.any() and float(mf[beside].max()) < 0.05, f"{int(beside.sum())} px, before max {float(base[beside].max()) if beside.any() else 0:.2f}, now {float(mf[beside].max()) if beside.any() else 0:.3f}")
E._P["own_near"] = 0.0
m_nn = comp.build_mask(f, M, S, track=None, aligned_target=al, rivals=[{"lmk": other, "front": True}])
check("own_near 0: that clamp is off", beside.any() and float(m_nn[beside].max()) > 0.2)
E._P["own_near"] = 0.12
E._P["own_pixels"] = 0.0
m_old = comp.build_mask(f, M, S, track=None, aligned_target=al, rivals=[{"lmk": other, "front": False}])
check("own_pixels 0: a face behind is ignored (v11.2.94 behaviour)", np.allclose(m_old, base))
E._P["own_pixels"] = 1.0
# --- objects at the face's edge: change, not position ----------------------------------------------------------------
E.occ_net_load("/tmp/claude-0/occmodels/xseg_2.onnx", threads=2)
tr = E.TrackState(0); S2 = 256; M2 = E.estimate_norm(f.kps, S2); pts2 = E._landmarks_to_aligned(f.landmark_2d_106, M2, S2)
def novel(frame):
    a2 = cv2.warpAffine(frame, M2, (S2, S2)); E._OCCNET_RAW["key"] = None; E.occ_net_keep(a2, pts2, S2)
    return E.occ_net_novel(tr, pts2, S2)
for _ in range(4): k0 = novel(img)
check("clean face, several frames: nothing cut at the edge", float(k0.min()) > 0.98, f"min {float(k0.min()):.3f}")
# an object on the brow / forehead edge: centred just above the top of the landmark outline, in the edge ring
hm2 = np.zeros((S2, S2), np.uint8); cv2.fillConvexPoly(hm2, cv2.convexHull(np.round(pts2).astype(np.int32)), 1)
top = int(np.where(hm2.any(1))[0][0]); cols = np.where(hm2[top + 4])[0]; cc = np.array([cols.mean(), top + 2.0])
IM2 = cv2.invertAffineTransform(M2); c = (IM2[:, :2] @ cc + IM2[:, 2]).astype(int); rad = max(3, int(14 * np.linalg.norm(IM2[:, 0])))
occ = img.copy(); cv2.circle(occ, tuple(int(v) for v in c), rad, (40, 160, 40), -1)
k1 = novel(occ)
check("object arriving at the forehead/brow edge: cut on that frame", float(k1.min()) < 0.2, f"min {float(k1.min()):.2f}")
for _ in range(10): k2 = novel(occ)
check("object that stays: still cut 10 frames later (never learned)", float(k2.min()) < 0.2, f"min {float(k2.min()):.2f}")
for _ in range(3): k3 = novel(img)
check("object gone: nothing cut", float(k3.min()) > 0.98, f"min {float(k3.min()):.3f}")
# --- front / behind from the outline edge on a real contact frame (kiss_cheek: B in front) -----------------------------
B = "/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/"
kp = np.load(B + "kiss_cheek_gt.npz")["kps"]; cap = cv2.VideoCapture(B + "kiss_cheek.mp4"); [cap.read() for _ in range(60)]; ok, fr = cap.read()
fs = fa.get(fr); fA = min(fs, key=lambda q: np.linalg.norm(q.kps.mean(0) - kp[60, 0].mean(0))); fB = min(fs, key=lambda q: np.linalg.norm(q.kps.mean(0) - kp[60, 1].mean(0)))
recs = {0: {"lmk": fA.landmark_2d_106, "bbox": fA.bbox}, 1: {"lmk": fB.landmark_2d_106, "bbox": fB.bbox}}
cp._order_cues(fr, recs); cp._ORDER_MEM.clear(); cp._ORDER_NOW[0] = 0
aA = float((fA.bbox[2] - fA.bbox[0]) * (fA.bbox[3] - fA.bbox[1])); aB = float((fB.bbox[2] - fB.bbox[0]) * (fB.bbox[3] - fB.bbox[1]))
check("contact frame: the outline-edge cue puts B (really in front) in front, though A's box is larger" if aA > aB else "contact frame: the outline-edge cue puts B (really in front) in front",
      cp._in_front(1, 0, aB, aA) is True, f"cue {cp._ORDER_CUE}")
print("ALL PASS" if fails == 0 else f"{fails} FAILED")
