"""usage: unit92.py PACK   Unit checks of the v11.2.92 additions (real photos from insightface's sample image; no user footage)."""
import sys, os, types, cv2, numpy as np
PACK = sys.argv[1]; sys.path.insert(0, PACK)
g = types.ModuleType("gradio"); g.Request = type("Request", (), {}); g.update = lambda **k: k; g.Error = Exception; sys.modules["gradio"] = g
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: "/tmp/claude-0/swapmodel/inswapper_128.onnx"; sys.modules["huggingface_hub"] = hf
os.environ.setdefault("PHOENIX_MODEL_DIR", "/tmp/claude-0/occmodels")
import swap_engine as E, core_pipeline as cp
from insightface.app import FaceAnalysis
fails = 0
def check(name, ok, info=""):
    global fails
    fails += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info else ""))
# --- side-view weight -------------------------------------------------------------------------
def kps_for(h):   # eyes 60 px apart, nose h eye-distances sideways
    return np.float32([[100, 100], [160, 100], [130 + 60 * h, 135], [110, 165], [150, 165]])
check("frontal face: side-view weight 0", E.profile_weight(kps_for(0.0)) == 0.0)
check("45 deg (0.45): weight 0", E.profile_weight(kps_for(0.45)) == 0.0)
w55 = E.profile_weight(kps_for(0.58)); check("between: weight rises smoothly", 0.2 < w55 < 0.8, f"{w55:.2f}")
check("side view (0.76): weight 1", E.profile_weight(kps_for(0.76)) == 1.0)
check("mirror image gives the same weight", E.profile_weight(kps_for(-0.6)) == E.profile_weight(kps_for(0.6)))
E._P["profile_mask"] = 0.0; check("profile_mask 0 switches it off", E.profile_weight(kps_for(1.0)) == 0.0); E._P["profile_mask"] = 1.0
check("no key points: weight 0, no error", E.profile_weight(None) == 0.0 and E.profile_weight(np.zeros((2, 2))) == 0.0)
# --- outline mask ------------------------------------------------------------------------------
check("outline mask without net and without landmarks: None", E.profile_mask(np.zeros((128, 128, 3), np.uint8), None, 128) is None)
pts = np.float32([[40 + 50 * np.cos(t), 64 + 40 * np.sin(t)] for t in np.linspace(0, 6.28, 106)])
pm = E.profile_mask(np.zeros((128, 128, 3), np.uint8), pts, 128)
check("outline mask without net = landmark outline", pm is not None and pm[64, 40] > 0.9 and pm[64, 120] < 0.01)
check("outline mask is zero at the crop border", pm is not None and float(pm[0].max()) == 0.0 and float(pm[:, -1].max()) == 0.0)
# --- eating rule -------------------------------------------------------------------------------
S = 128; lips = np.zeros((106, 2), np.float32); lips[:] = (64, 64)
ang = np.linspace(0, 6.28, 20, endpoint=False); lips[52:72] = np.stack([64 + 18 * np.cos(ang), 92 + 7 * np.sin(ang)], 1)
keep = np.ones((S, S), np.float32)
check("eating rule: nothing cut -> mask unchanged", np.array_equal(E._eating_mouth(np.ones((S, S), np.float32), keep, lips, S), np.ones((S, S), np.float32)))
k2 = keep.copy(); k2[88:96, 56:72] = 0.0
check("eating rule: cut only inside the lips (open mouth) -> unchanged", float(E._eating_mouth(np.ones((S, S), np.float32), k2, lips, S).min()) == 1.0)
k3 = keep.copy(); k3[95:125, 50:78] = 0.0
m3 = E._eating_mouth(np.ones((S, S), np.float32), k3, lips, S)
check("eating rule: object reaching out past the lips -> lips go back to the footage", float(m3[92, 64]) < 0.2 and float(m3[40, 64]) == 1.0, f"lip {m3[92,64]:.2f}")
class T: eat_w = 0.0
t = T(); seq = [float(E._eating_mouth(np.ones((S, S), np.float32), k3 if i < 6 else keep, lips, S, t)[92, 64]) for i in range(16)]
check("eating rule: fast in, slow out (no flash when the object leaves)", seq[2] < 0.3 and 0.2 < seq[8] < 0.95 and seq[-1] > seq[8], " ".join(f"{v:.2f}" for v in seq[::3]))
# --- GPEN ----------------------------------------------------------------------------------------
check("enhancer names: GPEN, GPEN + Cinematic, CodeFormer run GPEN", all(cp._is_gpen(n) for n in ("GPEN", "GPEN (fast restore)", "GPEN + Cinematic", "CodeFormer")))
check("enhancer names: Cinematic, None, GFPGAN do not", not any(cp._is_gpen(n) for n in ("Cinematic (clarity + smooth)", "None", "GFPGAN", None)))
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
img = cv2.imread("/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg")
img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
f = max(fa.get(img), key=lambda q: q.bbox[2] - q.bbox[0])
crop = cv2.warpAffine(img, E.estimate_norm(f.kps, 128), (128, 128))
soft = cv2.GaussianBlur(crop, (0, 0), 1.2)
cp._GPEN.update({"calls": 0, "ms": 0.0, "blend_sum": 0.0, "skipped": 0})
out, det = cp._gpen_restore(soft)
check("GPEN without footage reference: full blend, 256 crop", out is not None and out.shape == (256, 256, 3) and abs(cp._GPEN["blend_sum"] - 0.6) < 1e-6)
sharp_ref = cv2.resize(cv2.warpAffine(img, E.estimate_norm(f.kps, 512), (512, 512)), (128, 128), interpolation=cv2.INTER_AREA)
sharp_ref = cv2.addWeighted(sharp_ref, 1.8, cv2.GaussianBlur(sharp_ref, (0, 0), 1.0), -0.8, 0)
cp._GPEN.update({"calls": 0, "blend_sum": 0.0, "skipped": 0})
out2, det2 = cp._gpen_restore(soft, target=sharp_ref)
check("GPEN on sharp footage: restores (strength > 0)", out2 is not None and det2 is not None and cp._GPEN["blend_sum"] > 0.05, f"strength {cp._GPEN['blend_sum']:.2f}")
cp._GPEN.update({"calls": 0, "blend_sum": 0.0, "skipped": 0})
blurry = cv2.GaussianBlur(crop, (0, 0), 3.0)
out3, det3 = cp._gpen_restore(soft, target=blurry)
check("GPEN on blurry footage: network skipped, crop unchanged", det3 is None and out3 is not None and np.array_equal(out3, soft) and cp._GPEN["skipped"] == 1)
ga = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY).astype(np.float32); gt = cv2.cvtColor(cv2.resize(sharp_ref, (256, 256), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY).astype(np.float32)
check("GPEN result no sharper than 1.25 x the footage (+ the swap's own)", cp._lap_energy(ga) <= max(cp._lap_energy(gt) * 1.25, cp._lap_energy(cv2.cvtColor(cv2.resize(soft, (256, 256), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY).astype(np.float32))) * 1.0001)
o_a, d_a = cp._gpen_restore(soft, prev_detail=None); o_b, d_b = cp._gpen_restore(soft, prev_detail=np.zeros_like(d_a))
check("steady detail: previous detail is mixed in (half of it here)", np.allclose(d_b, d_a * 0.5, atol=1e-3))
print("ALL PASS" if fails == 0 else f"{fails} FAILED")
