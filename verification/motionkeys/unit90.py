import sys, types, numpy as np, cv2
PACK = sys.argv[1] if len(sys.argv) > 1 else '/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/pack90'
sys.path.insert(0, PACK)
g = types.ModuleType('gradio'); g.Request = type('Request', (), {}); g.update = lambda **k: k; g.Error = Exception; sys.modules['gradio'] = g
hf = types.ModuleType('huggingface_hub'); hf.hf_hub_download = lambda *a, **k: ''; sys.modules['huggingface_hub'] = hf
import swap_engine as E, config; E.configure(**config.ENGINE_TUNABLES)
import core_pipeline as C
ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name + (" | " + extra if extra else ""))
# a textured "face" photo to move around
src = cv2.imread('/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg')
face = src[145:267, 268:372].copy(); face = cv2.resize(face, (220, 260))
rng = np.random.default_rng(0)
def frame(cx, cy, ang=0.0, sx=1.0, noise=1.5):
    bg = np.full((720, 1280, 3), (120, 140, 150), np.uint8)
    M = cv2.getRotationMatrix2D((110, 130), ang, 1.0); M[0, 0] *= sx; M[0, 1] *= sx
    M[:, 2] += (cx - 110, cy - 130)
    m = cv2.warpAffine(np.ones(face.shape[:2], np.uint8) * 255, M, (1280, 720)); w = cv2.warpAffine(face, M, (1280, 720))
    out = np.where(m[:, :, None] > 0, w, bg).astype(np.float32) + rng.standard_normal((720, 1280, 3)).astype(np.float32) * noise
    return np.clip(out, 0, 255).astype(np.uint8)
def recs(cx, cy, det=True, w=220, h=260):
    return {0: {"bbox": np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32), "det": det, "alpha": 1.0}}
def run(seq, centres, det=True):
    """seq: frames; centres: face centre per frame. Returns the frame offsets at which an extra swap was asked for."""
    fmk = C._FaceMotionKeys(); fmk.mark_swap(recs(*centres[0]), 0); got = []
    for i in range(1, len(seq)):
        kind = fmk.check(seq[i - 1], seq[i], recs(*centres[i - 1], det=det), recs(*centres[i], det=det), i)
        if kind:
            fmk.commit(kind); got.append(i); fmk.mark_swap(recs(*centres[i]), i)
    return got, fmk
# --- still: only noise
still = [frame(640, 360) for _ in range(30)]
got, f = run(still, [(640, 360)] * 30); check("a still face asks for no extra key frame (30 frames, sensor noise)", got == [], f"{got}, peak change {f.peak_change:.3f}")
# --- slow drift
slow = [frame(640 + 0.5 * i, 360) for i in range(30)]
got, f = run(slow, [(640 + 0.5 * i, 360) for i in range(30)]); check("a slow drift (0.5 px/frame, 0.2 % of the face width): nothing inside the first 12 frames (longest preset gap is 6), at most one in 30", all(x > 12 for x in got) and len(got) <= 1, f"{got}")
# --- fast translation
fast = [frame(640 + 40.0 * i, 360) for i in range(12)]
got, f = run(fast, [(640 + 40.0 * i, 360) for i in range(12)]); check("a fast sweep (40 px/frame = 0.18 face widths) asks for an extra swap on every frame", len(got) >= 9 and all(b - a <= 1 for a, b in zip([0] + got, got)), f"{got}, by move {f.by_move}")
# --- turning in place: rotation + squeeze, no travel
turn = [frame(640, 360, ang=18 * np.sin(i * 0.9), sx=1 - 0.25 * abs(np.sin(i * 0.6))) for i in range(24)]
got, f = run(turn, [(640, 360)] * 24); check("turning/rolling in place (no travel) asks for key frames by change of look", len(got) >= 3 and f.by_change == len(got), f"{got}, peak change {f.peak_change:.2f}")
# --- translation alone with a good velocity estimate does not trigger the 'change' rule
fast2 = [frame(640 + 12.0 * i, 360) for i in range(30)]
got, f = run(fast2, [(640 + 12.0 * i, 360) for i in range(30)]); check("steady 12 px/frame travel (0.055 face widths) triggers by travel only, every ~2 frames, never by change", f.by_change == 0, f"{got}, by move {f.by_move}, peak change {f.peak_change:.3f}")
# --- minimum gap and cap
E._P['adaptive_key_min_gap'] = 3
got, f = run(fast, [(640 + 40.0 * i, 360) for i in range(12)]); check("min gap 3 keeps extra key frames 3 frames apart", all(b - a >= 3 for a, b in zip([0] + got, got)), f"{got}")
E._P['adaptive_key_min_gap'] = 1
E._P['adaptive_key_max_frac'] = 0.2
many = [frame(640 + 60.0 * i, 360) for i in range(40)]
got, f = run(many, [(640 + 60.0 * i, 360) for i in range(40)]); check("the cap (20 % of frames + 8) bounds the extra key frames", len(got) <= 0.2 * 40 + 8 + 1, f"{len(got)} of 40")
E._P['adaptive_key_max_frac'] = 0.6
# --- held / predicted geometry is not measured
got, f = run(fast, [(640 + 40.0 * i, 360) for i in range(12)], det=False); check("held or predicted geometry (det False) never triggers", got == [], f"{got}")
f = C._FaceMotionKeys()
check("before the first swap nothing is asked", not f.check(fast[0], fast[1], recs(640, 360), recs(680, 360), 1))
f.mark_swap({}, 0); check("a frame with no face records never asks", not f.check(fast[0], fast[1], {}, {}, 1))
# --- off switch
E._P['adaptive_key'] = 0.0; check("adaptive_key = 0 -> enabled() is False", not C._FaceMotionKeys.enabled()); E._P['adaptive_key'] = 1.0
# --- degenerate boxes do not raise
f = C._FaceMotionKeys(); f.mark_swap(recs(5000, 5000), 0)
check("a box outside the frame does not raise", f.check(fast[0], fast[1], recs(5000, 5000), recs(5100, 5000), 1) in (None, 'move', 'change'))
# --- the swap-gap function is unchanged
check("_adaptive_swap_gap unchanged (Balanced base 4, HIGH -> 2)", C._adaptive_swap_gap(4, "HIGH", "Balanced") == 2)
print("ALL PASS" if ok else "FAILURES")
