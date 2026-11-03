"""v11.2.103 unit checks (the paint stays on the body through fast moves: immediate re-detection, blur-tolerant acceptance,
smoothing that does not cut a curve).  usage: unit103.py PACKDIR      Synthetic data only; a stand-in inpainter (no model)."""
import sys, os, types, math, numpy as np, cv2
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


E.PATCH_INPAINTER = lambda crop, mk: cv2.inpaint(crop, (mk > 0).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
H, W = 480, 900
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
rng = np.random.default_rng(7)
tex = sum(cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), s) * a for s, a in ((1.2, 9), (3.0, 16), (9.0, 30)))
base = np.dstack([150 + 0.02 * xx, 168 + 0.01 * yy, 200 + 0 * xx]).astype(np.float32)
world = np.clip(base + tex[..., None] * 0.9, 0, 255)
cx0, cy0 = 300.0, 240.0
mole = (((yy - cy0) ** 2 + (xx - cx0) ** 2) <= 7 ** 2).astype(np.float32)
world_m = world * (1 - 0.55 * cv2.GaussianBlur(mole, (0, 0), 1.0)[..., None])


def view(dx, dy, ang=0.0, sc=1.0, blur=0):
    M = cv2.getRotationMatrix2D((cx0, cy0), ang, sc)
    M[0, 2] += dx; M[1, 2] += dy
    f = cv2.warpAffine(world_m, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if blur > 1:
        k = np.zeros((blur, blur), np.float32); k[blur // 2, :] = 1.0 / blur
        f = cv2.filter2D(f, -1, k)
    return np.clip(f, 0, 255).astype(np.uint8), M


u8 = lambda im: np.clip(im, 0, 255).astype(np.uint8)
hole = (((yy - cy0) ** 2 + (xx - cx0) ** 2) <= 13 ** 2)

# --- 1. smoothing never cuts a curve; still removes jitter -----------------------------------------------------------
tr = E.PatchTrack.__new__(E.PatchTrack)
tr.box, tr.size, tr.weak = (200, 280, 200, 280), 40, set()
n = 80
t = np.arange(n)
xs_curve = 230 * np.sin(2 * np.pi * t * 4 / 78.0)                        # a fast swing: peak speed ~72 px per frame
tr.A = {int(i): np.array([[1, 0, float(xs_curve[i])], [0, 1, 0]], float) for i in t}
tr.conf = {int(i): 1.0 for i in t}
E.PatchTrack.smooth(tr)
err = max(abs(tr.A[int(i)][0, 2] - xs_curve[i]) for i in t[3:-3])
check("smoothing: a fast swing (72 px/frame, turning round) is not cut at the turns (under 2.5 px off)", err < 2.5, f"max {err:.2f} px")
r = np.random.default_rng(1).normal(0, 0.6, n)
xs_line = 3.0 * t + r
tr.A = {int(i): np.array([[1, 0, float(xs_line[i])], [0, 1, 0]], float) for i in t}
tr.conf = {int(i): 1.0 for i in t}
E.PatchTrack.smooth(tr)
after = np.array([tr.A[int(i)][0, 2] for i in t]) - 3.0 * t
check("smoothing: jitter on a steady move is still removed (spread down by 35 % or more)", after[3:-3].std() < 0.65 * r[3:-3].std(), f"{r[3:-3].std():.2f} -> {after[3:-3].std():.2f} px")

# --- 2. local template search ---------------------------------------------------------------------------------------------
a_img, _ = view(0, 0)
trk = E.PatchTrack(0, a_img, hole)
shift = (58.0, -31.0)
cur, Mtrue = view(*shift)
gray = E._pr_gray(cur)
near = np.array([[1, 0, 22.0], [0, 1, -8.0]], np.float64)               # a prediction that is 40 px off
A_ = trk._redetect_tm(gray, around=near, radius=120.0)
c0 = np.array([cx0, cy0])
perr = lambda A: float(np.hypot(*((A[:, :2] @ c0 + A[:, 2]) - (Mtrue[:, :2] @ c0 + Mtrue[:, 2]))))
check("local template search: finds the patch 40 px from the prediction", A_ is not None and perr(A_) < 3.0, "none" if A_ is None else f"{perr(A_):.1f} px off")
far = np.array([[1, 0, -300.0], [0, 1, 0.0]], np.float64)
check("local template search: a prediction far from the patch (radius 60 px) finds nothing there", trk._redetect_tm(gray, around=far, radius=60.0) is None)

# --- 3. following a fast, blurred move: immediate re-detection vs the old wait --------------------------------------------
def run(redetect_now, blur_ok):
    E.PATCH_CFG["redetect_now"], E.PATCH_CFG["blur_ok"] = redetect_now, blur_ok
    tk = E.PatchTrack(0, a_img, hole)
    tk.A = {}; tk.conf = {}
    frames, truth = [], []
    for i in range(70):
        f = 0.0 if i < 8 else 1.0
        ph = max(0, i - 8)
        dx, dy = f * 150 * math.sin(ph / 3.2), f * 45 * math.sin(ph / 4.1 + 0.7)
        ang = f * 18 * math.sin(ph / 6.0)
        blur = int(min(15, abs(150 / 3.2 * math.cos(ph / 3.2)) * 0.5)) | 1 if f else 1
        im, M = view(dx, dy, ang, 1.0 + f * 0.10 * math.sin(ph / 5.0), blur)
        frames.append(im); truth.append(M)
    grays = [E._pr_gray(x) for x in frames]
    tk.track(lambda i: grays[i], list(range(0, 70)))
    ok = 0
    for i in range(70):
        A = tk.A.get(i)
        if A is None:
            continue
        pc = A[:, :2] @ c0 + A[:, 2]; pt = truth[i][:, :2] @ c0 + truth[i][:, 2]
        ok += float(np.hypot(*(pc - pt))) < 4.0
    return ok
old, new = run(False, False), run(True, True)
E.PATCH_CFG["redetect_now"] = E.PATCH_CFG["blur_ok"] = True
check("fast blurred move (peak ~47 px/frame, turning, zooming): followed within 4 px on 90 % of the 70 frames", new >= 63, f"{new}/70 (old way: {old}/70)")
check("...and clearly better than waiting for two lost frames", new >= old + 5 or new >= 68, f"{new} against {old}")
check("config: redetect_now, blur_ok and smooth_err are on", bool(E.PATCH_CFG["redetect_now"]) and bool(E.PATCH_CFG["blur_ok"]) and float(E.PATCH_CFG["smooth_err"]) > 0)

print("\n%d check(s) failed" % fails if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
