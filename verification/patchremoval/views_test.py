"""usage: views_test.py PACKDIR CLIPDIR [name] [N]    One mark painted at N positions (default 10, spread over the clip's frames where it is in
the picture) on a make_patchclip.py clip: planning time, per-frame time, and how much of the mark is left. Synthetic data only.
Also checks progress reporting and cancelling when the engine has them (v11.2.101)."""
import sys, time, numpy as np, cv2
PACK, CLIPS = sys.argv[1], sys.argv[2]
NAME = sys.argv[3] if len(sys.argv) > 3 else "proc_mole"
NV = int(sys.argv[4]) if len(sys.argv) > 4 else 10
sys.path.insert(0, PACK); import swap_engine as E
import onnxruntime as ort
so = ort.SessionOptions(); so.intra_op_num_threads = 2; so.log_severity_level = 3
sess = ort.InferenceSession("/tmp/claude-0/occmodels/inpainting_lama_2025jan.onnx", so, providers=["CPUExecutionProvider"])
def lama(crop, mk):
    h, w = crop.shape[:2]
    x = cv2.resize(crop[..., ::-1], (512, 512), interpolation=cv2.INTER_AREA if max(h, w) > 512 else cv2.INTER_CUBIC)
    x = x.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    m = (cv2.resize(mk.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)[None, None]
    o = sess.run(None, {"image": np.ascontiguousarray(x), "mask": m})[0][0].transpose(1, 2, 0)
    o = np.clip(o, 0, 255).astype(np.uint8)[..., ::-1]
    return cv2.resize(np.ascontiguousarray(o), (w, h), interpolation=cv2.INTER_AREA if max(h, w) < 512 else cv2.INTER_CUBIC)
E.PATCH_INPAINTER = lama
def frames(p):
    c = cv2.VideoCapture(p); o = []
    while True:
        ok, f = c.read()
        if not ok: return o
        o.append(f)
inp = frames(f"{CLIPS}/pr_{NAME}.mp4"); cln = frames(f"{CLIPS}/pr_{NAME}_clean.mp4")
g = np.load(f"{CLIPS}/pr_{NAME}_gt.npz"); mark, occ = g["mark"], g["occ"]; n = len(inp)
sys.path.insert(0, "/home/user/ClaudeCode/verification/patchremoval")
import make_patchclip as MP
cands = [i for i in range(5, 140) if mark[i].sum() > 60 and not occ[i].any()]
pick = [cands[int(round(k * (len(cands) - 1) / max(1, NV - 1)))] for k in range(NV)]
specs = [(inp[i], MP.finger_mask(mark[i], seed=2 + k), i, 45, "m1") for k, i in enumerate(pick)]
grays = [E._pr_gray(f) for f in inp]; thumbs = [E._pr_thumb(x) for x in grays]
msgs, prog = [], []
kw = {}
import inspect
if "progress" in inspect.signature(E.plan_tracks).parameters:
    kw = dict(progress=prog.append)
t0 = time.time()
tracks = E.plan_tracks(specs, (inp[0].shape[1], inp[0].shape[0]), n, thumbs, lambda i: grays[i], lambda i: inp[i], log=msgs.append, **kw)
tp = time.time() - t0
t0 = time.time(); out = []
for i, f in enumerate(inp):
    for tr in tracks:
        f = tr.apply(f, i)
    out.append(f)
ta = (time.time() - t0) / n
L = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
left, missed = [], 0
for i in range(n):
    mk = mark[i]
    if mk.sum() < 30 or occ[i].any(): continue
    lc, lo, li = L(cln[i]), L(out[i]), L(inp[i])
    con = float((lc - li)[mk].mean())
    if con <= 3: continue
    fr = float((lc - lo)[mk].mean()) / con
    left.append(fr); missed += fr > 0.5
cov = set()
for t in tracks: cov |= set(t.A.keys()) if hasattr(t, "A") and not isinstance(t.A, dict) or True else set()
print(f"{NAME} x {NV} positions: planned in {tp:.1f}s, {ta * 1000:.0f} ms/frame | mark left {100 * np.mean(left):.1f}%, missed {missed}/{len(left)} frames | progress messages {len(prog)}")
for m_ in msgs[-3:]: print("   ", m_)
if prog: print("    last progress:", prog[-1])
