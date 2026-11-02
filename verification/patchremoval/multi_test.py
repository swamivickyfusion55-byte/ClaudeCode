"""usage: multi_test.py PACKDIR CLIPDIR      Engine-level test of 10 marks at once (make_patchclip.py multi clip: ten moles/birthmarks on the
same skin, the skin moves, turns, changes brightness, an object passes at frames 40-59). Per mark: how much of it is left, frames missed.
Synthetic data only."""
import sys, time, numpy as np, cv2
PACK, CLIPS = sys.argv[1], sys.argv[2]
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
inp = frames(f"{CLIPS}/pr_multi.mp4"); cln = frames(f"{CLIPS}/pr_multi_clean.mp4")
g = np.load(f"{CLIPS}/pr_multi_gt.npz"); n = len(inp); K = int(g["nmarks"])
grays = [E._pr_gray(f) for f in inp]; thumbs = [E._pr_thumb(x) for x in grays]
specs = [(g["ref"], g[f"refmask_{k}"], 20, 45, f"m{k}") for k in range(K)]
t0 = time.time(); msgs = []
tracks = E.plan_tracks(specs, (inp[0].shape[1], inp[0].shape[0]), n, thumbs, lambda i: grays[i], lambda i: inp[i], log=msgs.append)
tp = time.time() - t0
t0 = time.time(); out = []
for i, f in enumerate(inp):
    for tr in tracks:
        f = tr.apply(f, i)
    out.append(f)
ta = (time.time() - t0) / n
L = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
bad = 0
print(f"{K} marks: {len(tracks)} followed, planned in {tp:.1f}s, {ta * 1000:.0f} ms/frame for all of them")
for k in range(K):
    ks = []
    for i in range(n):
        mk = g["marks_all"][k][i]
        if mk.sum() < 30: continue
        lc, lo, li = L(cln[i]), L(out[i]), L(inp[i])
        con = float((lc - li)[mk].mean())
        if con <= 3: continue
        ks.append(float((lc - lo)[mk].mean()) / con)
    miss = sum(1 for x in ks if x > 0.5)
    bad += 1 if miss > 0.25 * len(ks) else 0
    print(f"  mark {k}: left {100 * np.mean(ks):.1f}% of its contrast, missed on {miss}/{len(ks)} frames")
for m_ in msgs:
    if "WARNING" in m_ or "positions" in m_: print("   ", m_)
print("ALL MARKS REMOVED" if bad == 0 else f"{bad} MARK(S) NOT REMOVED")
