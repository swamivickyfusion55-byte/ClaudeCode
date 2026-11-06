"""v11.2.142: the belly clean through the JOB's own code - clean_belly.build_clean_belly (the adapter's step) -> core_pipeline._patch_specs
-> core_pipeline._patch_prepare (frames read as the render reads them, MediaPipe Pose in video mode on every 2nd frame, BodyPath,
swap_engine.anchor_to_body with the job's gray_at) -> PatchTrack.apply per frame, as the render does. Scored against the clip's clean twin.
usage: e2e142.py PACKDIR CLIPDIR   (make_bellytattoo.py clip; PHOENIX_MODEL_DIR holds the pose model, LAMA the fill model)"""
import sys, os, types, logging, io
PACK, CLIP = sys.argv[1], sys.argv[2]
sys.path.insert(0, PACK)
g = types.ModuleType("gradio"); g.Request = type("Request", (), {}); g.update = lambda **k: k; g.Error = Exception
sys.modules.setdefault("gradio", g)
buf = io.StringIO(); logging.basicConfig(level=logging.INFO, stream=buf, format="%(message)s")
import numpy as np, cv2
import core_pipeline as cp, clean_belly as CB
T = np.load(os.path.join(CLIP, "truth.npz"))
def frames(p):
    c = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(f)
    return fr
vp = os.path.join(CLIP, "clip.mp4"); src, cln = frames(vp), frames(os.path.join(CLIP, "clean.mp4"))
n = len(src); h, w = src[0].shape[:2]; fps = float(cv2.VideoCapture(vp).get(cv2.CAP_PROP_FPS) or 30.0)
pos = CB.build_clean_belly(vp)
specs = cp._patch_specs({"patches": pos})
print("positions from the job's belly clean: %d (%s); specs %d, groups %s" % (len(pos), ", ".join("t=%.2f" % p["t"] for p in pos), len(specs),
      sorted(set(s[3] for s in specs))), flush=True)
tracks = cp._patch_prepare(vp, specs, 0, n - 1, 1, n, w, h, lambda f: f, cv2.INTER_AREA, fps, fps, lambda p, m: None, lambda: False)
out = []
for i, f in enumerate(src):
    f = f.copy()
    for tr in tracks:
        f = tr.apply(f, i)
    out.append(f)
L = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
left = []
for i in range(n):
    m = T["mask"][i]
    con = float((L(cln[i]) - L(src[i]))[m].mean())
    left.append(float((L(cln[i]) - L(out[i]))[m].mean()) / max(1e-6, con))
left = np.array(left)
print("left of the marks: mean %.1f %%, worst %.1f %% (frame %d), frames over 25 %%: %d of %d" % (100 * left.mean(), 100 * left.max(), int(left.argmax()),
      int((left > 0.25).sum()), n))
for l in buf.getvalue().splitlines():
    if any(k in l for k in ("body anchor", "BODY", "clean belly", "PATCH-REMOVAL", "Traceback", "Error", "error")):
        print("  log: " + l[:400])
sys.stdout.flush(); os._exit(0)
