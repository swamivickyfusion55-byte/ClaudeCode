"""usage: bench_eval.py CLIPDIR RUN:name ...   Score a full-pipeline job (bench_pipeline.py patches=...) on a
make_patchclip.py clip: the job's final.mp4 against the clean twin, on the mark (frames where it is in the picture)."""
import sys, glob, numpy as np, cv2
def frames(p):
    c = cv2.VideoCapture(p); o = []
    while True:
        ok, f = c.read()
        if not ok: return o
        o.append(f)
L = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
for arg in sys.argv[2:]:
    run, name = arg.split(":")
    out = frames(glob.glob(f"/tmp/claude-0/bench_work/{run}/out/*/final.mp4")[0])
    inp, cln = frames(f"{sys.argv[1]}/pr_{name}.mp4"), frames(f"{sys.argv[1]}/pr_{name}_clean.mp4")
    g = np.load(f"{sys.argv[1]}/pr_{name}_gt.npz"); mark = g["mark"]
    n = min(len(out), len(inp), len(mark))
    left, missed, cnt = [], 0, 0
    for i in range(n):
        mk = mark[i]
        if mk.sum() <= 30: continue
        o = cv2.resize(out[i], (inp[i].shape[1], inp[i].shape[0])) if out[i].shape != inp[i].shape else out[i]
        lc, lo, li = L(cln[i]), L(o), L(inp[i])
        con = float((lc - li)[mk].mean()); rem = float((lc - lo)[mk].mean())
        f = rem / con if con > 3 else 0.0
        left.append(f); missed += f > 0.5
    left = np.array(left)
    print(f"{run} ({name}): {n} frames out of the job; mark left {100*left.mean():.1f}% on average, missed on {missed}/{len(left)} frames where it is in the picture")
