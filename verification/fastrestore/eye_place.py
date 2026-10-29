"""usage: eye_place.py RUN:CLIP ...   Where the swapped eyes end up against the real eyes, frame by frame (public videos).
Eye centres from the 106-point landmarks on the footage and on the output (the swap model keeps the target's pose, so
the drawn eyes should sit on the real ones; the replacement's eye SHAPE differs, its centre should not). Displacement
in % of the eye distance; the head is 'down' when the nose sits further below the eyes than the clip's median (x1.08)."""
import sys, glob, cv2, numpy as np
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.28)
B = "/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/"
def frames(p):
    c = cv2.VideoCapture(p); out = []
    while True:
        ok, f = c.read()
        if not ok: return out
        out.append(f)
def look(f):
    fs = fa.get(f)
    if not fs: return None
    q = max(fs, key=lambda q: (q.bbox[2] - q.bbox[0]) * (q.bbox[3] - q.bbox[1])); L = q.landmark_2d_106; k = q.kps
    ev = k[1] - k[0]; ed = np.linalg.norm(ev); r = np.arctan2(ev[1], ev[0]); c, s = np.cos(-r), np.sin(-r); rel = k[2] - (k[0] + k[1]) / 2
    return np.array([L[33:43].mean(0), L[87:97].mean(0)]), ed, (rel[0] * s + rel[1] * c) / ed
for arg in sys.argv[1:]:
    run, clip = arg.split(":")
    inp = frames(B + clip); out = frames(glob.glob(f"/tmp/claude-0/bench_work/{run}/out/*/final.mp4")[0])
    rows = []
    for t in range(min(len(inp), len(out))):
        o = out[t]
        if o.shape[:2] != inp[t].shape[:2]:      # rendered at another resolution: compare in the footage's pixels
            o = cv2.resize(o, (inp[t].shape[1], inp[t].shape[0]), interpolation=cv2.INTER_AREA)
        a, b = look(inp[t]), look(o)
        if a is None or b is None: continue
        d = np.linalg.norm(a[0] - b[0], axis=1).max() / a[1] * 100
        rows.append((t, d, a[2]))
    D = np.array([r[1] for r in rows]); V = np.array([r[2] for r in rows]); med = np.median(V); down = V > med * 1.08
    bad = [(t, d) for (t, d, v) in rows if d > max(10.0, 2.5 * np.median(D))]
    print(f"{run}: {len(rows)} frames, eye displacement median {np.median(D):.1f}% (head down {np.median(D[down]) if down.any() else float('nan'):.1f}%, "
          f"otherwise {np.median(D[~down]):.1f}%), 95% {np.percentile(D, 95):.1f}%, max {D.max():.1f}%; "
          f"frames with a displaced eye: {len(bad)} {[(t, round(d)) for t, d in bad[:10]]}; of them with the head down: {sum(1 for t, d in bad if down[[r[0] for r in rows].index(t)])}")
