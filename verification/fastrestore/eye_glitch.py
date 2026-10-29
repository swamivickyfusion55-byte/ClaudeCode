"""usage: eye_glitch.py VIDEO...   Public talking/nodding videos (LivePortrait's driving examples; no user footage).
Per frame, the detector's 5 points and the 106-point landmarks. A 'glitch' is a one-frame jump of the two EYE points that
the nose and mouth points do not share: second temporal difference of the eye points minus that of the nose/mouth,
in % of the eye distance, above 6 %. For each glitch: was the head tipped down (nose further below the eyes than the
clip's median: vert ratio) and were the eyes closing (106-point eye height / width vs median)? Also the same measure on
the 106-point eye centres, which are a second, independent estimate of the eyes."""
import sys, cv2, numpy as np
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.28)
def eye_open(L, idx):
    p = L[idx]; return (p[:, 1].max() - p[:, 1].min()) / max(1e-3, p[:, 0].max() - p[:, 0].min())
tot = dict(frames=0, g5=0, g106=0, g5_down=0, g5_closed=0)
for v in sys.argv[1:]:
    cap = cv2.VideoCapture(v); K, L106 = [], []
    while True:
        ok, fr = cap.read()
        if not ok: break
        fs = fa.get(fr)
        if not fs: K.append(None); L106.append(None); continue
        f = max(fs, key=lambda q: (q.bbox[2] - q.bbox[0]) * (q.bbox[3] - q.bbox[1]))
        K.append(f.kps.astype(np.float32)); L106.append(f.landmark_2d_106.astype(np.float32))
    n = len(K); ok = [i for i in range(1, n - 1) if K[i - 1] is not None and K[i] is not None and K[i + 1] is not None]
    if not ok: print(v, "no faces"); continue
    def vert(k):
        ev = k[1] - k[0]; ed = np.linalg.norm(ev); r = np.arctan2(ev[1], ev[0]); c, s = np.cos(-r), np.sin(-r); rel = k[2] - (k[0] + k[1]) / 2
        return (rel[0] * s + rel[1] * c) / ed
    vmed = np.median([vert(K[i]) for i in ok]); om = np.median([eye_open(L106[i], list(range(33, 43))) for i in ok])
    g5 = g106 = gd = gc = 0; ex = []
    for i in ok:
        ed = float(np.linalg.norm(K[i][1] - K[i][0])) + 1e-6
        d2 = np.linalg.norm(K[i] - 0.5 * (K[i - 1] + K[i + 1]), axis=1) / ed * 100
        spike = float(d2[:2].mean() - d2[2:].mean())
        e106 = lambda L: np.array([L[33:43].mean(0), L[87:97].mean(0)])
        if L106[i - 1] is not None:
            d3 = np.linalg.norm(e106(L106[i]) - 0.5 * (e106(L106[i - 1]) + e106(L106[i + 1])), axis=1) / ed * 100
            s106 = float(d3.mean() - d2[2:].mean())
        else: s106 = 0
        if spike > 6:
            g5 += 1; down = vert(K[i]) > vmed * 1.08; closed = eye_open(L106[i], list(range(33, 43))) < om * 0.8
            gd += down; gc += closed; ex.append(f"f{i}:{spike:.0f}%{' down' if down else ''}{' closing' if closed else ''}")
        if s106 > 6: g106 += 1
    print(f"{v.split('/')[-1]:8s} {len(ok):4d} frames: detector eye glitches {g5:3d} (head down {gd}, eyes closing {gc}); 106-point eye glitches {g106:3d}   {' '.join(ex[:6])}")
    tot["frames"] += len(ok); tot["g5"] += g5; tot["g106"] += g106; tot["g5_down"] += gd; tot["g5_closed"] += gc
print("TOTAL", tot)
