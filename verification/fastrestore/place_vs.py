"""v11.2.208: where one person's paint sits in run X against run Y (the same clip), frame by frame - for a person that is not
painted on frames 0-12, where contact_lag.py takes its calibration. Paint centres from the bench's MASK_DUMP; the person is
the slot whose first painted centre is nearest their true 5 points. usage: place_vs.py GT.npz DUMP_X.npz DUMP_Y.npz PERSON(0/1)"""
import sys, cv2, numpy as np
kps = np.load(sys.argv[1])["kps"]; p = int(sys.argv[4])
def centres(path):
    z = np.load(path); res = {}
    for k in range(len(z["slot"])):
        IM = cv2.invertAffineTransform(z["M"][k]); s = int(z["size"][k])
        res.setdefault(int(z["slot"][k]), {})[int(z["g"][k])] = IM[:, :2] @ np.array([s / 2, s / 2]) + IM[:, 2]
    best = min(res.items(), key=lambda kv: float(np.linalg.norm(kv[1][min(kv[1])] - kps[min(kv[1]), p].mean(0))))
    return best[1]
X, Y = centres(sys.argv[2]), centres(sys.argv[3])
both = sorted(set(X) & set(Y))
d = np.array([np.linalg.norm(X[t] - Y[t]) for t in both])
print("person %s: painted in X on %d frames (from %d), in Y on %d; on the %d frames both paint, X's paint centre is %.1f px from Y's on average, %.1f px at most"
      % ("AB"[p], len(X), min(X), len(Y), len(both), d.mean(), d.max()))
