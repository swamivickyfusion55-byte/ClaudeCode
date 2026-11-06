"""The v11.2.117 belly clip (its tattoo spans 0.03-0.17 above the hip line, partly below the new floor): how much of it each pack finds."""
import sys, os
sys.path.insert(0, sys.argv[1])
import numpy as np, cv2, belly_clear as BC
CL = sys.argv[2]; T = np.load(os.path.join(CL, "truth.npz"))
c = cv2.VideoCapture(os.path.join(CL, "clip.mp4")); fr = []
while True:
    ok, f = c.read()
    if not ok: break
    fr.append(f)
tot, above = [], []
for i in (0, 22, 45, 67):
    t = T["lm"][i]; L = {11: (*t[0], 1.0), 12: (*t[1], 1.0), 23: (*t[2], 1.0), 24: (*t[3], 1.0)}
    mk, _ = BC.find_marks(fr[i], L); mk = np.zeros(fr[i].shape[:2], bool) if mk is None else mk
    tat = T["mask"][i]
    lh, rh, ls, rs = (np.array(L[k][:2], np.float64) for k in (23, 24, 11, 12))
    pl, pr = lh + 0.10 * (ls - lh), rh + 0.10 * (rs - rh); v = (pr - pl) / np.linalg.norm(pr - pl); nrm = np.array([v[1], -v[0]])
    if np.dot(nrm, 0.5 * (ls + rs) - 0.5 * (lh + rh)) < 0: nrm = -nrm
    yy, xx = np.mgrid[0:tat.shape[0], 0:tat.shape[1]]; fl = (xx - pl[0]) * nrm[0] + (yy - pl[1]) * nrm[1]
    up = tat & (fl >= 8)
    tot.append((mk & tat).sum() / tat.sum()); above.append((mk & up).sum() / max(1, up.sum()))
    share = up.sum() / tat.sum()
print("%s: whole tattoo found %.1f %% (mean of 4 frames); its part above the floor (%.0f %% of it) found %.1f %%" % (
      os.path.basename(sys.argv[1]), 100 * np.mean(tot), 100 * share, 100 * np.mean(above)))
