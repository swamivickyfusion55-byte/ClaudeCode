"""v11.2.141: where each person's paint crop sits against the person's TRUE face, frame by frame (mask dump vs ground truth).
usage: geom_offset.py GT.npz MASK_DUMP.npz [MASK_DUMP2.npz ...] [--frames=a:b:step]
Offset = distance between the crop's centre mapped into the frame and the centre the same engine has on clean frames (the median
crop-centre minus true-5-point-centre over frames where the person is fully apart), so 0 px = where it belongs."""
import sys, cv2, numpy as np
args = [a for a in sys.argv[1:] if not a.startswith("--")]; fr = next((a.split("=")[1] for a in sys.argv if a.startswith("--frames=")), "36:92:3")
a0, a1, st = (int(x) for x in fr.split(":"))
g = np.load(args[0]); kps = g["kps"]; vis = g["a"]; full = (vis[:, 0] > 127).sum(axis=(1, 2)).max()
rows = {}
for dp in args[1:]:
    z = np.load(dp); sl = z["slot"]; idx = {int(s): np.nonzero(sl == s)[0] for s in np.unique(sl)}
    def centre(k):
        M = z["M"][k]; size = int(z["size"][k]); IM = cv2.invertAffineTransform(M); return IM[:, :2] @ np.array([size / 2, size / 2]) + IM[:, 2]
    sA = min(idx, key=lambda s: float(np.linalg.norm(centre(idx[s][0]) - kps[0, 0].mean(0))))
    off0 = np.median([centre(idx[sA][t]) - kps[t, 0].mean(0) for t in range(0, 12)], axis=0)
    rows[dp] = [float(np.linalg.norm(centre(idx[sA][t]) - kps[t, 0].mean(0) - off0)) for t in range(len(idx[sA]))]
print("frame  A visible  " + "  ".join("off(%d)" % i for i in range(len(rows))))
for t in range(a0, a1, st):
    print("%5d  %6.0f %%  " % (t, 100 * (vis[t, 0] > 127).sum() / full) + "  ".join("%6.0f" % rows[d][t] for d in args[1:]))
for i, d in enumerate(args[1:]):
    r = np.array(rows[d][30:95]); print("run %d: mean offset frames 30-94 %.1f px, worst %.1f px" % (i, r.mean(), r.max()))
