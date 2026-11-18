import sys, cv2, numpy as np
# sbs3.py OUT.png CLIP RES_A RES_B frames crop  (input | 212 | 213)
out, clip, ra, rb, fr = sys.argv[1:6]; crop = [int(v) for v in sys.argv[6].split(",")]
g = np.load(clip[:-4] + "_gt.npz")
def get(p, i):
    c = cv2.VideoCapture(p); c.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = c.read(); return f[crop[1]:crop[3], crop[0]:crop[2]].copy()
rows = []
for i in [int(v) for v in fr.split(",")]:
    a, b, c = get(clip, i), get(ra, i), get(rb, i)
    cv2.putText(a, "in %d: %.0f deg" % (i, abs(g["yaw"][i])), (8, 28), 0, 0.8, (0, 255, 255), 2)
    cv2.putText(b, "11.2.212", (8, 28), 0, 0.8, (0, 255, 255), 2); cv2.putText(c, "11.2.213", (8, 28), 0, 0.8, (0, 255, 255), 2)
    rows.append(np.hstack([a, b, c]))
img = np.vstack(rows); s = min(1.0, 1500 / img.shape[1]); cv2.imwrite(out, cv2.resize(img, None, fx=s, fy=s))
