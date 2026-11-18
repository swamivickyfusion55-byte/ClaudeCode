import sys, cv2, numpy as np
def frames(p):
    c = cv2.VideoCapture(p); out = []
    while True:
        ok, f = c.read()
        if not ok: break
        out.append(f)
    return out
for a, b in zip(sys.argv[1::2], sys.argv[2::2]):
    A, B = frames(a), frames(b)
    d = [float(np.abs(x.astype(np.int16) - y.astype(np.int16)).max()) for x, y in zip(A, B)]
    m1 = np.load(a.rsplit("/out/", 1)[0] + "/mask_dump.npz"); m2 = np.load(b.rsplit("/out/", 1)[0] + "/mask_dump.npz")
    md = max(float(np.abs(m1[k].astype(np.float32) - m2[k].astype(np.float32)).max()) for k in m1.files if k in m2.files)
    print("%s vs %s: frames %d/%d, largest pixel difference %.0f, frames differing %d, paste masks largest difference %.4f" % (
        a.split("/")[4], b.split("/")[4], len(A), len(B), max(d), sum(x > 0 for x in d), md))
