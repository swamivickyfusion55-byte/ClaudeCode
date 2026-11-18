import sys, glob, numpy as np, cv2
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "landmark_3d_68"], providers=["CPUExecutionProvider"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
def rd(img): return [(abs(float(q.pose[1])), abs(float(q.pose[0])), abs(float(q.pose[2]))) for q in fa.get(img)]
for p in sys.argv[1:]:
    rows = []
    if p.endswith(".mp4"):
        c = cv2.VideoCapture(p); t = 0
        while True:
            ok, fr = c.read()
            if not ok: break
            if t % 3 == 0: rows += rd(fr)
            t += 1
    else:
        for q in sorted(glob.glob(p)): rows += rd(cv2.imread(q))
    r = np.array(rows)
    both = int(((r[:, 0] >= 55) & (r[:, 1] >= 40) & (r[:, 2] >= 40)).sum())
    print("%-26s faces %4d | yaw max %5.1f | |pitch| max %5.1f | |roll| max %5.1f | read as past profile (yaw 55+, both 40+): %d" % (
        p.split("/")[-1][:26], len(r), r[:, 0].max(), r[:, 1].max(), r[:, 2].max(), both))
