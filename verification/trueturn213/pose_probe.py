import sys, glob, math, time, numpy as np, cv2, scipy.io as sio
sys.path.insert(0, "/home/user/ClaudeCode/verification/fastrestore")
from head3d import Head
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "landmark_3d_68"], providers=["CPUExecutionProvider"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
m3 = fa.models["landmark_3d_68"]
print("render (true yaw, pitch 0) -> the 3D-landmark model's pitch / yaw / roll, and its time")
h = Head("/tmp/claude-0/head3d/LeePerrySmith.glb", "/tmp/claude-0/head3d/Map-COL.jpg")
t1 = cv2.imread("/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg"); bg = cv2.GaussianBlur(cv2.resize(t1, (1280, 720)), (0, 0), 18)
for yaw in [0, 20, 40, 55, 62, 68, 72, 76, 80, 85, 90, 100, 110, 120, 135]:
    for side in (1, -1):
        im, cov, _ = h.render(side * yaw, 4, -3, 1280, 720, (640, 420), 760, bg)
        fs = fa.det_model.detect(im, max_num=0, metric="default")
        b, k = fs
        if b is None or len(b) == 0:
            print("  yaw %+4d: not detected" % (side * yaw)); continue
        from insightface.app.common import Face
        i = int(np.argmax(b[:, 4])); f = Face(bbox=b[i, :4], kps=k[i], det_score=b[i, 4])
        t = time.time(); m3.get(im, f); dt = (time.time() - t) * 1000
        print("  yaw %+4d: pose pitch %+6.1f yaw %+6.1f roll %+6.1f  (%.0f ms)" % (side * yaw, f.pose[0], f.pose[1], f.pose[2], dt))
print("AFLW2000 samples (annotated yaw)")
for p in sorted(glob.glob("/tmp/claude-0/prnet/TestImages/AFLW2000/*.jpg")):
    pp = sio.loadmat(p[:-4] + ".mat")["Pose_Para"][0]; im = cv2.imread(p)
    fs = fa.get(im)
    if not fs: print("  %s true yaw %+6.1f: not detected" % (p[-14:], math.degrees(pp[1]))); continue
    f = max(fs, key=lambda q: q.det_score)
    print("  %s true yaw %+6.1f pitch %+6.1f | model yaw %+6.1f pitch %+6.1f" % (p[-14:], math.degrees(pp[1]), math.degrees(pp[0]), f.pose[1], f.pose[0]))
