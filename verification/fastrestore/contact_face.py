"""usage: contact_face.py RUN:CLIP ...   (both faces swapped; mask dump of the run; make_kiss.py clips, synthetic, public photo)
Like contact_spill.py, but measured on each person's FACE (not the whole elliptical cut-out, which also holds wall, hair,
an arm): the face outline is the 106-point landmark hull (plus the usual forehead estimate) found once on the source photo
and moved into every frame by the exact similarity the clip used (fitted from the clip's true 5 points).
  seen on B's face    = A's paint left visible after B is painted over it, A x (1 - B), on B's visible face, %
  A on B's face       = A's raw paint weight there, % (before B covers it)
  B on A's face       = B's paint on A's visible face, % (B is painted last, so all of it is seen)
  own face painted    = each face's paint on its own visible face (eroded 6 px), %
Contact frames only (the two people within 12 px)."""
import sys, cv2, numpy as np
sys.path.insert(0, "/home/user/ClaudeCode/v11.2.95_OcclusionPro"); import swap_engine as E
from insightface.app import FaceAnalysis
B = "/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/"
FACE = {"f2": (466, 268, 573, 415), "f3": (745, 338, 845, 479)}            # make_kiss.py: A = f2, B = f3
src = cv2.imread("/usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg")
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"])
fa.prepare(ctx_id=-1, det_size=(640, 640)); fs = fa.get(src); PH = []
for nm in ("f2", "f3"):
    x1, y1, x2, y2 = FACE[nm]; c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    f = min(fs, key=lambda q: float(np.linalg.norm((q.bbox[:2] + q.bbox[2:]) / 2 - c)))
    PH.append((f.kps.astype(np.float32), E.with_forehead(f.landmark_2d_106).astype(np.float32)))
ero = np.ones((13, 13), np.uint8)
for arg in sys.argv[1:]:
    run, clip = arg.split(":")
    gt = np.load(B + f"kiss_{clip}_gt.npz"); vis, kps = gt["a"], gt["kps"]; n = len(vis)
    z = np.load(f"/tmp/claude-0/bench_work/{run}/mask_dump.npz"); ent = {}
    for m, M, size, sl in zip(z["m"], z["M"], z["size"], z["slot"]):
        ent.setdefault(int(sl), []).append((m, M, int(size)))
    smap = {}
    for sl, L in ent.items():
        votes = [0, 0]
        for k, (m, M, size) in enumerate(L[:8]):
            IM = cv2.invertAffineTransform(M); c = IM[:, :2] @ np.array([size / 2, size / 2]) + IM[:, 2]
            votes[int(np.argmin([np.linalg.norm(c - kps[k, p].mean(0)) for p in range(2)]))] += 1
        smap[sl] = int(np.argmax(votes))
    R = {k: [] for k in ("seen", "a_on_b", "b_on_a", "own_a", "own_b")}
    for t in range(n):
        a = vis[t, 0] > 127; b = vis[t, 1] > 127
        if (cv2.dilate(a.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) & b).sum() < 500:
            continue
        face = []
        for p in range(2):
            T, _ = cv2.estimateAffinePartial2D(PH[p][0], kps[t, p].astype(np.float32))
            pts = PH[p][1] @ T[:, :2].T + T[:, 2]; fm = np.zeros((720, 1280), np.uint8)
            cv2.fillConvexPoly(fm, cv2.convexHull(np.round(pts).astype(np.int32)), 1); face.append(fm.astype(bool))
        fa_v, fb_v = face[0] & a, face[1] & b                                # visible face of A, of B
        W = {}
        for sl, L in ent.items():
            if t < len(L):
                m, M, size = L[t]; W[smap[sl]] = cv2.warpAffine(cv2.resize(m, (size, size)), cv2.invertAffineTransform(M), (1280, 720))
        if 0 not in W or 1 not in W:
            continue
        if fb_v.sum() > 1500:
            R["seen"].append(float((W[0] * (1 - W[1]))[fb_v].mean()) * 100); R["a_on_b"].append(float(W[0][fb_v].mean()) * 100)
            R["own_b"].append(float(W[1][cv2.erode(fb_v.astype(np.uint8), ero).astype(bool)].mean()) * 100)
        if fa_v.sum() > 1500:
            R["b_on_a"].append(float(W[1][fa_v & ~b].mean()) * 100 if (fa_v & ~b).any() else 0.0)
            R["own_a"].append(float(W[0][cv2.erode(fa_v.astype(np.uint8), ero).astype(bool)].mean()) * 100)
    s = lambda k: f"{np.mean(R[k]):.1f}% (worst {np.max(R[k]):.1f}%)" if R[k] else "-"
    print(f"{run}: A's paint seen on B's face {s('seen')} [A raw {s('a_on_b')}]; B's paint on A's face {s('b_on_a')}; "
          f"own face painted A {np.mean(R['own_a']):.0f}% (lowest {np.min(R['own_a']):.0f}%), B {np.mean(R['own_b']):.0f}% (lowest {np.min(R['own_b']):.0f}%)")
