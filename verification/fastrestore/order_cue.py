"""usage: order_cue.py CLIP...   Which cue tells the face in FRONT when two faces overlap? (make_kiss.py clips: B is in
front; synthetic.) Per contact frame: (1) box area - the larger face is in front; (2) edge along the outline: mean image
gradient under each face's jaw/cheek outline points (106-point model, 0-32) that lie inside the OTHER face's outline -
the front face's outline sits on a real edge, the behind face's hidden outline is guessed across the other cheek."""
import sys, cv2, numpy as np
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
B = "/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/"
def edge_score(gm, L, other_hull):
    pts = [p for p in L[0:33] if cv2.pointPolygonTest(other_hull, (float(p[0]), float(p[1])), False) >= 0]
    if len(pts) < 3: return None
    v = []
    for p in pts:
        x, y = int(round(p[0])), int(round(p[1]))
        if 3 <= x < gm.shape[1] - 3 and 3 <= y < gm.shape[0] - 3: v.append(gm[y - 3:y + 4, x - 3:x + 4].max())
    return float(np.mean(v)) if v else None
for clip in sys.argv[1:]:
    cap = cv2.VideoCapture(B + f"kiss_{clip}.mp4"); kps = np.load(B + f"kiss_{clip}_gt.npz")["kps"]; t = 0
    area_ok = edge_ok = n = edge_n = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        fs = fa.get(fr)
        if len(fs) >= 2:
            # match detections to the true people
            fA = min(fs, key=lambda f: np.linalg.norm(f.kps.mean(0) - kps[t, 0].mean(0))); fB = min(fs, key=lambda f: np.linalg.norm(f.kps.mean(0) - kps[t, 1].mean(0)))
            if fA is not fB:
                hA = cv2.convexHull(fA.landmark_2d_106.astype(np.float32)); hB = cv2.convexHull(fB.landmark_2d_106.astype(np.float32))
                mA = np.zeros(fr.shape[:2], np.uint8); mB = mA.copy(); cv2.fillConvexPoly(mA, hA.astype(np.int32), 1); cv2.fillConvexPoly(mB, hB.astype(np.int32), 1)
                if (mA & mB).sum() > 300:
                    n += 1
                    aA = (fA.bbox[2] - fA.bbox[0]) * (fA.bbox[3] - fA.bbox[1]); aB = (fB.bbox[2] - fB.bbox[0]) * (fB.bbox[3] - fB.bbox[1])
                    area_ok += aB > aA
                    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    gm = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
                    eA, eB = edge_score(gm, fA.landmark_2d_106, hB), edge_score(gm, fB.landmark_2d_106, hA)
                    if eA is not None and eB is not None:
                        edge_n += 1; edge_ok += eB > eA
        t += 1
    print(f"{clip}: overlapping frames {n}: box area says B in front on {area_ok} ({100 * area_ok / max(1, n):.0f}%); outline edge says B in front on {edge_ok} of {edge_n} ({100 * edge_ok / max(1, edge_n):.0f}%)")
