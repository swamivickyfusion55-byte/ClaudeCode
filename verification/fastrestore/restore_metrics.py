"""usage: restore_metrics.py LABEL...  (label ts_<tag>_<clip>_<Q>, clip turn_<clip>.mp4 with true key points)
Per run: flicker of the pasted face in its own frame (motionkeys/flicker.py), sharpness of the face against the footage
(Laplacian variance ratio in the aligned central face), fine-detail change per frame (shimmer: frame-to-frame change of the
high-pass of the aligned face), and identity: ArcFace cosine of every painted face to the nearer replacement photo."""
import sys, cv2, numpy as np
sys.path.insert(0, "/home/user/ClaudeCode/verification/motionkeys"); import flicker as FL
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
B = "/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/"
srcs = [max(fa.get(cv2.imread(B + n)), key=lambda f: f.bbox[2] - f.bbox[0]).normed_embedding for n in ("src_a.jpg", "src_b.jpg")]
for lab in sys.argv[1:]:
    clip = lab.split("_")[2]; W = f"/tmp/claude-0/bench_work/{lab}"
    inp = FL.frames(B + f"turn_{clip}.mp4"); out = FL.frames(f"{W}/out/BENCHTS_/final.mp4"); kps = np.load(B + f"turn_{clip}_gt.npz")["kps"]
    n = min(len(inp), len(out)); fl = FL.analyse(lab) if True else {}
    sh, hp = [], []; prev = None
    for t in range(22, n):
        ai = FL.aligned(inp[t], kps[t]); ao = FL.aligned(out[t], kps[t])
        gi = cv2.cvtColor(ai, cv2.COLOR_BGR2GRAY).astype(np.float32); go = cv2.cvtColor(ao, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sh.append(cv2.Laplacian(go, cv2.CV_32F)[FL.core].var() / max(1e-3, cv2.Laplacian(gi, cv2.CV_32F)[FL.core].var()))
        h = go - cv2.GaussianBlur(go, (0, 0), 2.0)
        if prev is not None: hp.append(float(np.abs(h - prev)[FL.core].mean()))
        prev = h
    ids = []
    for t in range(22, n, 4):
        for f in fa.get(out[t]): ids.append(max(float(np.dot(f.normed_embedding, s)) for s in srcs))
    print("==", lab)
    for k, v in fl.items(): print(f"   {k}: {v}")
    print(f"   sharpness vs footage (Laplacian var ratio), median: {np.median(sh):.2f}")
    print(f"   fine-detail change per frame (shimmer): {np.mean(hp):.2f}")
    print(f"   identity (ArcFace cos to the replacement photo), mean / min: {np.mean(ids):.3f} / {np.min(ids):.3f}  ({len(ids)} faces)")
