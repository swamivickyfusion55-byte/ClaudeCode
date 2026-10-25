"""usage: bitemetrics.py TAG COLOUR [NONE_TAG]   Lip region of the partial-bite clip (make_bite.py), frames with the scoop
at the lip (38-84). Every pixel of the footage's own lips (outer lip contour, landmarks 52-63) not covered by the scoop is classed by which reference the output is nearer to:
the footage's own lip (food clip as filmed) or the swapped lip (the same job on the no-food clip, NONE_TAG run).
Reports the share of each; "two lips" = the smaller of the two shares (both kinds of lip visible at once); "ghost" =
pixels far from BOTH (a see-through mix of the two)."""
import sys, cv2, numpy as np
sys.path.insert(0, "/home/user/ClaudeCode/verification/motionkeys"); import flicker as FL
B = "/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/"
tag, col = sys.argv[1], sys.argv[2]; none_tag = sys.argv[3] if len(sys.argv) > 3 else "b92"
foot = FL.frames(B + f"bite_{col}.mp4"); out = FL.frames(f"/tmp/claude-0/bench_work/tb_{tag}_{col}/out/BENCHTB_/final.mp4")
ref = FL.frames(f"/tmp/claude-0/bench_work/tb_{none_tag}_none/out/BENCHTB_/final.mp4")
z = np.load(B + f"bite_{col}_gt.npz"); A = z["a"]; x0, y0, x1, y1 = z["lips"]; pw, ph = (x1 - x0) * 0.12, (y1 - y0) * 0.12
x0, y0, x1, y1 = int(x0 - pw), int(y0 - ph), int(x1 + pw), int(y1 + ph)
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
f0 = min(fa.get(FL.frames(B + "bite_none.mp4")[0]), key=lambda f: abs((f.bbox[0] + f.bbox[2]) / 2 - 640))
LIP = np.zeros((720, 1280), np.uint8); cv2.fillPoly(LIP, [np.round(f0.landmark_2d_106[52:64]).astype(np.int32)], 1)
LIP = LIP[y0:y1, x0:x1].astype(bool)     # the footage's own lips (outer contour), the only pixels counted
def blur(a): return cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.5)
rows = []
for t in range(38, 85):
    o, f, r = (blur(v[t][y0:y1, x0:x1]) for v in (out, foot, ref))
    free = (A[t][y0:y1, x0:x1] < 20) & LIP
    do, ds = np.abs(o - f).mean(2), np.abs(o - r).mean(2)
    orig = (do + 4 < ds) & free; sw = (ds + 4 < do) & free; ghost = (do > 12) & (ds > 12) & free
    nf = max(1, free.sum()); rows.append((orig.sum() / nf, sw.sum() / nf, ghost.sum() / nf))
R = np.array(rows) * 100
print(f"{tag} {col}: lip area showing the footage's lip {R[:,0].mean():.0f}%, the swapped lip {R[:,1].mean():.0f}%, "
      f"two lips (smaller share) mean {np.minimum(R[:,0], R[:,1]).mean():.0f}% worst {np.minimum(R[:,0], R[:,1]).max():.0f}%, ghost {R[:,2].mean():.1f}%")
