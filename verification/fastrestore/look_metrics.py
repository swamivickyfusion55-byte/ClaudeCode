"""usage: look_metrics.py LABEL...  (runs ts_<tag>_sharp_Ultra or any run on turn_sharp.mp4; public photo, no user footage)
How natural the painted face looks against the footage, in the aligned central face (true key points):
  brightness: mean L (LAB) of the output face minus the footage's (0 = as bright as the shot)
  highlights: share of face pixels above L 200, output vs footage
  texture: fine-detail energy (L minus its 1.5-px blur, mean abs) output / footage (1 = as much skin texture as the shot)
  colour: mean chroma output / footage"""
import sys, glob, cv2, numpy as np
sys.path.insert(0, "/home/user/ClaudeCode/verification/motionkeys"); import flicker as FL
B = "/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/"
inp = FL.frames(B + "turn_sharp.mp4"); K = np.load(B + "turn_sharp_gt.npz")["kps"]
def stats(img, k):
    a = cv2.cvtColor(FL.aligned(img, k), cv2.COLOR_BGR2LAB).astype(np.float32)
    L = a[:, :, 0] * (100 / 255.0) * 2.55; c = FL.core
    tex = np.abs(L - cv2.GaussianBlur(L, (0, 0), 1.5))[c].mean()
    chroma = np.hypot(a[:, :, 1] - 128, a[:, :, 2] - 128)[c].mean()
    return L[c].mean(), (L[c] > 200).mean(), tex, chroma
for lab in sys.argv[1:]:
    fs = glob.glob(f"/tmp/claude-0/bench_work/{lab}/out/*/final.mp4"); o = FL.frames(fs[0])
    S = np.array([stats(o[t], K[t]) for t in range(10, 60, 5)]); R = np.array([stats(inp[t], K[t]) for t in range(10, 60, 5)])
    print(f"{lab:22s} brightness {S[:,0].mean() - R[:,0].mean():+5.1f}  highlights {S[:,1].mean()*100:4.1f}% (footage {R[:,1].mean()*100:4.1f}%)  "
          f"texture x{S[:,2].mean() / R[:,2].mean():.2f}  colour x{S[:,3].mean() / R[:,3].mean():.2f}")
