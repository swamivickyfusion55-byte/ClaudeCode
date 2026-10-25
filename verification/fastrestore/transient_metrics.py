"""usage: transient_metrics.py cut TAG...   |   transient_metrics.py fast TAG...
cut:  runs tk_<TAG>_cut / tk_<TAG>_nocut against tk_<TAG>_ref (make_cutclip.py). Each output frame after the shot change is
      aligned to the true face position and compared with the same frame of the second shot rendered alone: a deformed
      face right after the cut shows as a large difference that settles back to the noise level.
fast: run tk_<TAG>_mix on turn_mix70: per frame the second temporal difference of the aligned painted face
      (|A_t - (A_t-1 + A_t+1)/2|), output against the footage's own: a one-frame deformation is a spike."""
import sys, cv2, numpy as np
sys.path.insert(0, "/home/user/ClaudeCode/verification/motionkeys"); import flicker as FL
B = "/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/"
import glob
def out(lab):
    fs = glob.glob(f"/tmp/claude-0/bench_work/tk_{lab}/out/*/final.mp4")
    assert len(fs) == 1, (lab, fs)
    return FL.frames(fs[0])
mode = sys.argv[1]
if mode == "cut":
    z = np.load(B + "cut_AB_gt.npz"); K = z["kps"]
    for tag in sys.argv[2:]:
        ref = out(f"{tag}_ref")
        for kind in ("cut", "nocut"):
            o = out(f"{tag}_{kind}")
            d = [float(np.abs(FL.aligned(o[t], K[t]).astype(np.float32) - FL.aligned(ref[t], K[t]).astype(np.float32)).mean(2)[FL.core].mean()) for t in range(30, 60)]
            print(f"{tag} {kind:5s}: after the shot change, difference to the second shot alone (grey levels) frames 30-37: "
                  + " ".join(f"{v:.1f}" for v in d[:8]) + f" | settled (frames 50-59) {np.mean(d[20:]):.1f}")
else:
    gt = np.load(B + "turn_mix70_gt.npz")["kps"]; inp = FL.frames(B + "turn_mix70.mp4")
    for tag in sys.argv[2:]:
        o = out(f"{tag}_mix"); n = min(len(o), len(inp), len(gt))
        ai = [FL.aligned(inp[t], gt[t]).astype(np.float32) for t in range(n)]; ao = [FL.aligned(o[t], gt[t]).astype(np.float32) for t in range(n)]
        si = np.array([np.abs(ai[t] - 0.5 * (ai[t - 1] + ai[t + 1])).mean(2)[FL.core].mean() for t in range(23, n - 1)])
        so = np.array([np.abs(ao[t] - 0.5 * (ao[t - 1] + ao[t + 1])).mean(2)[FL.core].mean() for t in range(23, n - 1)])
        r = so / np.maximum(si, 0.5); top = np.argsort(-r)[:5]
        print(f"{tag}: one-frame spikes of the painted face, output/footage: median x{np.median(r):.2f}, worst 5: "
              + ", ".join(f"frame {23 + i} x{r[i]:.1f}" for i in top) + f"; frames above x3: {(r > 3).sum()}")
