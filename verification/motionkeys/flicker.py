"""usage: flicker.py LABEL...  Temporal flicker of the PASTED face, measured in the face's own frame.
Each output (and input) frame is warped to the ArcFace-aligned 256 crop using the TRUE key points of the synthetic clip, so the
head's motion is removed. For the input, consecutive aligned crops differ only by noise, blur and the squeeze; for the output they also
differ by whatever the swap/paste does frame to frame. Reported: mean |change| between consecutive aligned crops inside the
central face (output, input, ratio), the 95th percentile frame, and the mean brightness pumping of the central face."""
import sys, cv2, numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/pack88')
import swap_engine as E
def frames(path):
    c = cv2.VideoCapture(path); out = []
    while True:
        ok, f = c.read()
        if not ok: break
        out.append(f)
    return out
S = 256
yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
core = (((xx - S * 0.5) / (S * 0.30)) ** 2 + ((yy - S * 0.52) / (S * 0.36)) ** 2) < 1.0
def aligned(img, k): return cv2.warpAffine(img, E.estimate_norm(k.astype(np.float32), S), (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
def analyse(label, start=22):
    clip = label.split('_')[2]; W = f"/tmp/claude-0/bench_work/{label}"
    inp = frames(f"turn_{clip}.mp4"); out = frames(f"{W}/out/BENCHTS_/final.mp4"); kps = np.load(f"turn_{clip}_gt.npz")['kps']; n = min(len(inp), len(out))
    ai = [aligned(inp[t], kps[t]).astype(np.float32) for t in range(n)]; ao = [aligned(out[t], kps[t]).astype(np.float32) for t in range(n)]
    di = np.array([np.abs(ai[t] - ai[t - 1]).mean(axis=2)[core].mean() for t in range(start, n)])
    do = np.array([np.abs(ao[t] - ao[t - 1]).mean(axis=2)[core].mean() for t in range(start, n)])
    pi = np.array([abs(ai[t].mean(axis=2)[core].mean() - ai[t - 1].mean(axis=2)[core].mean()) for t in range(start, n)])
    po = np.array([abs(ao[t].mean(axis=2)[core].mean() - ao[t - 1].mean(axis=2)[core].mean()) for t in range(start, n)])
    # step pattern: ratio of the biggest frame-to-frame change to the median (steps at the swap cadence)
    return {"aligned-face change per frame, output / input": f"{do.mean():.2f} / {di.mean():.2f}  (x{do.mean() / max(di.mean(), 1e-3):.2f})",
            "worst 5 % of frames, output / input": f"{np.percentile(do, 95):.2f} / {np.percentile(di, 95):.2f}",
            "brightness pumping of the aligned face, output / input": f"{po.mean():.2f} / {pi.mean():.2f}",
            "step ratio (max change / median) output": f"{do.max() / max(np.median(do), 1e-3):.1f}"}
if __name__ == "__main__":
    for lab in sys.argv[1:]:
        print("==", lab)
        for k, v in analyse(lab).items(): print(f"   {k}: {v}")
