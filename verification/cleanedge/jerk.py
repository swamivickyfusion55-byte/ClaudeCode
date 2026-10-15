import sys, cv2, numpy as np
exec(open('cheekmetrics.py').read().split('def metrics')[0])
def jerk(frs, sl, boxes):
    reg = [np.stack([f[b[1]:b[3], b[0]:b[2]].astype(np.float32).mean(axis=2) for f in frs]) for b in boxes]
    out = []
    for R in reg:
        d2 = np.abs(R[2:] - 2 * R[1:-1] + R[:-2]).mean(axis=(1, 2)); out.append(d2[sl].mean())
    return float(np.mean(out))
sl = slice(15, 81)
for lab in sys.argv[1:]:
    clip = "cheek_swell.mp4" if lab.endswith("swell") else "cheek_static.mp4"
    inp = frames(clip); out = frames(f"/tmp/claude-0/bench_work/{lab}/out/BENCHCK_/final.mp4"); n = min(len(inp), len(out))
    print(lab, "jerk in-band: input %.2f  output %.2f   (ratio %.2f)" % (jerk(inp[:n], sl, band_boxes()), jerk(out[:n], sl, band_boxes()), jerk(out[:n], sl, band_boxes()) / jerk(inp[:n], sl, band_boxes())))
