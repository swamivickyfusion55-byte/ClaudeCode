"""Image-domain metrics for the cheek clips: change inside the SCOOP and inside the CONE, against the input footage."""
import sys, cv2, numpy as np
exec(open('foodmetrics.py').read().split('def analyse')[0])
POS = {"L": (0.372, 0.579), "R": (0.621, 0.579), "M": (0.50, 0.667)}
def analyse2(label, spec, outdir="BENCHFD_"):
    W = f"/tmp/claude-0/bench_work/{label}"; name, pz = spec.split('_'); cx0, cy0 = POS[pz]
    inp = frames(f"food_{spec}.mp4"); out = frames(f"{W}/out/{outdir}/final.mp4"); gt = np.load(f"food_{spec}_gt.npz")["a"]
    ker = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    yy, xx = np.mgrid[0:720, 0:1280].astype(np.float32)
    res = {"scoop interior |out-in|": [], "scoop px |out-in|>12 (%)": [], "cone |out-in|": [], "cone px |out-in|>12 (%)": [], "face |out-in|": []}
    sup = np.zeros((720, 1280), bool); sup[BX[1]:BX[3], BX[0]:BX[2]] = True
    for i in range(45, 82):
        g = gt[i] > 127
        if g.sum() < 500: continue
        # scoop position in the hold phase
        cx = cx0 + 0.008 * np.sin((i - 35) / 3.0); cy = cy0 + 0.006 * np.cos((i - 35) / 4.0)
        sc = (((xx - (cx * 720 + 280)) / (0.134 * 720)) ** 2 + ((yy - cy * 720) / (0.134 * 0.92 * 720)) ** 2) < 1.0
        D = np.abs(out[i].astype(np.float32) - inp[i].astype(np.float32)).mean(axis=2)
        si = cv2.erode(sc.astype(np.uint8), ker(6)).astype(bool) & sup
        cone = g & ~cv2.dilate(sc.astype(np.uint8), ker(3)).astype(bool) & sup
        far = ~cv2.dilate(g.astype(np.uint8), ker(25)).astype(bool) & sup & (np.arange(720)[:, None] > 300)
        if si.sum() < 200 or far.sum() < 500: continue
        res["scoop interior |out-in|"].append(float(D[si].mean())); res["scoop px |out-in|>12 (%)"].append(100 * float((D[si] > 12).mean()))
        if cone.sum() > 300: res["cone |out-in|"].append(float(D[cone].mean())); res["cone px |out-in|>12 (%)"].append(100 * float((D[cone] > 12).mean()))
        res["face |out-in|"].append(float(D[far].mean()))
    return {k: round(float(np.mean(v)), 2) if v else float('nan') for k, v in res.items()}
if __name__ == "__main__":
    for lab in sys.argv[1:]:
        spec = "_".join(lab.split("_")[-2:]); print(lab, analyse2(lab, spec))
