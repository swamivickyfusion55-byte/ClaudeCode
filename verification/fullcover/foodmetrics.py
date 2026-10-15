import sys, cv2, numpy as np
def frames(path):
    c = cv2.VideoCapture(path); out = []
    while True:
        ok, f = c.read()
        if not ok: break
        out.append(f)
    return out
BX = (477, 144, 798, 585)
def analyse(label, name):
    W = f"/tmp/claude-0/bench_work/{label}"
    inp = frames(f"food_{name}.mp4"); out = frames(f"{W}/out/BENCHFD_/final.mp4"); gt = np.load(f"food_{name}_gt.npz")["a"]
    n = min(len(inp), len(out)); ker = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    sup = np.zeros((720, 1280), bool); sup[BX[1]:BX[3], BX[0]:BX[2]] = True
    # use only where the face is painted: lower-face rows of the bbox
    res = {"interior |out-in|": [], "interior px |out-in|>12 (%)": [], "face |out-in|": []}; prof = {d: [] for d in range(-8, 15)}
    for i in range(45, 82):
        g = gt[i] > 127
        if g.sum() < 500: continue
        D = np.abs(out[i].astype(np.float32) - inp[i].astype(np.float32)).mean(axis=2)
        gi = cv2.erode(g.astype(np.uint8), ker(6)).astype(bool) & sup
        far = ~cv2.dilate(g.astype(np.uint8), ker(25)).astype(bool) & sup & (np.arange(720)[:, None] > 300)
        if gi.sum() < 200 or far.sum() < 500: continue
        res["interior |out-in|"].append(float(D[gi].mean())); res["interior px |out-in|>12 (%)"].append(100 * float((D[gi] > 12).mean())); res["face |out-in|"].append(float(D[far].mean()))
        dout = cv2.distanceTransform((~g).astype(np.uint8), cv2.DIST_L2, 5); din = cv2.distanceTransform(g.astype(np.uint8), cv2.DIST_L2, 5)
        sd = np.where(g, -din, dout)
        for d in prof:
            m = sup & (sd >= d - 0.5) & (sd < d + 0.5) & (np.arange(720)[:, None] > 300)
            if m.sum() > 20: prof[d].append(float(D[m].mean()))
    r = {k: float(np.mean(v)) for k, v in res.items()}
    P = {d: np.mean(v) for d, v in prof.items() if v}
    plateau = np.mean([P[d] for d in range(10, 15) if d in P]); P = {d: v / plateau for d, v in P.items()}
    ds = sorted(P); 
    def cross(level):
        for a, b in zip(ds[:-1], ds[1:]):
            if P[a] < level <= P[b]: return a + (level - P[a]) / (P[b] - P[a]) * (b - a)
        return float('nan')
    r["edge 10-90% width (px)"] = cross(0.9) - cross(0.1); r["edge 50% position (px, + = face painted beyond the object)"] = cross(0.5)
    r["translucency (interior / face)"] = r["interior |out-in|"] / r["face |out-in|"]
    return r
if __name__ == "__main__":
    for lab in sys.argv[1:]:
        name = lab.split("_")[-1]
        print(lab, {k: round(v, 2) for k, v in analyse(lab, name).items()})
