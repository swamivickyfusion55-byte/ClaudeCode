"""Measure the final keep-map (occluder_keep_final) on real faces + realistic food, with the real TrackState."""
import sys, numpy as np, cv2, types, collections
PACK = sys.argv[1]; sys.path.insert(0, PACK)
import swap_engine as E, config; E.configure(**config.ENGINE_TUNABLES)
import os
for kv in filter(None, os.environ.get('OVR','').split(',')):
    k_, v_ = kv.split('='); E._P[k_] = float(v_)
sys.path.insert(0, '/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/lip'); sys.path.insert(0, '.')
sys.argv = [sys.argv[0]] + sys.argv[2:]
exec(open('/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/lip/realset.py').read())
from food import food
FOODS = {"vanilla": (235, 240, 248), "chocolate": (40, 55, 85), "strawberry": (150, 150, 235), "mint": (150, 215, 170), "mango": (60, 150, 240), "caramel": (70, 150, 215)}
def seq_positions():
    """(cx,cy,r_scale) per frame: still (learn) / enter / hold / leave."""
    pos = [None] * 40
    for i in range(8): u = (i + 1) / 8.0; pos.append((0.5 + 0.30 * (1 - u), 0.76 + 0.22 * (1 - u)))
    for i in range(24): pos.append((0.5 + 0.012 * np.sin(i / 3.0), 0.76 + 0.010 * np.cos(i / 4.0)))
    for i in range(8): u = (i + 1) / 8.0; pos.append((0.5 + 0.30 * u, 0.76 + 0.22 * u))
    return pos
def run(crop256, S, col, rng, r=0.19, track_cls=E.TrackState, fn=None):
    crop = cv2.resize(crop256, (S, S), interpolation=cv2.INTER_CUBIC) if S != 256 else crop256
    tr = track_cls(0); out = []
    for t, p in enumerate(seq_positions()):
        base = jitter256(crop, rng, S)
        if p is None: f = base; a = np.zeros((S, S), np.float32)
        else: f, a = food(base, p[0], p[1], r, col, rng)
        f = np.clip(f, 0, 255).astype(np.uint8)
        keep = (fn or E.AlignedCompositor.occluder_keep_final)(tr, f)
        out.append((t, p, keep.copy(), a, f))
    return out
def jitter256(crop, rng, S, amp=1.5):
    ang = rng.normal(0, 0.7); sc = 1 + rng.normal(0, 0.008); tx, ty = rng.normal(0, amp * S / 256.0, 2)
    M = cv2.getRotationMatrix2D((S / 2, S / 2), ang, sc); M[:, 2] += (tx, ty)
    o = cv2.warpAffine(crop, M, (S, S), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    return o + rng.standard_normal(o.shape).astype(np.float32) * 1.8
def metrics(frames, S):
    """Measured only where the face would actually be painted (canonical template > 0.5, above the chin ramp)."""
    ker = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    tm = E.canonical_template(S)
    rows = (np.arange(S)[:, None] + 0.5) / S
    support = (tm > 0.5) & (rows < 0.84)
    res = collections.defaultdict(list); det = []
    prev_keep = None
    for t, p, keep, a, f in frames:
        pk, prev_keep = prev_keep, keep
        if p is not None and 40 + 1 <= t < 40 + 8 + 3:                  # object still arriving
            g_ = (a > 0.9) & support
            if g_.sum() > 300: res["arrival: face alpha over object"].append(float(keep[g_].mean()))
        if p is None or t < 40 + 8 + 4 or t >= 40 + 8 + 24: continue   # steady hold frames only
        gt = (a > 0.5).astype(np.uint8)
        gts = gt.astype(bool) & support
        if gts.sum() < 200: continue
        er = cv2.erode(gt, ker(max(2, S // 50))).astype(bool) & support
        cut_share = float((keep[gts] < 0.5).mean())
        det.append(cut_share > 0.3)
        if cut_share <= 0.3: continue                                      # not detected: counted in "detected %", not in the quality numbers
        res["interior face alpha (mean)"].append(float(keep[er].mean()))
        res["interior px, face alpha>0.2 (%)"].append(100 * float((keep[er] > 0.2).mean()))
        edge = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, ker(1)).astype(bool) & support
        L = float(edge.sum()) / 2.0 + 1e-6                                # ~ boundary length in px
        band = cv2.dilate(gt, ker(max(2, S // 60))).astype(bool) & ~cv2.erode(gt, ker(max(2, S // 60))).astype(bool) & support
        res["soft edge width (px)"].append(float(((keep > 0.1) & (keep < 0.9) & band).sum() / L))
        pred = keep < 0.5
        res["edge error (px)"].append(float(((pred ^ (gt > 0)) & support).sum() / L))
        near = cv2.dilate(gt, ker(max(3, S // 20))).astype(bool) & ~cv2.dilate(gt, ker(max(2, S // 60))).astype(bool) & support
        far = ~cv2.dilate(gt, ker(max(3, S // 20))).astype(bool) & support
        if pk is not None:
            res["edge flicker (mean |dkeep|)"].append(float(np.abs(keep - pk)[band].mean()))
        inner = cv2.erode(gt, ker(max(1, S // 160))).astype(bool) & ~cv2.erode(gt, ker(max(3, S // 55))).astype(bool) & support
        if inner.sum() > 20: res["halo: face alpha in the object's inner edge band"].append(float(keep[inner].mean()))
        res["face wrongly cut near (%)"].append(100 * float((keep[near] < 0.5).mean()))
        res["face wrongly cut far (%)"].append(100 * float((keep[far] < 0.5).mean()))
    out = {k: float(np.mean(v)) for k, v in res.items()}
    out["detected frames (%)"] = 100 * float(np.mean(det)) if det else 0.0
    return out
if __name__ == "__main__":
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    foods = sys.argv[3].split(',') if len(sys.argv) > 3 else list(FOODS)
    faces = load_faces()[:nf]; rng = np.random.default_rng(11)
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for name in foods:
        col = FOODS[name]
        for fi, face in enumerate(faces):
            fr = run(face, S, col, rng)
            m = metrics(fr, S)
            for k, v in m.items(): agg[name][k].append(v)
    keys = ["arrival: face alpha over object","detected frames (%)","interior face alpha (mean)","interior px, face alpha>0.2 (%)","soft edge width (px)","edge error (px)","halo: face alpha in the object's inner edge band","edge flicker (mean |dkeep|)","face wrongly cut near (%)","face wrongly cut far (%)"]
    print(f"crop {S}px | per food (mean over 6 faces)")
    print(f"{'':>12}" + "".join(f"{k[:23]:>24}" for k in keys))
    for name in foods:
        print(f"{name:>12}" + "".join(f"{np.nanmean(agg[name].get(k,[np.nan])):>24.2f}" for k in keys))
    print(f"{'ALL':>12}" + "".join(f"{np.nanmean([np.nanmean(agg[n].get(k,[np.nan])) for n in foods]):>24.2f}" for k in keys))
