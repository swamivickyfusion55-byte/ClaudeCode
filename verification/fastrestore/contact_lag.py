"""v11.2.144: does the paint stay ON each face through contact? Frame by frame, from the engine's own paint dump (bench_pipeline.py
MASK_DUMP=1, with the frame number) against the clip's true 5 points.
usage: contact_lag.py GT.npz MASK_DUMP.npz [--span=a:b]
Per person: the paint crop's centre mapped into the frame minus the person's true 5-point centre, less that person's own constant
offset (its median on frames 0-12, where the faces are apart: how this engine always places that face), so 0 px = where it belongs.
  off      mean / 90th percentile / worst offset over the span (px): lag or drift of the paint against the moving face
  jump     the largest change of that offset from one frame to the next (px): a snap when a lost face is found again
  unpainted  frames of the span where the person is at least 30 % visible but has no paint at all (a revert)"""
import sys, cv2, numpy as np
args = [a for a in sys.argv[1:] if not a.startswith("--")]
span = next((a.split("=")[1] for a in sys.argv if a.startswith("--span=")), "20:110")
a0, a1 = (int(x) for x in span.split(":"))
g = np.load(args[0]); kps, vis = g["kps"], g["a"]; n = len(kps)
full = [(vis[:, p] > 127).sum(axis=(1, 2)).max() for p in range(2)]
z = np.load(args[1]); sl, gg = z["slot"], z["g"]
byg = {}
for k in range(len(sl)):
    byg.setdefault(int(sl[k]), {})[int(gg[k])] = k
def centre(k):
    M = z["M"][k]; size = int(z["size"][k]); IM = cv2.invertAffineTransform(M); return IM[:, :2] @ np.array([size / 2, size / 2]) + IM[:, 2]
slot_of = {}
for p in range(2):
    cands = [(s, min(d.items(), key=lambda kv: kv[0])) for s, d in byg.items() if d]
    slot_of[p] = min(cands, key=lambda c: float(np.linalg.norm(centre(c[1][1]) - kps[c[1][0], p].mean(0))))[0]
out = []
for p, nm in ((0, "A"), (1, "B")):
    d = byg.get(slot_of[p], {})
    ref = [centre(d[t]) - kps[t, p].mean(0) for t in range(0, 13) if t in d]
    off0 = np.median(ref, axis=0) if ref else np.zeros(2)
    offs, unp, prev, jump = [], [], None, 0.0
    for t in range(a0, min(a1, n)):
        share = float((vis[t, p] > 127).sum()) / max(1.0, float(full[p]))
        if t not in d:
            if share >= 0.30:
                unp.append(t)
            prev = None
            continue
        o = float(np.linalg.norm(centre(d[t]) - kps[t, p].mean(0) - off0))
        offs.append(o)
        if prev is not None:
            jump = max(jump, abs(o - prev))
        prev = o
    o = np.array(offs) if offs else np.zeros(1)
    print("%s: off mean %.1f px, 90th %.1f px, worst %.1f px | jump %.1f px | unpainted %d frames%s" % (
        nm, o.mean(), np.percentile(o, 90), o.max(), jump, len(unp), (" " + str(unp[:12]) + ("..." if len(unp) > 12 else "")) if unp else ""))
