"""v11.2.148 (SDOS-148): the kiss as one event. A partly hidden face whose boxes do not overlap (0.98 face widths apart,
vertically) is still taken back, because the pair is crossing. usage: unit148.py PACKDIR"""
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np, swap_engine as E
rng = np.random.default_rng(11)
def unit(v): return v / np.linalg.norm(v)
ea, eb = unit(rng.normal(size=512)), unit(rng.normal(size=512))
def part(e): return unit(0.12 * e + 0.85 * eb + rng.normal(size=512) * 0.02)   # partly hidden: its crop reads as the partner
class F:
    def __init__(s, b, emb):
        s.bbox = np.array(b, np.float32); x1, y1, x2, y2 = b; w = x2 - x1
        s.kps = np.array([[x1+.3*w, y1+.4*w], [x1+.7*w, y1+.4*w], [x1+.5*w, y1+.6*w], [x1+.35*w, y1+.8*w], [x1+.65*w, y1+.8*w]], np.float32)
        s.normed_embedding = emb; s.embedding = emb; s.det_score = 0.9; s.landmark_2d_106 = None; s.gender = 1; s.sex = "M"
T = E.MultiFaceTracker([0, 1]); xb = 700.0
# frames 0-21: a clean pair, side by side, overlapping as they come together; A ABOVE B from frame 22 (no box overlap, 0.98 apart)
for k in range(22):
    xb -= 6.0
    T.assign([F([xb - 250 + 6 * k, 300, xb + 6 * k, 550], ea), F([xb, 300, xb + 250, 550], eb)], {})
adm = 0
for k in range(22, 40):
    xb -= 1.0
    fa = F([xb, 55, xb + 250, 305], part(ea))            # A above B: boxes 0.98 face widths apart, overlap only 5 px
    fb = F([xb, 300, xb + 250, 550], eb)
    res = T.assign([fa, fb], {})
    adm += int(res.get(0) is fa)
print("partly hidden face above its partner, boxes apart: taken back on %d of 18 frames" % adm)
print("RESULT", adm)
