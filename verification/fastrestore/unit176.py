"""v11.2.176 (SDOS-176): two people by elimination, the sliver guard, and the longer carry in contact - on the 11.2.92 tracker.
Stand-in faces, no models. usage: unit176.py PACKDIR"""
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np, swap_engine as E
fails = 0
def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else ""), flush=True); fails += 0 if ok else 1
class F:
    def __init__(s, b, emb):
        s.bbox = np.array(b, np.float32); x1, y1, x2, y2 = b; w = x2 - x1
        s.kps = np.array([[x1+.3*w, y1+.4*w], [x1+.7*w, y1+.4*w], [x1+.5*w, y1+.6*w], [x1+.35*w, y1+.8*w], [x1+.65*w, y1+.8*w]], np.float32)
        s.normed_embedding = emb; s.embedding = emb; s.det_score = 0.9; s.landmark_2d_106 = None; s.gender = 1; s.sex = "M"
rng = np.random.default_rng(17)
def unit(v): return v / np.linalg.norm(v)
ea, eb, ec = unit(rng.normal(size=512)), unit(rng.normal(size=512)), unit(rng.normal(size=512))
def nobody(e): return unit(0.12 * e + rng.normal(size=512) * 0.05)       # turned away / blurred: ~0.13 to itself, ~0 to anyone
def box(cx, cy, w): return [cx - w / 2, cy - w / 2, cx + w / 2, cy + w / 2]
def run(two_elim=1, cf=1, shrink=1.0, weak=False, third=False):
    E._P["two_elim"], E._P["contact_follow"] = two_elim, cf
    T = E.MultiFaceTracker([0, 1]); got = wrong = 0
    for k in range(30):
        fa = F(box(325, 425, 250 * (shrink if k >= 15 else 1.0)), nobody(ea) if k >= 15 else ea)
        if weak and k >= 15:
            fa._kps_weak = True
        fb = F(box(925, 425, 250), eb)
        faces = [fa, fb] + ([F(box(625, 150, 200), ec)] if third and k >= 15 else [])
        res = T.assign(faces, {})
        if k >= 15:
            got += int(res.get(0) is fa); wrong += int(res.get(1) is fa or res.get(0) is fb)
    E._P["two_elim"], E._P["contact_follow"] = 1, 1
    return got, wrong
g, w = run()
check("2 people: a whole face turned away (reads as nobody, ~0.13) is kept by elimination, never swapped", g == 15 and w == 0, "%d of 15, %d wrong" % (g, w))
g, w = run(two_elim=0)
check("two_elim 0: the v11.2.92 behaviour (that face is refused)", g == 0, "%d of 15" % g)
g, w = run(shrink=0.75)
check("a sliver (its box 0.75 of its last width) is not taken by elimination", g == 0, "%d of 15" % g)
g, w = run(weak=True)
check("a face whose points are marked unreliable is not taken by elimination", g == 0, "%d of 15" % g)
g, w = run(third=True)
check("a third face in the frame: elimination does not act", g == 0 and w == 0, "%d of 15, %d wrong" % (g, w))
# the carry in contact: A missed for 60 calls while B (touching it) is found and moves
def carry_run(cf=1, b_missed=False):
    E._P["contact_follow"] = cf
    T = E.MultiFaceTracker([0, 1]); xb = 600.0; carried = 0
    for k in range(20):
        T.assign([F(box(450, 425, 250), ea), F(box(xb, 425, 250), eb)], {})
    for k in range(60):
        xb += 1.0
        T.assign([] if b_missed else [F(box(xb, 425, 250), eb)], {})
        carried += int(0 in T.carry(slots={0, 1}))
    E._P["contact_follow"] = 1
    return carried
c1, c0, cb = carry_run(), carry_run(cf=0), carry_run(b_missed=True)
check("in contact, a face missed for 60 frames is carried while its partner is seen (v11.2.92: 24)", c1 == 60 and c0 <= 25, "%d of 60 (contact_follow 0: %d)" % (c1, c0))
check("...but when both are lost, the v11.2.92 limit stands (no face kept on a guess for long)", cb <= 25, "%d of 60" % cb)
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
