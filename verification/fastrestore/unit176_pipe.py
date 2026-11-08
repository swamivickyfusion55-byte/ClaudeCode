"""v11.2.176 (SDOS-176): the steady front / behind decision on the 11.2.92 pipeline (core_pipeline._front_relations and the cut it
drives). Stand-in records, no models. usage: unit176_pipe.py PACKDIR"""
import sys, types
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception
sys.modules["gradio"] = g_
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: ""; sys.modules["huggingface_hub"] = hf
sys.path.insert(0, sys.argv[1])
import numpy as np, core_pipeline as cp
fails = 0
def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else ""), flush=True); fails += 0 if ok else 1
def rec(x, w, seen=True, xid=None):
    return {"bbox": np.array([x, 300, x + w, 300 + w], np.float32), "lmk": np.zeros((106, 2), np.float32), "det": seen,
            "seen_near": seen, "xid_near": xid}
def fresh():
    cp._FRONT_MEM.clear(); cp._FRONT_NOW.clear()
# 1) a side-view kiss: the two boxes within +-3 % of each other, changing every frame - v11.2.92 changed who is in front each time
fresh(); flips92 = flips = 0; prev92 = prev = None
for g in range(60):
    wa = 200 * (1.0 + 0.03 * (-1) ** g); wb = 200.0
    r = cp._front_relations({0: rec(500, wa), 1: rec(640, wb)}, g)[(0, 1)]
    a92 = wa * wa > wb * wb
    flips += int(prev is not None and r != prev); flips92 += int(prev92 is not None and a92 != prev92); prev, prev92 = r, a92
check("boxes the same size within 3 %: the face in front stays the same (v11.2.92 changed it on every frame)", flips == 0 and flips92 >= 50,
      "%d changes in 60 frames (v11.2.92 rule: %d)" % (flips, flips92))
# 2) identity: A's crop reads as B, B reads as itself -> A behind, at once, even if A's box is bigger
fresh()
r = cp._front_relations({0: rec(500, 230, xid=0.7), 1: rec(640, 200, xid=-0.9)}, 0)[(0, 1)]
check("a face whose crop reads as its partner is behind, whatever the box size", r is False, str(r))
held = [cp._front_relations({0: rec(500, 230), 1: rec(640, 200)}, g)[(0, 1)] for g in range(1, 50)]
check("...and that holds through frames without an identity reading (bigger box, 49 frames)", not any(held), "%d frames flipped" % sum(held))
# 3) sight: a face carried on a guess is never in front of one that is seen
fresh()
r = cp._front_relations({0: rec(500, 240, seen=False), 1: rec(640, 200, seen=True)}, 0)[(0, 1)]
check("a face not seen (carried on a guess) is never in front of one that is seen", r is False, str(r))
# 4) the cut: the rival list of each face follows the decision
fresh()
recs = {0: rec(500, 230, xid=0.7), 1: rec(640, 200, xid=-0.9)}
cp._FRONT_NOW.update(cp._front_relations(recs, 0))
ra, rb = cp._rival_landmarks(recs, 0), cp._rival_landmarks(recs, 1)
check("the face behind is cut by the face in front, never the other way round", len(ra) == 1 and len(rb) == 0, "behind cut by %d, front cut by %d" % (len(ra), len(rb)))
# 5) the switch off: the v11.2.92 rule (the bigger box in front)
fresh()
recs2 = {0: rec(500, 230, xid=0.7), 1: rec(640, 200, xid=-0.9)}
ra2, rb2 = cp._rival_landmarks(recs2, 0), cp._rival_landmarks(recs2, 1)
check("no decision made (contact_follow 0): the v11.2.92 rule, the bigger box in front", len(ra2) == 0 and len(rb2) == 1)
# 4b) two frontal faces first touching: the face going behind is 6-8 % BIGGER on the first frames, then 12-15 % smaller as it is
# covered (measured on the frontal smooch) - the first clear difference decides at once (it used to wait 4 frames)
fresh(); seq = [1.07, 1.06, 1.08, 0.85, 0.85, 0.88, 0.88, 0.82, 0.75, 0.70]
res = [cp._front_relations({0: rec(500, 200 * r ** 0.5), 1: rec(640, 200)}, g)[(0, 1)] for g, r in enumerate(seq)]
check("first touch a near tie, then the other face clearly bigger: it is in front from that frame on", res[:3] == [True] * 3 and not any(res[3:]),
      str(res))
fresh(); seq = [0.80] * 6 + [1.12] + [0.80] * 6
res = [cp._front_relations({0: rec(500, 200 * r ** 0.5), 1: rec(640, 200)}, g)[(0, 1)] for g, r in enumerate(seq)]
check("...but once decided on a clear difference, one contrary frame does not change it", not any(res), str(res))
# 5) a side-view kiss on the tracker: each face's 5 points all on its front half, so the two faces' points sit ~15 px apart while
# the boxes stay two faces (overlap ~0.15, each centre outside the other box); the face behind reads as its partner (0.85)
import swap_engine as E
class F:
    def __init__(s, b, emb, kc):
        s.bbox = np.array(b, np.float32); s.kps = np.array(kc, np.float32)[None, :] + np.array(
            [[-12, -20], [12, -20], [0, 0], [-10, 20], [10, 20]], np.float32)
        s.normed_embedding = emb; s.embedding = emb; s.det_score = 0.7; s.landmark_2d_106 = None; s.gender = 1; s.sex = "M"
rng = np.random.default_rng(5)
def unit(v): return v / np.linalg.norm(v)
ea, eb = unit(rng.normal(size=512)), unit(rng.normal(size=512))
def side_kiss(two_boxes=True):
    T = E.MultiFaceTracker([0, 1]); xa, xb = 300.0, 760.0; got = wrong = kept_two = 0
    for k in range(70):
        if k < 30:
            xa += 6.0; xb -= 6.0                                  # (approach, frontal: each face's points in its middle)
            fa = F([xa, 300, xa + 200, 500], ea, (xa + 100, 400)); fb = F([xb, 300, xb + 200, 500], eb, (xb + 100, 400))
        else:                                                    # the kiss: A's box x480-680, B's x630-830 (overlap 0.14)
            fa = F([480, 300, 680, 500], unit(0.12 * ea + 0.88 * eb + rng.normal(size=512) * 0.01), (648, 400))
            fb = F([630, 300, 830, 500], eb, (662, 400))
        faces = [fa, fb]
        if not two_boxes:
            E_tb = E._two_boxes; E._two_boxes = lambda a, b: False
        try:
            ap = cp._contact_apart(faces, T); kept = cp._dedupe_faces(faces, apart=ap)
            res = T.assign(kept, {})
        finally:
            if not two_boxes:
                E._two_boxes = E_tb
        if k >= 30:
            kept_two += int(len(kept) == 2); got += int(res.get(0) is fa and res.get(1) is fb)
            wrong += int(res.get(0) is fb or res.get(1) is fa)
    return kept_two, got, wrong
k2, g2, w2 = side_kiss()
check("side-view kiss (points of both faces ~15 px apart, boxes apart): both faces kept and each given to its own person",
      k2 == 40 and g2 == 40 and w2 == 0, "%d of 40 frames both kept, %d both right, %d wrong" % (k2, g2, w2))
k0, g0, w0 = side_kiss(two_boxes=False)
check("(without the box test - the points test only - the face behind was merged away or refused)", g0 < 20,
      "%d of 40 both kept, %d both right" % (k0, g0))
check("two boxes on one face (a cheek box, its centre inside the face's box) are not 'two faces'",
      not E._two_boxes([600, 300, 810, 600], [656, 300, 856, 600]) and not E._two_boxes([0, 0, 100, 100], [5, 5, 105, 105]))
check("a side-view pair (overlap 0.14, centres outside each other's box) is two faces", E._two_boxes([480, 300, 680, 500], [630, 300, 830, 500]))
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
