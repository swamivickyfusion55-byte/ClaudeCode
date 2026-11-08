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
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
