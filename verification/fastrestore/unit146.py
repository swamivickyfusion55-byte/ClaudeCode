"""v11.2.146 (SDOS-146): a partly hidden face in contact, and relative placement. Stand-in faces, no models. usage: unit146.py PACKDIR"""
import sys, types
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception
sys.modules["gradio"] = g_
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: ""; sys.modules["huggingface_hub"] = hf
sys.path.insert(0, sys.argv[1])
import numpy as np, core_pipeline as cp, swap_engine as E
fails = 0
def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else ""), flush=True); fails += 0 if ok else 1
class F:
    def __init__(s, b, emb):
        s.bbox = np.array(b, np.float32); x1, y1, x2, y2 = b; w = x2 - x1
        s.kps = np.array([[x1 + .3*w, y1 + .4*w], [x1 + .7*w, y1 + .4*w], [x1 + .5*w, y1 + .6*w], [x1 + .35*w, y1 + .8*w], [x1 + .65*w, y1 + .8*w]], np.float32)
        s.normed_embedding = emb; s.embedding = emb; s.det_score = 0.9; s.landmark_2d_106 = None; s.gender = 1; s.sex = "M"
rng = np.random.default_rng(7)
def unit(v): return v / np.linalg.norm(v)
ea, eb = unit(rng.normal(size=512)), unit(rng.normal(size=512))
def part(e):   # the partly hidden face: its crop reads as the partner (0.85)
    return unit(0.12 * e + 0.85 * eb + rng.normal(size=512) * 0.02)
# 1) A is lost while the pair comes together, then reappears partly hidden 120 px away from where it was last seen (a sudden approach)
T = E.MultiFaceTracker([0, 1]); xa, xb = 300.0, 700.0
for k in range(22):
    xa += 6.0; xb -= 6.0
    T.assign([F([xa, 300, xa + 250, 550], ea), F([xb, 300, xb + 250, 550], eb)], {})
for k in range(22, 40):
    xb -= 4.0
    T.assign([F([xb, 300, xb + 250, 550], eb)], {})
ok_n = wrong = 0
for k in range(40, 52):
    xb -= 4.0
    fa_ = F([xa + 120 + (k - 40) * 4, 300, xa + 370 + (k - 40) * 4, 550], part(ea))
    fb_ = F([xb, 300, xb + 250, 550], eb)
    res = T.assign([fa_, fb_], {})
    ok_n += int(res.get(0) is fa_); wrong += int(res.get(1) is fa_ or res.get(0) is fb_)
check("a partly hidden face reappearing far from where it was last seen (crop reading as its partner) is taken back, never swapped",
      ok_n >= 10 and wrong == 0, "%d of 12 calls, %d wrong" % (ok_n, wrong))
# 2) relative placement: A's last real box is from before the contact; the pair was seen touching at frames 10-20
def rec(cx, cy, w=200, det=True):
    return {"bbox": np.array([cx - w / 2, cy - w / 2, cx + w / 2, cy + w / 2], np.float32), "det": det, "alpha": 1.0}
hs = [(gg, rec(200, 200)) for gg in range(10, 21)] + [(5, rec(900, 600))]
hp = [(gg, rec(300, 200)) for gg in range(10, 21)]
Tr = cp._rel_translation(hs, hp, 40, rec(900, 600), rec(700, 400), 60)
check("relative placement: A is put where it stood against B (B now at 700,400; A was 100 px left of B), not where it was before",
      Tr is not None and abs(Tr[0, 2] + 300) < 1 and abs(Tr[1, 2] + 200) < 1, "shift %s" % (None if Tr is None else np.round(Tr[:, 2], 1).tolist()))
check("no touching sighting within the window: no relative placement (the v11.2.141 hold stands)",
      cp._rel_translation(hs, hp, 40, rec(900, 600), rec(700, 400), 10) is None)
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
