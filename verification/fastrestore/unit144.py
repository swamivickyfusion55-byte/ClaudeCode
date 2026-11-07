"""v11.2.144 KissFollow unit checks - the tracker's contact rules on stand-in faces (no models, no video). usage: unit144.py PACKDIR
Two faces approach at 6 px a frame and meet (a kiss: the approach stops dead), then sway together. Embeddings are random unit vectors
(the two people), and a profile view is imitated by a weak copy of a person's own embedding (similarity ~0.1)."""
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
        s.kps = np.array([[x1 + .3 * w, y1 + .4 * w], [x1 + .7 * w, y1 + .4 * w], [x1 + .5 * w, y1 + .6 * w],
                          [x1 + .35 * w, y1 + .8 * w], [x1 + .65 * w, y1 + .8 * w]], np.float32)
        s.normed_embedding = emb; s.det_score = 0.9; s.landmark_2d_106 = None; s.embedding = emb; s.gender = 1; s.sex = "M"
rng = np.random.default_rng(1)
def unit(v):
    return v / np.linalg.norm(v)
ea, eb, ec = unit(rng.normal(size=512)), unit(rng.normal(size=512)), unit(rng.normal(size=512))
def weak(e):
    return unit(0.15 * e + rng.normal(size=512) * 0.05)                     # a profile of that person: similarity ~0.1
def kiss(drop_a=range(22, 31), profile_after=30, extra=None, n=40):
    """-> per call: (A's box x1, A's true x1, slot0 got A's face, slot0 got B's face, slot1 got A's face), tracker"""
    T = E.MultiFaceTracker([0, 1]); xa, xb = 300.0, 700.0; out = []
    for k in range(n):
        if k < 20:
            xa += 6.0; xb -= 6.0
        else:
            xa += 3.0 * np.sin(k / 3.0); xb += 3.0 * np.sin(k / 3.0)
        fa_ = F([xa, 300, xa + 250, 550], weak(ea) if k > profile_after else ea)
        fb_ = F([xb, 300, xb + 250, 550], eb)
        faces = [fb_] if k in drop_a else [fa_, fb_]
        if extra is not None:
            faces = faces + extra(k, xa, xb)
        res = T.assign(faces, {})
        out.append((float(T.tracks[0].bbox[0]), xa, res.get(0) is fa_, res.get(0) is fb_, res.get(1) is fa_))
    return out, T
o141_note = "141: rematched 0 of 9, box 12 px off"
out, T = kiss()
err_miss = max(abs(a - t) for a, t, *_ in out[22:31])
check("in contact, a face the detector misses moves with its partner (the approach stops at the kiss: no overshoot into the partner)",
      err_miss <= 8.0, "worst %.1f px over the 9 missed calls" % err_miss)
re = sum(1 for (_, _, a0, _, _) in out[31:] if a0)
check("...and when it is found again, in profile (similarity ~0.1), it is taken back at once - no lock-out for the rest of the kiss",
      re == 9, "%d of 9 calls (%s)" % (re, o141_note))
wrong = sum(1 for (_, _, _, b0, a1) in out if b0 or a1)
check("...and no face ever goes to the other person's slot", wrong == 0, "%d wrong assignments in 40 calls" % wrong)
# a stranger appears right where A was last seen while A's own face is also there: A's own detection wins (mutual nearest)
def stranger(k, xa, xb):
    return [F([xa - 140, 290, xa + 110, 540], ec)] if k > 30 else []
out2, _ = kiss(extra=stranger)
st = sum(1 for (_, _, a0, _, _) in out2[31:] if a0)
check("a third face beside the kissing pair: A keeps its own (profile) face, the stranger is not given A's replacement", st >= 8,
      "A got its own face on %d of 9 calls" % st)
# B's face sits where A was last seen (A gone): it is not admitted to A - it is clearly B
T3 = E.MultiFaceTracker([0, 1]); xa, xb = 300.0, 700.0; stolen = 0
for k in range(40):
    if k < 20:
        xa += 6.0; xb -= 6.0
    fa_, fb_ = F([xa, 300, xa + 250, 550], ea), F([xb, 300, xb + 250, 550], eb)
    if k <= 25:
        faces = [fa_, fb_]
    else:                                                                   # A has left; B steps into A's place
        faces = [F([xa + 30, 300, xa + 280, 550], eb)]
    res = T3.assign(faces, {})
    stolen += int(k > 25 and res.get(0) is not None)
check("the other person (recognisable) standing where A was last seen is never taken for A", stolen == 0, "%d of 14 calls" % stolen)
# faces apart: nothing changes (no contact velocity is used)
T4 = E.MultiFaceTracker([0, 1]); used = 0
for k in range(30):
    fa_, fb_ = F([100 + 5 * k, 300, 350 + 5 * k, 550], ea), F([900, 300, 1150, 550], eb)
    T4.assign([fb_] if 15 <= k <= 20 else [fa_, fb_], {})
    used += int(T4.tracks[0]._contact_vel is not None)
check("two faces apart: the contact rules never act", used == 0 and not T4.tracks[0].crossing, "%d calls" % used)
E._P["contact_follow"] = 0
out5, _ = kiss()
re5 = sum(1 for (_, _, a0, _, _) in out5[31:] if a0)
check("contact_follow 0 restores the v11.2.141 behaviour (the face stays rejected after the miss)", re5 == 0, "%d of 9" % re5)
E._P["contact_follow"] = 1
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
