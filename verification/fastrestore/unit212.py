"""v11.2.212 (SDOS-212) unit checks: the gender of each person, as the user confirmed it in the app's Detect step. Stand-in faces and
embeddings, no models, no user media. The tracker (swap_engine.MultiFaceTracker.assign / carry), as unit209.py.
Slot 0 is the WOMAN (her replacement), slot 1 the MAN, in every case; person_gender = {0: 0, 1: 1} as the app sends it.
usage: unit212.py PACK211 PACK212"""
import sys, os, importlib.util
P211, P212 = sys.argv[1:3]
import numpy as np
import logging
logging.disable(logging.CRITICAL)


def load(name, pack):
    sys.path.insert(0, pack)
    spec = importlib.util.spec_from_file_location(name, os.path.join(pack, "swap_engine.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); sys.path.pop(0); return m


E1, E2 = load("se211", P211), load("se212", P212)
fails = 0


def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else ""), flush=True); fails += 0 if ok else 1


def pm(p):          # P(male) -> the gender model's score as the job keeps it (P = 1 / (1 + e^(-2 s)))
    p = min(max(p, 1e-6), 1 - 1e-6); return 0.5 * np.log(p / (1 - p))


class F:
    def __init__(s, b, emb, p_male, score=0.9, side=0.0):
        s.bbox = np.array(b, np.float32); x1, y1, x2, y2 = b; w = x2 - x1
        nx = x1 + (.5 + side * .4) * w
        s.kps = np.array([[x1+.3*w, y1+.4*w], [x1+.7*w, y1+.4*w], [nx, y1+.6*w], [x1+.35*w, y1+.8*w], [x1+.65*w, y1+.8*w]], np.float32)
        s.normed_embedding = emb; s.embedding = emb; s.det_score = score; s.landmark_2d_106 = None
        s.gender = int(p_male >= 0.5); s.sex = "M" if p_male >= 0.5 else "F"; s.gender_s = pm(p_male)


rng = np.random.default_rng(212)
def unit(v): return v / np.linalg.norm(v)
eW, eM = unit(rng.normal(size=512)), unit(rng.normal(size=512))
def mix(a, b, wa, wb): return unit(wa * a + wb * b + rng.normal(size=512) * 0.02)
def box(cx, cy, w=250): return [cx - w / 2, cy - w / 2, cx + w / 2, cy + w / 2]
PG = {0: 0, 1: 1}


def tracker(E, pg=PG):
    return E.MultiFaceTracker([0, 1], slot_gender={0: 0, 1: 1}, person_gender=pg) if E is E2 else E.MultiFaceTracker([0, 1], slot_gender={0: 0, 1: 1})


def run(E, frames, refs=None, pg=PG, tracker_=None):
    """frames: list of face lists (each face tagged .who = 'W' or 'M'). Returns paints: list of {slot: who} per call (carry incl.)"""
    T = tracker_ or tracker(E, pg)
    out = []
    for faces in frames:
        res = T.assign(faces, refs or {})
        got = {s: getattr(f, "who", "?") for s, f in res.items()}
        for s, pf in T.carry(slots={0, 1}).items():
            if s not in got and T.tracks[s].missed >= 1:
                got[s] = "carry"
        out.append(got)
    return out, T


def face(who, cx, cy=400, p_male=None, side=0.0, emb=None, w=250):
    f = F(box(cx, cy, w), emb if emb is not None else (eW if who == "W" else eM), p_male if p_male is not None else (0.02 if who == "W" else 0.99), side=side)
    f.who = who
    return f


def wrong(paints):   # the woman's replacement on the man, or the man's on the woman
    return sum(int(p.get(0) == "M") + int(p.get(1) == "W") for p in paints)


def right(paints):
    return sum(int(p.get(0) == "W") + int(p.get(1) == "M") for p in paints)


# 1. No Detect pictures for this part; the man stands on the LEFT. The first free slot took the leftmost face.
fr = [[face("M", 400), face("W", 900)] for _ in range(60)]
p1, _ = run(E1, fr); p2, _ = run(E2, fr)
check("no Detect pictures, the man on the left: 11.2.211 gives him the WOMAN's replacement (the report: a girl's face on a man)", wrong(p1) > 50, "%d wrong of 120" % wrong(p1))
check("...11.2.212 with the people's genders: each gets their own from the first frame, none wrong", wrong(p2) == 0 and right(p2) >= 118, "%d wrong, %d right of 120" % (wrong(p2), right(p2)))

# 2. Detect pictures that are poor for this part (another light, another angle): the woman reads slightly MORE like the man's picture
refW, refM = mix(eW, eM, 0.18, 0.10), mix(eM, eW, 0.30, 0.25)
fr = [[face("W", 400)] for _ in range(40)] + [[face("W", 400), face("M", 900)] for _ in range(40)]
p1, _ = run(E1, fr, refs={0: refW, 1: refM}); p2, _ = run(E2, fr, refs={0: refW, 1: refM})
check("a part opening on the woman alone, her picture poor (she reads more like the man's): 11.2.211 gives her the man's replacement",
      wrong(p1) > 20, "%d wrong" % wrong(p1))
check("...11.2.212: never (a clean front view of her reads as a woman)", wrong(p2) == 0, "%d wrong, %d right" % (wrong(p2), right(p2)))
first_w = next((k for k, p in enumerate(p2) if p.get(0) == "W"), None)
check("...and she is found by her gender (the only woman, read clean) within 3 detections, not left unpainted", first_w is not None and first_w <= 3,
      "hers from call %s; %d of her 80 calls painted" % (first_w, sum(p.get(0) == "W" for p in p2)))

# 2b. ...but not when a second woman is on screen (gender says nothing then), nor a stranger unlike her own picture
fr = [[face("W", 400), face("W", 900, emb=unit(rng.normal(size=512)))] for _ in range(30)]
fr[0][1].who = "X"
p2, T = run(E2, fr, refs={0: refW, 1: refM})
check("two women on screen and a poor picture: neither is found by gender (the picture or the elimination rule must decide)",
      T.gender_found_n == 0, "%d found by gender" % T.gender_found_n)
stranger = unit(rng.normal(size=512))
fr = [[face("W", 400, emb=stranger)] for _ in range(30)]
p2, T = run(E2, fr, refs={0: eW, 1: eM})
check("a woman unlike the woman's picture (a stranger), alone: not given her replacement", all(p.get(0) is None for p in p2),
      "%d calls painted" % sum(p.get(0) is not None for p in p2))

# 3. A person already bound to the wrong face (as after a swap in contact, where gender is not read): let go and found again
T = tracker(E2, pg={})
pre, T = run(E2, [[face("M", 400), face("W", 900)] for _ in range(20)], tracker_=T)
assert wrong(pre) > 30, "scenario: the binding should be wrong at first"
T.person_gender = dict(PG)
post, T = run(E2, [[face("M", 400), face("W", 900)] for _ in range(60)], tracker_=T)
first_bad = [k for k, p in enumerate(post) if p.get(0) == "M" or p.get(1) == "W"]
first_ok = next((k for k, p in enumerate(post) if p.get(0) == "W" and p.get(1) == "M"), None)
check("bound to the wrong people: from the first clean front view on, neither wrong replacement is painted (not even carried)",
      not first_bad, "wrong on calls %s" % first_bad[:5])
check("...and both are let go and found again: each with their own within 5 calls", first_ok is not None and first_ok <= 5 and T.gender_unbound_n >= 2,
      "own from call %s, let go %d" % (first_ok, T.gender_unbound_n))

# 4. In contact the woman's crop holds the man's face and reads as a man (measured: up to 4 of 34 frames at P >= 0.88): not refused
fr = [[face("W", 400), face("M", 900)] for _ in range(20)] + \
     [[face("W", 560, p_male=0.95 if k % 3 == 0 else 0.3), face("M", 740)] for k in range(40)]
p2, T = run(E2, fr)
lost = sum(1 for p in p2[20:] if p.get(0) != "W")
check("faces touching, the woman misread as a man: she keeps her replacement (contact readings are not used)", lost == 0 and wrong(p2) == 0,
      "%d calls without hers, %d wrong" % (lost, wrong(p2)))

# 5. A side view misread (nose 0.9 eye distances off the middle): not used either
fr = [[face("W", 400), face("M", 900)] for _ in range(20)] + [[face("W", 400, p_male=0.96, side=0.9), face("M", 900)] for _ in range(30)]
p2, _ = run(E2, fr)
check("a side view of the woman read as a man: she keeps her replacement", all(p.get(0) == "W" for p in p2[20:]), "%d of 30" % sum(p.get(0) == "W" for p in p2[20:]))

# 6. Memory: the man, once read on a clean view as a man for the woman's slot, is not given her replacement in contact either
T = tracker(E2)
p_a, T = run(E2, [[face("M", 400)] for _ in range(10)], tracker_=T)          # the man alone first: refused for her slot, remembered
p_b, T = run(E2, [[face("M", 560, p_male=0.5), face("W", 740, p_male=0.5)] for _ in range(30)], tracker_=T)   # in contact, unreadable
check("the man read once as a man: in contact (gender unreadable) he still never gets the woman's replacement", wrong(p_a + p_b) == 0,
      "%d wrong" % wrong(p_a + p_b))

# 7. Without the app's genders (an older app, or no Detect) nothing changes: the same decisions as 11.2.211
fr = [[face("M", 400 + 3 * k), face("W", 900 - 2 * k)] for k in range(40)] + [[face("W", 600)] for _ in range(20)]
p1, _ = run(E1, fr); p2, _ = run(E2, fr, pg={})
check("no genders from the app: every decision as in 11.2.211 (the gate is off)", p1 == p2)
E2._P["id_gender_gate"] = 0
p2b, _ = run(E2, [[face("M", 400), face("W", 900)] for _ in range(20)])
E2._P["id_gender_gate"] = 1
p1b, _ = run(E1, [[face("M", 400), face("W", 900)] for _ in range(20)])
check("id_gender_gate = 0: as 11.2.211 even with the genders given", p1b == p2b)

# 8. A man and a woman far apart, both read clean, each with a good picture: nobody loses a frame to the gate
fr = [[face("W", 400), face("M", 900)] for _ in range(80)]
p2, T = run(E2, fr, refs={0: eW, 1: eM})
check("the ordinary case (good pictures, both clean): every frame painted, none refused", right(p2) == 160 and T.gender_refused_n == 0,
      "%d right of 160, %d refused" % (right(p2), T.gender_refused_n))

# 8b. A hug (found on the bench's hug clip): she goes behind him and is not detected, while his face - alone, front, clean, a man -
# is where she was. That is HIS face, not hers read as a man: her binding is kept, and she is painted again the frame she is back.
fr = [[face("W", 400), face("M", 900)] for _ in range(20)]
fr += [[face("W", 400, p_male=0.5), face("M", 900 - 38 * k, p_male=0.5)] for k in range(1, 11)]      # coming together (contact)
fr += [[face("M", 410)] for _ in range(15)]                                                         # she is behind him, he is where she was
fr += [[face("W", 400), face("M", 900)] for _ in range(20)]
p2, T = run(E2, fr, refs={0: eW, 1: eM})
back = p2[45:]
check("a hug: her hidden, his face alone where she was - she is NOT let go, and has hers the frame she is back",
      T.gender_unbound_n == 0 and back[0].get(0) == "W" and all(p.get(0) == "W" for p in back) and wrong(p2) == 0,
      "let go %d, after: %s, wrong %d" % (T.gender_unbound_n, [p.get(0) for p in back[:4]], wrong(p2)))

# 9. The app's genders as they arrive
sys.path.insert(0, P212)
import types
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception
sys.modules["gradio"] = g_
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: ""; sys.modules["huggingface_hub"] = hf
import core_pipeline as cp
cases = [([0, 1], [0, 1, None, None]), ([1, None, 0], [1, None, 0, None]), ("[0, 1]", [0, 1, None, None]), (None, [None] * 4),
         ([True, False], [None] * 4), (["1", "x", 2], [1, None, None, None]), ("", [None] * 4)]
ok = all(cp._parse_person_gender(v) == want for v, want in cases)
check("settings.person_gender is read as the app sends it (0 / 1 / null per slot; anything else is 'not known')", ok,
      str([cp._parse_person_gender(v) for v, _ in cases]))
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
