"""v11.2.209 (SDOS-209): who a not-yet-found person may be given to. Stand-in faces and embeddings, no models, no user media.
The tracker only (swap_engine.MultiFaceTracker.assign), as unit176.py.
usage: unit209.py PACKDIR"""
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np, swap_engine as E
fails = 0
def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else ""), flush=True); fails += 0 if ok else 1
class F:
    def __init__(s, b, emb, gender=1, score=0.9, side=0.0):
        s.bbox = np.array(b, np.float32); x1, y1, x2, y2 = b; w = x2 - x1
        nx = x1 + (.5 + side * .4) * w                      # side: the nose's sideways offset in eye distances (0 = frontal)
        s.kps = np.array([[x1+.3*w, y1+.4*w], [x1+.7*w, y1+.4*w], [nx, y1+.6*w], [x1+.35*w, y1+.8*w], [x1+.65*w, y1+.8*w]], np.float32)
        s.normed_embedding = emb; s.embedding = emb; s.det_score = score; s.landmark_2d_106 = None
        s.gender = gender; s.sex = "M" if gender == 1 else "F"
rng = np.random.default_rng(29)
def unit(v): return v / np.linalg.norm(v)
eW, eM, eX, eC = (unit(rng.normal(size=512)) for _ in range(4))     # the woman, the man, a picture of nobody here, a bystander
def other_view(e): return unit(0.12 * e + rng.normal(size=512) * 0.05)   # the same person in an unfamiliar view: ~0.1 to their lock
def box(cx, cy, w=250): return [cx - w / 2, cy - w / 2, cx + w / 2, cy + w / 2]
# The tracker gives the first face to the first free slot (pictures aside), so each case names which slot has the MALE
# replacement: mslot. The other slot has the female one.
def tracker(mslot):
    return E.MultiFaceTracker([0, 1], slot_gender={mslot: 1, 1 - mslot: 0})

def woman_alone(refs, mslot, calls=260, gender_read=0, side=0.9):
    """The woman is found first. Then she is alone on screen; on every 4th call she shows an unfamiliar view somewhere else in
    the picture, which her own slot refuses. Is the MALE replacement ever put on her?"""
    T = tracker(mslot); wrong = 0; first = None; own = 0
    for k in range(calls):
        if k < 20 or k % 4:
            f = F(box(400, 400), eW, gender=0)
        else:
            f = F(box(900 + (k % 3) * 8, 420), other_view(eW), gender=gender_read, side=side)
        res = T.assign([f], refs)
        own += int(res.get(1 - mslot) is f)
        if res.get(mslot) is f:
            wrong += 1; first = k if first is None else first
    assert own >= 150, "the scenario is wrong: the woman was not found by her own slot (%d)" % own
    return wrong, first

w, first = woman_alone({0: eX}, mslot=0)
check("the woman alone, the man's Detect picture matching nobody (job DFA984A1): the male replacement is never put on her",
      w == 0, "%d calls with the male replacement on her%s" % (w, "" if first is None else ", first at call %d" % first))
w, first = woman_alone({}, mslot=1)
check("the woman alone, no Detect pictures: the male replacement is never put on her", w == 0,
      "%d calls with the male replacement on her%s" % (w, "" if first is None else ", first at call %d" % first))
w, first = woman_alone({}, mslot=1, gender_read=1)
check("...also when her side view is read as male (side views are not trusted for gender)", w == 0,
      "%d calls%s" % (w, "" if first is None else ", first at call %d" % first))

def both(refs, mslot, calls=120, man_gender=1, man_score=0.9, third=False):
    """Both on screen, apart. The woman is found; when is the man given the male replacement, and is anyone given the wrong one?"""
    T = tracker(mslot); first = None; wrong = 0
    for k in range(calls):
        fw, fm = F(box(400, 400), eW, gender=0), F(box(900, 400), eM, gender=man_gender, score=man_score)
        faces = [fw, fm] + ([F(box(650, 150, 200), eC, gender=1)] if third else [])
        res = T.assign(faces, refs)
        if res.get(mslot) is fm and first is None:
            first = k
        wrong += int(res.get(mslot) is fw or res.get(1 - mslot) is fm)
    return first, wrong

first, wrong = both({}, mslot=1)
check("both on screen, no pictures: the man gets the male replacement at once (as before)", first is not None and first <= 2 and wrong == 0,
      "first at call %s, %d wrong" % (first, wrong))
first, wrong = both({0: eX}, mslot=0)
check("both on screen, the man's picture matching nobody: he is found by elimination after about 45 sightings (job C77E081B)",
      first is not None and 40 <= first <= 60 and wrong == 0, "first at call %s, %d wrong" % (first, wrong))
first, wrong = both({0: eX}, mslot=0, man_gender=0)
check("...also when his gender is read wrong: with the woman found in the same frame, he can only be the man",
      first is not None and wrong == 0, "first at call %s, %d wrong" % (first, wrong))
first, wrong = both({0: eX}, mslot=0, man_score=0.35)
check("...but a weak detection (score 0.35: maybe not a face) is never taken this way", first is None and wrong == 0, "first %s" % first)
first, wrong = both({0: eX}, mslot=0, third=True)
check("...nor with a third face on screen (the picture must decide then)", first is None and wrong == 0, "first %s" % first)

def man_alone(side=0.0, gender_read=1, calls=60):
    """The woman was found (slot 0, the female replacement), then leaves. The man is alone on screen. No pictures."""
    T = tracker(1); first = None
    for k in range(20):
        T.assign([F(box(400, 400), eW, gender=0)], {})
    for k in range(calls):
        fm = F(box(900, 400), eM, gender=gender_read, side=side)
        if T.assign([fm], {}).get(1) is fm and first is None:
            first = k
    fw, fm = F(box(400, 400), eW, gender=0), F(box(900, 400), eM, gender=gender_read, side=side)
    back = T.assign([fw, fm], {})
    return first, back.get(1) is fm and back.get(0) is fw
first, back = man_alone()
check("the man alone after the woman left, facing the camera and read as male: found after a short check",
      first is not None and first <= 20, "first at call %s" % first)
first, back = man_alone(side=0.9)
check("the man alone in a side view: not taken on a guess; found at once when the woman is back in the picture",
      first is None and back, "alone: %s, with her back: %s" % (first, back))
first, back = man_alone(gender_read=0)
check("the man alone, read as female: not taken on a guess; found at once when the woman is back", first is None and back,
      "alone: %s, with her back: %s" % (first, back))

T = E.MultiFaceTracker([0], slot_gender={0: 1}); f = F(box(400, 400), eM)
check("a one-person job is not affected (the face is taken at once)", T.assign([f], {}).get(0) is f)
T = tracker(1); f = F(box(400, 400), eW, gender=0)
r = T.assign([f], {})
check("the first person of a two-person job is taken at once (as before)", r.get(0) is f and r.get(1) is None, str({k: (v is f) for k, v in r.items()}))
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
