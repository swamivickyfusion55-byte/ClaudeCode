"""v11.2.141 ContactOrder unit checks. usage: unit141.py PACKDIR   (no models needed)"""
import sys, types
PACK = sys.argv[1]; sys.path.insert(0, PACK)
g = types.ModuleType("gradio"); g.Request = type("Request", (), {}); g.update = lambda **k: k; g.Error = Exception
sys.modules.setdefault("gradio", g)
import numpy as np
import core_pipeline as cp, swap_engine as E
fails = 0
def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else "")); fails += 0 if ok else 1
# two faces overlapping, the same size; outline-edge evidence (if any) says slot 1 is in front
def face(x):
    lm = np.stack([np.linspace(x, x + 200, 106), 300 + 80 * np.sin(np.linspace(0, 6.28, 106))], 1).astype(np.float32)
    return {"bbox": np.array([x, 200, x + 200, 420], np.float32), "lmk": lm, "kps": lm[:5].copy()}
E._P["own_pixels"] = 1.0
cp._ORDER_MEM.clear(); cp._ORDER_CUE.clear(); cp._ORDER_CUE[(0, 1)] = 0.5          # evidence: 0 is BEHIND 1
seen = lambda r, s: dict(r, det=True, seen=s)
a, b = face(100), face(220)
r = cp._rival_landmarks({0: seen(a, True), 1: seen(b, False)}, 0)
check("a face nobody sees (held) is not in front of one that is seen", len(r) == 1 and r[0]["front"] is False, str([x["front"] for x in r]))
r = cp._rival_landmarks({0: seen(a, False), 1: seen(b, True)}, 0)
check("...and the seen face is in front of the held one", len(r) == 1 and r[0]["front"] is True, str([x["front"] for x in r]))
r = cp._rival_landmarks({0: seen(a, True), 1: dict(seen(b, True), contact_held=True)}, 0)
check("a contact-held face counts as not seen", len(r) == 1 and r[0]["front"] is False)
cp._ORDER_MEM.clear()
r = cp._rival_landmarks({0: seen(a, True), 1: seen(b, True)}, 0)
check("both seen: the outline-edge evidence decides, as before (1 in front)", len(r) == 1 and r[0]["front"] is True)
r = cp._rival_landmarks({0: {"bbox": a["bbox"], "lmk": a["lmk"], "det": True}, 1: {"bbox": b["bbox"], "lmk": b["lmk"], "det": True}}, 0)
check("records without the new 'seen' flag behave as before", len(r) == 1 and r[0]["front"] is True)
P = cp._parse_person_gender
check("person_gender from the app: [0, 1]", P([0, 1]) == [0, 1, None, None])
check("person_gender as JSON text with nulls", P("[1, null, 0]") == [1, None, 0, None])
check("person_gender junk -> nothing known", P("x") == [None] * 4 and P(None) == [None] * 4 and P([2, "m"]) == [None] * 4)
t = E.MultiFaceTracker([0, 1], slot_gender={0: 1, 1: None})
check("tracker: slot genders as given, vote store empty", t.slot_gender == {0: 1, 1: None} and t._g_votes == {} and t._g_fixed == set())
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
