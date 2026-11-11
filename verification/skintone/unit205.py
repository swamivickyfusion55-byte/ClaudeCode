"""v11.2.205 (SDOS-205): the skin tone without the models (setting, tone curve, fallback, hold and fade, cut), the adapter
(setting kept, slot matching) and a resumed job keeping the setting. Stand-in pictures, no user media.
usage: unit205.py PACKDIR WORKDIR"""
import sys, types, os, json, shutil
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception
sys.modules["gradio"] = g_
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: ""; sys.modules["huggingface_hub"] = hf
PACK, WORK = sys.argv[1], sys.argv[2]
os.environ["PHOENIX_PERSISTENT_ROOT"] = WORK + "/jobs"; os.environ["PHOENIX_PERSISTENT_JOB_STATE_DIR"] = WORK + "/jobs"
os.environ["PHOENIX_PERSISTENT_OUTPUT_DIR"] = WORK + "/out"
sys.path.insert(0, PACK)
import numpy as np, cv2, inspect
import skin_tone as ST
fails = 0


def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else ""), flush=True); fails += 0 if ok else 1


# ---- the setting ---------------------------------------------------------------------------------------------------------
pa = ST.parse_amounts
check("the app's list [a1..a4]: slot index = position, 0 = unchanged", pa([0.3, 0, -0.2, 0]) == {0: 0.3, 2: -0.2})
check("a dict of 1-based slot numbers ('1', 'slot2', '#3')", pa({"1": 0.1, "slot2": -0.5, "#3": 0.25}) == {0: 0.1, 1: -0.5, 2: 0.25})
check("a JSON string", pa('[0, 0.4]') == {1: 0.4})
check("out of range is clipped to -1..1; NaN, text and slot 5 are ignored",
      pa([2.0, float("nan"), "x", -3]) == {0: 1.0, 3: -1.0} and pa({"5": 0.3}) == {})
check("nothing, junk or all zeros: off", pa(None) == {} and pa("") == {} and pa("not json") == {} and pa([0, 0, 0, 0]) == {})

# ---- the tone curve -----------------------------------------------------------------------------------------------------
lL, lC = ST._luts(0.0)
check("0: lightness and colour unchanged", np.array_equal(lL, np.arange(256, dtype=np.uint8)) and np.array_equal(lC, np.arange(256, dtype=np.uint8)))
for a in (0.3, 1.0):
    L, C = ST._luts(a)
    check("brighter %+d %%: mid-tones up, black stays black, white stays white, a little less saturated" % (100 * a),
          L[128] > 128 and L[0] == 0 and L[255] == 255 and np.all(np.diff(L.astype(int)) >= 0) and abs(int(C[200]) - 128) < 72,
          "L 128 -> %d" % L[128])
for a in (-0.3, -1.0):
    L, C = ST._luts(a)
    check("darker %d %%: mid-tones down, order kept (no banding inversions), colour a little deeper" % (100 * a),
          L[128] < 128 and L[0] == 0 and np.all(np.diff(L.astype(int)) >= 0) and abs(int(C[200]) - 128) > 72, "L 128 -> %d" % L[128])
skin = np.zeros((40, 40, 3), np.uint8); skin[:] = (120, 150, 200)              # a warm skin colour (BGR)
tb, td = ST.toned(skin, 0.5), ST.toned(skin, -0.5)
lab0, labb, labd = (cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(int)[0, 0] for x in (skin, tb, td))
hue = lambda lab: np.degrees(np.arctan2(lab[2] - 128, lab[1] - 128))
check("toned keeps the hue (within 3 deg) and moves lightness the right way",
      labb[0] > lab0[0] > labd[0] and abs(hue(labb) - hue(lab0)) < 3 and abs(hue(labd) - hue(lab0)) < 3,
      "L %d / %d / %d" % (labd[0], lab0[0], labb[0]))

# ---- the per-video state without the models (colour fallback) ---------------------------------------------------------------
ST._STATE["skin"] = "unit test: no model"; ST._STATE["person"] = "unit test: no model"
H, W = 720, 1280


def scene(xa, xb):
    """Two flat 'people': a face ellipse + a body block in each person's own skin colour, on a grey-blue wall."""
    img = np.zeros((H, W, 3), np.uint8); img[:] = (110, 100, 90)
    recs = {}
    for slot, (x, col) in enumerate(((xa, (120, 150, 205)), (xb, (70, 95, 140)))):
        cv2.ellipse(img, (x, 220), (55, 72), 0, 0, 360, col, -1)
        cv2.rectangle(img, (x - 20, 290), (x + 20, 330), col, -1)                    # neck
        cv2.rectangle(img, (x - 95, 330), (x + 95, 600), (60, 60, 160), -1)            # a red shirt
        cv2.rectangle(img, (x - 135, 330), (x - 100, 560), col, -1)                    # bare arms
        cv2.rectangle(img, (x + 100, 330), (x + 135, 560), col, -1)
        k = np.array([[x - 22, 200], [x + 22, 200], [x, 228], [x - 18, 255], [x + 18, 255]], np.float32)
        recs[slot] = dict(kps=k, bbox=np.array([x - 55, 148, x + 55, 292], np.float32))
    return img, recs


img, recs = scene(380, 900)
st = ST.SkinTone({0: 0.6})
out = st.apply(img, img, recs)
d = np.abs(out.astype(int) - img.astype(int)).max(2)
own_face, oth_face = d[200:240, 360:400].mean(), d[200:240, 880:920].mean()
own_arm, oth_arm = d[400:500, 250:275].mean(), d[400:500, 1005:1030].mean()
shirt, wall = d[420:520, 340:420].mean(), d[40:100, 600:700].mean()
check("no models (colour fallback): the chosen person's face and arm change, the other's do not",
      own_face > 8 and own_arm > 4 and oth_face < 0.5 and oth_arm < 0.5,
      "face %.1f arm %.1f | other face %.2f arm %.2f" % (own_face, own_arm, oth_face, oth_arm))
check("no models: the shirt and the wall are not changed", shirt < 1.0 and wall < 0.5, "shirt %.2f wall %.2f" % (shirt, wall))
check("the summary says which method ran", "colour" in st.summary() and "SKIN-TONE" in st.summary(), st.summary()[:140])
# hold and fade: the person not found for a few frames keeps their place, fading; then nothing
ch = []
for k in range(ST.HOLD_FRAMES + 3):
    o2 = st.apply(img, img, {1: recs[1]})
    ch.append(float(np.abs(o2.astype(int) - img.astype(int)).max(2)[200:240, 360:400].mean()))
check("a chosen person not found keeps their tone %d frames, fading, then none (no sudden flash)" % ST.HOLD_FRAMES,
      ch[0] > 4 and all(ch[i + 1] <= ch[i] + 0.5 for i in range(len(ch) - 1)) and ch[-1] < 0.5,
      " ".join("%.1f" % c for c in ch[:6]) + " ... %.1f" % ch[-1])
st2 = ST.SkinTone({0: 0.6}); st2.apply(img, img, recs)
o3 = st2.apply(img, img, {}, cut=True)
check("a shot cut forgets the last place at once", np.array_equal(o3, img))
check("nothing chosen: the frame is the same object, untouched", ST.SkinTone({}).apply(img, img, recs) is img)
check("a broken record never stops the job (frame returned as is)",
      np.array_equal(ST.SkinTone({0: 0.5}).apply(img, img, {0: dict(kps=None, bbox=None)}), img))
_, ms, info = ST.preview(img, recs, {1: -0.5})
check("preview: a mask for the chosen person only, and its share", set(ms) == {1} and info["skin_share"]["2"] > 0.005,
      json.dumps(info["skin_share"]))
ST._STATE["skin"] = ""; ST._STATE["person"] = ""

# ---- the adapter -------------------------------------------------------------------------------------------------------
import phoenix_api_adapter as PA
import core_pipeline as cp
check("the job settings keep skin_tone", PA._job_config({"skin_tone": [0.2, 0, 0, 0], "quality": "Ultra"}).get("skin_tone") == [0.2, 0, 0, 0])
check("the adapter turns it into a plain list of 4, or None when all natural",
      PA._skin_setting({"2": -0.4}) == [0.0, -0.4, 0.0, 0.0] and PA._skin_setting([0, 0]) is None)


class F:
    def __init__(s, x, e=None):
        s.bbox = np.array([x, 100, x + 80, 200], np.float32); s.kps = np.zeros((5, 2), np.float32); s.normed_embedding = e


rng = np.random.default_rng(3)
unit = lambda v: v / np.linalg.norm(v)
ea, eb = unit(rng.normal(size=512)), unit(rng.normal(size=512))
fl, fr = F(100, eb), F(600, ea)                                     # B is on the left, A on the right
m1 = PA._slot_people([fl, fr], [], [0, 1])
check("no Detect references: slots left to right (the job's own start)", m1[0] is fl and m1[1] is fr)
m2 = PA._slot_people([fl, fr], [ea, eb, None, None], [0, 1])
check("with Detect references: each slot to its own person, whatever the position", m2[0] is fr and m2[1] is fl)
m3 = PA._slot_people([fl, fr], [ea, None, None, None], [0, 1])
check("one reference: that slot by identity, the other gets the remaining face", m3[0] is fr and m3[1] is fl)


class TF:
    """A face as the job's tracker sees it (as unit203.py)."""
    def __init__(s, x, emb, gender):
        s.bbox = np.array([x, 300, x + 200, 500], np.float32); w = 200.0; x1, y1 = float(x), 300.0
        s.kps = np.array([[x1 + .3 * w, y1 + .4 * w], [x1 + .7 * w, y1 + .4 * w], [x1 + .5 * w, y1 + .6 * w],
                          [x1 + .35 * w, y1 + .8 * w], [x1 + .65 * w, y1 + .8 * w]], np.float32)
        s.normed_embedding = s.embedding = emb; s.det_score = 0.9; s.landmark_2d_106 = None; s.gender = gender
        s.sex = "M" if gender == 1 else "F"


man, woman = TF(150, ea, 1), TF(900, eb, 0)
b1, how1 = PA._bind_people([man, woman], [], [0, 1], {0: 0, 1: 1})
check("with the replacement faces (1 a woman, 2 a man), no references: #1 on the woman although she is on the right "
      "(as the job paired them on the layered scene)", b1.get(0) is woman and b1.get(1) is man, how1)
b2, _ = PA._bind_people([man, woman], [ea, None, None, None], [0, 1], {0: 0, 1: 1})
check("a Detect reference outranks gender (#1 captured as the man)", b2.get(0) is man and b2.get(1) is woman)
b3, how3 = PA._bind_people([man, woman], [], [0, 1], {})
check("without the replacement faces: left to right, and says so", b3.get(0) is man and how3 == "left to right")
b4, how4 = PA._bind_people([man, TF(500, eb, 1)], [], [0, 1], {0: 1, 1: 1})
check("two men, no references: the first man from the left gets #1 (the job pairs same-gender people at the clip's start)",
      b4.get(0) is man and how4 == "the replacement faces' gender")
check("submit_video takes skin_tone and puts it in the job", "skin_tone" in inspect.signature(cp.submit_video).parameters
      and "skin_tone=skin_tone or None" in inspect.getsource(cp.submit_video))

# ---- a resumed job keeps the setting (saved inputs, v11.2.203) ------------------------------------------------------------
os.makedirs(WORK, exist_ok=True)
os.environ["PHOENIX_STORE_KEY"] = "unit-test-secret-0123456789abcdef"
src = {0: os.path.join(WORK, "src0.jpg")}
cv2.imwrite(src[0], (rng.random((64, 64, 3)) * 255).astype(np.uint8))
vp = os.path.join(WORK, "vid.mp4"); open(vp, "wb").write(os.urandom(50_000))
ok = cp._keep_job_inputs("UNIT0205", src, vp, dict(fps=30, skin_tone=[0.25, 0.0, 0.0, -0.5], password="pw"))
spec = cp._resume_spec("UNIT0205")
check("a job resumed after a restart still has its skin tone", ok and spec is not None and spec[2].get("skin_tone") == [0.25, 0.0, 0.0, -0.5])
shutil.rmtree(WORK, ignore_errors=True)
shutil.rmtree("/tmp/phoenix_resume/UNIT0205", ignore_errors=True)
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
