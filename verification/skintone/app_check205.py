"""v11.2.205: build the Space's real app.py Blocks (Gradio 4.44.1) with the face-swap model packages stubbed (MediaPipe and
scikit-image are REAL: the skin models run), launch it locally and call phoenix_skin_preview exactly as the phone app does
(gradio_client: the video file + a JSON string). The face detector is replaced by the scene's known face boxes. The video is a
layered two-person test scene (make_pair.py; people cut out of insightface's public sample photo, NOT user media).
usage: app_check205.py PACKDIR pair_<name>.npz pair_<name>.mp4 [genders of the two layers, e.g. 1,0 = back man, front woman]"""
import sys, types, importlib.abc, importlib.machinery, unittest.mock as um, os, json
PACK, NPZ, MP4 = sys.argv[1], sys.argv[2], sys.argv[3]
GEN = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
HEAVY = ("insightface", "onnxruntime", "torch", "torchvision", "basicsr", "gfpgan", "realesrgan", "facexlib", "spaces",
         "rembg", "timm", "kornia", "lama_cleaner", "simple_lama_inpainting")


class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in HEAVY:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        m = um.MagicMock(name=spec.name); m.__path__ = []; m.__spec__ = spec; return m

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _Stub())
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
sys.path.insert(0, PACK); os.chdir(PACK)
import numpy as np, cv2
import app
import core_pipeline as cp
demo = next(v for v in vars(app).values() if v.__class__.__name__ == "Blocks")
names = sorted(d.api_name for d in demo.fns.values() if getattr(d, "api_name", None))
print("API names:", names)
assert "phoenix_skin_preview" in names and "phoenix_patch_preview" in names and "phoenix_submit_video" in names
d = np.load(NPZ); KPS, BB, OWN, SK, FR = d["kps"], d["bbox"], d["owner"], d["skin"], d["frames"]
seen = {}
_grab = cp._grab_frame


def grab(video, pct):
    frm, idx, total = _grab(video, pct)
    seen["idx"] = idx
    return frm, idx, total


class Face:
    def __init__(s, kps, bbox, gender=None):
        s.kps, s.bbox, s.det_score, s.normed_embedding = np.asarray(kps, np.float32), np.asarray(bbox, np.float32), 0.9, None
        s.gender = gender


def fa_get(img, *a, **k):
    i = seen.get("idx", 0)
    if seen.get("none"):
        return []
    return [Face(KPS[i, s], BB[i, s], GEN[s] if GEN else None) for s in (0, 1)]


cp._grab_frame, cp._fa_get, cp._load = grab, fa_get, (lambda *a, **k: None)
# the replacement faces: a woman (#1) and a man (#2); the face model is stubbed, so their gender is given here
SRC = {"woman.jpg": 0, "man.jpg": 1}
_TMP = os.path.join(os.path.dirname(os.path.abspath(MP4)), "srcfaces"); os.makedirs(_TMP, exist_ok=True)
for _n in SRC:
    cv2.imwrite(os.path.join(_TMP, _n), np.full((64, 64, 3), 128, np.uint8))
_as_bgr = cp._as_bgr_image
cp._as_bgr_image = lambda p: (os.path.basename(str(p)) if isinstance(p, str) else _as_bgr(p))
cp._cached_source_face = lambda im: (Face(np.zeros((5, 2)), np.zeros(4), SRC.get(im)) if isinstance(im, str) and im in SRC else None)
cp._embed_faces = lambda faces: None
demo.queue().launch(server_name="127.0.0.1", server_port=7873, prevent_thread_lock=True, show_api=True)
from gradio_client import Client, handle_file
c = Client("http://127.0.0.1:7873/", verbose=False)
fails = 0


def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else ""), flush=True); fails += 0 if ok else 1


res = c.predict(handle_file(MP4), json.dumps({"position_pct": 50, "skin_tone": [0.5, 0, 0, 0]}), api_name="/phoenix_skin_preview")
info = json.loads(res[2])
print("JSON:", {k: v for k, v in info.items() if k != "people"}, info.get("people"))
check("phoenix_skin_preview answers through the real Gradio app: a toned frame, the untouched frame and JSON ok",
      info.get("ok") and os.path.isfile(res[0]) and os.path.isfile(res[1]))
i = seen["idx"]
after, before = cv2.imread(res[0]), cv2.imread(res[1])
L = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
dl = L(after) - L(before)
# slot 1 = the leftmost face (no Detect references sent): which layer is that?
left = 0 if BB[i, 0][0] < BB[i, 1][0] else 1
me, oth = (OWN[i] == left + 1) & (SK[i] > 0), (OWN[i] == 2 - left) & (SK[i] > 0)
# this frame is the scene's closest contact: the other woman's arm reaches across and touches person #1's hand
check("person #1 (the left one) is brighter on their skin; the other person's skin changes at most a quarter as much",
      dl[me].mean() > 5.0 and abs(dl[oth].mean()) < 0.25 * dl[me].mean(),
      "person #1 skin +%.1f levels, the other's skin %+.2f levels (%.0f %%)" % (dl[me].mean(), dl[oth].mean(), 100 * abs(dl[oth].mean()) / dl[me].mean()))
check("the untouched frame is the video's frame", float(np.abs(before.astype(int) - FR[i].astype(int)).mean()) < 3.0)
p1 = [p for p in info["people"] if p["slot"] == 1]
check("JSON names person #1 with its face box and amount", bool(p1) and abs(p1[0]["amount"] - 0.5) < 1e-6 and len(p1[0]["box"]) == 4)
# darker, for the other person
res2 = c.predict(handle_file(MP4), json.dumps({"position_pct": 50, "skin_tone": {"2": -0.6}}), api_name="/phoenix_skin_preview")
dl2 = L(cv2.imread(res2[0])) - L(cv2.imread(res2[1]))
check("person #2 darker (the setting as {\"2\": -0.6}): their skin darker; person #1's changes at most a quarter as much",
      dl2[oth].mean() < -5.0 and abs(dl2[me].mean()) < 0.25 * abs(dl2[oth].mean()),
      "person #2 %+.1f, person #1 %+.2f (%.0f %%)" % (dl2[oth].mean(), dl2[me].mean(), 100 * abs(dl2[me].mean()) / abs(dl2[oth].mean())))
if GEN:
    # the job pairs replacement #1 (a woman's face) with the woman - the preview must tone the same person
    res4 = c.predict(handle_file(MP4), json.dumps({"position_pct": 50, "skin_tone": [0.5, 0, 0, 0], "slots": [1, 2],
                                                   # the app uploads the faces first and sends the Space's paths; this
                                                   # server runs here, so the local paths are those paths
                                                   "faces": [os.path.join(_TMP, "woman.jpg"), os.path.join(_TMP, "man.jpg")]}),
                     api_name="/phoenix_skin_preview")
    i4 = json.loads(res4[2]); dl4 = L(cv2.imread(res4[0])) - L(cv2.imread(res4[1]))
    wl = GEN.index(0)
    wsk, msk = (OWN[i] == wl + 1) & (SK[i] > 0), (OWN[i] == 2 - wl) & (SK[i] > 0)
    check("with the replacement faces sent (#1 a woman's face): #1 is the woman, as the job pairs them, whatever her position",
          dl4[wsk].mean() > 5.0 and abs(dl4[msk].mean()) < 0.25 * dl4[wsk].mean() and "gender" in i4.get("by", ""),
          "woman %+.1f, man %+.2f, matched by %s" % (dl4[wsk].mean(), dl4[msk].mean(), i4.get("by")))
seen["none"] = True
res3 = c.predict(handle_file(MP4), json.dumps({"position_pct": 50, "skin_tone": [0.5, 0, 0, 0]}), api_name="/phoenix_skin_preview")
i3 = json.loads(res3[2])
check("a frame with no face: a clear message, no crash", i3.get("ok") is False and "No face" in i3.get("error", ""), i3.get("error", ""))
demo.close()
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
