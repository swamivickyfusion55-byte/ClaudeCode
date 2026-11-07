"""v11.2.144 KissFollow, the pipeline's part: the embedding de-duplication before the tracker (core_pipeline._dedupe_faces) must not
merge two people in contact whose embeddings agree because each one's ArcFace crop holds half of the other (a kiss in profile).
Stand-in faces, no models. usage: unit144_pipe.py PACKDIR"""
import sys, types
g = types.ModuleType("gradio"); g.Request = type("Request", (), {}); g.update = lambda **k: k; g.Error = Exception
sys.modules["gradio"] = g
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: ""; sys.modules["huggingface_hub"] = hf
sys.path.insert(0, sys.argv[1])
import numpy as np, core_pipeline as cp, swap_engine as E
fails = 0
def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else ""), flush=True); fails += 0 if ok else 1
class F:
    def __init__(s, b, emb, kps=None):
        s.bbox = np.array(b, np.float32); x1, y1, x2, y2 = b; w = x2 - x1
        s.kps = (np.array([[x1 + .3 * w, y1 + .4 * w], [x1 + .7 * w, y1 + .4 * w], [x1 + .5 * w, y1 + .6 * w],
                           [x1 + .35 * w, y1 + .8 * w], [x1 + .65 * w, y1 + .8 * w]], np.float32) if kps is None else kps)
        s.normed_embedding = emb; s.embedding = emb; s.det_score = 0.8; s.landmark_2d_106 = None; s.gender = 1; s.sex = "M"
rng = np.random.default_rng(3)
def unit(v):
    return v / np.linalg.norm(v)
ea, eb = unit(rng.normal(size=512)), unit(rng.normal(size=512))
# two people come together and kiss; the tracker follows both on clean frames
T = E.MultiFaceTracker([0, 1]); xa, xb = 300.0, 700.0
for k in range(20):
    xa += 6.0; xb -= 6.0
    T.assign([F([xa, 300, xa + 200, 600], ea), F([xb, 300, xb + 200, 600], eb)], {})
# the kiss: A's crop reads as B (0.85), the boxes overlap ~0.13
fa = F([xa + 4, 300, xa + 204, 600], unit(0.12 * ea + 0.88 * eb + rng.normal(size=512) * 0.01))
fb = F([xb - 4, 300, xb + 196, 600], eb)
sim = float(fa.normed_embedding @ fb.normed_embedding)
iou = cp._bbox_iou(fa.bbox, fb.bbox)
old = cp._dedupe_faces([fa, fb])
check("the stand-in kiss: the two faces' embeddings agree (%.2f) and v11.2.141-143's de-duplication keeps only one" % sim,
      len(old) == 1, "IoU %.2f, %d face(s) kept" % (iou, len(old)))
ap = cp._contact_apart([fa, fb], T)
new = cp._dedupe_faces([fa, fb], apart=ap)
check("the tracker sees two people, each where it was last seen: both faces are kept", len(new) == 2 and len(ap) == 2,
      "%d face(s) kept, %d pair entries" % (len(new), len(ap)))
# two boxes on ONE face (B's): a second box on B's cheek with B's facial points - still merged
fs = F([xb - 60, 310, xb + 150, 600], unit(eb + rng.normal(size=512) * 0.02), kps=fb.kps.copy())
ap2 = cp._contact_apart([fs, fb], T)
kept2 = cp._dedupe_faces([fs, fb], apart=ap2)
check("a second box on the partner's own face (the partner's points) is still merged", len(kept2) == 1 and len(ap2) == 0,
      "%d face(s) kept" % len(kept2))
# a box on top of another (IoU above 0.55) is merged whatever the tracker says
fo = F([xa + 10, 305, xa + 210, 605], fa.normed_embedding)
kept3 = cp._dedupe_faces([fa, fo, fb], apart=cp._contact_apart([fa, fo, fb], T))
check("two boxes on top of each other are still one face", len(kept3) == 2, "%d face(s) kept of 3" % len(kept3))
# the switch off: v11.2.141-143 exactly
cp.CONTACT_FOLLOW = False
ap4 = cp._contact_apart([fa, fb], T)
check("CONTACT_FOLLOW False: no pair is kept apart (v11.2.141-143)", len(ap4) == 0 and len(cp._dedupe_faces([fa, fb], apart=ap4)) == 1)
cp.CONTACT_FOLLOW = True
# no tracker history (first frame of a job): nothing is kept apart
ap5 = cp._contact_apart([fa, fb], E.MultiFaceTracker([0, 1]))
check("no one tracked yet: nothing is kept apart", len(ap5) == 0)
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
