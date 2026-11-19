"""v11.2.214 (SDOS-214) unit checks: an object at the mouth is taken out of the swap network's input (so the lips are drawn behind it,
not around it). Stand-in face maps and faces (no models, no user media) for the mask and the gates; then, when XSEG and CLIPDIR are
given, the real occlusion net on the synthetic bar clips (make_bar.py: a public AFLW2000 sample photo with an open mouth, a drawn bar).
usage: unit214.py PACK213 PACK214 [XSEG.onnx CLIPDIR]"""
import sys, os, types, inspect, re
P213, P214 = sys.argv[1:3]
XSEG = sys.argv[3] if len(sys.argv) > 3 else None
CLIPS = sys.argv[4] if len(sys.argv) > 4 else None
import numpy as np
import cv2
import logging
logging.disable(logging.CRITICAL)
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception
sys.modules["gradio"] = g_
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: ""; sys.modules["huggingface_hub"] = hf
sys.path.insert(0, P214)
import core_pipeline as cp
E = cp._E
fails = 0


def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else ""), flush=True); fails += 0 if ok else 1


# ---- a stand-in face map: a face oval (probability 1), the mouth's cavity (an island at 0.5), a bar from the right through the outline (0.0) ----
W = 256
SIZE = 128


def face_map(bar=True, cavity=True, island=False, tiny=False):
    p = np.zeros((W, W), np.float32)
    cv2.ellipse(p, (128, 128), (92, 112), 0, 0, 360, 1.0, -1)
    if cavity:
        cv2.ellipse(p, (100, 178), (22, 10), 0, 0, 360, 0.5, -1)               # the open mouth: a cavity the net half believes is face
    if bar:
        cv2.rectangle(p, (140, 166), (255, 192), 0.0, -1)                      # the bar: from the mouth out through the face outline
    if island:
        cv2.circle(p, (128, 190), 14, 0.0, -1)                                 # a hole inside the face that reaches no edge
    if tiny:
        cv2.rectangle(p, (212, 166), (218, 172), 0.0, -1)                      # a speck at the outline
    return p


def landmarks():
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    return np.stack([128 + 86 * np.cos(t), 126 + 106 * np.sin(t)], 1).astype(np.float32) * (SIZE / float(W))


E._OCCNET["sess"] = object()
cur = {"p": face_map()}
class FakeSess:
    class _I:
        name = "x"
    def get_inputs(self): return [FakeSess._I()]
    def run(self, a, feeds): return [np.full((1, 256, 256, 1), 0.7, np.float32)]


orig_raw = E.occ_net_raw
E._OCCNET["sess"] = FakeSess()
E._OCCNET_RAW["key"], E._OCCNET_RAW["m"] = "kept", "kept"
img_ = np.zeros((128, 128, 3), np.uint8)
r_ = orig_raw(img_, cache=False)
check("occ_net_raw(cache=False) (the erase, which runs in the swap workers): the net's map, and the shared memory of the last map untouched",
      r_.shape == (256, 256) and abs(float(r_[5, 5]) - 0.7) < 1e-6 and E._OCCNET_RAW["key"] == "kept" and E._OCCNET_RAW["m"] == "kept")
r2_ = orig_raw(img_)
check("occ_net_raw() as before: keeps the last map, and gives the same one back for the same crop",
      E._OCCNET_RAW["key"] != "kept" and orig_raw(img_) is r2_)
E._OCCNET["sess"] = object()
E.occ_net_raw = lambda aligned, cache=True: cur["p"].copy()
al = np.zeros((SIZE, SIZE, 3), np.uint8)
FW = (640, 480)

# 1. the mask
cur["p"] = face_map()
res = E.object_erase_mask(al, landmarks(), SIZE, M=None, frame_wh=None)
check("the bar through the outline gives a mask", res is not None and res[0].shape == (SIZE, SIZE) and 0.02 < res[1] < 0.4,
      "none" if res is None else "share %.3f" % res[1])
mask = cv2.resize(res[0].astype(np.uint8), (W, W), interpolation=cv2.INTER_NEAREST) > 0 if res is not None else np.zeros((W, W), bool)
bar = np.zeros((W, W), bool); bar[166:193, 140:256] = True
hull = np.zeros((W, W), np.uint8); cv2.fillConvexPoly(hull, cv2.convexHull(np.round(landmarks() * (W / float(SIZE))).astype(np.int32)), 1)
inside = bar & (hull > 0)
check("it covers the whole bar inside the face outline (edge and the way out through the outline included)",
      float((mask & inside).sum()) / max(1, int(inside.sum())) > 0.97, "%.3f" % (float((mask & inside).sum()) / max(1, int(inside.sum()))))
cavs = (face_map()[..., None][..., 0] == 0.5)
check("it does not touch the mouth's cavity (an island the net half believes is face, 14 px from the bar)", float((mask & cavs).sum()) / max(1, int(cavs.sum())) < 0.02,
      "%.3f" % (float((mask & cavs).sum()) / max(1, int(cavs.sum()))))
near = cv2.dilate(bar.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
check("nothing far from the bar is taken (within 15 px of it, or on its way out)", float((mask & ~near).sum()) / max(1, int(mask.sum())) < 0.02)
face_only = (face_map(bar=False, cavity=False) == 1.0) & ~(cv2.dilate(bar.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0)
check("no face pixel away from the bar is taken", float((mask & face_only).sum()) == 0.0)
cur["p"] = face_map(bar=False)
check("a face with an open mouth and no object: nothing is taken (None)", E.object_erase_mask(al, landmarks(), SIZE) is None)
cur["p"] = face_map(bar=False, island=True)
check("a hole inside the face outline that reaches no edge (a cavity): nothing is taken", E.object_erase_mask(al, landmarks(), SIZE) is None)
cur["p"] = face_map(bar=False, tiny=True)
check("a speck at the outline (under the minimum, short of the core): nothing is taken", E.object_erase_mask(al, landmarks(), SIZE) is None)
cur["p"] = face_map()
M0 = np.array([[1.0, 0, 0], [0, 1.0, 0]], np.float32)
check("an object in a crop that runs out of the frame (the border is smeared there): not taken",
      E.object_erase_mask(al, landmarks(), SIZE, M=M0, frame_wh=(100, 480)) is None)
check("... and in a crop inside the frame it is", E.object_erase_mask(al, landmarks(), SIZE, M=M0, frame_wh=(640, 480)) is not None)
E._P["obj_erase_edge_px"] = 0.0; E._P["obj_erase_reach"] = 0.0
res0 = E.object_erase_mask(al, landmarks(), SIZE)
E._P["obj_erase_edge_px"] = 0.025; E._P["obj_erase_reach"] = 0.16
m0 = cv2.resize(res0[0].astype(np.uint8), (W, W), interpolation=cv2.INTER_NEAREST) > 0
check("without the edge and the way out (their settings at 0) the mask stops at the eroded outline and leaves the bar's rim",
      float((m0 & inside).sum()) / max(1, int(inside.sum())) < 0.9 and (mask & inside).sum() > (m0 & inside).sum(),
      "%.3f against %.3f" % (float((m0 & inside).sum()) / max(1, int(inside.sum())), float((mask & inside).sum()) / max(1, int(inside.sum()))))
E.occ_net_raw = lambda aligned, cache=True: None
check("the net gives nothing: no mask (None)", E.object_erase_mask(al, landmarks(), SIZE) is None)
def boom(aligned, cache=True): raise RuntimeError("x")
E.occ_net_raw = boom
ok = True
try:
    r = E.object_erase_mask(al, landmarks(), SIZE); ok = r is None
except Exception:
    ok = False
check("a failing net never reaches the job (None)", ok)
E.occ_net_raw = lambda aligned, cache=True: cur["p"].copy()

# 2. the compositor: what the swap network is given
comp = E.AlignedCompositor(None)


class Face:
    def __init__(self, nose_dx=0.0):
        cx, cy, ed = 320.0, 240.0, 70.0
        self.kps = np.array([[cx - ed / 2, cy - 25], [cx + ed / 2, cy - 25], [cx + nose_dx * ed, cy + 5], [cx - 28, cy + 40], [cx + 28, cy + 40]], np.float32)
        self.landmark_2d_106 = np.random.default_rng(1).uniform(200, 440, (106, 2)).astype(np.float32)
        self.bbox = np.array([cx - 100, cy - 120, cx + 100, cy + 120], np.float32)


track = E.TrackState(0)
f0 = Face()
M = E.estimate_norm(f0.kps, 128)
IM = cv2.invertAffineTransform(M)
crop_mask = np.zeros((128, 128), bool); crop_mask[80:100, 60:128] = True                 # where the stub says the object is, in the crop
work = np.full((480, 640, 3), 150, np.uint8)
work[:, :, 1] = 140
frame_mask = cv2.warpAffine(crop_mask.astype(np.uint8) * 255, IM, (640, 480), flags=cv2.INTER_NEAREST) > 127
work[frame_mask] = (30, 30, 235)                                                         # the object: pure red
calls = []
def stub(al_, pts, size, M=None, frame_wh=None):
    calls.append(1)
    return crop_mask.copy(), float(crop_mask.mean())
E.object_erase_mask = stub
orig = work.copy()

E._P["obj_erase"] = 0
check("obj_erase = 0 (11.2.213): the network gets the frame itself", comp._erase_objects(work, orig, f0, track, None, True) is work and not calls)
E._P["obj_erase"] = 1
out = comp._erase_objects(work, orig, f0, track, None, True)
check("with an object: a new frame, the input frame itself untouched", out is not work and np.array_equal(work, orig))
inner = cv2.erode(frame_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
red = float(np.mean(out[inner][:, 2].astype(np.float32) - out[inner][:, 0].astype(np.float32)))
check("the object's colour is gone from the middle of its place (filled from around it, not left)", inner.sum() > 100 and red < 60, "red excess %.0f (was 205)" % red)
diff = np.abs(out.astype(np.int16) - work.astype(np.int16)).max(2) > 0
ys, xs = np.nonzero(diff)
check("only the object's place and a little around it changed", diff.sum() > 0 and float((diff & ~(cv2.dilate(frame_mask.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0)).sum()) == 0,
      "%d px changed, mask %d px" % (int(diff.sum()), int(frame_mask.sum())))
# a region larger than obj_erase_max_px (set to 200 here; 480 in the job): the fill is made at a smaller size and still takes the object out and changes nothing else
f4 = Face(); f4.kps = f0.kps * 3.0; f4.bbox = f0.bbox * 3.0
M4 = E.estimate_norm(f4.kps, 128); IM4 = cv2.invertAffineTransform(M4)
fm4 = cv2.warpAffine(crop_mask.astype(np.uint8) * 255, IM4, (1920, 1440), flags=cv2.INTER_NEAREST) > 127
work4 = np.full((1440, 1920, 3), 150, np.uint8); work4[:, :, 1] = 140; work4[fm4] = (30, 30, 235)
ys4, xs4 = np.nonzero(fm4)
E._P["obj_erase_max_px"] = 200
out4 = comp._erase_objects(work4, work4.copy(), f4, E.TrackState(9), None, True)
E._P["obj_erase_max_px"] = 480
inner4 = cv2.erode(fm4.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
red4 = float(np.mean(out4[inner4][:, 2].astype(np.float32) - out4[inner4][:, 0].astype(np.float32)))
d4 = np.abs(out4.astype(np.int16) - work4.astype(np.int16)).max(2) > 0
check("a region over the size limit (filled at a smaller size): the object is gone, only its place changed",
      max(ys4.max() - ys4.min(), xs4.max() - xs4.min()) > 200 and red4 < 60 and float((d4 & ~(cv2.dilate(fm4.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0)).sum()) == 0,
      "region %d x %d px, red excess %.0f" % (xs4.max() - xs4.min(), ys4.max() - ys4.min(), red4))
check("the track remembers it (erase_prev)", getattr(track, "erase_prev", False) is True and E.ERASE_STATS["erased"] >= 1)
n0 = len(calls)
trk2 = E.TrackState(1)
o2 = comp._erase_objects(work, orig, f0, trk2, None, False)
check("not Ultra / Best, nothing cut before and no earlier erase: not even looked for (no cost)", o2 is work and len(calls) == n0)
trk2.occn_keep = np.ones((64, 64), np.float32); trk2.occn_keep[40:60, :] = 0.0
o3 = comp._erase_objects(work, orig, f0, trk2, None, False)
check("not Ultra / Best, but the last frame's cut was big (over 2% of the crop): looked for", len(calls) == n0 + 1 and o3 is not work)
trk3 = E.TrackState(2); trk3.erase_prev = True
o4 = comp._erase_objects(work, orig, f0, trk3, None, False)
check("not Ultra / Best, an object was taken out on the last frame: looked for again (it holds)", len(calls) == n0 + 2 and o4 is not work)
n1 = len(calls)
check("two faces in contact (rivals): the contact rules decide, nothing taken", comp._erase_objects(work, orig, f0, track, [object()], True) is work and len(calls) == n1)
fp = Face(); fp._partners = [object()]
check("a partner in contact: nothing taken", comp._erase_objects(work, orig, fp, track, None, True) is work and len(calls) == n1)
ft = Face(nose_dx=0.7)
check("a face turned beyond obj_erase_yaw (the nose far off the eyes' middle): nothing taken (the profile mask decides there)", comp._erase_objects(work, orig, ft, track, None, True) is work and len(calls) == n1)
fn = Face(nose_dx=0.2)
check("a face turned a little: taken", comp._erase_objects(work, orig, fn, track, None, True) is not work and len(calls) == n1 + 1)
fe = Face(); fe.landmark_2d_106 = None
check("no landmarks: nothing taken", comp._erase_objects(work, orig, fe, track, None, True) is work)
E.object_erase_mask = lambda *a, **k: None
trk4 = E.TrackState(3); trk4.erase_prev = True
check("no object found: the frame itself, and the track forgets (erase_prev False)", comp._erase_objects(work, orig, f0, trk4, None, True) is work and trk4.erase_prev is False)
def boom2(*a, **k): raise RuntimeError("x")
E.object_erase_mask = boom2
check("a failing mask never reaches the job (the frame itself)", comp._erase_objects(work, orig, f0, track, None, True) is work)
E.object_erase_mask = stub
sess_keep = E._OCCNET["sess"]; E._OCCNET["sess"] = None
check("no occlusion net loaded: the frame itself", comp._erase_objects(work, orig, f0, track, None, True) is work)
E._OCCNET["sess"] = sess_keep

# 2b. the paste leaves the object alone where it was taken out (build_mask): the mask with and without the cut, the erase stubbed
rng_ = np.random.default_rng(2)
tgt = rng_.uniform(70, 190, (128, 128, 3)).astype(np.uint8)
stub_region = np.zeros((128, 128), bool); stub_region[70:95, 40:128] = True
cut_calls = []
cut_dil = []
def stub_cut(al_, pts, size, M=None, frame_wh=None, cache=False, dilate=None):
    cut_calls.append(cache)
    cut_dil.append(dilate)
    return stub_region.copy(), float(stub_region.mean())
E.object_erase_mask = stub_cut
E._P["occ_net"] = 1
def mk(**kw):
    f_ = kw.pop("face", None) or Face()
    if not hasattr(f_, "_erase_on"):
        f_._erase_on = True                     # (what _swap_one and _reuse_one do at Ultra, Best and Optimized)
    return comp.build_mask(f_, M, 128, track=None, aligned_target=tgt, frame_wh=(640, 480), **kw)
E._P["obj_erase_cut"] = 0
m0 = mk()
E._P["obj_erase_cut"] = 1
cut_calls.clear()
m1 = mk()
core_ = cv2.erode(stub_region.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
far_ = ~(cv2.dilate(stub_region.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0)
live = m0 > 0.5
check("build_mask: the object's place is cut from the paste (it was taken out of the network's input, so nothing is drawn there to show), the net's map shared, the cut 1% wider than the object",
      cut_calls == [True] and cut_dil == [E._P["obj_erase_cut_dilate"]] and abs(E._P["obj_erase_cut_dilate"] - 0.01) < 1e-9 and int((live & core_).sum()) > 50 and float(m1[live & core_].max()) < 0.1,
      "mask there %.2f -> %.2f" % (float(m0[live & core_].mean()) if (live & core_).any() else -1, float(m1[live & core_].mean()) if (live & core_).any() else -1))
check("...and nothing else changes (away from it the mask is the same, nowhere is it larger)", float(np.abs(m1 - m0)[far_].max()) < 1e-6 and float((m1 - m0).max()) < 1e-3,      # (the cut's edge snap smooths by a hair)
      "far max diff %.5f, largest increase %.5f" % (float(np.abs(m1 - m0)[far_].max()), float((m1 - m0).max())))
E._P["obj_erase_cut"] = 0
cut_calls.clear(); mk()
check("obj_erase_cut = 0 (or obj_erase = 0): the mask of 11.2.213, the erase not asked", cut_calls == [])
E._P["obj_erase_cut"] = 1; E._P["obj_erase"] = 0
cut_calls.clear(); mz = mk()
check("obj_erase = 0: not asked, the mask as without the cut", cut_calls == [] and float(np.abs(mz - m0).max()) < 1e-6)
E._P["obj_erase"] = 1
cut_calls.clear(); mr = mk(rivals=[np.random.default_rng(3).uniform(0, 300, (106, 2)).astype(np.float32)])
check("two faces in contact (rivals): the contact rules decide, the erase's cut stands aside", cut_calls == [])
fpp = Face(); fpp._partners = [{"kps": fpp.kps, "lmk": fpp.landmark_2d_106, "front": False}]
cut_calls.clear(); mk(face=fpp)
check("a partner touching: the erase's cut stands aside", cut_calls == [])
fnt = Face(); fnt._erase_on = False
cut_calls.clear(); mk(face=fnt)
check("a tier that does not erase (Fast, Balanced: face._erase_on False): the paste's cut is not asked either, the mask as in 11.2.213", cut_calls == [])
funtag = Face()
cut_calls.clear(); comp.build_mask(funtag, M, 128, track=None, aligned_target=tgt, frame_wh=(640, 480))
check("a face nobody tagged (a path that does not know the tier): the paste's cut is not asked either (11.2.213)", cut_calls == [])
fyes = Face(); fyes._erase_on = True
cut_calls.clear(); mk(face=fyes)
check("a tier that erases (face._erase_on True): asked", cut_calls == [True])
cut_calls.clear(); mk(face=Face(nose_dx=0.8))
check("a face turned beyond obj_erase_yaw: not asked", cut_calls == [])
E.object_erase_mask = lambda *a, **k: None
mn = mk()
check("no object found: the mask as without the cut", float(np.abs(mn - m0).max()) < 1e-6)
E.object_erase_mask = boom2
try:
    mb_ = mk(); okb = float(np.abs(mb_ - m0).max()) < 1e-6
except Exception:
    okb = False
check("a failing erase never reaches the mask (as without the cut)", okb)
E.object_erase_mask = stub

# 3. the wiring: run() hands the network the frame without the object; _swap_one asks for it at Ultra, Best and Optimized only
seen = {}
def fake_raw(img, face, src):
    seen["img"] = img
    return None, None
comp._raw_swap = fake_raw
E.object_erase_mask = stub
comp.run(work, orig, f0, object(), track=E.TrackState(4), erase_always=True)
check("run(erase_always=True) gives the swap network the erased frame, not the frame", seen["img"] is not work and not np.array_equal(seen["img"], work))
seen.clear()
comp.run(work, orig, f0, object(), track=E.TrackState(5))
check("run() with no flag and nothing cut before gives it the frame itself", seen["img"] is work)
check("run() has the erase_always flag, off by default", "erase_always" in inspect.signature(E.AlignedCompositor.run).parameters
      and inspect.signature(E.AlignedCompositor.run).parameters["erase_always"].default is False)
src = inspect.getsource(cp._swap_one)
check("_swap_one asks for the erase at Ultra, Best and Optimized only (Fast and Balanced as in 11.2.213)",
      cp._ERASE_TIERS == ("Ultra", "Best", "Optimized") and re.search(r'erase_always=\(str\(quality\) in _ERASE_TIERS\)', src) is not None
      and "face._erase_on = str(quality) in _ERASE_TIERS" in src)
cps = inspect.getsource(cp)
check("the job's key-swap loop (where the network really runs) hands it the erased frame at those tiers, with the contact rules (partners, rivals); the other tiers the frame itself",
      '_frm_in = comp._erase_objects(frm, frm, face, rec.get("track"), _rv, True)' in cps and "comp._raw_swap(_frm_in, face, src)" in cps
      and "_rv, _frm_in = None, frm" in cps and "if _erase_all:" in cps
      and "face._partners = _partner_info(records, slot)" in cps and "_rv = _rival_landmarks(records, slot)" in cps
      and "_erase_all = str(quality) in _ERASE_TIERS" in cps)
check("the job's composite of a cached swap (_reuse_one) tells the mask whether the tier erases (the cut follows the tier)",
      'face._erase_on = str(quality) in _ERASE_TIERS' in inspect.getsource(cp._reuse_one))
# the real PredictedFace (it has __slots__: an attribute that is not listed raises) through the two paths that tag it with the tier
class StubComp:
    def __init__(self): self.seen = []
    def run(self, work_, orig_, face_, src_, **kw):
        self.seen.append(("run", getattr(face_, "_erase_on", "missing"), kw.get("erase_always"))); return work_, True
    def reuse(self, work_, orig_, kps_, face_, **kw):
        self.seen.append(("reuse", getattr(face_, "_erase_on", "missing"), None)); return work_, True
stub_c = StubComp(); keep_comp = cp._compositor; cp._compositor = lambda: stub_c
w_ = np.zeros((480, 640, 3), np.uint8)
for tier, want in (("Ultra", True), ("Best", True), ("Optimized", True), ("Balanced", False), ("Fast", False)):
    pf = E.PredictedFace(Face().bbox, Face().kps, Face().landmark_2d_106)
    stub_c.seen.clear()
    try:
        cp._swap_one(w_, w_, pf, object(), tier)
        r1 = stub_c.seen[-1]
    except Exception as ex_:
        r1 = ("error", repr(ex_), None)
    trk_ = E.TrackState(7); trk_.fake = np.zeros((128, 128, 3), np.uint8)
    rec_ = {"bbox": Face().bbox, "kps": Face().kps, "lmk": Face().landmark_2d_106, "alpha": 1.0, "track": trk_}
    try:
        cp._reuse_one(w_, w_, rec_, tier)
        r2 = stub_c.seen[-1]
    except Exception as ex_:
        r2 = ("error", repr(ex_), None)
    check("a real PredictedFace through _swap_one and _reuse_one at %s: tagged erase %s, and the legacy path's flag %s" % (tier, want, want),
          r1 == ("run", want, want) and r2 == ("reuse", want, None), "%s | %s" % (r1, r2))
cp._compositor = keep_comp
check("the job resets and logs the erase counters", "ERASE_STATS.update" in inspect.getsource(cp) and "ERASE: an object at the mouth" in inspect.getsource(cp))
import config as cfg
check("the version is 11.2.214", cfg.VERSION == "v11.2.214", cfg.VERSION)

# 4. the real occlusion net on the synthetic bar clips (optional)
if XSEG and CLIPS:
    import importlib
    spec = importlib.util.spec_from_file_location("se214real", os.path.join(P214, "swap_engine.py"))
    Er = importlib.util.module_from_spec(spec); spec.loader.exec_module(Er)
    check("the real occlusion net loads", Er.occ_net_load(XSEG, threads=2))
    from insightface.app import FaceAnalysis
    fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "landmark_2d_106"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
    compr = Er.AlignedCompositor(None)
    allcov, allcav, none_px = [], [], []
    for clip, frames in (("bar_none.mp4", (30, 60, 100)), ("barh_choc.mp4", (50, 70, 90)), ("barh_cream.mp4", (70,)), ("barh_strawberry.mp4", (70,)),
                         ("barv_strawberry.mp4", (70,)), ("bar_choc.mp4", (70,)), ("bar_cream.mp4", (60, 90))):
        path = os.path.join(CLIPS, clip)
        if not os.path.exists(path):
            continue
        gt = np.load(path[:-4] + "_gt.npz") if os.path.exists(path[:-4] + "_gt.npz") else None
        cap = cv2.VideoCapture(path)
        for i in frames:
            cap.set(1, i); ok, f = cap.read()
            face = max(fa.get(f), key=lambda q: q.det_score)
            out_ = compr._erase_objects(f, f, face, None, None, True)
            chg = np.abs(out_.astype(np.int16) - f.astype(np.int16)).max(2) > 0
            if clip == "bar_none.mp4":
                none_px.append(int(chg.sum())); continue
            barm = cv2.resize(gt["a"][i], (1280, 720), interpolation=cv2.INTER_LINEAR) > 200
            opn = cv2.resize(gt["open"][i], (1280, 720), interpolation=cv2.INTER_LINEAR) > 100
            hl = np.zeros((720, 1280), np.uint8); cv2.fillConvexPoly(hl, cv2.convexHull(face.landmark_2d_106.astype(np.int32)), 1)
            inb = barm & (hl > 0)
            allcov.append(float((chg & inb).sum()) / max(1, int(inb.sum())))
            cavm = opn & ~(cv2.dilate(barm.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0)
            allcav.append(float((chg & cavm).sum()) / max(1, int(cavm.sum())))
    check("real net, frames with an open mouth and NO object: nothing is taken", len(none_px) > 0 and max(none_px) == 0, str(none_px))
    check("real net, a bar half in the open mouth (5 colours and angles): at least 95% of the bar inside the face outline is taken out (every frame)",
          len(allcov) > 0 and min(allcov) > 0.95, "min %.3f, mean %.3f over %d frames" % (min(allcov or [0]), float(np.mean(allcov or [0])), len(allcov)))
    check("real net: the open mouth's cavity (away from the bar) is not touched (under 5% on every frame)", len(allcav) > 0 and max(allcav) < 0.05,
          "max %.3f, mean %.3f" % (max(allcav or [1]), float(np.mean(allcav or [1]))))
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
