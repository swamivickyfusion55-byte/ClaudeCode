"""v11.2.210 (SDOS-210) unit checks: the new rules of the face mask, on synthetic clips (public photo; NOT user footage).
1. Eating (make_lick.py clip; the real detector and landmarks on every frame, the real build_mask with its memory):
   the replacement's lips stay on while the scoop is on the lips and between licks, the mouth does not change from one
   frame to the next, the scoop is not painted over, and with the new switches off the mask is v11.2.209's exactly.
2. Two side-view faces as partners (the two people of the make_pkiss.py clip): each face's paint stays on its own side of
   the line between them, its own side is painted as before, the occlusion model's cut is not run; a front-view partner
   leaves the mask as it was. (How the whole job then behaves in a side-view kiss is measured on the job itself.)
3. The helpers: the line between two faces, the cover of one outline by another, a least-squares similarity.
usage: unit210.py PACK209 PACK210 LICK.mp4 PKISS.mp4 XSEG.onnx"""
import sys, os, types, importlib.util
P209, P210, LICK, PKISS, XSEG = sys.argv[1:6]
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception
sys.modules["gradio"] = g_
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: ""; sys.modules["huggingface_hub"] = hf
import logging
logging.disable(logging.CRITICAL)
import numpy as np, cv2


def load(name, pack):
    spec = importlib.util.spec_from_file_location(name, os.path.join(pack, "swap_engine.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


E9, E0 = load("se209", P209), load("se210", P210)
for E in (E9, E0):
    assert E.occ_net_load(XSEG, threads=2), "occlusion model did not load"
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "landmark_2d_106"], providers=["CPUExecutionProvider"])
fa.prepare(ctx_id=-1, det_size=(640, 640))
fails = 0


def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else ""), flush=True); fails += 0 if ok else 1


def frames_of(p):
    c = cv2.VideoCapture(p); o = []
    while True:
        ok, f = c.read()
        if not ok:
            return o
        o.append(f)


# ---- 1. eating ---------------------------------------------------------------------------------------------------
fr = frames_of(LICK); g = np.load(LICK[:-4] + "_gt.npz"); A, LIPS, K0 = g["a"], g["lips"], g["kps"]
size = 256
dets, last = [], None
for f in fr:
    fs = fa.get(f)
    d = min(fs, key=lambda q: float(np.linalg.norm(q.kps.mean(0) - K0.mean(0)))) if fs else None
    if d is None or float(np.linalg.norm(d.kps.mean(0) - K0.mean(0))) > 80:
        d = last
    last = d; dets.append(d)
lipm = np.zeros(fr[0].shape[:2], np.uint8); cv2.fillPoly(lipm, [np.round(LIPS).astype(np.int32)], 1)


def run_lick(E, switches):
    keep = dict(E._P)
    for k, v in switches.items():
        E._P[k] = v
    comp = E.AlignedCompositor.__new__(E.AlignedCompositor); tr = E.TrackState(slot=0); out = []
    for f, d in zip(fr, dets):
        M = E.estimate_norm(d.kps, size)
        at = cv2.warpAffine(f, M, (size, size), borderMode=cv2.BORDER_REPLICATE)
        out.append((comp.build_mask(d, M, size, track=tr, aligned_target=at, frame_wh=(f.shape[1], f.shape[0])), M, at))
    E._P.clear(); E._P.update(keep)
    return out


def lick_numbers(E, res):
    face0 = cv2.erode((cv2.resize(E.occ_net_raw(res[0][2]), (size, size)) >= 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    rows = []
    for t, (m, M, at) in enumerate(res):
        a = cv2.warpAffine(A[t].astype(np.float32) / 255.0, M, (size, size))
        lp = cv2.warpAffine(lipm, M, (size, size), flags=cv2.INTER_NEAREST) > 0
        obj = cv2.erode((a > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & face0
        free = lp & (a < 0.05)
        rows.append((float(m[free].mean()) if free.sum() > 30 else np.nan, float(m[obj].mean()) if obj.sum() > 30 else np.nan,
                     float((lp & (a > 0.5)).sum()) / max(1.0, float(lp.sum())), int((a > 0.5).sum())))
    return np.array(rows, float)


OFF = {"eat_mouth": 1.0, "occ_net_lo": 0.5, "occ_net_hi": 1.0, "occ_net_follow": 0.0, "occ_net_below": 0.0, "occ_face_forget": 0.15,
       "contact_side": 0}
r9 = run_lick(E9, {}); r0off = run_lick(E0, OFF); r0 = run_lick(E0, {})
dmax = max(float(np.abs(a[0] - b[0]).max()) for a, b in zip(r9, r0off))
check("the new switches off: v11.2.210's mask equals v11.2.209's on all %d frames of the eating clip" % len(r9), dmax < 1e-6, "largest difference %.2e" % dmax)
check("a valid mask (0-1, finite) on every frame", all(np.isfinite(m[0]).all() and m[0].min() >= 0 and m[0].max() <= 1 for m in r0))
N9, N0 = lick_numbers(E9, r9), lick_numbers(E0, r0)
at = N0[:, 2] > 0.10; low = (N0[:, 3] > 200) & ~at
check("the scoop on the lips: the replacement's lips stay on (0.80 or more of the lips not under it; v11.2.209 showed the footage's own)",
      np.nanmean(N0[at, 0]) >= 0.80, "v11.2.210 %.2f, v11.2.209 %.2f" % (np.nanmean(N0[at, 0]), np.nanmean(N9[at, 0])))
check("between licks (the scoop under the lips): the lips fully the replacement's (0.95 or more; v11.2.209 faded them in and out)",
      np.nanmin(N0[low, 0]) >= 0.95, "lowest: v11.2.210 %.2f, v11.2.209 %.2f" % (np.nanmin(N0[low, 0]), np.nanmin(N9[low, 0])))
c0, c9 = float(np.nanmax(np.abs(np.diff(N0[:, 0])))), float(np.nanmax(np.abs(np.diff(N9[:, 0]))))
check("the mouth does not change between the two people from one frame to the next (lips paint changes by under 0.15)", c0 < 0.15,
      "largest change: v11.2.210 %.2f, v11.2.209 %.2f" % (c0, c9))
check("the scoop over the face is not painted over while it is on the lips (under 0.05)", np.nanmean(N0[at, 1]) < 0.05,
      "v11.2.210 %.2f, v11.2.209 %.2f" % (np.nanmean(N0[at, 1]), np.nanmean(N9[at, 1])))
check("...nor while it rests on the chin, under the lips (under 0.15; v11.2.209 painted the chin over it)", np.nanmean(N0[low, 1]) < 0.15,
      "v11.2.210 %.2f, v11.2.209 %.2f" % (np.nanmean(N0[low, 1]), np.nanmean(N9[low, 1])))
none = N0[:, 3] == 0
check("with nothing at the mouth the lips are fully painted", np.nanmin(N0[none, 0][:18]) >= 0.99, "%.3f" % np.nanmin(N0[none, 0][:18]))

# ---- 2. two side-view faces as partners ------------------------------------------------------------------------------
# The two people of the side-view clip on a frame where they are still apart (clean detections), declared partners: the
# rule acts on the geometry, whether or not they touch yet.
kf = frames_of(PKISS); kg = np.load(PKISS[:-4] + "_gt.npz"); KK = kg["kps"]
fs = fa.get(kf[8])
pp = [min(fs, key=lambda q: float(np.linalg.norm(q.kps.mean(0) - KK[8, p].mean(0)))) for p in range(2)]
check("the side-view clip's two faces are found apart and are side views (nose offset 0.45 or more)",
      pp[0] is not pp[1] and all(E0._yaw_offset(f.kps) >= 0.45 for f in pp), "offsets %.2f, %.2f" % tuple(E0._yaw_offset(f.kps) for f in pp))
calls = [0]
_onk = E0.occ_net_keep


def _count(*a, **k):
    calls[0] += 1
    return _onk(*a, **k)


E0.occ_net_keep = _count


def side_mask(p, partner_kps, front):
    comp = E0.AlignedCompositor.__new__(E0.AlignedCompositor)
    f = pp[p]; face = E0.PredictedFace(f.bbox, f.kps, f.landmark_2d_106, None, 0.5)
    if partner_kps is not None:
        face._partners = [{"kps": partner_kps, "lmk": pp[1 - p].landmark_2d_106, "front": front}]
    M = E0.estimate_norm(f.kps, size)
    at = cv2.warpAffine(kf[8], M, (size, size), borderMode=cv2.BORDER_REPLICATE)
    calls[0] = 0
    m = comp.build_mask(face, M, size, track=E0.TrackState(slot=p), aligned_target=at, frame_wh=(kf[8].shape[1], kf[8].shape[0]))
    return m, M, calls[0]


for p, nm in ((0, "the face behind"), (1, "the face in front")):
    m_no, M, n_no = side_mask(p, None, False)
    m_si, _, n_si = side_mask(p, pp[1 - p].kps, front=(p == 0))
    w = E0._own_side(pp[p].kps, pp[1 - p].kps, M, size, bias=(-1.0 if p == 0 else 1.0) * float(E0._P["contact_side_bias"]))
    far = w < 0.02
    check("side-view partners, %s: no paint on the partner's side of the line" % nm, float(m_si[far].max()) < 0.02 if far.any() else True,
          "largest paint there %.3f (without the rule %.3f)" % (float(m_si[far].max()) if far.any() else 0.0, float(m_no[far].max()) if far.any() else 0.0))
    own = w > 0.98
    check("side-view partners, %s: its own side painted as without a partner (within 3 %%)" % nm,
          abs(float(m_si[own].sum()) - float(m_no[own].sum())) <= 0.03 * float(m_no[own].sum()) + 1.0,
          "%.0f vs %.0f" % (float(m_si[own].sum()), float(m_no[own].sum())))
    check("side-view partners, %s: the occlusion model's cut is not run (it takes either face for the face there)" % nm, n_si == 0 and n_no >= 1,
          "calls with a partner %d, without %d" % (n_si, n_no))
# a front-view partner: the rule stands aside (the face in front cuts with its outline, as before)
m_no, M, _ = side_mask(1, None, False)
m_fr, _, n_fr = side_mask(1, K0, False)                # (K0: the eating clip's front-view face as the partner's points)
check("a front-view partner: the mask is as without the rule, and the occlusion model's cut runs",
      float(np.abs(m_fr - m_no).max()) < 1e-6 and n_fr >= 1, "largest difference %.2e, calls %d" % (float(np.abs(m_fr - m_no).max()), n_fr))
E0.occ_net_keep = _onk

# ---- 3. helpers -----------------------------------------------------------------------------------------------------
M0 = np.float32([[1, 0, 0], [0, 1, 0]])
ka = np.float32([[60, 100], [100, 100], [80, 125], [65, 150], [95, 150]]); kb = ka + np.float32([120, 0])
w = E0._own_side(ka, kb, M0, 256)
check("the line between two faces of the same size lies midway: own side painted, the partner's not",
      w is not None and w[125, 80] > 0.99 and w[125, 200] < 0.01 and abs(float(w[125, 140]) - 0.5) < 0.05, "at own centre %.2f, midway %.2f, partner %.2f" % (w[125, 80], w[125, 140], w[125, 200]))
sys.path.insert(0, P210)
import core_pipeline as cp
sq = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]] * 2)
check("cover of an outline by one shifted half its width is 0.5; apart, 0", abs(cp._hh_cover(sq, sq + np.float32([50, 0])) - 0.5) < 0.01 and cp._hh_cover(sq, sq + np.float32([200, 0])) == 0.0)
ang = np.radians(12.0); Rm = np.float32([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * 1.07
T = cp._hh_sim_fit(ka, ka @ Rm.T + np.float32([13, -7]))
check("a least-squares similarity recovers a turn of 12 deg, a size change of 7 % and a shift exactly",
      float(np.abs(cp._hh_apply(T, ka) - (ka @ Rm.T + np.float32([13, -7]))).max()) < 1e-2)
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
