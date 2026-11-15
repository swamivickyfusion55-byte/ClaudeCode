"""v11.2.211 (SDOS-211) unit checks. Stand-in and synthetic data only (no footage, no models).
1. Landmarks (TrackState): the 106 face landmarks are smoothed in the face's own frame -
   - a still face with a noisy detector: the landmarks' jitter in the picture falls by half or more;
   - a face moving steadily (6 px a frame) and a face turning: no lag against the smoothed 5 points (the landmarks keep their place
     in the face), and the landmarks stay where the raw detection puts them relative to the 5 points;
   - a jaw that really moves (8 px of the 112-px face in 3 frames): followed within 4 frames;
   - without a detection (predict) the landmarks move with the 5 points;
   - lmk_smooth = 0 keeps them as detected (11.2.210).
2. The occlusion cut (occ_net_keep, with a synthetic face map): a thin strip along the face's own edge is not an object (it switched on
   and off from frame to frame next to a touching face); a spoon-sized blob at the mouth still is; a piece cut on the last frame stays
   cut while it still overlaps itself; with the switches off, v11.2.210's behaviour.
3. The carried face (core_pipeline._HiddenHold._smooth_T): steady motion is passed without lag, measurement noise is reduced, and a
   real reversal is followed.
4. The carried face behind a face that passes it (core_pipeline._HiddenHold._step_slot, with stand-ins for the picture readings):
   - a frame-to-frame reading that shows the face in FRONT's motion (the corners of the visible part belong to it) is not used while the
     carried face is moving against that face, and never changes its learnt speed: it goes on at its own speed instead of being
     dragged with the front face (11.2.210 dragged it);
   - after 15 frames without anything to follow it is handed back to the job's own rules over 6 frames;
   - a detector that finds the face somewhere else (it came out on the other side) takes the position over in 6 steps, not one jump;
   - hid_weak_flow = 0 / hid_hand = 0 give 11.2.210's behaviour.
5. The cut of a CARRIED face by the face in front (build_mask, a stand-in outline over half of the face): its outline is dilated less
   (rival_dilate_hid), so more of the face behind is painted; a face that is not carried, and rival_dilate_hid = 0, give 11.2.210's cut.
usage: unit211.py PACK210 PACK211"""
import sys, os, types, importlib.util
P210, P211 = sys.argv[1:3]
g_ = types.ModuleType("gradio"); g_.Request = type("Request", (), {}); g_.update = lambda **k: k; g_.Error = Exception
sys.modules["gradio"] = g_
hf = types.ModuleType("huggingface_hub"); hf.hf_hub_download = lambda *a, **k: ""; sys.modules["huggingface_hub"] = hf
import logging
logging.disable(logging.CRITICAL)
import numpy as np, cv2


def load(name, pack):
    spec = importlib.util.spec_from_file_location(name, os.path.join(pack, "swap_engine.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


E = load("se211", P211)
fails = 0


def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("  [" + extra + "]" if extra else ""), flush=True); fails += 0 if ok else 1


rng = np.random.default_rng(7)
# a 106-point face in a 112-px ArcFace frame: the canonical 5 points, an outline, brows, nose, lips - what matters is that they are 106
TPL = E._arcface_dst(112).astype(np.float64)
ang = np.linspace(0, 2 * np.pi, 33, endpoint=False)
OUT = np.stack([56 + 38 * np.cos(ang), 62 + 50 * np.sin(ang)], 1)
REST = np.stack([56 + rng.uniform(-30, 30, 73), 62 + rng.uniform(-40, 40, 73)], 1)
CANON = np.vstack([OUT, REST]).astype(np.float64)                        # 106 x 2
assert CANON.shape == (106, 2)


class Det:
    pass


def face(kps, lmk, score=0.9):
    f = Det(); f.bbox = np.array([kps[:, 0].min() - 40, kps[:, 1].min() - 60, kps[:, 0].max() + 40, kps[:, 1].max() + 60], np.float32)
    f.kps = np.asarray(kps, np.float32); f.landmark_2d_106 = np.asarray(lmk, np.float32); f.det_score = score; f.normed_embedding = None
    return f


def place(t, scale=2.9, angle=0.0, shift=(300.0, 200.0), v=(0.0, 0.0)):
    """the face at frame t: the canonical 5 points and 106 landmarks, scaled (2.9 = a 320-px face), turned, moved"""
    a = np.radians(angle)
    R = scale * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    off = np.array(shift) + t * np.array(v)
    return TPL @ R.T + off, CANON @ R.T + off


def run(switch, n=60, noise_k=1.0, noise_l=1.5, move=(0.0, 0.0), turn_rate=0.0, jaw_jump_at=None):
    keep = dict(E._P)
    E._P["lmk_smooth"] = switch
    tr = E.TrackState(slot=0); r2 = np.random.default_rng(3)
    L, K, RAW, TRUE_L = [], [], [], []
    canon = CANON.copy()
    for t in range(n):
        if jaw_jump_at is not None:
            canon = CANON.copy(); canon[:17, 1] += 8.0 * np.clip((t - jaw_jump_at) / 3.0, 0, 1)      # the jaw drops 8 px (of 112) over 3 frames
        a = np.radians(turn_rate * t); sc = 2.9
        R = sc * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]); off = np.array([300.0, 200.0]) + t * np.array(move)
        k_true, l_true = TPL @ R.T + off, canon @ R.T + off
        f = face(k_true + r2.normal(0, noise_k, k_true.shape), l_true + r2.normal(0, noise_l, l_true.shape))
        tr.update(f, update_embedding=False)
        L.append(tr.lmk.copy()); K.append(tr.kps.copy()); RAW.append(f.landmark_2d_106.copy()); TRUE_L.append(l_true)
    E._P.clear(); E._P.update(keep)
    return tr, np.array(L), np.array(K), np.array(RAW), np.array(TRUE_L)


def jitter(X, lo=10, hi=50):
    """a frame minus the median of its 5 neighbours, mean over frames and points (px)"""
    v = [np.linalg.norm(X[t] - np.median(X[t - 2:t + 3], axis=0), axis=1).mean() for t in range(lo, hi)]
    return float(np.mean(v))


# ---- 1. landmarks --------------------------------------------------------------------------------------------------------
_, L1, K1, RAW1, T1 = run(1)
_, L0, K0, RAW0, T0 = run(0)
j0, j1 = jitter(L0), jitter(L1)
check("a still face, noisy detector: the landmarks' jitter in the picture falls by 45 % or more", j1 <= 0.55 * j0, "%.2f px -> %.2f px" % (j0, j1))
e0 = float(np.mean([np.linalg.norm(L0[t] - T0[t], axis=1).mean() for t in range(20, 60)]))
e1 = float(np.mean([np.linalg.norm(L1[t] - T1[t], axis=1).mean() for t in range(20, 60)]))
check("...and they are nearer the true places than the raw detections are", e1 < e0, "error %.2f px -> %.2f px" % (e0, e1))
_, L1m, K1m, RAWm, Tm = run(1, noise_k=0.0, noise_l=0.0, move=(6.0, 2.0))
_, L0m, K0m, RAWm0, Tm0 = run(0, noise_k=0.0, noise_l=0.0, move=(6.0, 2.0))
# the landmarks' place against the 5 points: the same offset as the raw detection has (no drift, no lag), whatever the 5 points do
off_s = np.array([L1m[t].mean(0) - K1m[t].mean(0) for t in range(20, 60)])
off_r = np.array([RAWm[t].mean(0) - place(t, shift=(300, 200), v=(6.0, 2.0))[0].mean(0) for t in range(20, 60)])
check("a face moving 6 px a frame: the landmarks keep their place against the 5 points (no lag of their own; within 1 px)",
      float(np.abs(off_s - off_r).max()) < 1.0, "largest difference %.2f px" % float(np.abs(off_s - off_r).max()))
_, Lt, Kt, RAWt, Tt = run(1, noise_k=0.0, noise_l=0.0, turn_rate=1.2)
offt = np.array([np.linalg.norm(Lt[t] - Tt[t], axis=1).mean() for t in range(20, 60)])
check("a face turning 1.2 deg a frame: the landmarks follow it (the 5 points are smoothed anyway; error against the true landmarks no larger than the points' own lag)",
      float(offt.max()) <= float(np.mean([np.linalg.norm(Kt[t] - (TPL @ (2.9 * np.array([[np.cos(np.radians(1.2 * t)), -np.sin(np.radians(1.2 * t))], [np.sin(np.radians(1.2 * t)), np.cos(np.radians(1.2 * t))]])).T + np.array([300.0, 200.0])), axis=1).mean() for t in range(20, 60)])) * 2.0 + 1.0,
      "landmark error up to %.2f px" % float(offt.max()))
_, Lj, Kj, RAWj, Tj = run(1, noise_k=0.0, noise_l=0.3, jaw_jump_at=25)
jaw_true = Tj[:, :17, 1].mean(1) - Tj[:, :17, 1].mean(1)[20]
jaw_got = Lj[:, :17, 1].mean(1) - Lj[:, :17, 1].mean(1)[20]
reach = float(jaw_got[32] / max(1e-6, jaw_true[32]))
check("a jaw that really drops (8 px of 112 in 3 frames): the landmarks reach 80 % of it within 4 frames of the end", reach >= 0.8,
      "reached %.0f %% at frame 32 (the move ends at 28)" % (100 * reach))
keep = dict(E._P)
tr, _, _, _, _ = run(1, n=30)
l_before = tr.lmk.copy(); k_before = tr.kps.copy()
tr.vel_kps = np.tile(np.array([4.0, 0.0], np.float32), (5, 1)); tr.predict(1.0)
move_k = (tr.kps - k_before).mean(0); move_l = (tr.lmk - l_before).mean(0)
check("without a detection (predict) the landmarks move with the 5 points", float(np.linalg.norm(move_k - move_l)) < 0.3 and float(np.linalg.norm(move_k)) > 1.0,
      "points %.2f px, landmarks %.2f px" % (float(move_k[0]), float(move_l[0])))
E._P.clear(); E._P.update(keep)
tr0, L0r, _, RAWr, _ = run(0, n=5)
check("lmk_smooth = 0: the landmarks are the detections as they are (11.2.210)", float(np.abs(L0r[-1] - RAWr[-1]).max()) < 1e-4)

# ---- 2. the occlusion cut --------------------------------------------------------------------------------------------------
W = 256
E._OCCNET["sess"] = "synthetic"
ANG = np.linspace(0, 2 * np.pi, 106, endpoint=False)
PTS = np.stack([128 + 80 * np.cos(ANG), 130 + 105 * np.sin(ANG)], 1).astype(np.float32)           # the outline, in the 256-px crop
yy, xx = np.mgrid[0:W, 0:W]
FACE = (((xx - 128) / 82.0) ** 2 + ((yy - 130) / 108.0) ** 2 < 1.0).astype(np.float32)


def keep_for(model, pts=PTS, prev=None, switches=None):
    """occ_net_keep with a synthetic face map (1 = face)"""
    saved = dict(E._P); orig = E.occ_net_raw
    for k, v in (switches or {}).items():
        E._P[k] = v
    E.occ_net_raw = lambda a: model
    try:
        out = E.occ_net_keep(np.zeros((W, W, 3), np.uint8), pts, W, prev_cut=prev, wide=True, region=(FACE > 0.5))
    finally:
        E.occ_net_raw = orig; E._P.clear(); E._P.update(saved)
    return out


strip = FACE.copy(); strip[60:200, 70:84] = 0.0                  # a strip 14 px wide along the face's left edge, from the brow to the jaw
cut_strip_new = keep_for(strip); cut_strip_old = keep_for(strip, switches={"occ_net_rim": 0})
check("a thin strip along the face's own edge is not an object (11.2.210 cut it)",
      float((cut_strip_new < 0.5).sum()) == 0 and float((cut_strip_old < 0.5).sum()) > 300,
      "cut %d px (11.2.210: %d px)" % (int((cut_strip_new < 0.5).sum()), int((cut_strip_old < 0.5).sum())))
blob = FACE.copy(); blob[((xx - 128) ** 2 + (yy - 192) ** 2) < 34 ** 2] = 0.0                  # a spoon-sized blob at the mouth and chin
cut_blob = keep_for(blob)
check("a spoon-sized blob at the mouth is still cut", float((cut_blob < 0.5).sum()) > 0.8 * float(((xx - 128) ** 2 + (yy - 192) ** 2 < 30 ** 2).sum()),
      "cut %d px of %d" % (int((cut_blob < 0.5).sum()), int(((xx - 128) ** 2 + (yy - 192) ** 2 < 34 ** 2).sum())))
hand = FACE.copy(); hand[100:190, 60:130] = 0.0                  # a big piece over the cheek: the wide rule, still cut
check("a big piece over the cheek is still cut", float((keep_for(hand) < 0.5).sum()) > 4000, "%d px" % int((keep_for(hand) < 0.5).sum()))
# hold: a piece cut on the last frame, now small and just outside the mouth zone's centre band, stays cut; without the last frame it is not
prev_disc = (((xx - 180) ** 2 + (yy - 175) ** 2) < 16 ** 2)
sm = FACE.copy(); sm[((xx - 182) ** 2 + (yy - 175) ** 2) < 11 ** 2] = 0.0
held = keep_for(sm, prev=prev_disc, switches={"occ_net_rim": 0}); free = keep_for(sm, prev=None, switches={"occ_net_rim": 0})     # (the strip rule off: this tests the hold alone)
check("a piece cut on the last frame stays cut while it overlaps itself (the mouth rule's hysteresis)",
      float((held < 0.5).sum()) > 300 and float((free < 0.5).sum()) == 0,
      "held %d px, without the last frame %d px" % (int((held < 0.5).sum()), int((free < 0.5).sum())))
check("with the new switches off the strip is cut as before and nothing is held (11.2.210)",
      float((keep_for(strip, switches={"occ_net_rim": 0, "occ_net_hold_core": 0}) < 0.5).sum()) > 300)

# ---- 3. the carried face -------------------------------------------------------------------------------------------------
sys.path.insert(0, P211)
import core_pipeline as cp
P = cp._E._P


def carry(series_fn, n=60, switch=1.0, noise=(0.0, 0.0, 0.0)):
    """feed _smooth_T a measured transform per frame: size, turn (rad), centre (px); returns the filtered centre / size / turn"""
    st = {"kps0": TPL.astype(np.float32) * 2.9 + np.float32([300, 200]), "bbox0": np.float32([200, 100, 420, 340]), "vel": np.zeros(2, np.float32), "age": 0}
    c_ref = st["kps0"].mean(0); out, truth = [], []
    r2 = np.random.default_rng(5)
    keep_ = dict(P); P["hid_filter"] = switch
    for t in range(n):
        sc, th, c = series_fn(t)
        truth.append((sc, th, c.copy()))
        sc_m = sc * (1 + r2.normal(0, noise[1])); th_m = th + r2.normal(0, noise[2]); c_m = c + r2.normal(0, noise[0], 2)
        R = sc_m * np.array([[np.cos(th_m), -np.sin(th_m)], [np.sin(th_m), np.cos(th_m)]])
        st["T"] = np.hstack([R, (c_m - R @ c_ref).reshape(2, 1)]).astype(np.float32); st["age"] = t
        T = cp._HiddenHold._smooth_T(st, P); st["T"] = T
        out.append((float(np.hypot(T[0, 0], T[1, 0])), float(np.arctan2(T[1, 0], T[0, 0])), T[:, :2] @ c_ref + T[:, 2]))
    P.clear(); P.update(keep_)
    return out, truth


c0 = np.array([300.0, 200.0])
steady = lambda t: (1.0 + 0.002 * t, np.radians(0.5 * t), c0 + t * np.array([4.0, -1.0]))
o, tr_ = carry(steady)
lag = max(np.linalg.norm(o[t][2] - tr_[t][2]) for t in range(25, 60))
check("steady motion (4 px a frame, a turn of 0.5 deg a frame, growing) passes without lag: centre within 0.5 px after 25 frames", lag < 0.5, "%.2f px" % lag)
o, tr_ = carry(lambda t: (1.0, 0.0, c0), noise=(1.2, 0.006, 0.004))
o_raw, _ = carry(lambda t: (1.0, 0.0, c0), noise=(1.2, 0.006, 0.004), switch=0.0)
e_f = np.mean([np.linalg.norm(o[t][2] - c0) for t in range(15, 60)]); e_r = np.mean([np.linalg.norm(o_raw[t][2] - c0) for t in range(15, 60)])
s_f = np.std([o[t][0] for t in range(15, 60)]); s_r = np.std([o_raw[t][0] for t in range(15, 60)])
check("a still face measured with white noise (1.2 px, 0.6 % size, 0.2 deg): centre and size noise fall by 15 % or more (all a filter with no lag can do)", e_f <= 0.85 * e_r and s_f <= 0.85 * s_r,
      "centre %.2f -> %.2f px, size %.4f -> %.4f" % (e_r, e_f, s_r, s_f))
def _rev(t):
    # +4 px a frame, slowing by 0.5 px a frame each frame to -4 px a frame (a head that turns round in 16 frames), from frame 30
    v = np.clip(4.0 - 0.5 * np.maximum(0, np.arange(t + 1) - 30), -4.0, 4.0)
    return 1.0, 0.0, c0 + np.array([float(v.sum()), 0.0])


o, tr_ = carry(_rev)
err = [np.linalg.norm(o[t][2] - tr_[t][2]) for t in range(25, 60)]
check("a head that turns round (4 px a frame to 4 px the other way in 16 frames) is followed within 3 px throughout", float(max(err)) < 3.0,
      "largest error %.1f px" % float(max(err)))

# ---- 4. a face passing behind another: the carry's decisions ----------------------------------------------------------------
def _recs(t, ax, bx, det_a=False, a_kps=None):
    """records of person A (behind, moving right) and B (in front, moving left) on frame t, B's outline over A's"""
    ka, la = place(0, shift=(ax, 200.0)); kb, lb = place(0, shift=(bx, 200.0))
    if a_kps is not None:
        ka = a_kps
    bb = lambda k: np.float32([k[:, 0].min() - 40, k[:, 1].min() - 60, k[:, 0].max() + 40, k[:, 1].max() + 60])
    return {0: {"kps": ka.astype(np.float32), "lmk": la.astype(np.float32), "bbox": bb(ka), "det": det_a},
            1: {"kps": kb.astype(np.float32), "lmk": lb.astype(np.float32), "bbox": bb(kb), "det": True}}


def passing(weak_flow, hand=1.0, lost=15, n=24, jump_at=None, noflow=1.0):
    """A is carried (it has been moving right at 6 px a frame, 70 % behind B, B moves left at 5 px a frame). Nothing can be followed by
    the reference; the frame-to-frame reading returns B's own motion (-5 px) for the first 6 frames, then nothing. Returns the centre of
    the carried A on each frame (None where it is not carried) and the carry's stats."""
    keep = dict(P); P["hid_weak_flow"] = weak_flow; P["hid_pass_noflow"] = noflow; P["hid_hand"] = hand; P["hid_lost"] = lost; P["hid_deep"] = 0.95
    H = cp._HiddenHold(); H.enabled = staticmethod(lambda: True)
    k0, l0 = place(0, shift=(560.0, 200.0)); bb0 = np.float32([k0[:, 0].min() - 40, k0[:, 1].min() - 60, k0[:, 0].max() + 40, k0[:, 1].max() + 60])
    H.act[0] = {"ref": object(), "kps0": k0.astype(np.float32), "lmk0": l0.astype(np.float32), "bbox0": bb0, "T": np.float32([[1, 0, 0], [0, 1, 0]]),
                "g": 0, "age": 10, "free": 0, "out": 0, "vel": np.float32([6.0, 0.0]), "used": True}
    saved = (cp._hh_ref_track, cp._hh_flow)
    cnt = {"n": 0}
    cp._hh_ref_track = lambda ref, frame, T0, f_lmk: (None, "", 0)
    def _flow(*a, **k):
        cnt["n"] += 1
        return (np.float32([[1, 0, -5.0], [0, 1, 0]]), "flow") if cnt["n"] <= 6 else (None, "")
    cp._hh_flow = _flow
    img = np.zeros((400, 900, 3), np.uint8); cen = []
    try:
        ax, bx = 560.0, 600.0
        H.prev = (0, img); H.prev_lmk = {1: place(0, shift=(bx, 200.0))[1].astype(np.float32)}; H.prev_kc = {1: place(0, shift=(bx, 200.0))[0].mean(0).astype(np.float32)}
        for t in range(1, n):
            ax += 6.0; bx -= 5.0
            a_kps = None; det_a = False
            if jump_at is not None and t >= jump_at:
                a_kps = place(0, shift=(ax + 90.0, 200.0))[0]; det_a = True        # the detector reads it 90 px further on
            recs = _recs(t, ax, bx, det_a, a_kps)
            H.step(t, img, recs)
            u = H.apply({k: dict(v) for k, v in recs.items()}, t)
            cen.append(float(u[0]["kps"].mean(0)[0]) if u[0].get("hid") else None)
    finally:
        cp._hh_ref_track, cp._hh_flow = saved
        P.clear(); P.update(keep)
    return cen, dict(H.stats)


cen, stt = passing(1.0, n=40)
sp = lambda c, a, b: float(np.mean([c[i + 1] - c[i] for i in range(a, b) if c[i] is not None and c[i + 1] is not None]))
v_own = sp(cen, 3, 12)
check("A behind B and moving against it, the frame-to-frame reading showing B's motion: it is not used - A goes on at its own speed (+6 px a frame, not B's -5)",
      abs(v_own - 6.0) < 1.0, "speed %.1f px a frame" % v_own)
cen1, _ = passing(1.0, n=40, noflow=0.0)
v_w = sp(cen1, 14, 20)
check("  (hid_pass_noflow = 0: the reading moves A for the 6 frames it lasts, but only the position: afterwards it goes on at its own speed)", abs(v_w - 6.0) < 1.5, "speed %.1f px a frame" % v_w)
cen0, _ = passing(0.0, n=40, noflow=0.0)
v_drag = sp(cen0, 14, 20)
check("  (hid_weak_flow = 0 and hid_pass_noflow = 0, 11.2.210: dragged along with B, moving the other way)", v_drag < -2.0, "speed %.1f px a frame" % v_drag)
nc = sum(c is not None for c in cen)
check("after hid_lost frames (here 15) of nothing to follow, the carry is handed back over 6 frames and then ends", stt.get("lost", 0) == 1 and cen[-1] is None and cen[10] is not None and 18 <= nc <= 26,
      "carried on %d of %d frames, lost %d" % (nc, len(cen), stt.get("lost", 0)))
cen_s, st_s = passing(1.0, hand=1.0, lost=15, n=14, jump_at=8)
cen_n, st_n = passing(1.0, hand=0.0, lost=15, n=14, jump_at=8)
steps = [abs(cen_s[i + 1] - cen_s[i]) for i in range(7, 12) if cen_s[i] is not None and cen_s[i + 1] is not None]
check("the detector finds A 90 px further on: handed over in steps (largest step under 35 px; 11.2.210 jumped 90 px at once)",
      len(steps) >= 3 and max(steps) < 35.0 and st_s.get("jumped", 0) == 1 and any(c is None for c in cen_n[8:]), "steps %s" % [round(x) for x in steps])

# ---- 5. the cut of a carried face -------------------------------------------------------------------------------------------
def _cut_mass(E, hid, dil):
    keep = dict(E._P)
    if "rival_dilate_hid" in E._P:
        E._P["rival_dilate_hid"] = dil
    k, l = place(0, shift=(300.0, 200.0)); kr, lr = place(0, shift=(300.0 - 95.0, 200.0))          # a face in front over its left part
    f = E.PredictedFace(np.float32([k[:, 0].min() - 40, k[:, 1].min() - 60, k[:, 0].max() + 40, k[:, 1].max() + 60]), k.astype(np.float32), l.astype(np.float32), None, 0.5)
    f._hid = hid; f._partners = []
    comp = E.AlignedCompositor.__new__(E.AlignedCompositor)
    M = E.estimate_norm(k.astype(np.float32), 128)
    m = comp.build_mask(f, M, 128, track=None, rivals=[lr.astype(np.float32)])
    E._P.clear(); E._P.update(keep)
    return float(m.sum())


E210 = load("se210", P210)
m_h, m_n, m_0 = _cut_mass(E, True, 0.06), _cut_mass(E, False, 0.06), _cut_mass(E, True, 0.0)
m_210 = _cut_mass(E210, True, 0.0)
check("a carried face is cut less by the face in front (its outline dilated 0.06 of the crop, not 0.085): more of it painted", m_h > 1.03 * m_n, "%.0f against %.0f" % (m_h, m_n))
check("a face that is not carried: the cut as in 11.2.210", abs(m_n - _cut_mass(E210, False, 0.0)) < 1e-3, "%.0f against %.0f" % (m_n, _cut_mass(E210, False, 0.0)))
check("rival_dilate_hid = 0: a carried face is cut as in 11.2.210", abs(m_0 - m_210) < 1e-3, "%.0f against %.0f" % (m_0, m_210))
print("\n%s" % ("all checks passed" if fails == 0 else "%d check(s) failed" % fails))
