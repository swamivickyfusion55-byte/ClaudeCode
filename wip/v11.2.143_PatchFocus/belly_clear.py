"""v11.2.110 (SDOS-110) Lower-belly clear skin. v11.2.112 (SDOS-112): the pose model on MediaPipe 0.10.30+ (Tasks API).
v11.2.140 (SDOS-140) BellyTattoo: the band is the whole belly - from just under the chest (0.56 of the way up from the hip
joints to the shoulders) down to 0.10 above the hip line, which keeps it above the bikini line - and the navel is recognised
wherever it sits on the body's middle line, not only at the band's top. Tattoos and marks only.

Opt-in (job setting ``belly_clear``). On the uncovered skin from just below the navel down to the waist line (the hip line),
every tattoo, mole or other mark is removed and the skin is evened out:

1. **Where.** MediaPipe Pose (already a Space dependency) gives the shoulders and hips on sampled frames. The band is taken
   between the hip line and a line ``t_top`` of the way up from the hips to the shoulders (the navel sits about 0.27 of the way
   up; the band starts a little below it), widened sideways beyond the hip joints, which lie inside the body's outline.
2. **Which pixels.** Only skin: a colour model of this person's skin is fitted on the band itself. A mark is (a) anything that is
   not skin but is surrounded by skin (ink of any colour: tattoos), or (b) skin-coloured but clearly darker than the skin around
   it (moles, pigment). Clothing (the waistband, shorts) is not surrounded by skin and is left alone; so is the navel.
3. **How.** On a few key frames the marks found become positions of one mark for the existing Patch Removal engine: filled once
   (LaMa), then carried with the body and relit on every frame, exactly like a mark the user painted by hand.
4. **Even skin.** On every frame the band's skin (not clothing, not the navel's surroundings outside the band) gets a gentle
   edge-preserving smoothing (``smooth``), feathered at the band's edges.

It never paints outside the band, never on non-skin that touches the band's edge (clothing), and does nothing on frames where
the body is not found. Text-only diagnostics go to the job log (BELLY-CLEAR lines)."""
from __future__ import annotations

import math
import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

CFG = {
    "every": 3,              # pose on every 3rd frame (interpolated in between)
    "pose_max_w": 640,       # the pose model sees the frame at most this wide
    "vis_min": 0.5,          # landmark visibility needed for shoulders and hips
    "t_top": 0.56,           # v11.2.140: the whole belly - up to just under the chest (the underbust is ~0.62)
    "t_bot": 0.10,           # v11.2.140: ...down to 0.10 above the hip-joint line: above the bikini line, never below it
    "navel_t": (0.18, 0.46), # v11.2.140: the navel is looked for this far up the body (0.27-0.36 on most bodies) ...
    "navel_x": 0.15,         #   ...within this share of the hip width of the body's middle line
    "ink_dl": 8.0,           # v11.2.140: a non-skin piece is ink when this many levels darker than the skin ring around it ...
    "ink_m": 6.0,            #   ...or this far from the skin's colour (Mahalanobis; skin itself is under 3.5)
    "rescue_t": 0.15,        # v11.2.140: thin strokes running into cloth are rescued only this share of the torso above the floor ...
    "rescue_side": 0.15,     #   ...and this share of the band's width away from its sides (not the waist's outline)
    "hug_max": 0.6,          # v11.2.140: a thin piece with more than this share within 5 px of the cloth's body is cloth, not ink
    "floor_px": 4,           # v11.2.140: marks stop this far above the floor, so the fill's 3 px halo stays above it too
    "edge_k": 0.025,         # v11.2.140: a darker-skin mark lies at least this share of the band's size inside the skin (not the outline)
    "side_ext": 0.30,        # widened sideways by this share of the hip width on each side
    "keys_max": 8,           # key frames on which marks are looked for (positions of one Patch Removal mark)
    "key_gap_s": 1.5,        # at least this far apart
    "smooth": 0.5,           # even-skin strength (0 = off, 1 = full edge-preserving smoothing)
    "mark_dilate": 4,        # px added around every mark found (soft ink edges)
    "patch_dl": 4.0,         # v11.2.143: inside the pentagon, the dark patch: pixels this many grey levels darker than the skin around
    "patch_k": 3.0,          #   the pentagon (a smooth fit of it), and this many noise sigmas of that skin - a faint patch, soft edges
    "patch_ring": 0.08,      #   the skin around: a ring this share of the hip width outside the pentagon (at least 10 px)
    "patch_min_px": 20,      #   a dark piece smaller than this (and not touching a mark) is skin texture, not the patch
    "patch_grow": 2,         #   px added around the dark patch (its last, faintest edge)
    "patch_navel_px": 4,     #   ...and never within this many px of the navel
    "patch_open": 2,         #   a patch is a blob: thinner dark pieces are taken away (opening, disc of this radius) - the soft shadow
                             #   ring round the navel, a crease, a hair (measured: a 2-3 px arc beside the navel pulled the pentagon over it)
    "dark_k": 3.0,           # a skin-coloured pixel is a mark when it is this many noise sigmas darker than the skin around it
    "dark_min": 12.0,        #   ...and at least this many grey levels darker
    "min_region_px": 1500,   # a band smaller than this (far away, or mostly covered) is skipped
    "max_mark_share": 0.35,  # a non-skin piece larger than this share of the band is clothing, not a mark
    "enclosed_min": 0.70,    # a non-skin piece is a mark when this share of its rim (inside the band) is skin
    "hold_s": 0.7,           # landmarks are carried this long over frames where the body is not found
    "cleanup": True,         # v11.2.112: on every frame, a mark still showing in the band after the carried fill is filled there
    "cleanup_min_px": 30,    #   ...when at least this many px of marks are found
    "hip_torso_min": 0.20,   # v11.2.112: hip width under this share of the torso's length = a side view, or a pose guessed with the head
                             #   out of the frame (MediaPipe finds the person by the face; its shoulders / hips are then squeezed together)
}

L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24
STATS = {"frames": 0, "cleaned": 0, "cleaned_px": 0}   # frames the even-skin pass changed; frames / px the per-frame clean-up filled


# v11.2.112: the pose model of MediaPipe's Tasks API (MediaPipe 0.10.30 and later no longer have the legacy ``mp.solutions``; the
# Space resolves ``mediapipe<0.11`` to 0.10.35). BlazePose "full" (the same network as the legacy model_complexity=1), Apache-2.0,
# 9.4 MB, downloaded once into the first writable model folder (or taken from the Space's own folder when uploaded there).
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
POSE_MODEL_SHA256 = "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1"
POSE_MODEL_BYTES = 9398198
_LIB_HINT = " (the Space needs the system packages libgles2 and libegl1: add them to packages.txt)"


def _model_dirs():
    import os
    return [x for x in (os.environ.get("PHOENIX_MODEL_DIR", ""), "/data/phoenix_models", os.path.expanduser("~/.phoenix_models"),
                        "/tmp/phoenix_models") if x]


def _file_ok(fp, want, nbytes):
    import os, hashlib
    try:
        if nbytes and os.path.getsize(fp) != nbytes:
            return False
        h = hashlib.sha256()
        with open(fp, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return (not want) or h.hexdigest() == want
    except Exception:
        return False


def pose_model_path():
    """The pose model file: the Space's own folder (uploaded by hand), else the model folders, else downloaded once. -> path."""
    import os, shutil, urllib.request
    name = os.path.basename(POSE_MODEL_URL)
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if os.path.isfile(here) and os.path.getsize(here) > 1_000_000:
        return here
    for d in _model_dirs():
        fp = os.path.join(d, name)
        if os.path.isfile(fp) and _file_ok(fp, POSE_MODEL_SHA256, POSE_MODEL_BYTES):
            return fp
    path = None
    for d in _model_dirs():
        try:
            os.makedirs(d, exist_ok=True)
            if os.access(d, os.W_OK):
                path = os.path.join(d, name)
                break
        except Exception:
            continue
    if path is None:
        raise RuntimeError("no writable model folder for the pose model")
    url = os.environ.get("PHOENIX_POSE_URL", "") or POSE_MODEL_URL
    tmp = path + ".part%d" % os.getpid()
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        if not _file_ok(tmp, POSE_MODEL_SHA256 if url == POSE_MODEL_URL else "", POSE_MODEL_BYTES if url == POSE_MODEL_URL else 0):
            raise RuntimeError("the downloaded pose model failed its size / sha256 check")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
    return path


def canonical(lm):
    """v11.2.112: the same side order on every frame. MediaPipe can take a front view for a back view from one frame to the next
    (left and right swapped); the band does not care, but a blend of two swapped samples between them squeezes the band to nothing.
    Order: across the body from 11/23 to 12/24 runs clockwise of the torso's up direction (a front view, in picture coordinates)."""
    if lm is None:
        return None
    try:
        p = {k: np.array(lm[k][:2], np.float64) for k in (L_SH, R_SH, L_HIP, R_HIP)}
    except Exception:
        return lm
    u = 0.5 * (p[L_SH] + p[R_SH]) - 0.5 * (p[L_HIP] + p[R_HIP])
    d = (p[R_SH] - p[L_SH]) + (p[R_HIP] - p[L_HIP])
    if float(u[0] * d[1] - u[1] * d[0]) > 0.0:
        return {L_SH: lm[R_SH], R_SH: lm[L_SH], L_HIP: lm[R_HIP], R_HIP: lm[L_HIP]}
    return lm


def clean_samples(samples, window=4, max_dev=0.5):
    """v11.2.112: drop a sampled pose that jumps away from the poses around it (more than max_dev of the hip width from the median of
    up to ``window`` samples on each side) - MediaPipe, unsure (the head hidden, a hand over the body), can put a body somewhere else
    for one frame, with full confidence. -> (kept samples, number dropped)."""
    keys = sorted(samples)
    pts = {k: np.array([samples[k][j][:2] for j in (L_SH, R_SH, L_HIP, R_HIP)], np.float64) for k in keys}
    out, dropped = {}, 0
    for n, k in enumerate(keys):
        nb = [pts[keys[m]] for m in range(max(0, n - window), min(len(keys), n + window + 1)) if m != n]
        if len(nb) < 2:
            out[k] = samples[k]
            continue
        med = np.median(np.array(nb), axis=0)
        hw = max(8.0, float(np.median([np.linalg.norm(q[2] - q[3]) for q in nb])))
        if float(np.abs(pts[k] - med).max()) > max_dev * hw:
            dropped += 1
            continue
        out[k] = samples[k]
    return out, dropped


class MediaPipePose:
    """Lazy MediaPipe Pose. detect(bgr, t_ms=None) -> {11, 12, 23, 24: (x, y, visibility)} in pixels, or None.
    v11.2.112: the legacy ``mp.solutions.pose`` where the installed MediaPipe still has it, else the Tasks API's PoseLandmarker
    (MediaPipe 0.10.30+, e.g. the Space's 0.10.35). ``backend`` says which; ``error`` why none could be loaded."""

    def __init__(self, still=False):
        self._pose = None
        self._mp = None
        self.error = None
        self.backend = None
        self.still = bool(still)                             # (one picture: the preview)
        self._ts = -1

    def _get(self):
        if self._pose is None and self.error is None:
            try:
                import mediapipe as mp
                self._mp = mp
                ver = str(getattr(mp, "__version__", "?"))
                sol = getattr(mp, "solutions", None)
                if sol is not None and hasattr(sol, "pose"):
                    self._pose = sol.pose.Pose(static_image_mode=self.still, model_complexity=1, enable_segmentation=False,
                                               min_detection_confidence=0.5, min_tracking_confidence=0.5)
                    self.backend = "MediaPipe %s (solutions)" % ver
                else:
                    from mediapipe.tasks.python import vision as _v
                    from mediapipe.tasks.python.core.base_options import BaseOptions as _BO
                    mpath = pose_model_path()
                    try:
                        opts = _v.PoseLandmarkerOptions(base_options=_BO(model_asset_path=mpath),
                                                        running_mode=_v.RunningMode.IMAGE if self.still else _v.RunningMode.VIDEO,
                                                        num_poses=1, min_pose_detection_confidence=0.5,
                                                        min_pose_presence_confidence=0.5, min_tracking_confidence=0.5,
                                                        output_segmentation_masks=False)
                        self._pose = _v.PoseLandmarker.create_from_options(opts)
                    except OSError as e:
                        if "GLES" in str(e) or "EGL" in str(e):
                            raise OSError(str(e) + _LIB_HINT)
                        raise
                    self.backend = "MediaPipe %s (tasks, %s)" % (ver, __import__("os").path.basename(mpath))
                log.info("BELLY-CLEAR pose model: %s%s", self.backend, " - still image" if self.still else "")
            except Exception as e:                 # (the feature is then off for this job; the job itself continues)
                self.error = "%s: %s" % (type(e).__name__, e)
                self._pose = None
        return self._pose

    def detect(self, bgr, t_ms=None):
        pose = self._get()
        if pose is None:
            return None
        h, w = bgr.shape[:2]
        s = min(1.0, float(CFG["pose_max_w"]) / float(max(1, w)))
        small = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1.0 else bgr
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        if hasattr(pose, "process"):                         # legacy solutions
            r = pose.process(rgb)
            if not r.pose_landmarks:
                return None
            lm = r.pose_landmarks.landmark
        else:                                                # tasks
            mi = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            if self.still:
                r = pose.detect(mi)
            else:                                            # (video mode: timestamps must rise)
                t = int(t_ms) if t_ms is not None else self._ts + 33
                t = max(t, self._ts + 1)
                self._ts = t
                r = pose.detect_for_video(mi, t)
            if not r.pose_landmarks:
                return None
            lm = r.pose_landmarks[0]

        def _vis(p):
            v = getattr(p, "visibility", None)
            if v is None:
                v = getattr(p, "presence", None)
            return 1.0 if v is None else float(v)
        return canonical({k: (float(lm[k].x) * w, float(lm[k].y) * h, _vis(lm[k])) for k in (L_SH, R_SH, L_HIP, R_HIP)})

    def close(self):
        try:
            if self._pose is not None:
                self._pose.close()
        except Exception:
            pass
        self._pose = None


LENGTH_RANGE = (0.5, 1.5)     # v11.2.111: the band's length and breadth can be scaled by the user (1 = as found)
WIDTH_RANGE = (0.5, 1.5)
T_TOP_MAX, T_BOT_MIN = 0.62, 0.10       # v11.2.140: never onto the chest (underbust ~0.62), never below 0.10 above the hip line
                                        #   (Length cannot take it there either)


def clamp_scale(v, rng):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(v):
        return 1.0
    return float(min(rng[1], max(rng[0], v)))


_STILL = {"cls": None, "pose": None}


def still_pose():
    """v11.2.111: one still-image pose model kept for the previews (loading it costs seconds; previews run one at a time)."""
    if _STILL["pose"] is None or _STILL["cls"] is not MediaPipePose:
        _STILL["pose"], _STILL["cls"] = MediaPipePose(still=True), MediaPipePose
    return _STILL["pose"]


def region_polygon(lm, length=1.0, width=1.0):
    """The band (4 points, float) from shoulder / hip landmarks, or None when they are not reliable.
    length / width (v11.2.111): scale the band's height (about its middle) and breadth (about the body's middle line); the band
    never reaches above T_TOP_MAX or below T_BOT_MIN of the way from the hips to the shoulders."""
    if lm is None:
        return None
    try:
        if min(lm[k][2] for k in (L_SH, R_SH, L_HIP, R_HIP)) < CFG["vis_min"]:
            return None
        ls, rs = np.array(lm[L_SH][:2]), np.array(lm[R_SH][:2])
        lh, rh = np.array(lm[L_HIP][:2]), np.array(lm[R_HIP][:2])
    except Exception:
        return None
    hip_w = float(np.linalg.norm(lh - rh))
    torso = float(np.linalg.norm(0.5 * (ls + rs) - 0.5 * (lh + rh)))
    if hip_w < 8 or torso < 16 or hip_w > 3.0 * torso:      # (a side view or a broken pose: no band)
        return None
    if hip_w < float(CFG["hip_torso_min"]) * torso:        # (v11.2.112: side view, or the head out of the frame)
        return None
    length = clamp_scale(length, LENGTH_RANGE)
    width = clamp_scale(width, WIDTH_RANGE)
    e = float(CFG["side_ext"]) * hip_w
    P = lambda h_, s_, t: h_ + t * (s_ - h_)
    tc = 0.5 * (float(CFG["t_top"]) + float(CFG["t_bot"]))
    hh = 0.5 * (float(CFG["t_top"]) - float(CFG["t_bot"])) * length
    t_top, t_bot = min(T_TOP_MAX, tc + hh), max(T_BOT_MIN, tc - hh)
    pts = []
    for t in (t_top, t_bot):
        pl, pr = P(lh, ls, t), P(rh, rs, t)
        d = pl - pr
        nd = d / max(1e-6, float(np.linalg.norm(d)))
        m = 0.5 * (pl + pr)
        half = (0.5 * float(np.linalg.norm(d)) + e) * width
        pts.append((m + nd * half, m - nd * half))
    (tl, tr), (bl, br) = pts
    return np.array([tl, tr, br, bl], np.float32)


def region_mask(poly, shape):
    m = np.zeros(shape[:2], np.uint8)
    if poly is not None:
        cv2.fillConvexPoly(m, np.round(poly).astype(np.int32), 255)
    return m > 0


def _lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


def skin_model(img, band):
    """This person's skin colour, fitted on the band: (mean ab, inverse covariance ab, median L, L spread), or None."""
    if band.sum() < 200:
        return None
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    # v11.2.112: Cr from 133 (the usual skin range starts there) - from 128, neutral black / grey clothing (Cr ~128) at the band's
    # bottom joined the skin model, and the top of black shorts was then taken for dark marks on skin
    broad = band & (cr >= 133) & (cr <= 185) & (cb >= 70) & (cb <= 140) & (y >= 30)
    if broad.sum() < 150:
        return None
    lab = _lab(img)
    ab = lab[..., 1:][broad]
    med = np.median(ab, axis=0)
    d = np.linalg.norm(ab - med, axis=1)
    core = ab[d < np.percentile(d, 75)]                     # (robust: the 3/4 nearest the median)
    if len(core) < 100:
        return None
    mu = core.mean(axis=0)
    cov = np.cov(core.T) + np.eye(2) * 4.0
    L = lab[..., 0][broad]
    lmed = float(np.median(L))
    lmad = float(np.median(np.abs(L - lmed))) * 1.4826 + 2.0
    return mu, np.linalg.inv(cov), lmed, lmad


def skin_pixels(img, band, model):
    mu, icov, lmed, lmad = model
    lab = _lab(img)
    dab = lab[..., 1:] - mu
    m2 = np.einsum("...i,ij,...j->...", dab, icov, dab)
    L = lab[..., 0]
    return band & (m2 < 3.5 ** 2) & (L > lmed - 6.0 * lmad - 25.0) & (L < lmed + 6.0 * lmad + 30.0)


def _navel_pick(mask, lm):
    """The most navel-like spot of a mask: a compact, filled dark spot on the middle line, navel_t of the way up from the hips.
    -> (its piece of the opened mask, bool, or None; the opening size kk)."""
    hip_w0 = max(1e-3, float(np.linalg.norm(np.array(lm[R_HIP][:2], np.float64) - np.array(lm[L_HIP][:2], np.float64))))
    kk = max(5, int(round(0.035 * hip_w0)) | 1)
    core = cv2.morphologyEx(np.asarray(mask).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
    n2, lab2, st2, cen2 = cv2.connectedComponentsWithStats(core, connectivity=8)
    ls_, rs_ = np.array(lm[L_SH][:2], np.float64), np.array(lm[R_SH][:2], np.float64)
    lh_, rh_ = np.array(lm[L_HIP][:2], np.float64), np.array(lm[R_HIP][:2], np.float64)
    hm, sm = 0.5 * (lh_ + rh_), 0.5 * (ls_ + rs_)
    up = sm - hm
    tl = max(1e-3, float(np.linalg.norm(up)))
    u = up / tl
    across = (rh_ - lh_) - float(np.dot(rh_ - lh_, u)) * u
    across = across / max(1e-3, float(np.linalg.norm(across)))
    hip_w = max(1e-3, float(np.linalg.norm(rh_ - lh_)))
    lo, hi = CFG["navel_t"]
    best, best_d = None, 1e9
    for k in range(1, n2):
        c = cen2[k] - hm
        tt = float(np.dot(c, u)) / tl
        xx = float(np.dot(c, across)) / hip_w
        a = float(st2[k, cv2.CC_STAT_AREA])
        ww, hh = float(st2[k, cv2.CC_STAT_WIDTH]), float(st2[k, cv2.CC_STAT_HEIGHT])
        roundish = (0.45 < ww / max(hh, 1.0) < 2.2 and 6.0 <= a < (0.12 * hip_w) ** 2 and a >= 0.40 * ww * hh)
        if lo <= tt <= hi and abs(xx) <= CFG["navel_x"] and roundish:
            d = abs(tt - 0.31) / (hi - lo) + abs(xx) / CFG["navel_x"]
            if d < best_d:
                best, best_d = k, d
    return (None if best is None else lab2 == best), kk


def _above_floor(poly, shape):
    """Pixels at least floor_px above the band's bottom edge (the line 0.10 above the hip joints, parallel to the hip line)."""
    bl, br = np.asarray(poly[3], np.float64), np.asarray(poly[2], np.float64)
    v = br - bl
    v = v / max(1e-6, float(np.linalg.norm(v)))
    nrm = np.array([v[1], -v[0]])
    if float(np.dot(nrm, np.asarray(poly[0], np.float64) - bl)) < 0:
        nrm = -nrm
    yy_, xx_ = np.mgrid[0:shape[0], 0:shape[1]]
    return ((xx_ - bl[0]) * nrm[0] + (yy_ - bl[1]) * nrm[1]) >= float(CFG["floor_px"])


def _disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(r) + 1, 2 * int(r) + 1))


def _dark_patch(L, band, skin, clothing, marks_all, out_pieces, navel_zone, inside, floor_ok, lm, info, away_ok=None):
    """v11.2.143: the dark patch INSIDE the pentagon, as a whole - every pixel clearly darker than the skin AROUND the pentagon.
    The marks finder needs a mark to be ink, or 12 levels darker than the skin around it: a faint patch (pigment, a birthmark, a
    healed scar) and the soft edge of any patch fall short and were left as a ring. Here the user's (or the automatic) pentagon
    says where the patch is: the skin in a ring just outside it is fitted (a smooth surface - the light across the belly), and
    inside it every pixel patch_dl levels and patch_k noise sigmas below that fit is the patch. Not the navel, not clothing, not
    below the floor, not a mark the pentagon left out (it stays whole), and - as for the darker-skin marks - not on the body's
    outline (away_ok: edge_px inside the skin; measured: the shaded side of the waist under a pentagon hanging over it). -> bool mask, or None."""
    hip_w = max(1e-3, float(np.linalg.norm(np.array(lm[R_HIP][:2], np.float64) - np.array(lm[L_HIP][:2], np.float64))))
    dil = lambda m, r: cv2.dilate(np.asarray(m).astype(np.uint8), _disk(r)) > 0
    cand = inside & band & ~clothing & floor_ok
    if away_ok is not None:
        cand &= away_ok
    if navel_zone is not None:
        cand &= ~dil(navel_zone, int(CFG["patch_navel_px"]))
    if int(cand.sum()) < 20:
        return None
    excl = dil(marks_all, 3) | clothing
    nvz = None if navel_zone is None else dil(navel_zone, int(CFG["patch_navel_px"]))
    if nvz is not None:
        excl |= nvz
    r_out = max(10, int(round(float(CFG["patch_ring"]) * hip_w)))
    ring = dil(inside, r_out) & ~dil(inside, 3) & band & skin & ~excl
    if int(ring.sum()) < 60:
        info["patch"] = "too little skin around the pentagon"
        return None
    Ls = cv2.GaussianBlur(L.astype(np.float32), (0, 0), 1.5)
    ys, xs = np.nonzero(ring)
    xm, ym, sc = float(xs.mean()), float(ys.mean()), float(max(np.ptp(xs), np.ptp(ys), 1))
    def basis(x_, y_):
        x_ = (x_ - xm) / sc
        y_ = (y_ - ym) / sc
        cols = [np.ones_like(x_), x_, y_]
        if len(xs) >= 200:                                   # the light across the belly, with its curve
            cols += [x_ * x_, x_ * y_, y_ * y_]
        return np.stack(cols, 1)
    A = basis(xs.astype(np.float64), ys.astype(np.float64))
    v = Ls[ys, xs].astype(np.float64)
    keep = np.ones(len(v), bool)
    s = 1.0
    coef = None
    for _ in range(3):                                       # robust: a dark speck in the ring does not pull the fit down
        coef = np.linalg.lstsq(A[keep], v[keep], rcond=None)[0]
        res = v - A @ coef
        s = max(1.0, 1.4826 * float(np.median(np.abs(res[keep] - np.median(res[keep])))))
        keep = np.abs(res) <= 2.5 * s
        if keep.sum() < 30:
            break
    thr = max(float(CFG["patch_dl"]), float(CFG["patch_k"]) * s)
    cy, cx = np.nonzero(cand)
    dk = np.zeros(band.shape, bool)
    dk[cy, cx] = (basis(cx.astype(np.float64), cy.astype(np.float64)) @ coef - Ls[cy, cx]) > thr
    dk = cv2.morphologyEx(dk.astype(np.uint8), cv2.MORPH_OPEN, _disk(int(CFG["patch_open"])))
    dk = (cv2.morphologyEx(dk, cv2.MORPH_CLOSE, _disk(2)) > 0) & cand
    near_mark = dil(marks_all & ~out_pieces, 1)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dk.astype(np.uint8), connectivity=8)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < int(CFG["patch_min_px"]) and not (near_mark & (lab == k)).any():
            dk[lab == k] = False
    if navel_zone is None and dk.any():                      # the navel was not among the marks: is it in the patch? (never filled)
        nv, kk = _navel_pick(dk, lm)
        if nv is not None:
            dk &= ~dil(nv, kk + int(CFG["patch_navel_px"]))
            info["navel"] = True
            info["_navel_px"] = dil(nv, kk)
    if not dk.any():
        info["patch_px"] = 0
        return None
    stay = dil(out_pieces, 4)
    if nvz is not None:
        stay |= nvz
    dk = dil(dk, int(CFG["patch_grow"])) & band & ~clothing & floor_ok & ~stay
    info["patch_px"] = int(dk.sum())
    info["patch_thr"] = round(float(thr), 1)
    return dk


def find_marks(img, lm, length=1.0, width=1.0, region=None, taken=None):
    """-> (mark mask bool, info dict). Marks: non-skin surrounded by skin, or skin-coloured but clearly darker (inside the band)."""
    info = {"band_px": 0, "skin_px": 0, "marks_px": 0, "pieces": 0, "clothing_px": 0, "navel": False}
    poly = region_polygon(lm, length, width)
    if poly is None:
        return None, info
    band = region_mask(poly, img.shape)
    info["band_px"] = int(band.sum())
    if band.sum() < CFG["min_region_px"]:
        return None, info
    model = skin_model(img, band)
    if model is None:
        return None, info
    skin = skin_pixels(img, band, model)
    info["skin_px"] = int(skin.sum())
    if skin.sum() < 0.25 * band.sum():
        return None, info                                   # mostly covered: nothing to do
    # (a) non-skin pieces surrounded by skin (ink of any colour)
    non = (band & ~skin).astype(np.uint8)
    # v11.2.140: and clearly ink. Skin at the waist's shaded sides can fall just outside the colour model (measured 3.7-4.0 against
    # the cut at 3.5) while being no darker than the skin around it (0-2 levels); ink is far darker (89-104) or far from the skin's
    # colour (12-14). A piece counts when it is ink_dl darker than its ring of skin, or ink_m from the skin's colour (coloured ink).
    lab_a = _lab(img)
    mu_a, icov_a = model[0], model[1]
    dab_a = lab_a[..., 1:] - mu_a
    chroma_a = np.sqrt(np.maximum(0.0, np.einsum("...i,ij,...j->...", dab_a, icov_a, dab_a)))
    non = cv2.morphologyEx(non, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab_, st, cen = cv2.connectedComponentsWithStats(non, connectivity=8)
    marks = np.zeros(band.shape, bool)
    clothing = np.zeros(band.shape, bool)
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for k in range(1, n):
        a = int(st[k, cv2.CC_STAT_AREA])
        if a < 4:
            continue
        if a > CFG["max_mark_share"] * band.sum():
            info["clothing_px"] += a
            clothing |= lab_ == k
            continue
        piece = lab_ == k
        x, y, w, h = st[k, cv2.CC_STAT_LEFT], st[k, cv2.CC_STAT_TOP], st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]
        pad = 5
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(band.shape[1], x + w + pad), min(band.shape[0], y + h + pad)
        pc = piece[y0:y1, x0:x1].astype(np.uint8)
        rim = (cv2.dilate(pc, ring_k) > 0) & ~piece[y0:y1, x0:x1]
        rim_in = rim & band[y0:y1, x0:x1]
        out_of_band = rim & ~band[y0:y1, x0:x1]
        if rim_in.sum() == 0:
            continue
        sk = float((rim_in & skin[y0:y1, x0:x1]).sum()) / float(rim_in.sum())
        # a piece that reaches the band's edge and is not small is clothing (a waistband, shorts): the band's outline cuts it, so
        # what is left of its rim inside the band can be all skin (v11.2.111: a waistband was taken for a mark when the band was
        # narrowed); thin ink strokes attached to it are recovered below
        touches_edge = int(out_of_band.sum()) >= 8
        if touches_edge and a > 0.02 * band.sum():
            pc_body = cv2.morphologyEx(pc, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
            if int(pc_body.sum()) >= 0.5 * a:                  # thick: cloth (thin tattoo strokes cut by the band's edge are not)
                info["clothing_px"] += a
                clothing |= piece
                continue
        if sk >= CFG["enclosed_min"] and not (out_of_band.sum() > 0.4 * rim.sum() and a > 0.05 * band.sum()):
            ring_sk = rim_in & skin[y0:y1, x0:x1]
            pl = lab_a[y0:y1, x0:x1, 0][piece[y0:y1, x0:x1]]
            dl = float(np.median(lab_a[y0:y1, x0:x1, 0][ring_sk]) - np.median(pl)) if ring_sk.any() else 0.0
            ch = float(np.median(chroma_a[y0:y1, x0:x1][piece[y0:y1, x0:x1]]))
            if dl >= CFG["ink_dl"] or ch >= CFG["ink_m"]:
                marks |= piece
            else:
                info["not_ink"] = info.get("not_ink", 0) + a
        else:
            info["clothing_px"] += a
            clothing |= piece
    # ink that touches clothing joins it into one piece. Cloth is thick; a tattoo stroke is thin: an opening with an 11 px disc keeps
    # the cloth's body (and its straight edges) and strips the thin strokes attached to it - those are ink (unless they lie on the
    # band's own edge, where the band's outline cuts the cloth into slivers)
    if clothing.any():
        body = cv2.morphologyEx(clothing.astype(np.uint8), cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
        # v11.2.112: the slivers on the band's own edge are cut off first, then each thin piece is judged on its own (a tattoo stroke
        # joined to black shorts by the shorts' ragged top edge, which runs out to the band's sides, was all skipped before)
        edge = band & ~(cv2.erode(band.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0)
        thin = (clothing & ~body & ~edge).astype(np.uint8)
        n3, lab3, st3, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
        # v11.2.140: a thin piece that lies ALONG the cloth's edge is the cloth's own ragged edge, or the background showing along
        # the body's outline (the whole-belly band reaches the waist's sides) - not ink. A tattoo stroke sticks out of the cloth:
        # only its end touches it. Measured: a 9 px sliver of sea along the waist was taken for ink and then painted on every frame.
        hug_zone = cv2.dilate(body.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
        # v11.2.140: the rescue is for strokes running into the waistband or shorts at the band's BOTTOM (SDOS-112). With the
        # whole belly in the band, its sides reach the waist's outline, where the dark gap between body and background, or bits of
        # the background, are thin pieces too (measured: 260, 41 and 36 px on one frame, then painted on every frame). Only thin
        # pieces in the band's lower part (rescue_t of the torso above the floor) and away from its sides (rescue_side of its
        # width) are rescued.
        P0, P1, P2, P3 = (np.asarray(poly[j], np.float64) for j in range(4))
        def _line_dist(a_, b_):
            v_ = b_ - a_
            v_ = v_ / max(1e-6, float(np.linalg.norm(v_)))
            yy_, xx_ = np.mgrid[0:band.shape[0], 0:band.shape[1]]
            return np.abs((xx_ - a_[0]) * v_[1] - (yy_ - a_[1]) * v_[0])
        side_d = np.minimum(_line_dist(P0, P3), _line_dist(P1, P2))
        floor_d = _line_dist(P3, P2)
        lm_sh = 0.5 * (np.array(lm[L_SH][:2], np.float64) + np.array(lm[R_SH][:2], np.float64))
        lm_hp = 0.5 * (np.array(lm[L_HIP][:2], np.float64) + np.array(lm[R_HIP][:2], np.float64))
        torso_px = float(np.linalg.norm(lm_sh - lm_hp))
        rescue_ok = (floor_d <= float(CFG["rescue_t"]) * torso_px) & (side_d >= float(CFG["rescue_side"]) * float(np.linalg.norm(P2 - P3)))
        for k in range(1, n3):
            if st3[k, cv2.CC_STAT_AREA] < 10:
                continue
            comp = lab3 == k
            if float((comp & hug_zone).sum()) > float(CFG["hug_max"]) * float(comp.sum()):
                continue
            if float((comp & rescue_ok).sum()) < 0.5 * float(comp.sum()):
                continue
            # v11.2.140: and it must be ink, like any other piece (section (a)): shaded skin along the body's outline falls just
            # outside the skin model, joins the background, and its strip came back here as a "stroke" (measured: 147 px on the
            # waist's side, then painted on every frame)
            ring_c = (cv2.dilate(comp.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & ~comp & skin
            dl_c = float(np.median(lab_a[..., 0][ring_c]) - np.median(lab_a[..., 0][comp])) if ring_c.any() else 0.0
            ch_c = float(np.median(chroma_a[comp]))
            if dl_c < CFG["ink_dl"] and ch_c < CFG["ink_m"]:
                info["not_ink"] = info.get("not_ink", 0) + int(comp.sum())
                continue
            marks |= comp
            clothing &= ~comp
    # (b) skin-coloured but clearly darker than the skin around it (moles, pigment)
    L = _lab(img)[..., 0]
    ys, xs = np.nonzero(band)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    Lc, sc = L[y0:y1, x0:x1], skin[y0:y1, x0:x1]
    size = max(y1 - y0, x1 - x0)
    sig = max(4.0, size / 12.0)
    w_ = cv2.GaussianBlur(sc.astype(np.float32), (0, 0), sig)
    bg = cv2.GaussianBlur(Lc * sc, (0, 0), sig) / np.maximum(w_, 1e-3)
    fine = Lc - cv2.GaussianBlur(Lc, (0, 0), 1.2)
    noise = float(np.median(np.abs(fine[sc] - np.median(fine[sc])))) * 1.4826 + 1.0 if sc.any() else 3.0
    dark = sc & ((bg - Lc) > max(CFG["dark_min"], CFG["dark_k"] * noise * 2.0))
    dk = np.zeros(band.shape, bool)
    dk[y0:y1, x0:x1] = dark
    dk = cv2.morphologyEx(dk.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    # v11.2.140: not on the body's outline. With the whole belly in the band, the band reaches the waist's sides, where the skin
    # fades into the background (or a waistband's shadow): those soft edges are darker than the skin around them but are no mark.
    # A darker-skin mark must lie at least edge_px inside the skin - away from clothing, background and the band's own edge.
    # (Ink, route (a), is judged by its rim of skin and is not affected.)
    away = (band & ~clothing).astype(np.uint8)
    away[non.astype(bool) & ~marks] = 0
    edge_px = max(4.0, float(CFG["edge_k"]) * size)
    inside = cv2.distanceTransform(away, cv2.DIST_L2, 3) >= edge_px
    marks |= dk & inside
    away_ok = inside
    # the navel is not a mark. v11.2.140: with the whole belly in the band the navel can be anywhere in it, so it is placed on the
    # body - a compact, filled dark spot on the middle line, navel_t of the way up from the hips - not at the band's top edge. Only
    # the one most navel-like spot is kept: a mole or a tattoo dot elsewhere on the middle line is still a mark.
    navel_zone = None
    if marks.any():
        nv_, kk = _navel_pick(marks, lm)
        if nv_ is not None:
            navel_zone = (cv2.dilate(nv_.astype(np.uint8),
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kk + 1, 2 * kk + 1))) > 0) & marks
            marks &= ~navel_zone
            info["navel"] = True
    if marks.any():                                         # (grown into skin to cover soft ink edges, never into clothing)
        d = int(CFG["mark_dilate"])
        marks = (cv2.dilate(marks.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))) > 0) & band & ~clothing
        if navel_zone is not None:                          # (v11.2.140: a mark grown next to the navel does not grow over it)
            marks &= ~(cv2.dilate(navel_zone.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))) > 0)
        # v11.2.140: the floor is a hard limit. The fill grows every hole by its halo (PATCH_CFG halo_px, 3 px), so marks stop
        # floor_px above the band's bottom edge (the line 0.10 above the hip joints, parallel to the hip line): nothing below it changes.
        marks &= _above_floor(poly, marks.shape)
    if region is not None:
        # v11.2.142: only the marks inside the pentagon (the user's, or the automatic one), each piece WHOLE or not at all: a
        # piece is in when its middle is. A mark the pentagon's line cuts is never half removed (half a mole left, and which
        # half would change with the pose's few px from frame to frame). The skin model and the clothing test see the whole band.
        # taken (the job): the pieces to keep, as decided once for the whole video (clean_belly._pentagon_hits).
        marks_all = marks
        if marks.any():
            if taken is not None:
                marks = marks & np.asarray(taken, bool)
            else:
                rpoly = np.asarray(region, np.float32).reshape(-1, 1, 2)
                n_, lab_, _st, cen_ = cv2.connectedComponentsWithStats(marks.astype(np.uint8), connectivity=8)
                keep = np.zeros(n_, bool)
                for k in range(1, n_):
                    keep[k] = cv2.pointPolygonTest(rpoly, (float(cen_[k][0]), float(cen_[k][1])), False) >= 0
                marks = keep[lab_]
        info["_kept"] = marks.copy()
        info["marks_outside"] = int(cv2.connectedComponents((marks_all & ~marks).astype(np.uint8))[0] - 1) if marks_all.any() else 0
        # v11.2.143: and the dark patch inside the pentagon as a whole (its faint parts and soft edges too)
        rmask = np.zeros(band.shape, np.uint8)
        cv2.fillPoly(rmask, [np.round(np.asarray(region, np.float32)).astype(np.int32)], 1)
        pt = _dark_patch(L, band, skin, clothing, marks_all, marks_all & ~marks, navel_zone, rmask > 0,
                         _above_floor(poly, band.shape), lm, info, away_ok=away_ok)
        if pt is not None:
            marks = marks | pt
    # v11.2.143: what the video's paint must never cover, however the patch engine grows the hole (its halo, the hair halo, the
    # per-frame seal: 8-14 px): below the floor, clothing, the navel, and a mark the pentagon left out. Written into the mask the
    # engine gets (clean_belly: value 64); measured before: on a moving test clip the paint reached past the floor's 4 px margin.
    ko = ~_above_floor(poly, band.shape)
    if clothing.any():
        ko |= cv2.dilate(clothing.astype(np.uint8), _disk(2)) > 0
    if navel_zone is not None:
        ko |= cv2.dilate(navel_zone.astype(np.uint8), _disk(int(CFG["patch_navel_px"]))) > 0
    if info.get("_navel_px") is not None:
        ko |= cv2.dilate(info.pop("_navel_px").astype(np.uint8), _disk(int(CFG["patch_navel_px"]))) > 0
    if region is not None and "_kept" in info:
        lo_ = marks_all & ~info["_kept"]
        if lo_.any():
            ko |= cv2.dilate(lo_.astype(np.uint8), _disk(3)) > 0
    info["_keepout"] = ko & ~marks
    info["marks_px"] = int(marks.sum())
    info["pieces"] = int(cv2.connectedComponents(marks.astype(np.uint8))[0] - 1) if marks.any() else 0
    return marks, info


# ======================================================================================================================
# v11.2.142 BellyAnchor (SDOS-142): the belly patch in BODY coordinates.
#   u: across the body, in hip widths from the middle line (the hip joints at u = -0.5 / +0.5)
#   t: up the body, 0 at the hip-joint line, 1 at the shoulder line (the navel is about 0.27-0.36)
# A point (u, t) is x = hip_mid + u * (right_hip - left_hip) + t * (shoulder_mid - hip_mid): it moves, turns and scales with the
# hips and shoulders, so a patch held in (u, t) follows the body - it cannot stay behind when the body moves.
# ======================================================================================================================
PENT = {
    "margin": 0.06,        # v11.2.143: the automatic pentagon lies this share of the patch's size outside it ...
    "margin_min": 6.0,     #   ...at least this many px ...
    "margin_max": 14.0,    #   ...at most this many (it was 8 % and grown about its middle: 2-3 x wider than the patch)
    "join": 0.06,          # v11.2.143: marks this share of the hip width apart (x2) are one patch (the strokes of one tattoo)
    "also": 0.5,           #   the automatic pentagon goes round the main dark patch (most area x darkness), and another one only
    "near": 0.30,          #   when it is this dark-heavy (share of the main one) and this close (share of the hip width)
    "default": [[-0.30, 0.30], [0.30, 0.30], [0.38, 0.18], [0.0, 0.13], [-0.38, 0.18]],   # no mark found: a pentagon below the navel
    "u_max": 1.2,          # a pentagon corner may not be farther than this from the middle line (hip widths)
}


def mark_pieces(marks, lm):
    """The mark mask's connected pieces -> (labels int32, [(label, middle (u, t) in body coordinates, area px)])."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(np.asarray(marks).astype(np.uint8), connectivity=8)
    if n <= 1:
        return lab, []
    ut = to_body(np.asarray(cen[1:], np.float64), lm)
    return lab, [(k, ut[k - 1], int(st[k, cv2.CC_STAT_AREA])) for k in range(1, n)]


def parse_marks(v):
    """The app's belly_marks (what the last preview found): [[u, t, taken 1|0, px], ...] as a list or JSON text -> (n, 4) array, or None."""
    try:
        if isinstance(v, str):
            import json as _json
            v = _json.loads(v)
        a = np.asarray(v, np.float64)
        if a.ndim != 2 or a.shape[1] < 3 or len(a) == 0 or len(a) > 200 or not np.isfinite(a[:, :3]).all():
            return None
        if a.shape[1] == 3:
            a = np.hstack([a, np.full((len(a), 1), 100.0)])
        return a[:, :4]
    except Exception:
        return None


def in_pentagon_body(ut, pent_ut):
    """Is the body point (u, t) inside the pentagon given in body coordinates (on its line counts as inside)?"""
    return cv2.pointPolygonTest(np.asarray(pent_ut, np.float32).reshape(-1, 1, 2), (float(ut[0]), float(ut[1])), False) >= 0


def body_affine(lm):
    """2x3 affine (u, t, 1) -> (x, y) from the shoulders and hips, or None."""
    try:
        ls, rs = np.array(lm[L_SH][:2], np.float64), np.array(lm[R_SH][:2], np.float64)
        lh, rh = np.array(lm[L_HIP][:2], np.float64), np.array(lm[R_HIP][:2], np.float64)
    except Exception:
        return None
    hm, sm = 0.5 * (lh + rh), 0.5 * (ls + rs)
    U, T = rh - lh, sm - hm
    if float(np.linalg.norm(U)) < 4.0 or float(np.linalg.norm(T)) < 8.0:
        return None
    if abs(float(U[0] * T[1] - U[1] * T[0])) < 1e-3:
        return None
    return np.array([[U[0], T[0], hm[0]], [U[1], T[1], hm[1]]], np.float64)


def to_body(pts, lm):
    """image px (N, 2) -> body (u, t) (N, 2), or None."""
    M = body_affine(lm)
    if M is None:
        return None
    Mi = cv2.invertAffineTransform(M)
    P = np.asarray(pts, np.float64).reshape(-1, 2)
    return P @ Mi[:, :2].T + Mi[:, 2]


def from_body(ut, lm):
    """body (u, t) (N, 2) -> image px (N, 2), or None."""
    M = body_affine(lm)
    if M is None:
        return None
    P = np.asarray(ut, np.float64).reshape(-1, 2)
    return P @ M[:, :2].T + M[:, 2]


def clamp_pentagon_body(ut, length=1.0, width=1.0):
    """Corners kept on the belly: t between the band's floor (+ floor_px, as the marks) and its top, u within u_max.
    The floor (0.10 above the hip line, the bikini line) is a hard limit (v11.2.140); no corner can be moved below it."""
    P = np.asarray(ut, np.float64).reshape(-1, 2).copy()
    lo, hi = float(T_BOT_MIN), float(T_TOP_MAX)
    P[:, 1] = np.clip(P[:, 1], lo, hi)
    P[:, 0] = np.clip(P[:, 0], -float(PENT["u_max"]), float(PENT["u_max"]))
    return P


def _five(poly):
    """A convex polygon -> exactly 5 corners (fewer: the longest edges split; more: simplified, then grown to enclose)."""
    P = np.asarray(poly, np.float64).reshape(-1, 2)
    if len(P) > 5:
        peri = float(cv2.arcLength(P.astype(np.float32), True))
        eps = 0.01 * peri
        Q = P
        for _ in range(60):
            Q = cv2.approxPolyDP(P.astype(np.float32), eps, True).reshape(-1, 2).astype(np.float64)
            if len(Q) <= 5:
                break
            eps *= 1.25
        P = Q
    while len(P) < 5:
        e = [float(np.linalg.norm(P[(k + 1) % len(P)] - P[k])) for k in range(len(P))]
        k = int(np.argmax(e))
        P = np.insert(P, k + 1, 0.5 * (P[k] + P[(k + 1) % len(P)]), axis=0)
    return P[:5]


def focus_marks(marks, lm, img=None):
    """v11.2.143: the main dark patch among the marks. Marks within 2 x join of the hip width of each other are one patch; each
    patch weighs its area x its darkness (against the skin just around it); the heaviest is the main one, and another patch
    joins it only when it weighs at least `also` of it and lies within `near` of the hip width. A stray speck or a small mole
    elsewhere on the belly no longer stretches the pentagon over the whole belly.
    -> (the chosen marks, bool; number of patches found; number chosen)."""
    m = np.asarray(marks).astype(bool)
    if not m.any():
        return m, 0, 0
    hip_w = max(1e-3, float(np.linalg.norm(np.array(lm[R_HIP][:2], np.float64) - np.array(lm[L_HIP][:2], np.float64))))
    r = max(4, int(round(float(PENT["join"]) * hip_w)))
    ng, lg = cv2.connectedComponents((cv2.dilate(m.astype(np.uint8), _disk(r)) > 0).astype(np.uint8), connectivity=8)
    L = _lab(img)[..., 0] if img is not None else None
    mass = np.zeros(ng)
    for g in range(1, ng):
        pix = m & (lg == g)
        a = float(pix.sum())
        if a == 0:
            continue
        con = 1.0
        if L is not None:
            rg = (cv2.dilate(pix.astype(np.uint8), _disk(5)) > 0) & ~(cv2.dilate(pix.astype(np.uint8), _disk(1)) > 0) & ~m
            if rg.any():
                con = max(1.0, float(np.median(L[rg])) - float(L[pix].mean()))
        mass[g] = a * con
    best = int(np.argmax(mass))
    if best == 0:
        return m, ng - 1, ng - 1
    dist = cv2.distanceTransform((~(m & (lg == best))).astype(np.uint8), cv2.DIST_L2, 3)
    keep = [best]
    for g in range(1, ng):
        if g == best or mass[g] <= 0:
            continue
        if mass[g] >= float(PENT["also"]) * mass[best] and float(dist[m & (lg == g)].min()) <= float(PENT["near"]) * hip_w:
            keep.append(g)
    return m & np.isin(lg, keep), int((mass > 0).sum()), len(keep)


def _min_pentagon(pts):
    """The smallest pentagon around points (nearly): their convex hull, then the edge whose removal adds the least area is removed,
    again and again, by extending its two neighbours to where they meet - until five corners are left."""
    P = cv2.convexHull(np.asarray(pts, np.float32).reshape(-1, 1, 2)).reshape(-1, 2).astype(np.float64)
    if len(P) < 3:
        c = P.mean(axis=0)
        P = c + np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])
    while len(P) > 5:
        n = len(P)
        best = None
        for i in range(n):
            a0, a1, b0, b1 = P[i - 1], P[i], P[(i + 1) % n], P[(i + 2) % n]
            da, db = a1 - a0, b1 - b0
            den = da[0] * db[1] - da[1] * db[0]
            if abs(den) < 1e-9:
                continue
            w = b0 - a1
            s_ = (w[0] * db[1] - w[1] * db[0]) / den          # X = a1 + s da (beyond a1), X = b0 + r db (before b0: r <= 0)
            r_ = (w[0] * da[1] - w[1] * da[0]) / den
            if s_ < -1e-9 or r_ > 1e-9:
                continue
            X = a1 + s_ * da
            area = 0.5 * abs((X[0] - a1[0]) * (b0[1] - a1[1]) - (X[1] - a1[1]) * (b0[0] - a1[0]))
            if best is None or area < best[0]:
                best = (area, i, X)
        if best is None:
            return _five(P)
        _, i, X = best
        j = (i + 1) % n
        P = np.array([X if k == i else P[k] for k in range(n) if k != j])
    while len(P) < 5:
        e = [float(np.linalg.norm(P[(k + 1) % len(P)] - P[k])) for k in range(len(P))]
        k = int(np.argmax(e))
        P = np.insert(P, k + 1, 0.5 * (P[k] + P[(k + 1) % len(P)]), axis=0)
    return P


def _offset_poly(P, m):
    """A convex polygon with every edge moved m px outwards (corners where the moved edges meet)."""
    P = np.asarray(P, np.float64)
    n = len(P)
    c = P.mean(axis=0)
    nr = []
    for i in range(n):
        d = P[(i + 1) % n] - P[i]
        v = np.array([d[1], -d[0]]) / max(1e-9, float(np.linalg.norm(d)))
        if float(np.dot(v, 0.5 * (P[i] + P[(i + 1) % n]) - c)) < 0:
            v = -v
        nr.append(v)
    Q = []
    for i in range(n):
        n1, n2 = nr[i - 1], nr[i]                      # the edges before and after corner i
        p1, p2 = P[i - 1] + m * n1, P[i] + m * n2
        d1, d2 = P[i] - P[i - 1], P[(i + 1) % n] - P[i]
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-9 * max(1.0, float(np.linalg.norm(d1)) * float(np.linalg.norm(d2))):
            Q.append(P[i] + m * n2)
            continue
        w = p2 - p1
        s_ = (w[0] * d2[1] - w[1] * d2[0]) / den
        Q.append(p1 + s_ * d1)
    return np.array(Q)


def auto_pentagon(marks, lm, img=None):
    """The automatic pentagon (image px, 5 x 2): v11.2.143 tight round the main dark patch (focus_marks) - the smallest pentagon
    round it with a margin of `margin` of its size (6-14 px) - or the default one below the navel when there are no marks."""
    if marks is not None and int(np.asarray(marks).sum()) >= 6:
        sel, _, _ = focus_marks(marks, lm, img)
        ys, xs = np.nonzero(sel)
        P = _min_pentagon(np.stack([xs, ys], 1))
        size = max(8.0, float(np.ptp(np.stack([xs, ys], 1), axis=0).max()))
        mg = min(float(PENT["margin_max"]), max(float(PENT["margin_min"]), float(PENT["margin"]) * size))
        return _offset_poly(P, mg)
    D = from_body(np.array(PENT["default"], np.float64), lm)
    return D


def fit_pentagon(img, lm, length=1.0, width=1.0):
    """v11.2.143: the automatic pentagon in two passes. 1) Tight round the main dark patch (its marks). 2) The dark patch is then
    looked for inside that pentagon grown by 3 margins (its faint outer edge is no mark: it would be cut off), and the pentagon is
    fitted tight round all of it. -> (pentagon px, source "auto" | "default", number of patches found, number inside)."""
    mk, _ = find_marks(img, lm, length, width)
    if mk is None or int(mk.sum()) < 6:
        return auto_pentagon(None, lm), "default", 0, 0
    sel, n_all, n_sel = focus_marks(mk, lm, img)
    P1 = auto_pentagon(mk, lm, img)
    ys, xs = np.nonzero(sel)
    size = max(8.0, float(np.ptp(np.stack([xs, ys], 1), axis=0).max()))
    mg = min(float(PENT["margin_max"]), max(float(PENT["margin_min"]), float(PENT["margin"]) * size))
    mk2, _ = find_marks(img, lm, length, width, region=_offset_poly(_min_pentagon(np.stack([xs, ys], 1)), 3.0 * mg))
    if mk2 is None or int(mk2.sum()) < 6:
        return P1, "auto", n_all, n_sel
    ys2, xs2 = np.nonzero(mk2 | sel)
    return _offset_poly(_min_pentagon(np.stack([xs2, ys2], 1)), mg), "auto", n_all, n_sel


def pentagon_info(pent_px, lm, length=1.0, width=1.0):
    """(clamped pentagon in px, the same in body coordinates) - the floor and the band's top applied. Either may be None."""
    if pent_px is None:
        return None, None
    ut = to_body(pent_px, lm)
    if ut is None:
        return np.asarray(pent_px, np.float64), None
    ut = clamp_pentagon_body(ut, length, width)
    return from_body(ut, lm), ut


def parse_pentagon(v):
    """settings value -> (N, 2) float array (5 corners), or None. Accepts [[a, b] x 5] or JSON text of it."""
    try:
        if isinstance(v, str):
            import json
            v = json.loads(v) if v.strip() else None
        if v is None:
            return None
        P = np.asarray(v, np.float64).reshape(-1, 2)
        if len(P) != 5 or not np.all(np.isfinite(P)):
            return None
        return P
    except Exception:
        return None


class BodyPath:
    """v11.2.142: the body's shoulders and hips on every frame - from pose samples (every few frames), outliers dropped
    (clean_samples), smoothed over time (a centred Gaussian, zero lag), interpolated between samples; held at most hold_s past
    the last sample. at(i) -> landmark dict or None."""

    def __init__(self, samples, n_frames, fps, sigma_s=0.10):
        kept, self.dropped = clean_samples({int(k): v for k, v in samples.items() if v is not None})
        self.keys = sorted(kept)
        self.n = int(n_frames)
        self.hold = max(1, int(round(float(CFG["hold_s"]) * float(fps or 30.0))))
        sig = max(0.5, float(sigma_s) * float(fps or 30.0))           # (frames)
        pts = {k: np.array([kept[k][j][:2] for j in (L_SH, R_SH, L_HIP, R_HIP)], np.float64) for k in self.keys}
        self.sm = {}
        for k in self.keys:
            w, acc = 0.0, np.zeros((4, 2))
            for k2 in self.keys:
                d = abs(k2 - k)
                if d > 3.0 * sig:
                    continue
                if d > self.hold:                                   # (never smooth across a gap the hold would not bridge)
                    continue
                ww = float(np.exp(-0.5 * (d / sig) ** 2))
                acc += ww * pts[k2]
                w += ww
            self.sm[k] = acc / w
        self.sample_count = len(self.keys)

    def at(self, idx):
        import bisect
        if not self.keys:
            return None
        j = bisect.bisect_left(self.keys, idx)
        a = self.keys[j - 1] if j > 0 else None
        b = self.keys[j] if j < len(self.keys) else None
        P = None
        if b is not None and b == idx:
            P = self.sm[b]
        elif a is not None and b is not None and (b - a) <= 2 * self.hold:
            tt = (idx - a) / float(b - a)
            P = (1 - tt) * self.sm[a] + tt * self.sm[b]
        elif a is not None and idx - a <= self.hold:
            P = self.sm[a]
        elif b is not None and b - idx <= self.hold:
            P = self.sm[b]
        if P is None:
            return None
        return {L_SH: (P[0, 0], P[0, 1], 1.0), R_SH: (P[1, 0], P[1, 1], 1.0), L_HIP: (P[2, 0], P[2, 1], 1.0), R_HIP: (P[3, 0], P[3, 1], 1.0)}

    def affine(self, idx):
        lm = self.at(idx)
        return None if lm is None else body_affine(lm)

    def below_floor(self, idx, x0, y0, w, h):
        """v11.2.143: the pixels of the box (x0, y0, w, h) of frame idx that lie below the belly's floor on THIS frame's body (the line
        T_BOT_MIN above the hip joints, plus floor_px), or None where there is no body. The video's belly paint is cut there on every
        frame: a hard limit that holds whatever the patch tracker does (measured: on a fast-moving test clip the paint, carried by the
        tracker, reached up to 31 px below the line on every frame; held by the anchor frame's mask alone it cannot be stopped)."""
        M = self.affine(idx)
        if M is None:
            return None
        try:
            Mi = np.linalg.inv(np.vstack([np.asarray(M, np.float64), [0.0, 0.0, 1.0]]))
        except np.linalg.LinAlgError:
            return None
        tl = max(1.0, float(np.linalg.norm(np.asarray(M)[:, 1])))
        yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w]
        tt = Mi[1, 0] * xx + Mi[1, 1] * yy + Mi[1, 2]
        return tt < (T_BOT_MIN + float(CFG["floor_px"]) / tl)


class BellyTrack:
    """Per-frame even-skin pass over the band (applied after Patch Removal, before the face swap). apply(frame, idx)."""

    def __init__(self, samples, n_frames, fps, length=1.0, width=1.0, smooth=None):
        self.length, self.width = clamp_scale(length, LENGTH_RANGE), clamp_scale(width, WIDTH_RANGE)
        # (its own copy: a preview with another strength while this job runs must not change it)
        self.smooth = float(CFG["smooth"] if smooth is None else min(1.0, max(0.0, float(smooth))))
        self.samples = dict(samples)                         # frame index -> landmarks
        self.keys = sorted(self.samples)
        self.n = int(n_frames)
        self.hold = max(1, int(round(float(CFG["hold_s"]) * float(fps or 30.0))))
        self.model = None
        self.frames_smoothed = 0
        STATS["frames"] = 0
        STATS["cleaned"] = 0
        STATS["cleaned_px"] = 0
        self.cleanup = bool(CFG.get("cleanup", True))

    def landmarks_at(self, idx):
        """Linear between the two nearest samples (each no further than hold frames), else None."""
        if not self.keys:
            return None
        import bisect
        j = bisect.bisect_left(self.keys, idx)
        a = self.keys[j - 1] if j > 0 else None
        b = self.keys[j] if j < len(self.keys) else None
        if b is not None and b == idx:
            return self.samples[b]
        if a is not None and b is not None and (b - a) <= 2 * self.hold:
            t = (idx - a) / float(b - a)
            la, lb = self.samples[a], self.samples[b]
            return {k: tuple((1 - t) * np.array(la[k]) + t * np.array(lb[k])) for k in la}
        if a is not None and idx - a <= self.hold:
            return self.samples[a]
        if b is not None and b - idx <= self.hold:
            return self.samples[b]
        return None

    def _clean_marks(self, frame, lm, x0, y0, x1, y1):
        """v11.2.112: the marks still showing in the band on this frame (where the fill carried from the key frames lost the mark:
        the body turned, the tracker held its position) are found again and filled here. Thin strokes and small marks come out
        clean; a large filled tattoo comes out softer than the key frames' fill."""
        H, W = frame.shape[:2]
        p = 24
        X0, Y0, X1, Y1 = max(0, x0 - p), max(0, y0 - p), min(W, x1 + p), min(H, y1 + p)
        roi = frame[Y0:Y1, X0:X1]
        lr = {k: (float(v[0]) - X0, float(v[1]) - Y0, float(v[2])) for k, v in lm.items()}
        try:
            mk, _ = find_marks(roi, lr, self.length, self.width)
        except Exception:
            return frame
        if mk is None or int(mk.sum()) < int(CFG["cleanup_min_px"]):
            return frame
        fill = cv2.inpaint(roi, mk.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
        out = frame.copy()
        r = out[Y0:Y1, X0:X1]
        r[mk] = fill[mk]
        STATS["cleaned"] += 1
        STATS["cleaned_px"] += int(mk.sum())
        return out

    def apply(self, frame, idx):
        s = float(self.smooth)
        lm = self.landmarks_at(idx)
        poly = region_polygon(lm, self.length, self.width)
        if poly is None:
            return frame
        H, W = frame.shape[:2]
        x0, y0 = int(max(0, np.floor(poly[:, 0].min()) - 8)), int(max(0, np.floor(poly[:, 1].min()) - 8))
        x1, y1 = int(min(W, np.ceil(poly[:, 0].max()) + 9)), int(min(H, np.ceil(poly[:, 1].max()) + 9))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return frame
        if self.cleanup:
            frame = self._clean_marks(frame, lm, x0, y0, x1, y1)
        if s <= 0.0:
            return frame
        roi = frame[y0:y1, x0:x1]
        band = region_mask(poly - np.array([x0, y0], np.float32), roi.shape)
        if band.sum() < CFG["min_region_px"]:
            return frame
        model = self.model or skin_model(roi, band)
        if model is None:
            return frame
        if self.model is None:
            self.model = model
        skin = skin_pixels(roi, band, model)
        if skin.sum() < 0.25 * band.sum():
            return frame
        size = max(x1 - x0, y1 - y0)
        feather = max(2.0, size * 0.03)
        w = cv2.GaussianBlur(skin.astype(np.float32), (0, 0), feather) * skin
        er = int(max(1, round(2.0 * feather)))
        inner = cv2.erode(band.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * er + 1, 2 * er + 1)))
        w *= cv2.GaussianBlur(inner.astype(np.float32), (0, 0), feather) * band   # (fades out INSIDE the band: nothing outside it changes)
        sm = cv2.bilateralFilter(roi, d=9, sigmaColor=22, sigmaSpace=max(3.0, size / 80.0))
        out = frame.copy()
        a = (np.clip(w, 0.0, 1.0) * s)[..., None]
        out[y0:y1, x0:x1] = np.clip(roi.astype(np.float32) * (1.0 - a) + sm.astype(np.float32) * a, 0, 255).astype(np.uint8)
        self.frames_smoothed += 1
        STATS["frames"] += 1
        return out


def choose_keys(samples, n_frames, fps, keys_max=None, length=1.0, width=1.0):
    """Key frames for the mark search: spread over the video, at least key_gap_s apart, the band largest in its stretch."""
    keys_max = int(keys_max if keys_max is not None else CFG["keys_max"])
    good = []
    for i, lm in sorted(samples.items()):
        poly = region_polygon(lm, length, width)
        if poly is not None:
            good.append((i, float(cv2.contourArea(poly.astype(np.float32)))))
    if not good or keys_max <= 0:
        return []
    gap = max(1, int(round(float(CFG["key_gap_s"]) * float(fps or 30.0))))
    span = max(1, int(math.ceil(n_frames / float(keys_max))))
    out = []
    for s0 in range(0, n_frames, span):
        cand = [g for g in good if s0 <= g[0] < s0 + span and all(abs(g[0] - o) >= gap for o in out)]
        if cand:
            out.append(max(cand, key=lambda g: g[1])[0])
    return sorted(out)[:keys_max]


def preview(img, lm, length=1.0, width=1.0, smooth=None, remove_fn=None):
    """v11.2.111: belly clean on ONE frame, for the app's preview. img: BGR; lm: landmarks (or None: no body found); remove_fn(img,
    mask_u8) -> (img, info): the still-frame mark removal (the Patch Removal preview's own fill). -> (after, after with the band
    outlined, info dict). Nothing is changed when no body is found."""
    info = {"body": lm is not None, "band": None, "length": clamp_scale(length, LENGTH_RANGE), "width": clamp_scale(width, WIDTH_RANGE)}
    poly = region_polygon(lm, length, width)
    if poly is None:
        info["reason"] = "no body found" if lm is None else "shoulders and hips not clear enough (a side view, or the head out of the frame)"
        return img, img.copy(), info
    info["band"] = [[round(float(x), 1), round(float(y), 1)] for x, y in poly]
    mk, mi = find_marks(img, lm, length, width)
    info.update({k: mi[k] for k in ("band_px", "skin_px", "marks_px", "pieces", "clothing_px", "navel")})
    out = img
    if mk is not None and int(mk.sum()) >= 30 and remove_fn is not None:
        out, _ = remove_fn(img, mk.astype(np.uint8) * 255)
    bt = BellyTrack({0: lm}, 1, 30.0, length, width, smooth)
    info["smooth"] = bt.smooth
    out = bt.apply(out, 0)
    outlined = out.copy()
    th = max(2, int(round(max(img.shape[:2]) / 400.0)))
    cv2.polylines(outlined, [np.round(poly).astype(np.int32)], True, (255, 200, 40), th, cv2.LINE_AA)
    return out, outlined, info
