"""v11.2.110 (SDOS-110) Lower-belly clear skin. v11.2.112 (SDOS-112): the pose model on MediaPipe 0.10.30+ (Tasks API).

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
    "t_top": 0.22,           # the band starts this far up from the hip line towards the shoulders (navel ~0.27)
    "t_bot": -0.03,          # ...and ends a little below the hip line (the lowest part of the waist)
    "side_ext": 0.30,        # widened sideways by this share of the hip width on each side
    "keys_max": 8,           # key frames on which marks are looked for (positions of one Patch Removal mark)
    "key_gap_s": 1.5,        # at least this far apart
    "smooth": 0.5,           # even-skin strength (0 = off, 1 = full edge-preserving smoothing)
    "mark_dilate": 4,        # px added around every mark found (soft ink edges)
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
T_TOP_MAX, T_BOT_MIN = 0.40, -0.05      # never higher than 0.40 of the way up to the shoulders, never lower than just below the hip line


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


def find_marks(img, lm, length=1.0, width=1.0):
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
            marks |= piece
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
        for k in range(1, n3):
            if st3[k, cv2.CC_STAT_AREA] < 10:
                continue
            comp = lab3 == k
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
    marks |= dk
    # the navel is not a mark: a compact dark piece near the middle of the band's top edge is left alone
    if marks.any():
        n2, lab2, st2, cen2 = cv2.connectedComponentsWithStats(marks.astype(np.uint8), connectivity=8)
        tm = 0.5 * (poly[0] + poly[1])
        bm = 0.5 * (poly[2] + poly[3])
        hgt = float(np.linalg.norm(tm - bm))
        wid = float(np.linalg.norm(poly[0] - poly[1]))
        for k in range(1, n2):
            c = cen2[k]
            a = float(st2[k, cv2.CC_STAT_AREA])
            ww, hh = float(st2[k, cv2.CC_STAT_WIDTH]), float(st2[k, cv2.CC_STAT_HEIGHT])
            near_top = float(np.dot(c - tm, (bm - tm) / max(hgt, 1e-3))) < 0.22 * hgt
            mid = abs(float(np.dot(c - tm, (poly[1] - poly[0]) / max(wid, 1e-3)))) < 0.12 * wid
            # (v11.2.112: and filled - a navel is a solid dark spot; a V of tattoo strokes at the band's top middle is not)
            roundish = 0.45 < ww / max(hh, 1.0) < 2.2 and a < (0.10 * wid) ** 2 and a >= 0.40 * ww * hh
            if near_top and mid and roundish:
                marks[lab2 == k] = False
                info["navel"] = True
    if marks.any():                                         # (grown into skin to cover soft ink edges, never into clothing)
        d = int(CFG["mark_dilate"])
        marks = (cv2.dilate(marks.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))) > 0) & band & ~clothing
    info["marks_px"] = int(marks.sum())
    info["pieces"] = int(cv2.connectedComponents(marks.astype(np.uint8))[0] - 1) if marks.any() else 0
    return marks, info


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
