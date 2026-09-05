"""
Swamitech v11 "Aequus" — aligned-space swap engine.

WHY THIS MODULE EXISTS
======================
Every visual defect Swamitech has fought for months (flicker on profile turns,
the brown patch, waxy/flat skin, face-1/face-2 mix-ups during a hug) traces back
to two structural decisions in the older pipeline:

  1. Masking and colour matching were done in IMAGE SPACE using the detector's
     axis-aligned bounding box. A bbox is pose-dependent: it balloons when the
     head rolls, and it contains hair / neck / background / the *other* person.
     Statistics computed over it are contaminated, and a mask derived from it
     changes shape every frame -> visible flicker and colour casts.

  2. Whenever anything was uncertain (detector skipped, low confidence, faces
     overlapping) the pipeline fell back to the ORIGINAL frame. A single
     original frame between swapped frames is the most visible artefact a face
     swap can produce. Reverting is *worse* than a slightly imperfect swap.

This module fixes both. All compositing happens in the 128x128 ArcFace-aligned
crop that inswapper already produces internally. That crop is pose-normalised by
construction, so:

  * a mask defined there has stable geometry regardless of yaw/roll;
  * an EMA over that mask is meaningful (you are averaging the same anatomy);
  * colour statistics gathered there, inside the mask, never see background.

The module depends only on numpy + cv2 so it can be unit-tested without
insightface, onnxruntime or gradio present.

Public API
----------
    AlignedCompositor          - the swap+blend core
    TrackState                 - per-identity temporal memory
    MultiFaceTracker           - occlusion-aware slot assignment
    PredictedFace              - lightweight face stand-in for skipped frames
    swap_and_composite(...)    - convenience wrapper used by core_pipeline
"""

from __future__ import annotations

import logging
import math
from itertools import permutations

import cv2
import numpy as np

__all__ = [
    "AlignedCompositor",
    "estimate_norm",
    "landmark_fit_error",
    "ARCFACE_DST",
    "TrackState",
    "MultiFaceTracker",
    "PredictedFace",
    "swap_and_composite",
    "ENGINE_VERSION",
]

ENGINE_VERSION = "aequus-1.1.4-continuous"

log = logging.getLogger("swamitech.engine")


# --------------------------------------------------------------------------
# Tunables. Overridable from config.py via configure().
# --------------------------------------------------------------------------
_P = {
    # canonical mask shape (fractions of the aligned crop)
    "mask_cx": 0.500,
    "mask_cy": 0.545,
    "mask_rx": 0.435,
    "mask_ry": 0.495,
    "mask_power": 2.55,      # superellipse exponent; >2 = squarer, <2 = pointier
    "mask_feather": 0.16,    # gaussian sigma as fraction of crop size
    # Force the template to zero at the crop border. The superellipse is
    # centred at cy=0.545 with ry=0.495, so it reaches 1.04 - it runs off the
    # bottom of the crop and was 0.881 there (top 0.250, sides 0.111). That is
    # a HARD EDGE in the composite, not a feathered one. It normally hides
    # because the chin edge lands on a neck, but it is what turns any
    # mis-placed paste into a visible straight-edged rectangle. The chin sits
    # at about y=0.875 in ArcFace-128 space, so an 0.08 rolloff clears it.
    "mask_border": 0.08,
    # landmark-hull refinement
    "hull_dilate": 0.085,    # dilate hull by this fraction of crop size
    "hull_floor": 0.30,      # hull never removes more than (1-floor) of template
    "mask_ema": 0.36,        # SoftStable: match config (0.36)
    # colour transfer (SoftStable brightness-only)
    "cm_strength_ab": 0.85,  # chroma follows the target scene strongly
    "cm_strength_l": 0.45,   # SoftStable: was 0.55
    "cm_std_lo": 0.72,       # contrast ratio clamp - never crush face contrast
    "cm_std_hi": 1.45,
    "cm_ema": 0.18,          # SoftStable: was 0.28
    "cm_max_shift": 20.0,    # SoftStable: was 26
    "cm_delta_clamp": 5.0,   # SoftStable: max |dmean| step vs prior smoothed
    # occlusion guard (off by default)
    "occl_min_keep": 0.35,
    # Occlusion strength is a CONTINUOUS weight in [0,1], not a boolean.
    # A binary guard changed the mask area by several percent from one frame
    # to the next every time it toggled, and the mask is what the colour
    # statistics are weighted by - so a toggle moved both the silhouette and
    # the brightness at once. The guard now ramps over occl_ramp frames.
    "occl_ramp": 0.25,
    # Rival-face subtraction. During a kiss/hug the other person's cheek lands
    # inside this face's aligned crop with near-identical chroma, so
    # skin_confidence() cannot see it. Their landmark hull can be projected in
    # geometrically, which can.
    "rival_cut": 0.85,       # how hard a rival hull is removed (0 = off)
    "rival_feather": 0.09,   # softness of that cut, fraction of crop size
    # Hull temporal stability. The 106-point hull is the only pose-DEPENDENT
    # term in an otherwise pose-normalised mask, so it is what makes the
    # silhouette breathe on a yaw turn. Smooth it on its own, slower clock.
    "hull_ema": 0.22,
    # tracker
    "trk_w_id": 0.52,
    "trk_w_iou": 0.33,
    "trk_w_dist": 0.15,
    "trk_gate_new": 0.30,    # identity sim needed to CREATE a slot binding
    "trk_gate_hold": 0.10,   # identity sim needed to KEEP an established one
    "trk_lock_hits": 4,      # frames before a track is "established"
    "trk_cross_iou": 0.12,   # tracks this close are considered "crossing"
    "trk_flip_margin": 0.14, # identity margin needed to justify a label flip
    "trk_flip_frames": 5,    # ...sustained for this many frames
    "trk_max_missed": 24,  # aligned with config ENGINE_TUNABLES / predicted hold budget
    "trk_alpha": 0.42,       # bbox smoothing when updating from a detection
    # Landmark (kps) smoothing: One Euro filter, adaptive to motion speed.
    # The existing trk_alpha above is a FIXED-rate EMA - same smoothing
    # strength whether the face is still or moving fast. That is the known
    # limitation the One Euro filter (Casiez/Roussel/Vogel 2012) exists to
    # fix, and it is what MediaPipe's own production face landmarker uses
    # internally for exactly this reason.
    #
    # Values below were swept empirically (test_smoothing.py), not just
    # solved theoretically: 0.1153 is the min_cutoff that reproduces
    # trk_alpha's exact behaviour at rest, but 0.06 measured BETTER on both
    # axes at once - still-phase jitter slightly below the old fixed EMA
    # (not just similar), while fast-motion lag dropped 78.7% (33px -> 7px
    # in the simulated head-turn). kps_beta controls how fast smoothing
    # relaxes as measured per-frame speed increases - this is what targets
    # flicker specifically during fast movement without under-smoothing a
    # still or slow-moving face.
    "kps_min_cutoff": 0.06,
    "kps_beta": 0.015,
}


def configure(**kw):
    """Override tunables (called once from core_pipeline with config values)."""
    for k, v in kw.items():
        if k in _P and v is not None:
            _P[k] = v


# --------------------------------------------------------------------------
# Canonical mask construction (cached — shape depends only on crop size)
# --------------------------------------------------------------------------
_TEMPLATE_CACHE: dict[int, np.ndarray] = {}


def canonical_template(size: int) -> np.ndarray:
    """Soft superelliptical face template in aligned space, float32 in [0,1].

    The ArcFace 5-point similarity transform puts eyes/nose/mouth at fixed
    canonical positions, so this one shape fits every pose. That is precisely
    what makes it flicker-free: nothing about it varies frame to frame.
    """
    cached = _TEMPLATE_CACHE.get(size)
    if cached is not None:
        return cached

    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    xs = (xs + 0.5) / size
    ys = (ys + 0.5) / size
    dx = np.abs(xs - _P["mask_cx"]) / _P["mask_rx"]
    dy = np.abs(ys - _P["mask_cy"]) / _P["mask_ry"]
    n = float(_P["mask_power"])
    r = np.power(np.power(dx, n) + np.power(dy, n), 1.0 / n)

    # soft rolloff: 1 inside, 0 outside, smooth band in between
    mask = np.clip((1.12 - r) / 0.28, 0.0, 1.0).astype(np.float32)

    # ...and a second rolloff that guarantees the template reaches zero at the
    # crop border, so the paste never has a hard edge to give itself away.
    b = float(_P.get("mask_border", 0.08) or 0.0)
    edge_roll = None
    if b > 0.0:
        edge = np.minimum(np.minimum(xs, 1.0 - xs), np.minimum(ys, 1.0 - ys))
        edge_roll = np.clip(edge / b, 0.0, 1.0).astype(np.float32)
        mask *= edge_roll

    k = int(max(3, round(size * _P["mask_feather"]))) | 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    if edge_roll is not None:
        # Applied again AFTER the feather: the Gaussian has a ~21 px kernel and
        # smears interior weight back out to the border, which left 0.296 there
        # on the first pass. Re-applying pins the border to exactly zero.
        mask *= edge_roll
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    _TEMPLATE_CACHE[size] = mask
    return mask


# --------------------------------------------------------------------------
# ArcFace 5-point alignment.
#
# inswapper builds its own 128x128 aligned crop from the 5 keypoints and hands
# back the affine it used. That affine is only available on frames where the
# ONNX forward pass actually ran. To composite a CACHED aligned result onto a
# LATER frame - which is what makes every output frame a real composite rather
# than a stale ROI pasted from another moment in time - the same affine has to
# be derivable from keypoints alone. This is that derivation: the canonical
# ArcFace destination points plus a Umeyama similarity fit, i.e. exactly what
# insightface does internally, reimplemented here so the module keeps its
# numpy+cv2-only dependency footprint and stays unit-testable.
#
# Any residual disagreement with the installed insightface build is cancelled
# out at runtime by aligned_correction() below, so this never has to match
# bit-for-bit to be safe.
# --------------------------------------------------------------------------
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _umeyama(src: np.ndarray, dst: np.ndarray):
    """Least-squares similarity (scale+rotation+translation) src -> dst."""
    src = np.asarray(src, np.float64).reshape(-1, 2)
    dst = np.asarray(dst, np.float64).reshape(-1, 2)
    num = src.shape[0]
    if num < 2:
        return None
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_d = src - src_mean
    dst_d = dst - dst_mean
    A = (dst_d.T @ src_d) / num
    d = np.ones(2, np.float64)
    if np.linalg.det(A) < 0:
        d[1] = -1.0
    try:
        U, S, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return None
    if rank == 1:
        if np.linalg.det(U) * np.linalg.det(Vt) > 0:
            R = U @ Vt
        else:
            keep = d[1]
            d[1] = -1.0
            R = U @ np.diag(d) @ Vt
            d[1] = keep
    else:
        R = U @ np.diag(d) @ Vt
    var_src = src_d.var(axis=0).sum()
    scale = 1.0 if var_src <= 1e-12 else float((S @ d) / var_src)
    M = np.zeros((2, 3), np.float32)
    M[:, :2] = (scale * R).astype(np.float32)
    M[:, 2] = (dst_mean - scale * (R @ src_mean)).astype(np.float32)
    return M


def _arcface_dst(image_size: int):
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    return ARCFACE_DST * ratio + np.array([diff_x, 0.0], np.float32)


def estimate_norm(kps, image_size: int = 128):
    """Image-space -> aligned-crop affine for 5-point ArcFace keypoints."""
    try:
        lmk = np.asarray(kps, np.float32).reshape(-1, 2)
    except Exception:
        return None
    if lmk.shape[0] < 5:
        return None
    lmk = lmk[:5]
    dst = _arcface_dst(image_size)
    return _umeyama(lmk, dst)


def landmark_fit_error(kps, image_size: int = 128):
    """0..~1+ : how badly the 5 keypoints disagree with ANY single rigid pose.

    A real face's 5 landmarks - however extreme the pose - come from one rigid
    structure, so a similarity transform (rotation+scale+translation) can
    always be found that lands them close to the ArcFace canonical template.
    When a detector's landmark regression is unreliable - pushed past where it
    was trained by an extreme yaw/pitch, an eye guessed from hair, a mouth
    placed by the shape prior rather than the image - the 5 points stop being
    mutually consistent with any single pose, and the BEST-FIT residual spikes
    even though det_score and each individual coordinate can look
    unremarkable in isolation. That is exactly the failure this measures, and
    exactly what malforms compositing: fitting an affine to inconsistent
    points does not fail loudly, it silently returns a plausible-looking M
    that does not match the real head, so the aligned crop reprojects onto
    the wrong place at the wrong rotation and scale - a rotated, misplaced
    rectangle is the visible result.

    Unlike _frontal_score/_pitch_score (heuristics on where individual points
    sit), this is a direct geometric consistency check, so it also catches
    configurations that do not trip either heuristic's simple thresholds.

    Returns None if a transform cannot even be attempted (e.g. degenerate
    points that make the source covariance singular) - that is itself a
    reliability failure and should be treated as maximally unreliable by the
    caller.
    """
    try:
        lmk = np.asarray(kps, np.float32).reshape(-1, 2)[:5]
    except Exception:
        return None
    if lmk.shape[0] < 5:
        return None
    dst = _arcface_dst(image_size)
    M = _umeyama(lmk, dst)
    if M is None:
        return None
    try:
        proj = lmk @ M[:, :2].T + M[:, 2]
        err = np.linalg.norm(proj - dst, axis=1)
        eye_dist = float(np.linalg.norm(dst[0] - dst[1])) + 1e-6
        return float(np.mean(err) / eye_dist)
    except Exception:
        return None


def _as3x3(M):
    T = np.eye(3, dtype=np.float32)
    T[:2, :] = np.asarray(M, np.float32).reshape(2, 3)
    return T


def aligned_correction(kps, M_actual, image_size: int):
    """Aligned-space correction C with  C @ estimate_norm(kps) == M_actual.

    Guarantees the reprojected crop lines up exactly with whatever affine the
    installed inswapper build actually used, so a cached aligned result can be
    re-pasted on a later frame with no seam at the hand-over.
    """
    if M_actual is None:
        return None
    est = estimate_norm(kps, image_size)
    if est is None:
        return None
    try:
        C = _as3x3(M_actual) @ np.linalg.inv(_as3x3(est))
        return C[:2, :].astype(np.float32)
    except Exception:
        return None


def apply_correction(C, M):
    if C is None:
        return M
    try:
        return (_as3x3(C) @ _as3x3(M))[:2, :].astype(np.float32)
    except Exception:
        return M


def _landmarks_to_aligned(lmk, M, size):
    """Project image-space landmarks into the aligned crop using affine M."""
    if lmk is None:
        return None
    try:
        pts = np.asarray(lmk, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 8:
            return None
        A = np.asarray(M, dtype=np.float32).reshape(2, 3)
        out = pts @ A[:, :2].T + A[:, 2]
        # keep only points that landed near the crop
        ok = (out[:, 0] > -size * 0.35) & (out[:, 0] < size * 1.35) & \
             (out[:, 1] > -size * 0.35) & (out[:, 1] < size * 1.35)
        out = out[ok]
        return out if len(out) >= 8 else None
    except Exception:
        return None


def hull_mask(pts_aligned, size) -> np.ndarray | None:
    """Convex hull of aligned landmarks, dilated + feathered, float32 [0,1]."""
    if pts_aligned is None:
        return None
    try:
        m = np.zeros((size, size), np.uint8)
        hull = cv2.convexHull(np.round(pts_aligned).astype(np.int32))
        cv2.fillConvexPoly(m, hull, 255)
        d = int(max(1, round(size * _P["hull_dilate"])))
        m = cv2.dilate(m, np.ones((d * 2 + 1, d * 2 + 1), np.uint8), 1)
        k = int(max(3, round(size * 0.10))) | 1
        m = cv2.GaussianBlur(m, (k, k), 0)
        return (m.astype(np.float32) / 255.0)
    except Exception:
        return None


# --------------------------------------------------------------------------
# Occlusion guard (optional): drop clearly non-skin pixels from the mask.
# --------------------------------------------------------------------------
def skin_confidence(aligned_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-pixel [0,1] confidence that a pixel belongs to the tracked face.

    Uses the face's OWN median chroma (not a fixed skin model), so it works
    across skin tones. Intended to suppress a hand, a microphone or another
    person's cheek that intrudes into the aligned crop during a hug.
    """
    try:
        full = aligned_bgr.shape[0]
        # The result is Gaussian-blurred to a twelfth of the crop before use, so
        # gathering it at half scale and scaling back changes it by at most
        # ~0.08 (mean 0.014) while costing 45% less.
        if full >= 96:
            small = cv2.resize(aligned_bgr, (full // 2, full // 2), interpolation=cv2.INTER_AREA)
            m_small = cv2.resize(mask, (full // 2, full // 2), interpolation=cv2.INTER_AREA)
            conf = skin_confidence(small, m_small)
            return cv2.resize(conf, (full, full), interpolation=cv2.INTER_LINEAR)
        ycc = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        cr, cb = ycc[:, :, 1], ycc[:, :, 2]
        core = mask > 0.75
        if int(core.sum()) < 64:
            return np.ones(mask.shape, np.float32)
        mcr, mcb = float(np.median(cr[core])), float(np.median(cb[core]))
        scr = float(np.std(cr[core])) + 3.0
        scb = float(np.std(cb[core])) + 3.0
        d = np.sqrt(((cr - mcr) / scr) ** 2 + ((cb - mcb) / scb) ** 2)
        conf = np.clip(1.0 - (d - 2.2) / 2.6, 0.0, 1.0).astype(np.float32)
        k = max(3, (aligned_bgr.shape[0] // 12) | 1)
        conf = cv2.GaussianBlur(conf, (k, k), 0)
        floor = float(_P["occl_min_keep"])
        return (floor + (1.0 - floor) * conf).astype(np.float32)
    except Exception:
        return np.ones(mask.shape, np.float32)


# --------------------------------------------------------------------------
# Per-identity temporal memory
# --------------------------------------------------------------------------
class TrackState:
    """Temporal memory for one identity slot.

    Holds everything that must NOT be recomputed independently per frame:
    the canonical mask, the colour correction, bbox/kps velocity, and the
    long-term identity embedding.
    """

    __slots__ = ("slot", "bbox", "obs_bbox", "kps", "lmk", "vel_bbox", "vel_kps",
                 "emb", "hits", "missed", "mask_ema", "cm_dmean", "cm_ratio",
                 "flip_votes", "crossing", "last_face", "det_score", "last_hit_bbox",
                 "hull_ema", "occl_w", "alpha_ema", "last_dt", "last_hit_kps",
                 "fake", "fake_corr", "fake_size", "fake_frame")

    def __init__(self, slot=0):
        self.slot = slot
        self.bbox = None
        self.obs_bbox = None      # last RAW detection (unsmoothed) — association
        self.kps = None
        self.lmk = None
        self.vel_bbox = np.zeros(4, np.float32)
        self.vel_kps = None
        self.emb = None
        self.hits = 0
        self.missed = 0
        self.mask_ema = None
        self.cm_dmean = None     # np.float32[3]  LAB mean delta
        self.cm_ratio = None     # np.float32[3]  LAB std ratio
        self.flip_votes = 0
        self.crossing = False
        self.last_face = None
        self.det_score = 0.0
        self.last_hit_bbox = None
        self.last_dt = 1.0        # output frames between the last two detections
        self.last_hit_kps = None  # raw keypoints at the last real detection
        self.hull_ema = None      # smoothed landmark hull (aligned space)
        self.occl_w = 0.0         # CONTINUOUS occlusion weight in [0,1]
        self.alpha_ema = 1.0      # smoothed composite opacity
        # Cached aligned swap result. Holding the 128x128 crop (not a
        # full-frame paste) is what lets a later frame be composited with its
        # OWN background, its OWN geometry and its OWN lighting while reusing
        # the expensive ONNX forward pass.
        self.fake = None
        self.fake_corr = None     # aligned-space correction for this build
        self.fake_size = 0
        self.fake_frame = -10 ** 9

    # -- geometry -------------------------------------------------------
    @property
    def established(self) -> bool:
        return self.hits >= int(_P["trk_lock_hits"])

    @property
    def match_bbox(self):
        """One-step-ahead prediction from the last RAW observation.

        Association must never use the EMA-smoothed bbox: smoothing lags, and
        when two people close on each other at different speeds the lagged
        estimates can cross over each other while the real faces have not.
        That lag was a direct cause of face-1/face-2 swapping in a hug.
        """
        base = self.obs_bbox if self.obs_bbox is not None else self.bbox
        if base is None:
            return None
        # Velocity is per OUTPUT frame, so the prediction has to span the
        # detector's actual interval - otherwise, at a cadence above 1, the
        # gate that decides who a detection belongs to is comparing against a
        # position the face left several frames ago.
        return (np.asarray(base, np.float32)
                + self.vel_bbox * float(max(1.0, self.last_dt))).astype(np.float32)

    def predict(self, n_frames: float = 1.0):
        """Advance geometry by ``n_frames`` output frames at constant velocity.

        ``n_frames`` exists because velocity is stored PER OUTPUT FRAME (see
        update()) while the caller may only get to advance a track once per
        detector interval. Assuming those are the same thing - which the
        previous signature forced - made the prediction lag by exactly the
        detection cadence, so the carried face trailed the real one during
        every fast movement and then snapped forward on the next detection.
        That snap is visible as a flick.
        """
        n = max(0.0, float(n_frames))
        if n <= 0.0:
            return self.bbox
        steps = max(1, int(round(n)))
        # Confidence in an extrapolation decays per FRAME advanced, so the
        # damping has to be summed over the frames being covered. Applying the
        # end-of-interval damping factor to the whole interval at once (n * damp)
        # under-advances a multi-frame catch-up badly - a 5-frame advance moved
        # about 2.2 frames' worth - which shows up as the carried face trailing
        # the real one and then snapping forward on the next detection.
        m0 = self.missed
        self.missed = m0 + steps
        step = float(sum(0.85 ** min(m0 + i, 12) for i in range(1, steps + 1)))
        if self.bbox is not None:
            self.bbox = (self.bbox + self.vel_bbox * step).astype(np.float32)
        if self.obs_bbox is not None:
            self.obs_bbox = (self.obs_bbox + self.vel_bbox * step).astype(np.float32)
        if self.kps is not None and self.vel_kps is not None:
            self.kps = (self.kps + self.vel_kps * step).astype(np.float32)
        return self.bbox

    def update(self, face, update_embedding=True, dt_frames: float = 1.0):
        a = float(_P["trk_alpha"])
        dt = max(1.0, float(dt_frames or 1.0))
        self.last_dt = dt
        # Frames since this track last had a REAL detection. predict() adds to
        # `missed`; a hit resets it. `dt` covers the interval that has not been
        # counted yet.
        elapsed = max(1.0, float(self.missed) + dt)
        bb = np.asarray(face.bbox, np.float32).reshape(4).copy()
        if self.last_hit_bbox is None:
            self.vel_bbox = np.zeros(4, np.float32)
        else:
            # Measure against the last OBSERVED box, not against obs_bbox -
            # predict() advances obs_bbox, so (bb - obs_bbox) is the prediction
            # RESIDUAL. Feeding a residual back as velocity means an accurate
            # prediction halves the velocity, and a few accurate predictions in
            # a row drive it to zero: the carried face stops moving mid-gap and
            # then jumps when the detector next reports. Dividing by the real
            # elapsed frames also keeps the unit at pixels per OUTPUT frame,
            # whatever the detector cadence is.
            inst = (bb - self.last_hit_bbox) / elapsed
            self.vel_bbox = (self.vel_bbox * 0.5 + inst * 0.5).astype(np.float32)
        self.obs_bbox = bb.copy()
        self.last_hit_bbox = bb.copy()
        if self.bbox is None:
            self.bbox = bb
        else:
            self.bbox = (self.bbox * (1.0 - a) + bb * a).astype(np.float32)

        kps = getattr(face, "kps", None)
        if kps is not None:
            kp = np.asarray(kps, np.float32).copy()
            if self.kps is not None and self.kps.shape == kp.shape:
                # One Euro filter, t_e = 1 frame. A raw per-frame speed
                # estimate (this frame's delta) drives the cutoff, not the
                # already-smoothed vel_kps - the filter needs to react to
                # how fast things are moving RIGHT NOW, not a lagged
                # estimate of that, or it would itself lag exactly when
                # responsiveness matters most.
                raw_speed = float(np.mean(np.abs(kp - self.kps))) / elapsed
                cutoff = float(_P["kps_min_cutoff"]) + float(_P["kps_beta"]) * raw_speed
                # t_e = elapsed frames, not a hard-coded 1. With a detector
                # cadence above 1 the old form under-smoothed by that factor.
                r = 2.0 * math.pi * cutoff * elapsed
                a = r / (r + 1.0)
                sm = self.kps * (1.0 - a) + kp * a
                # Same reasoning as vel_bbox: measure against the last observed
                # keypoints, never against the predicted ones.
                if self.last_hit_kps is not None and self.last_hit_kps.shape == kp.shape:
                    v = (kp - self.last_hit_kps) / elapsed
                    self.vel_kps = (v if self.vel_kps is None
                                    else self.vel_kps * 0.6 + v * 0.4)
                self.kps = sm.astype(np.float32)
            else:
                self.kps = kp
                self.vel_kps = np.zeros_like(kp)
            self.last_hit_kps = kp.copy()

        lmk = getattr(face, "landmark_2d_106", None)
        if lmk is not None:
            self.lmk = np.asarray(lmk, np.float32).copy()

        if update_embedding:
            e = getattr(face, "normed_embedding", None)
            if e is not None:
                e = np.asarray(e, np.float32)
                n = float(np.linalg.norm(e)) + 1e-6
                e = e / n
                if self.emb is None:
                    self.emb = e
                else:
                    mix = self.emb * 0.90 + e * 0.10
                    self.emb = mix / (float(np.linalg.norm(mix)) + 1e-6)

        self.det_score = float(getattr(face, "det_score", 0.5) or 0.5)
        self.last_face = face
        self.hits += 1
        self.missed = 0

    def predicted_face(self):
        """A stand-in face object usable by inswapper on a skipped frame."""
        if self.bbox is None or self.kps is None:
            return None
        return PredictedFace(self.bbox, self.kps, self.lmk,
                             getattr(self.last_face, "normed_embedding", None),
                             self.det_score)

    # -- appearance -----------------------------------------------------
    def smooth_hull(self, hull: np.ndarray | None):
        """EMA the landmark hull on its own, slower clock.

        The hull is the only POSE-DEPENDENT term in an otherwise
        pose-normalised mask, so it is the term that makes the silhouette
        breathe when the head yaws. Smoothing the composed mask (as before)
        could not separate this from legitimate scale changes; smoothing the
        hull itself can.
        """
        if hull is None:
            return self.hull_ema
        a = float(_P.get("hull_ema", 0.22) or 0.22)
        if a <= 0 or self.hull_ema is None or self.hull_ema.shape != hull.shape:
            self.hull_ema = hull.astype(np.float32).copy()
        else:
            self.hull_ema = (self.hull_ema * (1.0 - a) + hull * a).astype(np.float32)
        return self.hull_ema

    def ramp_occlusion(self, target: float) -> float:
        """Move the occlusion weight toward ``target`` at a bounded rate.

        A hard on/off guard moved the mask area by several percent between
        consecutive frames, and the mask is exactly what the colour statistics
        are weighted by - so one toggle shifted the silhouette AND the
        brightness at the same time. Ramping removes both steps.
        """
        r = float(_P.get("occl_ramp", 0.25) or 0.25)
        t = float(np.clip(target, 0.0, 1.0))
        self.occl_w = float(self.occl_w + (t - self.occl_w) * np.clip(r, 0.01, 1.0))
        return self.occl_w

    def smooth_alpha(self, target: float) -> float:
        """EMA the composite opacity.

        det_score wobbles by several hundredths between consecutive frames on
        a profile turn. Driving alpha straight from it made the replacement
        fade partly back toward the real face and out again, several times a
        second - seen as brightness flicker and as the original face
        'showing through'. Opacity now moves smoothly or not at all.
        """
        t = float(np.clip(target, 0.0, 1.0))
        self.alpha_ema = float(self.alpha_ema * 0.72 + t * 0.28)
        return self.alpha_ema

    def cache_fake(self, fake, corr, frame_ord):
        self.fake = fake
        self.fake_corr = corr
        self.fake_size = int(fake.shape[0]) if fake is not None else 0
        self.fake_frame = int(frame_ord)

    def smooth_mask(self, mask: np.ndarray) -> np.ndarray:
        a = float(_P["mask_ema"])
        if a <= 0 or self.mask_ema is None or self.mask_ema.shape != mask.shape:
            self.mask_ema = mask.astype(np.float32).copy()
        else:
            self.mask_ema = (self.mask_ema * (1.0 - a) + mask * a).astype(np.float32)
        return self.mask_ema

    def smooth_colour(self, dmean, ratio):
        """EMA colour correction with per-frame delta clamp (SoftStable).

        Clamp each channel's dmean to ±cm_delta_clamp from the previous
        *smoothed* value before EMA so lighting/pose jumps cannot pop
        brightness. SoftStable only — no hold-everywhere paste logic.
        """
        a = float(_P["cm_ema"])
        dmean = np.asarray(dmean, np.float32).reshape(-1).copy()
        ratio = np.asarray(ratio, np.float32).reshape(-1).copy()
        # NOTE: the previous version also re-seeded outright whenever
        # missed > 8. Dropping a converged correction and replacing it with a
        # single frame's raw measurement is a step change in face brightness,
        # and it fired precisely on the frames a long occlusion ended - which
        # is when a brightness pop is most visible. Only seed when there is
        # genuinely nothing to carry.
        if self.cm_dmean is None:
            self.cm_dmean = dmean.astype(np.float32).copy()
            self.cm_ratio = ratio.astype(np.float32).copy()
        else:
            max_step = float(_P.get("cm_delta_clamp", 5.0) or 5.0)
            prev = self.cm_dmean
            clamped = np.clip(dmean, prev - max_step, prev + max_step)
            self.cm_dmean = (prev * (1.0 - a) + clamped * a).astype(np.float32)
            self.cm_ratio = (self.cm_ratio * (1.0 - a) + ratio * a).astype(np.float32)
        return self.cm_dmean, self.cm_ratio

    def reset_appearance(self, hard: bool = False):
        """Let the appearance memory re-converge; do not delete it.

        This is called whenever the tracker merely SUSPECTS a relabel, which
        on a geometric test happens routinely during a crossing. Clearing the
        colour EMA outright produced a measured ~4x frame-to-frame luma step
        on the very next frame - a visible brightness pop in exactly the hug /
        kiss shots it was meant to protect. Halving the correction lets it
        re-converge over a few frames instead, which is invisible. ``hard``
        remains available for a genuine identity change.
        """
        self.mask_ema = None
        self.hull_ema = None
        self.fake = None
        self.fake_corr = None
        self.fake_size = 0
        if hard:
            self.cm_dmean = None
            self.cm_ratio = None
        else:
            if self.cm_dmean is not None:
                self.cm_dmean = (self.cm_dmean * 0.5).astype(np.float32)
            if self.cm_ratio is not None:
                self.cm_ratio = (1.0 + (self.cm_ratio - 1.0) * 0.5).astype(np.float32)


class PredictedFace:
    """Minimal duck-typed stand-in matching insightface's Face attributes."""

    # NOTE: `_slot` belongs here. Without it, a caller tagging a carried face
    # with its identity slot raised AttributeError - silently, because those
    # call sites are inside try/except - so a carried face reached the renderer
    # with no slot and was dropped. The effect was that key frames where the
    # detector was deliberately skipped contributed no geometry at all.
    __slots__ = ("bbox", "kps", "landmark_2d_106", "normed_embedding",
                 "det_score", "predicted", "_track", "_slot", "_occlusion_guard")

    def __init__(self, bbox, kps, lmk=None, emb=None, det_score=0.5):
        self.bbox = np.asarray(bbox, np.float32).reshape(4).copy()
        self.kps = np.asarray(kps, np.float32).copy()
        self.landmark_2d_106 = None if lmk is None else np.asarray(lmk, np.float32).copy()
        self.normed_embedding = emb
        self.det_score = float(det_score)
        self.predicted = True
        self._track = None
        self._slot = None
        self._occlusion_guard = 0.0   # continuous weight, not a flag


# --------------------------------------------------------------------------
# The compositor
# --------------------------------------------------------------------------
class AlignedCompositor:
    """Swap + blend entirely inside the ArcFace-aligned crop."""

    def __init__(self, swapper):
        self.swapper = swapper
        self._warned_no_M = False

    # -- inswapper interop ---------------------------------------------
    def _raw_swap(self, img, face, src_face):
        """Return (bgr_fake_128, M) or (None, None).

        insightface's INSwapper.get(..., paste_back=False) returns
        (bgr_fake, M). Older/forked builds may return only bgr_fake; we detect
        that and fall back to the paste_back path in the caller.
        """
        try:
            out = self.swapper.get(img, face, src_face, paste_back=False)
        except Exception as e:  # pragma: no cover - depends on runtime build
            log.debug("paste_back=False unsupported (%s)", e)
            return None, None
        if isinstance(out, (tuple, list)) and len(out) == 2:
            fake, M = out
            if fake is not None and M is not None:
                return np.asarray(fake), np.asarray(M, np.float32).reshape(2, 3)
        if not self._warned_no_M:
            log.warning("inswapper did not return an affine matrix; "
                        "falling back to legacy paste-back path")
            self._warned_no_M = True
        return None, None

    # -- mask ------------------------------------------------------------
    def build_mask(self, face, M, size, track=None, occlusion_guard=0.0,
                   aligned_target=None, rivals=None):
        """Face mask in aligned space.

        ``occlusion_guard`` is a CONTINUOUS weight in [0,1], not a flag: a
        boolean guard changed the mask silhouette (and therefore the colour
        statistics weighted by it) in a single frame every time it toggled.

        ``rivals`` are other faces' image-space landmark sets belonging to
        people who are IN FRONT of this one. Two faces in contact have
        essentially identical chroma, so skin_confidence() is blind to a cheek
        pressed against this face - but the rival's own landmarks say exactly
        where it is, and they project into this crop through the same affine.
        """
        mask = canonical_template(size).copy()

        pts = _landmarks_to_aligned(getattr(face, "landmark_2d_106", None), M, size)
        h = hull_mask(pts, size)
        if track is not None:
            h = track.smooth_hull(h)
        if h is not None:
            floor = float(_P["hull_floor"])
            mask = mask * (floor + (1.0 - floor) * h)

        if rivals:
            cut = float(_P.get("rival_cut", 0.85) or 0.0)
            if cut > 0.0:
                block = np.zeros((size, size), np.float32)
                for rl in rivals:
                    rp = _landmarks_to_aligned(rl, M, size)
                    rh = hull_mask(rp, size)
                    if rh is not None:
                        np.maximum(block, rh, out=block)
                if float(block.max()) > 0.02:
                    k = int(max(3, round(size * float(_P.get("rival_feather", 0.09))))) | 1
                    block = cv2.GaussianBlur(block, (k, k), 0)
                    mask = mask * (1.0 - cut * np.clip(block, 0.0, 1.0))

        g = float(np.clip(occlusion_guard, 0.0, 1.0))
        if g > 0.01 and aligned_target is not None:
            conf = skin_confidence(aligned_target, mask)
            mask = mask * (1.0 - g + g * conf)

        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        if track is not None:
            mask = track.smooth_mask(mask)
        return mask

    # -- colour ----------------------------------------------------------
    @staticmethod
    def _masked_stats(lab: np.ndarray, w: np.ndarray):
        """Mask-weighted per-channel mean and std.

        Deliberately a per-channel loop over contiguous 2D slices. A "vectorised"
        version that broadcasts the weights over all three channels at once was
        measured 5x SLOWER (1.69 ms vs 0.34 ms): it allocates two full
        HxWx3 float arrays and reduces over a non-contiguous axis, whereas each
        2D slice here stays in cache.
        """
        wsum = float(w.sum())
        if wsum < 32.0:
            return None, None
        mean = np.empty(3, np.float32)
        std = np.empty(3, np.float32)
        for c in range(3):
            ch = lab[:, :, c]
            m = float((ch * w).sum() / wsum)
            v = float((((ch - m) ** 2) * w).sum() / wsum)
            mean[c] = m
            std[c] = math.sqrt(max(v, 1e-6))
        return mean, std

    def colour_match(self, fake_bgr, target_bgr, mask, track=None, strength=1.0):
        """Match the swap to the scene using FACE-ONLY statistics.

        The historical bug: statistics were taken over the whole detector bbox,
        which on a profile turn is mostly hair and background. Matching to that
        pushed the face toward the background colour -> the brown patch. Here
        every statistic is weighted by the face mask, inside a crop that
        contains nothing but the face.
        """
        if strength <= 0:
            return fake_bgr
        w = mask.astype(np.float32)
        if float(w.sum()) < 32.0:
            return fake_bgr

        f_lab = cv2.cvtColor(fake_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        t_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Statistics are gathered at FULL resolution on purpose. Gathering them
        # on a half-scale copy saves only ~0.1 ms and area-averaging destroys
        # variance: the measured per-channel std came out ~15 LAB levels low.
        # std drives `ratio`, the contrast-matching term - the exact quantity
        # whose mis-scaling produced the flat "brown patch" this engine was
        # written to fix. Not a safe place to trade accuracy for speed.
        fm, fs = self._masked_stats(f_lab, w)
        tm, ts = self._masked_stats(t_lab, w)
        if fm is None or tm is None:
            return fake_bgr

        dmean = np.clip(tm - fm, -_P["cm_max_shift"], _P["cm_max_shift"]).astype(np.float32)
        ratio = np.clip(ts / (fs + 1e-6), _P["cm_std_lo"], _P["cm_std_hi"]).astype(np.float32)

        if track is not None:
            dmean, ratio = track.smooth_colour(dmean, ratio)

        s = np.array([_P["cm_strength_l"], _P["cm_strength_ab"],
                      _P["cm_strength_ab"]], np.float32) * float(strength)
        eff_ratio = 1.0 + (ratio - 1.0) * s
        eff_dmean = dmean * s

        # In-place on f_lab — no full LAB duplicate (CPU bandwidth). Per-channel
        # for the same cache reason as _masked_stats: broadcasting this over all
        # three channels at once measured 2.6x slower (0.37 ms vs 0.14 ms).
        for c in range(3):
            f_lab[:, :, c] = (f_lab[:, :, c] - fm[c]) * eff_ratio[c] + fm[c] + eff_dmean[c]
        np.clip(f_lab, 0, 255, out=f_lab)
        return cv2.cvtColor(f_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # -- paste back --------------------------------------------------------
    @staticmethod
    def paste_back(img, fake_bgr, mask, M, alpha=1.0):
        """Warp the aligned result back using a face-sized destination ROI.

        The old implementation called ``warpAffine(..., (W, H))`` twice for
        every face swap. At 720p/1080p that means allocating and filling two
        full-resolution images even though the actual replacement occupies a
        small face ROI. The affine transform is identical; only the destination
        coordinate system is translated to the ROI. This preserves the visual
        result while removing most of the per-swap memory bandwidth.
        """
        H, W = img.shape[:2]
        IM = cv2.invertAffineTransform(np.asarray(M, np.float32).reshape(2, 3))

        # Transform the aligned-crop corners to obtain a conservative image-space
        # ROI. The aligned crop is only 128x128 for the standard inswapper model,
        # so this bounds the replacement without ever creating a W×H warp buffer.
        size_h, size_w = fake_bgr.shape[:2]
        corners = np.array([[[0.0, 0.0], [size_w - 1.0, 0.0],
                            [size_w - 1.0, size_h - 1.0], [0.0, size_h - 1.0]]],
                           dtype=np.float32)
        dst = cv2.transform(corners, IM)[0]
        x1 = max(0, int(np.floor(dst[:, 0].min())) - 2)
        y1 = max(0, int(np.floor(dst[:, 1].min())) - 2)
        x2 = min(W, int(np.ceil(dst[:, 0].max())) + 3)
        y2 = min(H, int(np.ceil(dst[:, 1].max())) + 3)
        if x2 <= x1 or y2 <= y1:
            return img

        rw, rh = x2 - x1, y2 - y1
        # cv2.warpAffine treats the supplied matrix as a source->destination
        # transform and internally inverts it. To make the destination origin
        # equal to (x1,y1), translate the source->destination matrix by -ROI
        # origin; this is equivalent to cropping the full-frame warp without
        # changing any sampled pixels.
        local_M = IM.copy()
        local_M[:, 2] -= np.array([float(x1), float(y1)], np.float32)

        # The two warps must keep DIFFERENT border modes and therefore cannot be
        # packed into one 4-channel call. The canonical template is 0.881 at the
        # bottom-centre edge of the aligned crop (the chin runs right off it), so
        # warping the face with BORDER_CONSTANT would blend black in under the
        # chin at ~0.9 alpha - a dark fringe exactly where a seam is most
        # visible. BORDER_REPLICATE on the face is load-bearing.
        warp_face = cv2.warpAffine(fake_bgr, local_M, (rw, rh),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        warp_mask = cv2.warpAffine(mask, local_M, (rw, rh),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if float(warp_mask.max()) * float(alpha) <= 0.004:
            return img

        out = img if img.flags.writeable else img.copy()
        roi = out[y1:y2, x1:x2]
        # 8-bit alpha composite through OpenCV's SIMD paths rather than
        # promoting the whole destination ROI to float32. The ROI is an order of
        # magnitude larger than the 128x128 aligned crop (typically ~290x290 at
        # 720p), so the float round-trip was the single most expensive step in
        # the per-frame composite: 1.44 ms of it, against 0.26 ms here.
        # Difference against the float path is at most 2/255 on a handful of
        # pixels - below the quantisation of the 8-bit output either way.
        m3 = cv2.cvtColor(cv2.convertScaleAbs(warp_mask, alpha=255.0 * float(alpha)),
                          cv2.COLOR_GRAY2BGR)
        cv2.add(cv2.multiply(warp_face, m3, scale=1.0 / 255.0),
                cv2.multiply(roi, cv2.bitwise_not(m3), scale=1.0 / 255.0),
                dst=roi)
        return out

    # -- composite tail (shared by the swap and reuse paths) ----------------
    def _composite(self, work, orig, face, fake, M, *, track, alpha,
                   colour_strength, occlusion_guard, rivals):
        size = int(fake.shape[0])
        aligned_target = cv2.warpAffine(orig, M, (size, size),
                                        borderMode=cv2.BORDER_REPLICATE)
        mask = self.build_mask(face, M, size, track=track,
                               occlusion_guard=occlusion_guard,
                               aligned_target=aligned_target,
                               rivals=rivals)
        toned = self.colour_match(fake, aligned_target, mask, track=track,
                                  strength=colour_strength)
        return self.paste_back(work, toned, mask, M, alpha=alpha)

    # -- one-shot -----------------------------------------------------------
    def run(self, work, orig, face, src_face, *, track=None, alpha=1.0,
            colour_strength=1.0, occlusion_guard=0.0, rivals=None,
            frame_ord=0, post=None):
        """Swap `face` in `work` with `src_face`. Returns (image, ok).

        The aligned result is cached on ``track`` so later frames can be
        composited from it without another ONNX forward pass - see
        ``reuse()``. ``post`` is an optional callable applied ONCE to the
        aligned crop (this is where a face enhancer belongs: enhancing the
        cached crop means every frame that reuses it is enhanced identically,
        instead of enhanced and unenhanced frames alternating at the swap
        cadence and pulsing).
        """
        fake, M = self._raw_swap(work, face, src_face)
        if fake is None:
            return work, False

        size = int(fake.shape[0])
        if post is not None:
            try:
                p = post(fake)
                if p is not None and p.shape == fake.shape:
                    fake = p
            except Exception as e:
                log.debug("aligned post-process skipped: %s", e)

        if track is not None:
            track.cache_fake(fake, aligned_correction(getattr(face, "kps", None), M, size),
                             frame_ord)

        out = self._composite(work, orig, face, fake, M, track=track, alpha=alpha,
                              colour_strength=colour_strength,
                              occlusion_guard=occlusion_guard, rivals=rivals)
        return out, True

    # -- reuse a cached aligned result on a later frame ---------------------
    def reuse(self, work, orig, kps, face, *, track, alpha=1.0,
              colour_strength=1.0, occlusion_guard=0.0, rivals=None):
        """Composite the cached aligned swap onto THIS frame's geometry.

        This is what replaces "paste the face ROI copied out of a different
        frame". The expensive part of a swap is the ONNX forward pass; the
        placement, the mask, the background and the colour match are cheap and
        are all recomputed here from the CURRENT frame. So a frame the swapper
        did not run on still gets its face in the right place, at the right
        angle, lit by its own scene - rather than a rectangle of some other
        moment stamped on top of it.
        """
        if track is None or track.fake is None:
            return work, False
        size = int(track.fake_size or track.fake.shape[0])
        M = estimate_norm(kps, size)
        if M is None:
            return work, False
        M = apply_correction(track.fake_corr, M)
        out = self._composite(work, orig, face, track.fake, M, track=track,
                              alpha=alpha, colour_strength=colour_strength,
                              occlusion_guard=occlusion_guard, rivals=rivals)
        return out, True


def swap_and_composite(swapper, work, orig, face, src_face, **kw):
    """Convenience wrapper. Returns (image, ok)."""
    return AlignedCompositor(swapper).run(work, orig, face, src_face, **kw)


# --------------------------------------------------------------------------
# Multi-face association
# --------------------------------------------------------------------------
def _iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    u = ua + ub - inter
    return float(inter / u) if u > 0 else 0.0


def _cdist_norm(a, b):
    if a is None or b is None:
        return 9.0
    ca = np.array([(a[0] + a[2]) * .5, (a[1] + a[3]) * .5], np.float32)
    cb = np.array([(b[0] + b[2]) * .5, (b[1] + b[3]) * .5], np.float32)
    diag = max(1.0, float((a[2] - a[0] + a[3] - a[1]) * 0.5))
    return float(np.linalg.norm(ca - cb) / diag)


def optimal_assignment(cost, forbidden=None):
    """Exact minimum-cost assignment for the tiny matrices we deal with.

    Still exhaustive - slots <= 4 - but over COLUMN choices only. The previous
    implementation looped over permutations of rows AND columns, which
    enumerates every assignment r! times over: at 4 slots x 12 detections that
    measured 209 ms of pure Python per frame, on a build whose whole point is
    CPU throughput. Iterating rows in fixed order and permuting only the
    columns visits each distinct assignment exactly once.

    Partial assignments (fewer pairs than min(n, m)) are still reachable:
    the search descends row by row and may leave a row unassigned, so a row
    with no admissible column no longer blocks the rows after it.
    """
    n = len(cost)
    if n == 0:
        return {}
    m = len(cost[0])
    if m == 0:
        return {}
    forbidden = forbidden or set()

    best = {"pairs": 0, "cost": float("inf"), "map": {}}
    cur = {}

    def rec(i, used, total, pairs):
        if i == n:
            if pairs > best["pairs"] or (pairs == best["pairs"] and total < best["cost"]):
                best["pairs"], best["cost"], best["map"] = pairs, total, dict(cur)
            return
        # The objective is lexicographic - most pairs first, then least cost -
        # so a branch may only be pruned on cost once it is established that it
        # cannot beat the incumbent on pair count either.
        reach = pairs + (n - i)
        if reach < best["pairs"]:
            return
        if reach == best["pairs"] and total >= best["cost"]:
            return
        for j in range(m):
            if j in used or (i, j) in forbidden:
                continue
            c = cost[i][j]
            if c >= 1e6:
                continue
            cur[i] = j
            used.add(j)
            rec(i + 1, used, total + c, pairs + 1)
            used.discard(j)
            cur.pop(i, None)
        rec(i + 1, used, total, pairs)     # leave row i unassigned

    rec(0, set(), 0.0, 0)
    return best["map"]


class MultiFaceTracker:
    """Occlusion-aware assignment of detections to replacement slots.

    Design notes
    ------------
    * Global optimum, not greedy — see optimal_assignment().
    * Hysteresis: an established slot keeps its detection unless a rival is
      better by `trk_flip_margin` for `trk_flip_frames` consecutive frames.
      Embedding noise on a profile turn lasts 1-3 frames, so it can no longer
      cause a label flip.
    * Crossing lock: while two tracks overlap (a hug), embeddings are frozen
      and label changes are blocked outright. ArcFace embeddings are least
      reliable exactly when two faces are cheek to cheek, so we stop believing
      them and trust motion instead.
    """

    def __init__(self, slots):
        self.tracks = {s: TrackState(s) for s in slots}

    # ------------------------------------------------------------------
    def _mark_crossings(self):
        ids = [s for s, t in self.tracks.items() if t.bbox is not None]
        for s in ids:
            self.tracks[s].crossing = False
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                ta, tb = self.tracks[a], self.tracks[b]
                if _iou(ta.match_bbox, tb.match_bbox) >= _P["trk_cross_iou"] or \
                   _cdist_norm(ta.match_bbox, tb.match_bbox) < 1.25:
                    ta.crossing = True
                    tb.crossing = True

    def _identity(self, face, track, ref):
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            return -1.0
        emb = np.asarray(emb, np.float32)
        # FIX: previously took max(ref_sim, track_sim) - but ref is the
        # REPLACEMENT photo's own embedding (confirmed unrelated to whoever
        # is actually in the video; nothing currently populates a genuine
        # per-slot reference), essentially noise, while track.emb is a real,
        # learned signal built up from this track's own accumulated
        # detections once established. Taking the max let an unrelated,
        # essentially random ref similarity spike above the reliable track
        # similarity on any given frame - worst exactly in the noisiest
        # moments (side poses, crossings), which is precisely when this
        # mattered most and likely a real contributor to both the
        # intermittent identity swaps and some of the side-pose flicker.
        # Once a track has learned its own embedding, trust that over the
        # unrelated reference; only fall back to ref before any real signal
        # exists (a track with no detections yet - the one-time bootstrap
        # case, which the positional fallback above already handles
        # separately, so this is now mostly a defensive fallback).
        if track is not None and track.emb is not None:
            return float(np.dot(emb, track.emb))
        if ref is not None:
            return float(np.dot(emb, np.asarray(ref, np.float32)))
        return -1.0

    def _belongs_to_other(self, slot, face):
        """True if this detection sits closer to a different established track.

        This is the flip detector. It compares the candidate against every
        track's one-step prediction, so it catches the case where slot #0 is
        about to take the detection that is plainly slot #1's face — the exact
        failure that shows up as the two faces trading places mid-hug.
        """
        tr = self.tracks[slot]
        own_box = tr.match_bbox
        if own_box is None:
            return False
        own = _iou(own_box, face.bbox)
        own_d = _cdist_norm(own_box, face.bbox)
        for s2, t2 in self.tracks.items():
            if s2 == slot or not t2.established or t2.match_bbox is None:
                continue
            o = _iou(t2.match_bbox, face.bbox)
            od = _cdist_norm(t2.match_bbox, face.bbox)
            if o > own + 0.12 or (od + 0.20 < own_d):
                return True
        return False

    # ------------------------------------------------------------------
    def assign(self, faces, refs, dt_frames: float = 1.0):
        """faces: list of detections. refs: {slot: reference embedding}.

        Returns {slot: face} for slots that got a detection this frame. Slots
        that missed are advanced by prediction, and the caller can still swap
        them via TrackState.predicted_face().
        """
        self._mark_crossings()
        slots = sorted(self.tracks.keys())
        if not faces:
            for t in self.tracks.values():
                t.predict(dt_frames)
            return {}

        n, m = len(slots), len(faces)
        cost = [[1e9] * m for _ in range(n)]
        sim_tab = [[-1.0] * m for _ in range(n)]

        for i, s in enumerate(slots):
            tr = self.tracks[s]
            pred = tr.match_bbox
            for j, f in enumerate(faces):
                sim = self._identity(f, tr, refs.get(s))
                sim_tab[i][j] = sim
                iou = _iou(pred, f.bbox) if pred is not None else 0.0
                dist = _cdist_norm(pred, f.bbox) if pred is not None else 9.0
                spatial = max(iou, max(0.0, 1.0 - dist * 0.7))

                gate = _P["trk_gate_hold"] if tr.established else _P["trk_gate_new"]
                if tr.crossing and tr.established:
                    gate = -1.0          # trust motion, not embeddings
                if sim < gate and not (tr.established and spatial >= 0.55):
                    continue

                w_id, w_iou, w_d = _P["trk_w_id"], _P["trk_w_iou"], _P["trk_w_dist"]
                if tr.crossing and tr.established:
                    w_id, w_iou, w_d = 0.18, 0.55, 0.27
                cost[i][j] = (w_id * (1.0 - max(sim, -0.2)) +
                              w_iou * (1.0 - iou) +
                              w_d * min(1.0, dist / 3.0))

        pairing = optimal_assignment(cost)

        # ---- positional fallback for never-established slots --------------
        # _identity() above compares each detected face against refs.get(s) -
        # but refs is only ever populated by a caller that actually sends
        # per-slot reference images. Without one, it falls back to the
        # REPLACEMENT face's own embedding, compared against a video face
        # that has no reason to resemble it. Result: a brand-new slot can
        # never clear trk_gate_new, cost[i][j] stays at the 1e9 sentinel for
        # every candidate, and the slot is permanently unassigned for the
        # entire video - not a wrong pairing, no pairing at all.
        #
        # Researched the established convention for exactly this "no
        # reference available" case: production face-swap tools (a
        # documented Next Diffusion workflow; the face_analyser.py pattern
        # shared across multiple face-swap Spaces including deep-live-cam)
        # converge on left-to-right positional assignment as the standard
        # fallback - first replacement face maps to the leftmost detected
        # face, second to the next, and so on. This only ever fires for
        # slots that are BOTH unresolved by the identity/spatial cost above
        # AND never-established - an already-tracked slot always keeps using
        # the real spatial/identity continuity logic untouched. Once a slot
        # establishes this way, every later frame tracks it normally through
        # the same code path as before; this is purely a one-time bootstrap.
        unresolved = [i for i, s in enumerate(slots) if pairing.get(i) is None and not self.tracks[s].established]
        if unresolved:
            claimed = set(pairing.values())
            available = sorted(
                (j for j in range(m) if j not in claimed),
                key=lambda j: faces[j].bbox[0]
            )
            for i, j in zip(sorted(unresolved), available):
                pairing[i] = j

        # ---- hysteresis: veto flips that are not sustained ----------------
        # "Changed" is decided GEOMETRICALLY, never by python object identity:
        # detections are fresh objects every frame, so an identity test would
        # fire on literally every frame and permanently stall the tracker.
        result = {}
        for i, s in enumerate(slots):
            tr = self.tracks[s]
            j = pairing.get(i)
            if j is None:
                tr.predict(dt_frames)
                tr.flip_votes = max(0, tr.flip_votes - 1)
                continue
            f = faces[j]
            changed = self._belongs_to_other(s, f)

            if changed and tr.established:
                # The class docstring specifies that during a crossing "label
                # changes are blocked outright" - because this is exactly the
                # window where ArcFace embeddings are least reliable, so there
                # is no trustworthy basis for relabelling anyone. The previous
                # implementation only DELAYED the flip by trk_flip_frames (5)
                # frames: flip_votes ticked up every crossing frame and, once
                # it hit 5, the relabel went through anyway. Any real crossing
                # - two people walking past each other, a hug, one turning
                # across the other - lasts well over 5 frames, so in practice
                # the lock never held and the labels traded places mid-
                # crossing. Block outright while crossing and carry the
                # existing binding on motion, as designed; trk_max_missed
                # (24 frames) remains the safety valve if a "crossing" state
                # somehow persists, and it now preserves the learned
                # embedding so the track can re-acquire the RIGHT person by
                # identity afterwards instead of guessing.
                if tr.crossing:
                    tr.predict(dt_frames)
                    continue
                if sim_tab[i][j] < _P["trk_flip_margin"] + 0.30:
                    tr.flip_votes += 1
                    if tr.flip_votes < _P["trk_flip_frames"]:
                        # Keep the previous binding for now; geometry carries it.
                        tr.predict(dt_frames)
                        continue
                tr.flip_votes = 0
            else:
                tr.flip_votes = max(0, tr.flip_votes - 1)

            # Never poison the long-term identity with a profile/crossing frame.
            good = sim_tab[i][j] >= 0.26 and not tr.crossing

            # ROOT-CAUSE FIX for identities trading places mid-video.
            #
            # The >= 0.26 guard exists to stop a bad frame from POISONING an
            # established identity. But it is evaluated against sim_tab, and
            # sim_tab comes from _identity(), which needs tr.emb to produce a
            # meaningful number. Before a track has ever learned an embedding
            # there is nothing to compare against, so sim is either -1.0 (no
            # ref at all) or noise (a ref that is the unrelated replacement
            # photo). Either way it is far below 0.26, so update_embedding
            # stays False, so tr.emb is never set, so the next frame is in
            # exactly the same position. tr.emb stays None for the ENTIRE
            # video: a guard meant to protect an embedding was preventing the
            # embedding from ever existing.
            #
            # Consequence: multi-face tracking ran on pure geometry, with
            # zero identity signal. _belongs_to_other() is also purely
            # geometric (IoU + centre distance), so when two people cross or
            # pass close, nothing in the system can tell who is who - the
            # slots follow whichever box is nearest, trade places, and can
            # NEVER recover, because recovery would need the identity signal
            # that was never built. That is precisely "the faces interchanged
            # midway and stayed interchanged".
            #
            # This also explains why single-face jobs were always clean: that
            # path goes through bind(), whose update_embedding defaults to
            # True, so it seeds the embedding unconditionally and never hits
            # this trap. Only assign() - the multi-face path - is affected.
            #
            # Seeding when tr.emb is None is safe by construction: there is
            # no prior identity to corrupt. Still refuse to seed mid-crossing,
            # where a cheek-to-cheek embedding may blend two people - that is
            # the one case where a first impression really can be wrong.
            if tr.emb is None and not tr.crossing:
                good = True

            tr.update(f, update_embedding=good, dt_frames=dt_frames)
            if changed:
                tr.reset_appearance()
            result[s] = f

        for s in slots:
            if self.tracks[s].missed > _P["trk_max_missed"]:
                keep_emb = self.tracks[s].emb
                self.tracks[s] = TrackState(s)
                self.tracks[s].emb = keep_emb
        return result

    def bind(self, slot, face, update_embedding=True, dt_frames: float = 1.0):
        """Attach a detection to a slot without running association.

        Used by the single-face path, which has its own well-tested pairing
        logic (startup identity lock, reference embeddings). We still want the
        track's temporal memory — mask EMA, colour EMA, velocity for carry —
        so the appearance of a single-face job is stabilised the same way.
        """
        tr = self.tracks.get(slot)
        if tr is None:
            tr = self.tracks[slot] = TrackState(slot)
        tr.crossing = False
        tr.update(face, update_embedding=update_embedding, dt_frames=dt_frames)
        return tr

    # ------------------------------------------------------------------
    def carry(self, slots=None):
        """Predicted faces for slots with no detection this frame.

        This is what replaces "fall back to the original frame". A predicted
        face still has valid 5-point kps, which is all inswapper needs, so the
        swap keeps running through detector gaps instead of blinking.
        """
        out = {}
        for s, tr in self.tracks.items():
            if slots is not None and s not in slots:
                continue
            # NOTE: this used to also require ``tr.missed != 0``. That made
            # carry() return NOTHING on the single-face path, where a failed
            # pairing never advances the track - so the one code path that
            # exists to keep the swap alive through a bad frame was unreachable
            # exactly when it was called for, and the frame fell back to the
            # real face. A track with missed == 0 has CURRENT geometry, which
            # is the best case for carrying, not a reason to refuse.
            if not tr.established:
                continue
            if tr.missed > _P["trk_max_missed"]:
                continue
            pf = tr.predicted_face()
            if pf is not None:
                out[s] = pf
        return out

# ════════════════════════════════════════════════════════════════════════════════
# PHASE 1: OUT-OF-FRAME BOUNDARY VALIDATION (Non-intrusive Addition)
# ════════════════════════════════════════════════════════════════════════════════
# This section adds boundary confidence validation WITHOUT modifying existing code.
# It works with the existing TrackState, PredictedFace, and swap logic.

from enum import Enum

class TrackingState(Enum):
    """State machine for face tracking at frame boundaries."""
    ACTIVE = "active"
    EDGE_RISK = "edge_risk"
    LOST = "lost"
    SLEEPING = "sleeping"


class FaceTrackState:
    """Per-face state tracking for boundary validation."""
    __slots__ = ["track_id", "state", "confidence_history", "bbox_history", "frames_in_state"]
    
    def __init__(self, track_id: str):
        self.track_id = track_id
        self.state = TrackingState.ACTIVE
        self.confidence_history: list = []
        self.bbox_history: list = []
        self.frames_in_state = 0
    
    def update_state(self, confidence: float, bbox: tuple):
        """Update state based on sustained confidence levels."""
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > 30:
            self.confidence_history.pop(0)
        self.bbox_history.append(bbox)
        if len(self.bbox_history) > 5:
            self.bbox_history.pop(0)
        self.frames_in_state += 1
        
        # Hysteresis: require 3+ frames of sustained confidence for state change
        if len(self.confidence_history) >= 3:
            recent_conf = np.mean(self.confidence_history[-3:])
            threshold_high = 0.70  # Transition to EDGE_RISK
            threshold_low = 0.80   # Recover to ACTIVE
            
            if self.state == TrackingState.ACTIVE and recent_conf < threshold_high:
                self.state = TrackingState.EDGE_RISK
                self.frames_in_state = 0
            elif self.state == TrackingState.EDGE_RISK and recent_conf > threshold_low:
                self.state = TrackingState.ACTIVE
                self.frames_in_state = 0


def validate_detection_confidence(bbox, frame_shape, landmarks=None):
    """
    Calculate confidence penalty for faces at frame edges.
    Returns confidence multiplier [0.0, 1.0].
    """
    if bbox is None or frame_shape is None:
        return 1.0
    
    H, W = frame_shape[:2]
    if H <= 0 or W <= 0:
        return 1.0
    
    x1, y1, x2, y2 = bbox
    margin_edge = max(1, int(0.05 * min(W, H)))    # 5% margin
    margin_near = max(1, int(0.12 * min(W, H)))    # 12% margin
    
    penalty = 1.0
    
    # Horizontal edges
    if x1 < margin_edge or x2 > (W - margin_edge):
        penalty *= 0.50
    elif x1 < margin_near or x2 > (W - margin_near):
        penalty *= 0.60
    
    # Vertical edges
    if y1 < margin_edge or y2 > (H - margin_edge):
        penalty *= 0.50
    elif y1 < margin_near or y2 > (H - margin_near):
        penalty *= 0.60
    
    return float(np.clip(penalty, 0.0, 1.0))

