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

ENGINE_VERSION = "aequus-1.2.72-plate"

log = logging.getLogger("swamitech.engine")


def _fnum_local(x, default=0.5):
    if x is None:
        return float(default)
    try:
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return float(default)
            x = x.reshape(-1)[0]
        return float(x)
    except Exception:
        return float(default)


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
    "cm_strength_l": 0.42,   # CinemaQA: steadier than SoftStable 0.45
    "cm_std_lo": 0.72,       # contrast ratio clamp - never crush face contrast
    "cm_std_hi": 1.45,
    "cm_ema": 0.15,          # CinemaQA: steadier than SoftStable 0.18
    "cm_max_shift": 20.0,    # SoftStable: was 26
    "cm_delta_clamp": 4.0,   # CinemaQA: tighter than SoftStable 5.0
    # occlusion guard (off by default)
    "occl_min_keep": 0.45,  # SolidFace: keep center opaque under marginal occl_guard
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
    # --- plate matching (SDOS-071 F2/F3) ---------------------------------
    # A restored face is DENOISED and SHARP. The plate it is pasted into is
    # neither: it carries sensor grain, and during movement it carries motion
    # blur. A clean, sharp face inside a grainy, blurred frame is the single
    # most reliable "that is CGI" tell there is, and no amount of colour
    # matching hides it because it is a spatial-frequency mismatch, not a
    # tonal one. These two passes measure the plate and match it.
    # Hard ceiling on the retained restored crop. The adaptive chooser will
    # never go above the face's on-screen span anyway, but a 2-vCPU box
    # rendering close-ups pays about +30 ms per output frame at 512 and this
    # is the dial that buys that back. 128 reproduces the old behaviour
    # exactly; 256 keeps most of the visible gain for a third of the cost.
    "restore_crop_max": 512,
    "grain_strength": 0.90,  # fraction of the measured grain deficit restored
    "grain_max": 9.0,        # hard cap on added sigma (8-bit levels)
    "grain_chroma": 0.35,    # chroma grain relative to luma grain
    "grain_ema": 0.25,       # temporal smoothing of the sigma estimate
    "mblur_strength": 0.85,  # fraction of the measured sharpness excess removed
    "mblur_max": 0.055,      # blur length cap as a fraction of crop size
    "mblur_ema": 0.30,
    # How an ANISOTROPY deficit converts to kernel length. Measured on
    # line-blurred plates (perp/along Sobel energy under the face mask):
    # L=3 -> 1.42, L=5 -> 1.92, L=7 -> 2.38, L=11 -> 3.06, L=21 -> 4.18, i.e.
    # about 0.14 of anisotropy per pixel of kernel. 6.0 is the inverse of that
    # rounded DOWN on purpose: this pass should under-blur rather than over-,
    # because a face softer than its plate reads as out of focus while one
    # slightly sharper just reads as a good restoration.
    "mblur_k": 6.0,
    "rival_feather": 0.09,   # softness of that cut, fraction of crop size
    # Hull temporal stability. The 106-point hull is the only pose-DEPENDENT
    # term in an otherwise pose-normalised mask, so it is what makes the
    # silhouette breathe on a yaw turn. Smooth it on its own, slower clock.
    "hull_ema": 0.22,
    # tracker
    "trk_w_id": 0.52,
    "trk_w_iou": 0.33,
    "trk_w_dist": 0.15,
    "trk_gate_new": 0.12,    # virgin slots bind on spatial; identity locks after
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
    # --- alpha-beta motion filter (bbox + kps velocity) --------------------
    # Replaces the flat EMA over MEASURED velocity that this class used up to
    # v11.2.5. That EMA had no notion of acceleration or of its own prediction
    # being wrong: it averaged raw consecutive-detection deltas, so on a fast
    # head turn (accelerate - travel - decelerate) the velocity it handed to
    # predict() was always about one EMA time-constant stale, and on a
    # direction reversal it kept pointing the old way until enough new
    # measurements outvoted the old ones. With a sparse detector cadence
    # "enough measurements" is a large fraction of a second, which is exactly
    # the interval over which the carried face ends up pasted far from the
    # real one.
    #
    # An alpha-beta filter (Kalman with fixed gains; no covariance to carry,
    # so it costs nothing per frame) closes that loop: it PREDICTS where the
    # face should be, then splits the prediction error between a position and
    # a velocity correction. A prediction that comes back right leaves the
    # velocity alone - which is the correct response to constant motion, and
    # the thing the old EMA got wrong in the other direction (see the note at
    # the update site about never feeding a residual back AS velocity).
    #
    # ab_beta is set from ab_alpha by the Benedict-Bordner critically-damped
    # relation beta = a^2/(2-a) when left at 0 (the default), which is the
    # standard choice and keeps the filter from ringing after a turn.
    # ab_alpha 0.85 converges to a true constant velocity within two
    # detections; the old EMA needed four to get within 10%.
    "ab_alpha": 0.85,
    "ab_beta": 0.0,          # 0 = derive from ab_alpha (critically damped)
}


# How fast the landmark-size EMA follows a real change in apparent face size.
# Per DETECTION, not per frame, so the time constant does not move with the
# detector cadence. 0.20 is about a five-detection response - fast enough to
# follow a subject walking toward the camera, slow enough that a detector box
# wobbling frame to frame does not reach the paste.
_KPS_SCALE_EMA = 0.20


def _ab_beta() -> float:
    """Velocity gain for the alpha-beta filter, derived if not set."""
    b = float(_P.get("ab_beta", 0.0) or 0.0)
    if b > 0.0:
        return b
    a = float(_P.get("ab_alpha", 0.85) or 0.85)
    a = min(0.999, max(0.001, a))
    return (a * a) / (2.0 - a)


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
        # The jaw is not rigid. A wide-open mouth pulls the two mouth
        # corners far from the closed-mouth template and the whole face
        # was rejected, so the original showed. Eyes and nose stay put
        # on a real face and still fail when the read is hair or skull.
        err = err[:3]
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


def scale_affine(M, k):
    """Scale an image->crop affine so it targets a crop k times larger.

    The swapper hands back an affine that maps image space into ITS OWN crop
    (128px). When the restored crop is kept at a higher resolution, every
    later consumer - aligned_correction, the mask, paste_back - must be told
    about a crop that is k times bigger, or the face is pasted at 1/k size.
    Pre-multiplying by a uniform scale is exactly that statement.
    """
    if M is None or k == 1.0:
        return M
    try:
        S = np.array([[float(k), 0.0, 0.0],
                      [0.0, float(k), 0.0],
                      [0.0, 0.0, 1.0]], np.float32)
        return (S @ _as3x3(M))[:2, :].astype(np.float32)
    except Exception:
        return M


def crop_upscale_ok(old_size: int, new_size: int) -> bool:
    """May a restored crop of new_size be kept in place of one of old_size?

    Keeping the bigger crop only works if the ArcFace template itself scales
    by the same factor, because every later consumer recomputes
    estimate_norm(kps, new_size) and the correction must cancel exactly.
    _arcface_dst switches branch on image_size % 112, so 512 from 128 scales
    perfectly (ratio 4, diff_x 32 both ways) while 448 from 128 does NOT -
    it lands on the 112 branch with diff_x 0 and the face would be pasted
    offset by ~46px of crop space. Verify the identity rather than trusting
    the ratio; it costs two tiny array builds and it is the difference
    between a sharper face and a misplaced one.
    """
    try:
        old_size = int(old_size)
        new_size = int(new_size)
    except Exception:
        return False
    if new_size == old_size:
        return True
    if old_size <= 0 or new_size < old_size:
        return False
    k = float(new_size) / float(old_size)
    try:
        return bool(np.allclose(_arcface_dst(new_size),
                                _arcface_dst(old_size) * k, atol=1e-3))
    except Exception:
        return False


def native_crop_span(M, size: int) -> float:
    """How many IMAGE pixels the aligned crop of `size` actually covers.

    M maps image space into the crop, so its linear part's scale is
    crop-pixels-per-image-pixel. Inverting that gives the face's true
    on-screen span - the only honest ceiling on useful crop resolution,
    because paste_back resamples the crop down to exactly this many pixels.
    """
    try:
        A = np.asarray(M, np.float32).reshape(2, 3)[:, :2]
        s = math.sqrt(abs(float(np.linalg.det(A))))
        if not (s > 1e-6):
            return float(size)
        return float(size) / s
    except Exception:
        return float(size)


def choose_crop_size(M, base: int, max_size: int) -> int:
    """Smallest admissible crop size that does not throw plate detail away.

    Keeping every restored crop at the restorer's native 512 would be 16x the
    compositing work of 128 on EVERY output frame (measured 28.6 ms vs 6.85 ms
    at 1080p) and most of it wasted: paste_back immediately resamples the crop
    down to the face's on-screen span, so resolution above that span buys
    nothing visible. Resolution BELOW it is the real defect - that is the
    enhancer's output being crushed to 128 and then re-upsampled by the paste.
    So: pick the smallest size at or above the on-screen span, clamp to what
    the restorer gave us, and only ever use sizes whose ArcFace template
    scales exactly (crop_upscale_ok).
    """
    base = int(base)
    max_size = int(max_size)
    try:
        cap = int(_P.get("restore_crop_max", 512) or 0)
        if cap > 0:
            max_size = min(max_size, cap)
    except Exception:
        pass
    if max_size <= base:
        return base
    # Half a pixel of slack: the span comes out of a sqrt of a determinant,
    # so an exact 3x face measures 384.0000001 and without the slack it would
    # be rounded UP to the next size and cost 16x the compositing for nothing.
    native = native_crop_span(M, base) - 0.5
    if base >= native:
        return base
    best = base
    k = 2
    while base * k <= max_size:
        cand = base * k
        if crop_upscale_ok(base, cand):
            best = cand
            if cand >= native:
                break
        k += 1
    return best


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


def occluder_keep(aligned_bgr: np.ndarray) -> np.ndarray:
    """1 on the face, 0 only on a solid object in front of the mouth.

    The oval used to paint the new face over ice cream or food. Cutting every
    pixel that was slightly off skin colour also punched the lower lip and
    the chin, and that edge moved every frame. Only a large, clearly foreign
    patch in the mouth is removed. The lip and chin stay fully swapped.
    """
    try:
        h, w = aligned_bgr.shape[:2]
        ycc = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        yy, cr, cb = ycc[:, :, 0], ycc[:, :, 1], ycc[:, :, 2]
        row = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        col = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
        upper = (row < 0.42) & (yy > 40.0)
        if int(upper.sum()) < 80:
            return np.ones((h, w), np.float32)
        mcr = float(np.median(cr[upper]))
        mcb = float(np.median(cb[upper]))
        scr = float(np.std(cr[upper])) + 6.0
        scb = float(np.std(cb[upper])) + 6.0
        dist = np.sqrt(((cr - mcr) / scr) ** 2 + ((cb - mcb) / scb) ** 2)
        # Mouth cavity only. The lower lip is near y=0.72 and the chin
        # is below that; cutting either of them is what looked thin and shaky.
        mouth = (row >= 0.58) & (row <= 0.66) & (col >= 0.36) & (col <= 0.64)
        foreign = mouth & (yy >= 90.0) & (dist > 5.0)
        raw = foreign.astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
        keep_src = np.zeros((h, w), np.uint8)
        min_area = max(80, int(0.04 * h * w))
        for i in range(1, n):
            if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
                keep_src[labels == i] = 255
        if int(keep_src.max()) == 0:
            return np.ones((h, w), np.float32)
        k = max(3, (h // 18) | 1)
        soft = cv2.GaussianBlur(keep_src, (k, k), 0).astype(np.float32) / 255.0
        return np.clip(1.0 - soft, 0.0, 1.0).astype(np.float32)
    except Exception:
        return np.ones(aligned_bgr.shape[:2], np.float32)


def _bb_iou(a, b) -> float:
    """Axis-aligned IoU for two [x1,y1,x2,y2] boxes. 0 on any failure."""
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in np.asarray(a, np.float32).reshape(4)]
        bx1, by1, bx2, by2 = [float(v) for v in np.asarray(b, np.float32).reshape(4)]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            return 0.0
        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
        return float(inter / (area_a + area_b - inter + 1e-6))
    except Exception:
        return 0.0


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
                 "emb", "id_lock", "hits", "missed", "mask_ema", "cm_dmean", "cm_ratio",
                 "flip_votes", "crossing", "last_face", "det_score", "last_hit_bbox",
                 "hull_ema", "occl_w", "alpha_ema", "last_dt", "last_hit_kps",
                 "fake", "fake_corr", "fake_size", "fake_frame",
                 "_frame_wh", "_paste_frozen", "_reacquired",
                 "confirm_hits", "_confirm_ease", "ab_bbox", "ab_kps",
                 "skin_ref", "skin_hits", "kps_scale", "skin_miss",
                 "pose_ok", "occ_keep", "grain_ema", "mblur_ema")

    def __init__(self, slot=0):
        self.slot = slot
        self.bbox = None
        self.obs_bbox = None      # last RAW detection (unsmoothed) — association
        self.kps = None
        self.lmk = None
        self.vel_bbox = np.zeros(4, np.float32)
        self.vel_kps = None
        # Alpha-beta filter position states. Deliberately NOT the same thing
        # as bbox/kps (display-smoothed, and advanced with confidence damping
        # by predict()) nor as last_hit_bbox/last_hit_kps (raw measurements).
        # The filter needs its own undamped estimate to predict from, or the
        # residual it measures is contaminated by the display damping.
        self.ab_bbox = None
        self.ab_kps = None
        # This identity's own remembered face chroma (median Cr, Cb sampled at
        # its landmarks), learned only from frames where the face was actually
        # visible. skin_confidence() in this module answers a DIFFERENT
        # question: it takes its reference from the mask core of the crop it is
        # handed, so when an occluder covers that core the occluder's colour
        # BECOMES the reference and it confidently keeps the occluder while
        # trimming the real skin at the edges. It cannot detect occlusion of
        # the centre by construction. A reference carried across time can.
        self.skin_ref = None
        self.skin_hits = 0
        # Consecutive content-gate rejections. The gate learns its reference
        # ONLY from accepted frames, so without this the reference freezes the
        # moment it starts rejecting and can never recover. See _content_visible.
        self.skin_miss = 0
        # Did this identity's LAST accepted read describe a good pose? The
        # pose gate keeps painting through a marginal pose only for a head it
        # was already painting - see _kps_reliable's second threshold.
        self.pose_ok = False
        self.occ_keep = None
        # Plate-matching state. Both are per-frame MEASUREMENTS of the plate,
        # and a raw per-frame measurement jitters - which would make the grain
        # amplitude and the blur length crawl, a worse artefact than the one
        # being fixed. Smoothed, they track the shot instead of the frame.
        self.grain_ema = None
        self.mblur_ema = None
        # Slowly-smoothed SIZE of this identity's landmark constellation.
        # Position and size do not deserve the same filter; sharing one is
        # what makes a pasted face pulse. See the rescale site below.
        self.kps_scale = None
        self.emb = None
        # Frozen gallery identity. Set on the first confident bind and never
        # replaced by a different person (that was the male↔female re-entry
        # swap: a new face overwrote the slot, then the original person came
        # back looking "wrong" for their own slot).
        self.id_lock = None
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
        # Last known frame size (W, H) from core assign/bind — used to freeze
        # constant-velocity prediction once the face has mostly left the shot.
        self._frame_wh = None
        self._paste_frozen = False  # True: stop advancing; dead for paste
        self._reacquired = False    # set by update() for core geom-stub clear
        self.confirm_hits = 0       # v11.1.10: reliable hits needed after reacquire
        self._confirm_ease = 0      # v11.2.1: neutralized (was CinemaQA soft ease-in)

    # -- geometry -------------------------------------------------------
    @property
    def established(self) -> bool:
        return self.hits >= int(_P["trk_lock_hits"]) or self.id_lock is not None

    def _commit_identity(self, e):
        """Seed or refresh identity. Never replace a lock with a different person."""
        if e is None:
            return
        e = np.asarray(e, np.float32)
        n = float(np.linalg.norm(e)) + 1e-6
        e = e / n
        if self.id_lock is None:
            self.id_lock = e
            self.emb = e
            return
        sim = float(np.dot(self.id_lock, e))
        if sim < 0.38:
            return
        if self.emb is None:
            self.emb = e
        else:
            mix = self.emb * 0.90 + e * 0.10
            self.emb = mix / (float(np.linalg.norm(mix)) + 1e-6)
        if sim >= 0.55:
            mix2 = self.id_lock * 0.97 + e * 0.03
            self.id_lock = mix2 / (float(np.linalg.norm(mix2)) + 1e-6)

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

        v11.1.9 ReentrySafe: once the bbox is mostly outside the last known
        frame, stop advancing and mark paste-dead. Constant-velocity carry
        while off-screen compounds into empty-space / body paste on return.
        """
        n = max(0.0, float(n_frames))
        if n <= 0.0:
            return self.bbox
        steps = max(1, int(round(n)))
        m0 = self.missed
        self.missed = m0 + steps
        # Already left the frame — count the miss but do not keep coasting.
        if self._paste_frozen:
            return self.bbox
        # Confidence in an extrapolation decays per FRAME advanced, so the
        # damping has to be summed over the frames being covered. Applying the
        # end-of-interval damping factor to the whole interval at once (n * damp)
        # under-advances a multi-frame catch-up badly - a 5-frame advance moved
        # about 2.2 frames' worth - which shows up as the carried face trailing
        # the real one and then snapping forward on the next detection.
        # Fast motion must not be damped. 0.85 per frame made a 5-frame gap
        # travel only ~3 frames, so the paste trailed a quick head turn and
        # then snapped. Slow motion keeps the old damping so a still face
        # does not drift.
        speed = 0.0
        if self.vel_bbox is not None:
            speed = float(np.hypot(float(self.vel_bbox[0]), float(self.vel_bbox[1])))
        damp = 0.98 if speed > 3.5 else 0.85
        step = float(sum(damp ** min(m0 + i, 12) for i in range(1, steps + 1)))
        if self.bbox is not None:
            self.bbox = (self.bbox + self.vel_bbox * step).astype(np.float32)
        if self.obs_bbox is not None:
            self.obs_bbox = (self.obs_bbox + self.vel_bbox * step).astype(np.float32)
        if self.kps is not None and self.vel_kps is not None:
            self.kps = (self.kps + self.vel_kps * step).astype(np.float32)
        # Freeze when mostly outside the last known frame (set by core).
        if self._frame_wh is not None and self.bbox is not None:
            try:
                W, H = float(self._frame_wh[0]), float(self._frame_wh[1])
                x1, y1, x2, y2 = [float(v) for v in self.bbox]
                area = max(1.0, (x2 - x1) * (y2 - y1))
                ix = max(0.0, min(x2, W) - max(x1, 0.0))
                iy = max(0.0, min(y2, H) - max(y1, 0.0))
                contain = float(max(0.0, min(1.0, (ix * iy) / area)))
                if contain < 0.38:
                    self._paste_frozen = True
                    self.confirm_hits = 0  # v11.1.10: must re-confirm on return
            except Exception:
                pass
        return self.bbox

    def flow_correct(self, prev_bgr, curr_bgr) -> bool:
        """Move this face by what the pixels actually did since the last key frame.

        Constant-velocity coast lags a real acceleration: a quick nod or turn
        changes speed inside one detector gap, and the paste arrives late.
        Lucas-Kanade on the five landmarks is what live swap tools use for
        that gap. It is a few points, not a full-frame flow, so it stays cheap
        on CPU. A disagreed or low-confidence track returns False and the
        caller keeps the velocity coast.
        """
        if self._paste_frozen or self.kps is None or prev_bgr is None or curr_bgr is None:
            return False
        if prev_bgr.shape[:2] != curr_bgr.shape[:2]:
            return False
        pts = np.asarray(self.kps, np.float32).reshape(-1, 1, 2)
        if pts.shape[0] < 3:
            return False
        try:
            prev_g = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
            curr_g = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
            nxt, st, err = cv2.calcOpticalFlowPyrLK(
                prev_g, curr_g, pts, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
            )
        except Exception:
            return False
        if nxt is None or st is None or err is None:
            return False
        ok = (st.reshape(-1) == 1) & (err.reshape(-1) < 18.0)
        if int(np.count_nonzero(ok)) < 3:
            return False
        delta = (nxt.reshape(-1, 2) - pts.reshape(-1, 2))[ok]
        med = np.median(delta, axis=0).astype(np.float32)
        spread = float(np.median(np.linalg.norm(delta - med, axis=1)))
        if spread > 14.0 or float(np.hypot(med[0], med[1])) > 96.0:
            return False
        measured = nxt.reshape(-1, 2)
        original = pts.reshape(-1, 2)
        new_kps = original.copy()
        for i, good in enumerate(ok):
            new_kps[i] = measured[i] if good else original[i] + med
        self.kps = new_kps.astype(np.float32)
        slide = np.array([med[0], med[1], med[0], med[1]], np.float32)
        if self.bbox is not None:
            self.bbox = (np.asarray(self.bbox, np.float32) + slide).astype(np.float32)
        if self.obs_bbox is not None:
            self.obs_bbox = (np.asarray(self.obs_bbox, np.float32) + slide).astype(np.float32)
        dt = max(1.0, float(self.last_dt or 1.0))
        per = (med / dt).astype(np.float32)
        self.vel_kps = np.broadcast_to(per, self.kps.shape).copy()
        self.vel_bbox = np.array([per[0], per[1], per[0], per[1]], np.float32)
        return True

    def update(self, face, update_embedding=True, dt_frames: float = 1.0, kps_ok: bool = True):
        a = float(_P["trk_alpha"])
        dt = max(1.0, float(dt_frames or 1.0))
        self.last_dt = dt
        # Frames since this track last had a REAL detection. predict() adds to
        # `missed`; a hit resets it. `dt` covers the interval that has not been
        # counted yet.
        elapsed = max(1.0, float(self.missed) + dt)
        bb = np.asarray(face.bbox, np.float32).reshape(4).copy()

        # --- v11.1.9 ReentrySafe: reacquire snap ---------------------------
        # After a long miss / off-screen / round-trip, EMA-blending the new
        # detection toward the stale *predicted* bbox (and deriving velocity
        # from (new - last_hit)/elapsed across the gap) paints the first
        # returned frames in the wrong place. Snap hard instead.
        # Prevents off-screen carry → wrong-place paste on return.
        # --- v11.1.10 HairGate: keep _paste_frozen until confirm_hits >= 2
        # reliable frames so a hair/skull/neck first-hit cannot paint.
        reacquire = False
        if self.last_hit_bbox is not None:
            lh = np.asarray(self.last_hit_bbox, np.float32).reshape(4)
            lw = max(1.0, float(lh[2] - lh[0]))
            lhgt = max(1.0, float(lh[3] - lh[1]))
            cx0 = 0.5 * (float(lh[0]) + float(lh[2]))
            cy0 = 0.5 * (float(lh[1]) + float(lh[3]))
            cx1 = 0.5 * (float(bb[0]) + float(bb[2]))
            cy1 = 0.5 * (float(bb[1]) + float(bb[3]))
            # v11.2.2: judge displacement against where this track's OWN
            # tracked velocity says it should be after `elapsed` frames, not
            # against the raw last-hit position. The un-scaled version
            # (comparing raw consecutive-detection IOU/distance regardless of
            # how many frames separated them) fired on ordinary continuous
            # motion across a sparse detector cadence - a subject crossing a
            # whole box-width between two detector calls (normal at
            # Fast/Optimized SKIP_N with real movement) looked identical to a
            # genuine re-acquisition elsewhere, snapping velocity to zero and
            # re-freezing paste on exactly the "fast movement" footage this
            # project has been trying to fix. Verified directly: the
            # project's own regression suite (t_final.py) caught this -
            # constant 10px/frame motion at a 10-frame detector cadence was
            # being LEARNED AS ZERO velocity because every update reacquired.
            exp_cx, exp_cy = cx0, cy0
            if self.vel_bbox is not None:
                exp_cx += 0.5 * (float(self.vel_bbox[0]) + float(self.vel_bbox[2])) * elapsed
                exp_cy += 0.5 * (float(self.vel_bbox[1]) + float(self.vel_bbox[3])) * elapsed
            cdist = float(math.hypot(cx1 - exp_cx, cy1 - exp_cy))
            # v11.2.3: this OR clause's own flat "8" disagreed with
            # trk_max_missed (24, the budget _predicted_miss_budget() and
            # this same class' own .missed > trk_max_missed checks use for
            # "how long may a gap be trusted"). The position-based cdist
            # check above is the primary signal for "did the identity
            # actually move elsewhere"; missed alone should only force a
            # re-confirm once a gap is long enough that even good position
            # tracking stops being trustworthy - the same bar the rest of
            # this class already uses, not a stricter, unrelated one.
            # Measured directly: on sustained rapid oscillating motion, a
            # handful of motion-veto-rejected frames alone could push
            # missed past 8 while the track was still being correctly
            # predicted through the gap, forcing an unnecessary reacquire
            # (and its 2-hit HairGate confirm delay) at the exact moment a
            # good real detection arrived to end the gap.
            if (int(self.missed) >= int(_P.get("trk_max_missed", 24) or 24)
                    or cdist > 0.75 * max(lw, lhgt)):
                reacquire = True
        # v11.2.4 HoldThrough: overlapping boxes are a turn / talk / open
        # mouth, not an exit. HairGate's freeze-until-2-hits is what left
        # the original face on screen for seconds after a rapid head turn
        # even though the detector was already reporting the same head.
        # Only a true teleport (near-zero overlap with both last hit AND
        # the predicted box) may freeze paste.
        same_head = False
        if self.last_hit_bbox is not None:
            try:
                same_head = _bb_iou(self.last_hit_bbox, bb) >= 0.18
            except Exception:
                same_head = False
        if (not same_head) and self.bbox is not None:
            try:
                same_head = _bb_iou(self.bbox, bb) >= 0.18
            except Exception:
                pass
        if same_head:
            reacquire = False
            if self._paste_frozen:
                self._paste_frozen = False
                self.confirm_hits = 2
                self.alpha_ema = 1.0
        # v11.2.0 CinemaQA: sticky until core wipes geom timeline.
        # Previously `= reacquire` cleared the flag on the confirming update
        # *before* _record_geometry could scrub stubs → exit/return ghost glide.
        if reacquire:
            self._reacquired = True

        if reacquire:
            # v11.2.3 tried keeping the pre-gap vel_bbox here instead of
            # zeroing it, reasoning the subject likely kept moving through
            # the gap. Measured directly and found WORSE on sustained
            # oscillating motion: at a reversal point in the motion (the
            # exact moment a reacquire is likely, since that is where the
            # constant-velocity assumption breaks hardest) the pre-gap
            # velocity points the WRONG WAY, so projecting forward with it
            # overshoots in the opposite direction - worse than assuming
            # no velocity at all. Reverted to zero; the real fix for this
            # class of failure is a motion model that represents
            # acceleration/reversal, not a better guess at which stale
            # velocity to keep, which is out of scope here.
            self.vel_bbox = np.zeros(4, np.float32)
            self.ab_bbox = bb.copy()
            self.obs_bbox = bb.copy()
            self.last_hit_bbox = bb.copy()
            self.bbox = bb.copy()
            kps = getattr(face, "kps", None)
            if kps is not None:
                kp = np.asarray(kps, np.float32).copy()
                self.kps = kp
                self.ab_kps = kp.copy()
                # v11.2.3: None, not zeros. update()'s own velocity EMA
                # ("v if self.vel_kps is None else vel_kps*0.6 + v*0.4")
                # treats an existing vel_kps as a real prior to blend with -
                # a zeroed vector is not "no estimate yet", it is "was
                # stationary", and got treated as such: the FIRST genuine
                # velocity reading after a reacquire was damped 60% toward
                # that false zero, understating a subject who kept moving
                # right through the reacquire. That understated velocity is
                # exactly what core_pipeline.py's motion-consistency veto
                # uses for its expected-position budget, so a second real,
                # correct detection could still get vetoed on nothing more
                # than this residual damping - measured directly as the
                # repeated short-recovery/long-dropout cycle a sustained
                # fast, oscillating motion produced even after the veto's
                # own zero-velocity skip (immediately below/elsewhere) was
                # added. None restores the same fresh-start treatment a
                # genuine first-time establishment already gets.
                self.vel_kps = None
                self.kps_scale = None
                self.last_hit_kps = kp.copy()
            lmk = getattr(face, "landmark_2d_106", None)
            if lmk is not None:
                self.lmk = np.asarray(lmk, np.float32).copy()
            if update_embedding:
                self._commit_identity(getattr(face, "normed_embedding", None))
            self.det_score = _fnum_local(getattr(face, "det_score", 0.5), 0.5)
            self.last_face = face
            self.hits += 1
            self.missed = 0
            # v11.2.4: a CLEAN first return (kps_ok) paints immediately.
            # Waiting for a second consecutive reliable hit is what made
            # the original face linger for seconds after the subject was
            # already back. Unreliable first hit still waits one more.
            if kps_ok:
                self.confirm_hits = 2
                self._paste_frozen = False
                self.alpha_ema = 1.0
            else:
                self.confirm_hits = 0
                self._paste_frozen = True
            self._confirm_ease = 0
            return

        # --- alpha-beta motion filter (v11.2.6) ---------------------------
        # PRE-v11.2.6 NOTE, kept because it is the trap this code has to keep
        # avoiding: velocity must never be measured against obs_bbox. predict()
        # advances obs_bbox, so (bb - obs_bbox) is the prediction RESIDUAL, and
        # feeding a residual back AS velocity means an accurate prediction
        # halves the velocity - a few accurate predictions in a row drive it to
        # zero, the carried face stops moving mid-gap, then jumps when the
        # detector next reports.
        #
        # An alpha-beta filter uses that same residual, but as a CORRECTION
        # added to the velocity rather than as the velocity itself:
        #
        #     x_pred = x + v*dt         (where the filter thought it would be)
        #     r      = z - x_pred       (how wrong that was)
        #     x      = x_pred + a*r     (position: believe the measurement)
        #     v      = v + (b/dt)*r     (velocity: accelerate toward the truth)
        #
        # r = 0 therefore leaves v untouched, which is the correct response to
        # motion the filter is already predicting - the exact opposite of the
        # degenerate case above. What it buys over the old flat EMA of measured
        # deltas is acceleration: the residual is large and signed while the
        # subject is speeding up, slowing down or reversing, so v is corrected
        # in the right direction on the FIRST detection after the change
        # instead of being averaged toward it over several. That interval is
        # what a zoomed shot magnifies (the same head turn covers far more
        # pixels), and it is where the held face was ending up hundreds of
        # pixels from the real one.
        #
        # Dividing by `elapsed` (not by 1) keeps the unit at pixels per OUTPUT
        # frame whatever the detector cadence is, and makes the velocity gain
        # correct for the gap actually being corrected over: the same residual
        # accumulated across ten frames implies a tenth of the per-frame
        # velocity error it would imply across one.
        _ab_a = float(_P.get("ab_alpha", 0.85) or 0.85)
        _ab_b = _ab_beta()
        if self.last_hit_bbox is None or self.ab_bbox is None:
            self.vel_bbox = np.zeros(4, np.float32)
            self.ab_bbox = bb.copy()
        else:
            x_pred = self.ab_bbox + self.vel_bbox * elapsed
            r = bb - x_pred
            self.vel_bbox = (self.vel_bbox + (_ab_b / elapsed) * r).astype(np.float32)
            self.ab_bbox = (x_pred + _ab_a * r).astype(np.float32)
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
                # Same alpha-beta filter as vel_bbox, on its own state. Note
                # this is a SECOND, independent filter and not a duplicate of
                # the One Euro smoothing just above: One Euro produces the
                # DISPLAY keypoints (self.kps, jitter-free but deliberately
                # lagged), while this produces the MOTION MODEL (vel_kps, what
                # predict() extrapolates with and what core_pipeline's
                # motion-consistency veto sizes its budget from). Smoothing the
                # motion model with the display filter would make the veto
                # reject exactly the fast, correct detections it exists to let
                # through.
                if (self.last_hit_kps is not None
                        and self.last_hit_kps.shape == kp.shape
                        and self.ab_kps is not None
                        and self.ab_kps.shape == kp.shape
                        and self.vel_kps is not None
                        and self.vel_kps.shape == kp.shape):
                    kx_pred = self.ab_kps + self.vel_kps * elapsed
                    kr = kp - kx_pred
                    self.vel_kps = (self.vel_kps + (_ab_b / elapsed) * kr).astype(np.float32)
                    self.ab_kps = (kx_pred + _ab_a * kr).astype(np.float32)
                else:
                    # No usable prior: seed from the raw measured delta rather
                    # than from zero, so a subject who is already moving does
                    # not spend a detection interval being modelled as still.
                    if self.last_hit_kps is not None and self.last_hit_kps.shape == kp.shape:
                        self.vel_kps = ((kp - self.last_hit_kps) / elapsed).astype(np.float32)
                    else:
                        self.vel_kps = np.zeros_like(kp)
                    self.ab_kps = kp.copy()
                # --- damp SIZE separately from position -------------------
                # One Euro deliberately relaxes its smoothing as speed rises,
                # so a fast-moving face is not lagged. That is right for WHERE
                # the landmarks are and wrong for HOW BIG they are: a head's
                # apparent size does not change because it moved sideways, so
                # scale was being speed-relaxed for no reason and the
                # detector's own box error passed straight into the size of
                # the pasted face. Measured with a jittering detector box
                # (t_jitter): the paste never drops out - no flicker in the
                # presence sense - but its area swings up to 31% frame to
                # frame, which is the reported "little flicker on movement".
                #
                # The constellation is rescaled about its own centroid toward
                # a slow EMA of its size. Position and rotation are untouched,
                # so nothing here adds lag to a moving face; only the
                # breathing is removed. The EMA runs per DETECTION, not per
                # frame, so its time constant does not shift with cadence.
                _c = sm.mean(axis=0)
                _sc = float(np.mean(np.linalg.norm(sm - _c, axis=1)))
                if _sc > 1e-3:
                    if self.kps_scale is None:
                        self.kps_scale = _sc
                    else:
                        self.kps_scale = float(self.kps_scale * (1.0 - _KPS_SCALE_EMA)
                                               + _sc * _KPS_SCALE_EMA)
                    sm = _c + (sm - _c) * (self.kps_scale / _sc)
                self.kps = sm.astype(np.float32)
            else:
                self.kps = kp
                self.vel_kps = np.zeros_like(kp)
                self.ab_kps = kp.copy()
            self.last_hit_kps = kp.copy()

        lmk = getattr(face, "landmark_2d_106", None)
        if lmk is not None:
            self.lmk = np.asarray(lmk, np.float32).copy()

        if update_embedding:
            self._commit_identity(getattr(face, "normed_embedding", None))

        self.det_score = _fnum_local(getattr(face, "det_score", 0.5), 0.5)
        self.last_face = face
        self.hits += 1
        self.missed = 0
        # v11.1.10 HairGate: while paste-frozen (exit or reacquire), require
        # confirm_hits >= 2 reliable updates before allowing paste again.
        if self._paste_frozen:
            if kps_ok:
                self.confirm_hits = int(getattr(self, "confirm_hits", 0) or 0) + 1
                if self.confirm_hits >= 2:
                    self._paste_frozen = False
                    # v11.2.1 SolidFace: binary paste at full strength — no soft
                    # ease / no alpha_ema reseed to 0.70 (that made original bleed).
                    self._confirm_ease = 0
                    self.alpha_ema = 1.0
            else:
                self.confirm_hits = 0
                # keep _paste_frozen True
        else:
            if kps_ok:
                self.confirm_hits = max(int(getattr(self, "confirm_hits", 0) or 0), 2)

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
        """Composite opacity EMA — snap-up to full strength, slow fade-out only.

        v11.2.1 SolidFace: when gates say YES the swap must sit at ~1.0, not
        linger half-transparent via a slow 0.72/0.28 climb. Prefer snap-up
        toward 1.0; use the slower EMA only when intentionally fading out.
        """
        t = float(np.clip(target, 0.0, 1.0))
        cur = float(self.alpha_ema)
        if t >= 0.94:
            # Snap-up / fast catch-up — never cap below full during stable track.
            self.alpha_ema = max(cur, float(cur * 0.45 + t * 0.55), t)
        elif t < cur:
            # Intentional fade-out (exit / miss budget) — keep slower EMA.
            self.alpha_ema = float(cur * 0.72 + t * 0.28)
        else:
            self.alpha_ema = float(cur * 0.45 + t * 0.55)
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
    # v11.2.2: `_frame_shape` added for the same reason `_slot` was above -
    # core_pipeline.py's ReentrySafe containment check tags carried faces
    # with it (`pf._frame_shape = ...`), and the assignment was silently
    # swallowed by the same try/except pattern, falling back to (correct,
    # but only by luck of a second code path existing) TrackState._frame_wh.
    __slots__ = ("bbox", "kps", "landmark_2d_106", "normed_embedding",
                 "embedding", "det_score", "predicted", "_track", "_slot",
                 "_occlusion_guard", "_frame_shape")

    def __init__(self, bbox, kps, lmk=None, emb=None, det_score=0.5):
        self.bbox = np.asarray(bbox, np.float32).reshape(4).copy()
        self.kps = np.asarray(kps, np.float32).copy()
        self.landmark_2d_106 = None if lmk is None else np.asarray(lmk, np.float32).copy()
        self.normed_embedding = emb
        self.embedding = emb
        self.det_score = float(det_score)
        self.predicted = True
        self._track = None
        self._slot = None
        self._occlusion_guard = 0.0   # continuous weight, not a flag


# --------------------------------------------------------------------------
# The compositor
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Plate matching: grain and motion blur  (SDOS-071 F2 / F3)
# --------------------------------------------------------------------------
# Both passes answer the same question - "what does the ORIGINAL footage look
# like at high spatial frequencies, right here, right now?" - and make the
# swap agree with the answer. Both must run per OUTPUT FRAME, never on the
# cached crop: grain baked into a crop that is reused for twenty frames stops
# being grain and becomes dirt on the lens, and blur baked in would persist
# after the head stopped moving.

_IMMERKAER = np.array([[1.0, -2.0, 1.0],
                       [-2.0, 4.0, -2.0],
                       [1.0, -2.0, 1.0]], np.float32)


_GRAIN_POOL = None
_GRAIN_POOL_N = 0


def _grain_field(h: int, w: int, rng=None):
    """A unit-variance noise tile, sampled at a random offset from a pool.

    Drawing fresh Gaussians every frame is the single most expensive thing in
    the whole plate-matching path: four 512x512 draws measured ~11 ms, more
    than the entire rest of the composite. A pool generated once and read at a
    random offset per frame is visually indistinguishable - the eye cannot
    detect reuse of a 1024-wide noise field offset differently each frame -
    and costs a memory copy. The variance is exactly 1 either way, so nothing
    downstream needs to change.
    """
    global _GRAIN_POOL, _GRAIN_POOL_N
    need = int(max(h, w)) * 2
    if _GRAIN_POOL is None or _GRAIN_POOL_N < need:
        _GRAIN_POOL_N = max(1024, need)
        _GRAIN_POOL = np.random.standard_normal(
            (_GRAIN_POOL_N, _GRAIN_POOL_N)).astype(np.float32)
    P = _GRAIN_POOL_N
    r = rng if rng is not None else np.random
    oy = int(r.integers(0, P - h)) if hasattr(r, "integers") else \
        int(r.randint(0, P - h))
    ox = int(r.integers(0, P - w)) if hasattr(r, "integers") else \
        int(r.randint(0, P - w))
    return _GRAIN_POOL[oy:oy + h, ox:ox + w]


def noise_sigma(gray: np.ndarray, w: np.ndarray) -> float:
    """Immerkaer noise estimate, restricted to the mask.

    Deliberately NOT "std of a high-pass", which a face defeats: pores, lashes
    and the lid crease are high-frequency STRUCTURE and would be counted as
    noise, so the estimate would ride up on detailed faces and the matcher
    would inject grain that is not there. Immerkaer's kernel is built to have
    zero response to any locally-quadratic surface, which is what smooth skin
    plus soft shading is, so structure largely cancels and noise does not.
    Reference: J. Immerkaer, "Fast Noise Variance Estimation", CVIU 1996.
    """
    try:
        if gray is None or w is None:
            return 0.0
        r = cv2.filter2D(gray.astype(np.float32), -1, _IMMERKAER,
                         borderType=cv2.BORDER_REFLECT101)
        wsum = float(w.sum())
        if wsum < 64.0:
            return 0.0
        # mean |response| under the mask, scaled to a Gaussian sigma
        return float(math.sqrt(math.pi / 2.0) * float((np.abs(r) * w).sum())
                     / (6.0 * wsum))
    except Exception:
        return 0.0


def motion_anisotropy(gray, ux: float, uy: float, w) -> float:
    """Gradient energy PERPENDICULAR to a direction, over energy ALONG it.

    This replaces an absolute sharpness comparison, which was wrong in a way
    that mattered: a restored face is SUPPOSED to be sharper than the plate -
    that is what the restorer is for - so "fake is sharper than target" is the
    normal case even on a locked-off tripod shot, and a pass driven by it
    would blur the restoration away on every frame with any movement at all.

    Motion blur is not a loss of sharpness, it is a DIRECTIONAL loss of
    sharpness: it suppresses detail along the travel axis and leaves detail
    across it untouched. So the quantity to match is the asymmetry, which is
    ~1.0 for any unblurred image at any detail level and climbs with blur
    length. Measured across detail amplitudes 8, 20 and 40 the three curves
    agree to within 3%, so this is genuinely independent of how much texture
    the face has - which is exactly the property the absolute measure lacked.
    """
    try:
        g = np.asarray(gray, np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        along = np.abs(gx * ux + gy * uy)
        perp = np.abs(gx * (-uy) + gy * ux)
        wsum = float(w.sum())
        if wsum < 64.0:
            return 1.0
        a = float((along * w).sum() / wsum)
        pq = float((perp * w).sum() / wsum)
        if a < 1e-6:
            return 1.0
        return pq / a
    except Exception:
        return 1.0


def _ema(prev, cur, a):
    if prev is None:
        return float(cur)
    return float(prev) * (1.0 - a) + float(cur) * a


def match_grain(face_bgr, plate_bgr, w, track=None, rng=None):
    """Add the grain the plate has and the restored face does not - IN IMAGE SPACE.

    This deliberately does NOT run on the aligned crop, and that is the whole
    point. Grain is what the sensor wrote onto the final pixel grid, so both
    halves of the job have to happen on that grid:

      * MEASURED on the untouched plate. The aligned target is produced by a
        bilinear warp, and bilinear interpolation averages neighbours, which
        is a low-pass filter - it destroys roughly half the grain amplitude
        before anything can measure it.
      * INJECTED after the paste warp, not before. Grain added to the crop
        goes through that same filter on the way back out, losing about half
        again, and is also geometrically stretched by the warp - which is
        something a camera cannot do.

    Measured end to end, doing this in crop space closed a 95% grain mismatch
    only as far as 85%. The reference region is the plate UNDER the mask: that
    is the original face, so it is the same subject at the same distance under
    the same lighting, which is the fairest grain reference available.
    """
    try:
        amt = float(_P.get("grain_strength", 0.0) or 0.0)
        if amt <= 0.0 or w is None:
            return face_bgr
        wsum = float(w.sum())
        if wsum < 64.0:
            return face_bgr

        s_t = noise_sigma(cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY), w)
        s_f = noise_sigma(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY), w)

        # Quadrature, not subtraction: noise powers add, amplitudes do not.
        deficit = math.sqrt(max(s_t * s_t - s_f * s_f, 0.0)) * amt
        deficit = min(deficit, float(_P.get("grain_max", 9.0)))

        if track is not None:
            track.grain_ema = _ema(getattr(track, "grain_ema", None), deficit,
                                   float(_P.get("grain_ema", 0.25)))
            deficit = float(track.grain_ema)
        if deficit < 0.35:          # below the 8-bit quantiser - invisible
            return face_bgr

        h, wd = face_bgr.shape[:2]
        chroma = float(_P.get("grain_chroma", 0.35))
        out = face_bgr.astype(np.float32)
        # Sensor grain is mostly luma; adding equal independent noise to all
        # three BGR channels produces coloured speckle that reads as
        # compression damage rather than film. One shared luma field plus a
        # weak independent chroma field per channel is what real grain looks
        # like. The mask weight keeps it inside the face, so it cannot put a
        # rectangle of sparkle on the background.
        lw = _grain_field(h, wd, rng) * (deficit * w)
        for c in range(3):
            out[:, :, c] += lw
            if chroma > 0.0:
                out[:, :, c] += _grain_field(h, wd, rng) * (deficit * chroma * w)
        np.clip(out, 0, 255, out=out)
        return out.astype(np.uint8)
    except Exception as e:
        log.debug("grain match skipped: %s", e)
        return face_bgr


def match_motion_blur(fake_bgr, target_bgr, mask, M, track=None):
    """Soften the swap to the plate's own sharpness, along the motion axis.

    The length is not guessed from shutter physics - it is MEASURED, from how
    DIRECTIONAL the plate's own blur is (see motion_anisotropy). Comparing
    absolute sharpness instead would be wrong in a way that defeats the whole
    build: the restored face is meant to be sharper than the plate, so that
    comparison fires on every moving frame and blurs the restoration away.
    Anisotropy is ~1.0 for any unblurred face regardless of how much texture
    it has, so on a tripod shot this pass measures no deficit and does
    nothing at all, while on a whip pan it measures a large one.

    The direction comes from the track's image-space keypoint velocity pushed
    through M's linear part, because the blur has to lie along the face's
    motion IN THE CROP, which rotates with the head.
    """
    try:
        amt = float(_P.get("mblur_strength", 0.0) or 0.0)
        if amt <= 0.0 or track is None or mask is None:
            return fake_bgr
        v = getattr(track, "vel_kps", None)
        if v is None:
            return fake_bgr
        vel = np.asarray(v, np.float32).reshape(-1, 2).mean(axis=0)
        A = np.asarray(M, np.float32).reshape(2, 3)[:, :2]
        va = A @ vel                      # image-space velocity -> crop space
        speed = float(np.hypot(va[0], va[1]))
        size = int(fake_bgr.shape[0])
        if speed < max(1.0, size * 0.004):
            return fake_bgr               # effectively still - never soften

        w = mask.astype(np.float32)
        if float(w.sum()) < 64.0:
            return fake_bgr
        ux, uy = float(va[0]) / speed, float(va[1]) / speed
        a_t = motion_anisotropy(cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY),
                                ux, uy, w)
        a_f = motion_anisotropy(cv2.cvtColor(fake_bgr, cv2.COLOR_BGR2GRAY),
                                ux, uy, w)
        # The plate must be measurably MORE smeared along the travel axis than
        # the swap is. On a sharp plate both sides read ~1.0, the deficit is
        # zero, and nothing happens - which is the behaviour that makes this
        # safe to leave on.
        deficit = a_t - a_f
        if deficit <= 0.05:
            return fake_bgr

        L = 1.0 + deficit * float(_P.get("mblur_k", 6.0)) * amt
        # Never blur further than the motion itself could have smeared, and
        # never past the global cap - an over-long kernel turns a fast turn
        # into a smear that no real camera would produce.
        L = min(L, speed, size * float(_P.get("mblur_max", 0.055)))
        if track is not None:
            track.mblur_ema = _ema(getattr(track, "mblur_ema", None), L,
                                   float(_P.get("mblur_ema", 0.30)))
            L = float(track.mblur_ema)
        k = int(round(L))
        if k < 2:
            return fake_bgr
        k = min(k | 1, 31)                # odd, and bounded for cost

        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0
        ang = math.degrees(math.atan2(float(va[1]), float(va[0])))
        R = cv2.getRotationMatrix2D((k / 2.0 - 0.5, k / 2.0 - 0.5), -ang, 1.0)
        kern = cv2.warpAffine(kern, R, (k, k), flags=cv2.INTER_LINEAR)
        ssum = float(kern.sum())
        if ssum <= 1e-6:
            return fake_bgr
        kern /= ssum

        blurred = cv2.filter2D(fake_bgr, -1, kern,
                               borderType=cv2.BORDER_REFLECT101)
        # Feather the swap-in by the mask so the crop border never shows a
        # step between blurred and unblurred pixels.
        w3 = w[:, :, None]
        return (blurred.astype(np.float32) * w3
                + fake_bgr.astype(np.float32) * (1.0 - w3)).astype(np.uint8)
    except Exception as e:
        log.debug("motion-blur match skipped: %s", e)
        return fake_bgr


class AlignedCompositor:
    """Swap + blend entirely inside the ArcFace-aligned crop."""

    def __init__(self, swapper):
        self.swapper = swapper
        self._warned_no_M = False

    # -- inswapper interop ---------------------------------------------
    def _raw_swap(self, img, face, src_face):
        """Return (bgr_fake_128, M) or (None, None).

        insightface's INSwapper.get(..., paste_back=False) returns
        (bgr_fake, M). Older/forked builds may return only bgr_fake. If M
        is missing we recover it from the 5-point ArcFace template. If get()
        fails entirely we run the ONNX session ourselves with the swapper's
        own emap — that is the actual "no swap / original face" recovery,
        not a rewrite of the pipeline.
        """
        size = 128
        insz = getattr(self.swapper, "input_size", None)
        if insz is not None:
            size = int(insz[0] if not isinstance(insz, int) else insz)
        M_est = estimate_norm(getattr(face, "kps", None), size)

        out = None
        try:
            out = self.swapper.get(img, face, src_face, paste_back=False)
        except Exception as e:
            log.debug("paste_back=False unsupported (%s)", e)

        fake, M = None, None
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            fake = out[0]
            if len(out) >= 2:
                M = out[1]
        elif isinstance(out, np.ndarray):
            fake = out

        if fake is not None:
            fake = np.asarray(fake)
            if fake.ndim == 4:
                fake = fake[0]
            if fake.ndim == 3 and fake.shape[0] in (1, 3) and fake.shape[-1] not in (1, 3):
                fake = np.transpose(fake, (1, 2, 0))
            if fake.dtype != np.uint8:
                if float(np.max(fake)) <= 1.5:
                    fake = fake * 255.0
                fake = np.clip(fake, 0, 255).astype(np.uint8)
            # A full-frame paste-back is not an aligned crop.
            if fake.ndim == 3 and fake.shape[0] == img.shape[0] and fake.shape[1] == img.shape[1]:
                fake = None
            elif fake.ndim != 3 or min(fake.shape[:2]) < 64:
                fake = None

        if M is not None:
            try:
                M = np.asarray(M, np.float32).reshape(2, 3)
            except Exception:
                M = None
        if M is None:
            M = M_est

        if fake is not None and M is not None:
            return fake, M

        bound = self._raw_swap_session(img, face, src_face, M_est, size)
        if bound[0] is not None:
            return bound

        if not self._warned_no_M:
            log.warning("inswapper did not return an aligned crop; "
                        "caller will use the paste-back path")
            self._warned_no_M = True
        return None, None

    def _raw_swap_session(self, img, face, src_face, M, size):
        """Direct ONNX forward using the loaded inswapper session + emap."""
        sw = self.swapper
        sess = getattr(sw, "session", None)
        kps = getattr(face, "kps", None)
        latent = getattr(src_face, "normed_embedding", None)
        if latent is None:
            latent = getattr(src_face, "embedding", None)
        if sess is None or kps is None or latent is None:
            return None, None
        if M is None:
            M = estimate_norm(kps, size)
        if M is None:
            return None, None
        try:
            aimg = cv2.warpAffine(
                img, M, (size, size),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
            lat = np.asarray(latent, np.float32).reshape(1, -1)
            nrm = float(np.linalg.norm(lat))
            if nrm > 1e-6:
                lat = lat / nrm
            emap = getattr(sw, "emap", None)
            if emap is not None:
                lat = np.dot(lat, np.asarray(emap, np.float32))
                lat = lat / (np.linalg.norm(lat) + 1e-6)
            mean = float(getattr(sw, "input_mean", 0.0) or 0.0)
            std = float(getattr(sw, "input_std", 255.0) or 255.0)
            blob = cv2.dnn.blobFromImage(
                aimg, 1.0 / std, (size, size), (mean, mean, mean), swapRB=True,
            )
            names = list(getattr(sw, "input_names", None) or [i.name for i in sess.get_inputs()])
            onames = list(getattr(sw, "output_names", None) or [o.name for o in sess.get_outputs()])
            feeds = {names[0]: np.ascontiguousarray(blob)}
            if len(names) > 1:
                feeds[names[1]] = np.ascontiguousarray(lat.astype(np.float32))
            pred = sess.run([onames[0]], feeds)[0]
            fake = np.asarray(pred)
            if fake.ndim == 4:
                fake = fake[0]
            if fake.ndim == 3 and fake.shape[0] in (1, 3):
                fake = np.transpose(fake, (1, 2, 0))
            if float(np.max(fake)) <= 1.5:
                fake = fake * 255.0
            fake = np.clip(fake, 0, 255).astype(np.uint8)
            if fake.shape[-1] == 3:
                fake = fake[:, :, ::-1]
            return fake, np.asarray(M, np.float32).reshape(2, 3)
        except Exception as e:
            log.debug("session swap skipped: %s", e)
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
            shaped = floor + (1.0 - floor) * h
            # The jaw hull follows the lips and was opening and closing
            # the lower lip and chin. Below the mouth the steady oval
            # is the mask, so that edge does not breathe.
            row = np.linspace((0.5 / size), 1.0 - (0.5 / size), size, dtype=np.float32)[:, None]
            lower = np.clip((row - 0.68) / 0.08, 0.0, 1.0).astype(np.float32)
            shaped = shaped * (1.0 - lower) + lower
            mask = mask * shaped
        else:
            row = None

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
        # After the temporal mask, so a cone or scoop is removed on the
        # frame it appears instead of being smoothed back in.
        if aligned_target is not None:
            keep = occluder_keep(aligned_target)
            if track is not None:
                prev = getattr(track, "occ_keep", None)
                if prev is None or getattr(prev, "shape", None) != keep.shape:
                    track.occ_keep = np.ones_like(keep)
                else:
                    # A hole opens slowly. The lip going back to solid is instant,
                    # so a talking mouth cannot leave a flickering edge.
                    track.occ_keep = np.where(
                        keep < track.occ_keep,
                        track.occ_keep * 0.90 + keep * 0.10,
                        keep,
                    ).astype(np.float32)
                keep = track.occ_keep
            mask = mask * keep
        if row is not None:
            # Longer falloff under the chin only. The lip stays fully painted.
            t = np.clip((row - 0.86) / 0.14, 0.0, 1.0)
            ramp = np.cos(t * (np.pi * 0.5)).astype(np.float32)
            mask = np.minimum(mask, ramp)
        return np.clip(mask, 0.0, 1.0).astype(np.float32)

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
    def paste_back(img, fake_bgr, mask, M, alpha=1.0, orig=None, track=None):
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

        # Grain, on the output pixel grid, with the plate under the mask as
        # the reference. Here and nowhere earlier - see match_grain.
        if orig is not None:
            try:
                warp_face = match_grain(warp_face, orig[y1:y2, x1:x2],
                                        warp_mask, track=track)
            except Exception as e:
                log.debug("grain stage skipped: %s", e)

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
        # Motion blur belongs HERE, in aligned space: it is a property of the
        # optics, so it must act on the face before the sensor stage, and the
        # aligned crop is the frame the head itself lives in, so one direction
        # vector describes it correctly however the head is rotated.
        toned = match_motion_blur(toned, aligned_target, mask, M, track=track)
        # Grain belongs in paste_back, on the output pixel grid, which is where
        # the sensor put it. Passing `orig` is what switches that stage on.
        return self.paste_back(work, toned, mask, M, alpha=alpha,
                               orig=orig, track=track)

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
                # Accept a restorer that returns a LARGER square crop and keep
                # it. The old guard required the same shape, so a 512 restored
                # face was silently discarded and the 128 one used instead -
                # the enhancer's entire output thrown away. Everything
                # downstream is already size-parameterised; only the affine has
                # to be told the crop grew.
                if p is not None and getattr(p, "ndim", 0) == 3 \
                        and p.shape[0] == p.shape[1] and p.shape[0] >= size:
                    want = choose_crop_size(M, size, int(p.shape[0]))
                    if want != int(p.shape[0]):
                        p = cv2.resize(p, (want, want),
                                       interpolation=cv2.INTER_AREA)
                    if want != size:
                        M = scale_affine(M, float(want) / float(size))
                        size = want
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


def _det_gender(face):
    """InsightFace buffalo: 0 = female, 1 = male. Always a Python int or None."""
    if face is None:
        return None
    g = getattr(face, "gender", None)
    if g is None:
        g = getattr(face, "sex", None)
    if g is None:
        return None
    try:
        if isinstance(g, (bytes, str)):
            s = str(g).lower()
            if "female" in s or s in ("f", "w"):
                return 0
            if "male" in s or s == "m":
                return 1
            return None
        arr = np.asarray(g).reshape(-1)
        if arr.size == 0:
            return None
        if arr.size > 1:
            iv = int(np.argmax(arr))
        else:
            iv = int(arr[0])
        if iv in (0, 1):
            return iv
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------
# Multi-face association
# --------------------------------------------------------------------------
def _iou(a, b):
    if a is None or b is None:
        return 0.0
    try:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
        bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    except Exception:
        return 0.0
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
    try:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        ca = np.array([(a[0] + a[2]) * .5, (a[1] + a[3]) * .5], np.float32)
        cb = np.array([(b[0] + b[2]) * .5, (b[1] + b[3]) * .5], np.float32)
        diag = max(1.0, float((a[2] - a[0] + a[3] - a[1]) * 0.5))
        return float(np.linalg.norm(ca - cb) / diag)
    except Exception:
        return 9.0


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


# Picked-slot admission (SDOS-022). Absolute 0.32 was a cliff: a man picked
# on one frame and seen later at another distance scored 0.20–0.31 against
# his own crop and was never swapped. Relative: this slot must prefer the
# face more than any other picked slot, above a low stranger floor.
_PICK_FLOOR = 0.18
_PICK_MARGIN = 0.05
_PICK_SEP = 0.08


def _ref_cos(emb, ref):
    if emb is None or ref is None:
        return -1.0
    try:
        ev = np.asarray(emb, np.float32).reshape(-1)
        rv = np.asarray(ref, np.float32).reshape(-1)
        k = min(ev.size, rv.size)
        if k < 8:
            return -1.0
        return float(np.dot(ev[:k], rv[:k]))
    except Exception:
        return -1.0


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

    def __init__(self, slots, slot_gender=None):
        self.tracks = {s: TrackState(s) for s in slots}
        # 0 = female, 1 = male, None = unknown (from the replacement photo)
        self.slot_gender = {s: slot_gender.get(s) if slot_gender else None for s in slots}

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
        lock = getattr(track, "id_lock", None) if track is not None else None
        if lock is not None:
            return float(np.dot(emb, lock))
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
        if faces is None or (hasattr(faces, "__len__") and len(faces) == 0):
            for t in self.tracks.values():
                t.predict(dt_frames)
            self._preserve_identity_on_expire()
            return {}
        faces = list(faces)

        n, m = len(slots), len(faces)

        def _cos_emb(a, b):
            if a is None or b is None:
                return -1.0
            va = np.asarray(a, np.float32).reshape(-1)
            vb = np.asarray(b, np.float32).reshape(-1)
            k = min(va.size, vb.size)
            if k < 8:
                return -1.0
            return float(np.dot(va[:k], vb[:k]))

        # Duplicate woman-box locked two slots. Free the clone so the man
        # who joins later still has a virgin slot.
        for i, s in enumerate(slots):
            e1 = getattr(self.tracks[s], "id_lock", None)
            if e1 is None:
                e1 = self.tracks[s].emb
            if e1 is None:
                continue
            ref1 = refs.get(s) if refs else None
            for i2 in range(i + 1, n):
                s2 = slots[i2]
                e2 = getattr(self.tracks[s2], "id_lock", None)
                if e2 is None:
                    e2 = self.tracks[s2].emb
                if e2 is None:
                    continue
                if _cos_emb(e1, e2) < 0.38:
                    continue
                ref2 = refs.get(s2) if refs else None
                keep = s
                if ref1 is not None or ref2 is not None:
                    if _cos_emb(ref2, e2) > _cos_emb(ref1, e1):
                        keep = s2
                drop_s = s2 if keep == s else s
                trd = self.tracks[drop_s]
                trd.emb = None
                trd.id_lock = None
                trd.hits = 0

        # A slot must not keep a lock that belongs to another slot's Detect ref.
        for i, s in enumerate(slots):
            ref = refs.get(s) if refs else None
            lock = getattr(self.tracks[s], "id_lock", None)
            if lock is None:
                lock = self.tracks[s].emb
            if ref is None or lock is None:
                continue
            own = _cos_emb(lock, ref)
            other = -1.0
            for s2 in slots:
                if s2 == s:
                    continue
                r2 = refs.get(s2) if refs else None
                if r2 is not None:
                    other = max(other, _cos_emb(lock, r2))
            if other > own + 0.04:
                trd = self.tracks[s]
                trd.emb = None
                trd.id_lock = None
                trd.hits = 0
                logging.info("slot %s lock belonged to another ref (own=%.2f other=%.2f) — reset",
                             s, own, other)

        cost = [[1e9] * m for _ in range(n)]
        sim_tab = [[-1.0] * m for _ in range(n)]
        ref_sim = [[-1.0] * m for _ in range(n)]
        for i, s_ in enumerate(slots):
            _r = refs.get(s_) if refs else None
            if _r is None:
                continue
            for j, f in enumerate(faces):
                ref_sim[i][j] = _ref_cos(getattr(f, "normed_embedding", None), _r)

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
                has_id = (getattr(tr, "id_lock", None) is not None) or (tr.emb is not None)
                long_gap = int(getattr(tr, "missed", 0) or 0) >= 8 or bool(getattr(tr, "_paste_frozen", False))
                # Gender is not a reason to drop a face. InsightFace often
                # calls the man a woman, and the woman a man for a few frames.
                # The Detect-frame photo is the identity. A wrong gender tag
                # must not skip him or flash her back to the original.
                if has_id:
                    if long_gap:
                        if sim < 0.40:
                            continue
                    elif sim < 0.38 and not (spatial >= 0.55 and sim >= 0.22):
                        continue
                else:
                    ref = refs.get(s) if refs else None
                    if ref is not None:
                        rs = ref_sim[i][j]
                        if rs < _PICK_FLOOR:
                            continue
                        best_other = -1.0
                        for i2 in range(n):
                            if i2 != i:
                                best_other = max(best_other, ref_sim[i2][j])
                        if best_other > rs:
                            continue
                    else:
                        skip = False
                        for s2 in slots:
                            if s2 == s:
                                continue
                            le = getattr(self.tracks[s2], "id_lock", None)
                            if le is None:
                                le = self.tracks[s2].emb
                            if le is None:
                                continue
                            if _cos_emb(getattr(f, "normed_embedding", None), le) >= 0.45:
                                skip = True
                                break
                        if skip:
                            continue

                w_id, w_iou, w_d = _P["trk_w_id"], _P["trk_w_iou"], _P["trk_w_dist"]
                if tr.crossing and tr.established:
                    w_id, w_iou, w_d = 0.18, 0.55, 0.27
                if not has_id:
                    w_id, w_iou, w_d = 0.10, 0.55, 0.35
                cost[i][j] = (w_id * (1.0 - max(sim, -0.2)) +
                              w_iou * (1.0 - iou) +
                              w_d * min(1.0, dist / 3.0))

        pairing = optimal_assignment(cost)

        # Virgin slots only (no gallery lock yet). Never L-R rebind a slot
        # that already knows its person — that is how F1 leaving caused
        # Male/F3 to steal slot 0, then F1 coming back got the male paste.
        unresolved = [
            i for i, s in enumerate(slots)
            if pairing.get(i) is None
            and getattr(self.tracks[s], "id_lock", None) is None
            and self.tracks[s].emb is None
        ]
        if unresolved:
            claimed = set(pairing.values())
            leftovers = [j for j in range(m) if j not in claimed]
            # A leftover that still matches a locked slot is that person
            # (profile / return) — do not give them to a virgin slot.
            def _min_lock_sim(j):
                best = 1.0
                any_lock = False
                for i2, s2 in enumerate(slots):
                    if getattr(self.tracks[s2], "id_lock", None) is None and self.tracks[s2].emb is None:
                        continue
                    any_lock = True
                    best = min(best, sim_tab[i2][j])
                if not any_lock:
                    return 0.0
                return best
            leftovers = [j for j in leftovers if _min_lock_sim(j) < 0.45]
            leftovers.sort(key=lambda j: (_min_lock_sim(j), float(np.asarray(faces[j].bbox).reshape(-1)[0])))
            used_j = set()
            for i in unresolved:
                pick = None
                ref = refs.get(slots[i]) if refs else None
                if ref is not None:
                    best = _PICK_FLOOR
                    for j in leftovers:
                        if j in used_j:
                            continue
                        sim = ref_sim[i][j]
                        if sim <= best:
                            continue
                        best_other = -1.0
                        for i2 in range(n):
                            if i2 != i:
                                best_other = max(best_other, ref_sim[i2][j])
                        if best_other > sim:
                            continue
                        best, pick = sim, j
                if pick is None and ref is None:
                    for j in leftovers:
                        if j in used_j:
                            continue
                        emb = getattr(faces[j], "normed_embedding", None)
                        if emb is None:
                            continue
                        clone = False
                        for s2 in slots:
                            le = getattr(self.tracks[s2], "id_lock", None)
                            if le is None:
                                le = self.tracks[s2].emb
                            if le is None:
                                continue
                            if _cos_emb(emb, le) >= 0.45:
                                clone = True
                                break
                        if not clone:
                            pick = j
                            break
                if pick is not None:
                    pairing[i] = pick
                    used_j.add(pick)

        # Same person must not fill two slots (two boxes on the woman at start).
        taken = [i for i in range(n) if pairing.get(i) is not None]
        drop = set()
        for a in range(len(taken)):
            for b in range(a + 1, len(taken)):
                i, i2 = taken[a], taken[b]
                j, j2 = pairing[i], pairing[i2]
                e1 = getattr(faces[j], "normed_embedding", None)
                e2 = getattr(faces[j2], "normed_embedding", None)
                if e1 is None or e2 is None:
                    continue
                a1 = np.asarray(e1, np.float32).reshape(-1)
                a2 = np.asarray(e2, np.float32).reshape(-1)
                k = min(a1.size, a2.size)
                if k < 8:
                    continue
                if float(np.dot(a1[:k], a2[:k])) < 0.50:
                    continue
                t1, t2 = self.tracks[slots[i]], self.tracks[slots[i2]]
                r1 = refs.get(slots[i]) if refs else None
                r2 = refs.get(slots[i2]) if refs else None

                def _rsim(r, e):
                    if r is None or e is None:
                        return -1.0
                    rv = np.asarray(r, np.float32).reshape(-1)
                    ev = np.asarray(e, np.float32).reshape(-1)
                    kk = min(rv.size, ev.size)
                    if kk < 8:
                        return -1.0
                    return float(np.dot(ev[:kk], rv[:kk]))

                keep_i = None
                if r1 is not None or r2 is not None:
                    s1 = max(_rsim(r1, e1), _rsim(r1, e2))
                    s2 = max(_rsim(r2, e1), _rsim(r2, e2))
                    if s1 != s2:
                        keep_i = i if s1 > s2 else i2
                if keep_i is None:
                    keep_i = i if (t1.emb is not None or getattr(t1, "id_lock", None) is not None) else i2
                    if t2.emb is not None and t1.emb is None:
                        keep_i = i2
                drop.add(i2 if keep_i == i else i)
        for i in drop:
            pairing.pop(i, None)

        # LATE JOIN: an empty slot takes a face that is clearly NOT anyone
        # already locked. This is the woman-first / man-later path and does
        # not depend on Detect-frame refs or the 0.32 cliff.
        claimed = set(pairing.values())
        empty_i = [
            i for i, s in enumerate(slots)
            if pairing.get(i) is None
            and getattr(self.tracks[s], "id_lock", None) is None
            and self.tracks[s].emb is None
        ]
        if empty_i:
            locked_e = []
            for s in slots:
                e = getattr(self.tracks[s], "id_lock", None)
                if e is None:
                    e = self.tracks[s].emb
                if e is not None:
                    locked_e.append(e)
            for j in range(m):
                if not empty_i:
                    break
                if j in claimed:
                    continue
                emb = getattr(faces[j], "normed_embedding", None)
                if emb is None:
                    continue
                mx = max((_cos_emb(emb, e) for e in locked_e), default=-1.0)
                if locked_e and mx >= 0.42:
                    continue
                pick_i = None
                best_rs = -1.0
                any_ref = False
                for i in list(empty_i):
                    r = refs.get(slots[i]) if refs else None
                    if r is None:
                        continue
                    any_ref = True
                    rs = ref_sim[i][j]
                    if rs < _PICK_FLOOR:
                        continue
                    other = max((ref_sim[i2][j] for i2 in range(n) if i2 != i), default=-1.0)
                    if other > rs:
                        continue
                    if rs > best_rs:
                        best_rs, pick_i = rs, i
                if pick_i is None and not any_ref:
                    got_g = _det_gender(faces[j])
                    pool = list(empty_i)
                    if len(self.tracks) >= 2 and got_g in (0, 1):
                        matched = [
                            i for i in empty_i
                            if self.slot_gender.get(slots[i]) == got_g
                        ]
                        wrong = [
                            i for i in empty_i
                            if self.slot_gender.get(slots[i]) in (0, 1)
                            and self.slot_gender.get(slots[i]) != got_g
                        ]
                        if matched:
                            pool = matched
                        # Do not refuse the face when every empty slot
                        # disagrees on gender. A wrong tag was leaving the
                        # man original for the whole clip.
                        if not pool:
                            pool = list(empty_i)
                    if pool:
                        pick_i = pool[0]
                if pick_i is None:
                    continue
                empty_i.remove(pick_i)
                pairing[pick_i] = j
                claimed.add(j)
                logging.info("late-join slot %s ← face (ref_sim=%.2f locked=%.2f)",
                             slots[pick_i], best_rs, mx)

        # Re-entry steal: if a detection is clearly a locked slot's person,
        # give it back even if Hungarian/bootstrap parked it elsewhere.
        for i, s in enumerate(slots):
            lock = getattr(self.tracks[s], "id_lock", None)
            if lock is None:
                lock = self.tracks[s].emb
            if lock is None:
                continue
            best_j, best_sim = None, 0.40
            for j in range(m):
                if sim_tab[i][j] > best_sim:
                    best_sim, best_j = sim_tab[i][j], j
            if best_j is None:
                continue
            owner = next((i2 for i2 in range(n) if pairing.get(i2) == best_j), None)
            if owner is None:
                pairing[i] = best_j
            elif owner != i and best_sim >= sim_tab[owner][best_j] + 0.08:
                pairing.pop(owner, None)
                pairing[i] = best_j

        # Un-cross two established slots if swapping them raises identity
        # scores. This is the male↔female paste: Hungarian + left-to-right
        # bootstrap can lock the wrong pair, then hysteresis keeps it.
        established = [i for i, s in enumerate(slots)
                       if self.tracks[s].emb is not None or getattr(self.tracks[s], "id_lock", None) is not None]
        if len(established) >= 2:
            for a in range(len(established)):
                for b in range(a + 1, len(established)):
                    i, i2 = established[a], established[b]
                    j, j2 = pairing.get(i), pairing.get(i2)
                    if j is None or j2 is None:
                        continue
                    def _rs(ii, jj):
                        try:
                            return float(ref_sim[ii][jj])
                        except Exception:
                            return -1.0
                    cur = _rs(i, j) + _rs(i2, j2)
                    swp = _rs(i, j2) + _rs(i2, j)
                    # Swap only when the Detect-frame picks say the faces
                    # are crossed. A one-frame gender flicker must not
                    # trade them or drop either face to the original.
                    if swp > cur + 0.08:
                        pairing[i], pairing[i2] = j2, j

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

            # Never poison the long-term identity with a profile/crossing frame
            # or with a *different person* after a gap (the 3rd-character steal).
            lock_sim = sim_tab[i][j]
            good = lock_sim >= 0.38 and not tr.crossing
            if getattr(tr, "id_lock", None) is None and tr.emb is None and not tr.crossing:
                good = True
            elif getattr(tr, "id_lock", None) is not None and lock_sim < 0.32:
                # Detection is not this slot's person. Hold geometry, don't bind.
                tr.predict(dt_frames)
                continue

            # v11.2.0: fail-closed if landmarks missing (core usually pre-filters
            # with _kps_reliable; this is defense-in-depth for HairGate confirm).
            _kps = getattr(f, "kps", None)
            _kps_ok = _kps is not None and len(_kps) >= 5
            tr.update(f, update_embedding=good, dt_frames=dt_frames, kps_ok=_kps_ok)
            if changed:
                tr.reset_appearance()
            result[s] = f

        self._preserve_identity_on_expire()
        return result

    def _preserve_identity_on_expire(self):
        """Drop coasted geometry after a long miss; keep the gallery lock.

        Geometry must die or the next person who walks into that region
        inherits the slot. Identity must live or the original person cannot
        reclaim the slot on return.
        """
        for s in list(self.tracks.keys()):
            tr = self.tracks[s]
            if tr.missed > _P["trk_max_missed"]:
                keep_emb = tr.emb
                keep_lock = getattr(tr, "id_lock", None)
                nw = TrackState(s)
                nw.emb = keep_emb
                nw.id_lock = keep_lock
                if keep_lock is not None or keep_emb is not None:
                    nw.hits = max(int(_P["trk_lock_hits"]), 1)
                self.tracks[s] = nw

    def bind(self, slot, face, update_embedding=True, dt_frames: float = 1.0,
             kps_ok: bool = True):
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
        tr.update(face, update_embedding=update_embedding, dt_frames=dt_frames,
                  kps_ok=kps_ok)
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
            # v11.1.9: frozen after leaving frame — do not offer for paste.
            if getattr(tr, "_paste_frozen", False):
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

