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
import os
import math
from itertools import permutations

import cv2
import time
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

ENGINE_VERSION = "aequus-1.2.104-anchor"

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
    "rival_cut": 1.0,        # how hard a rival hull is removed (0 = off); v11.2.95: 1.0 (was 0.85: 15 % of one swap on the other face)
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
    # --- occluder rejection (SDOS-074) -----------------------------------
    # Thresholds are in units of the FACE'S OWN chroma/luma spread, so they
    # mean the same thing on any skin tone under any light. See occluder_keep.
    "occ_chroma": 2.6,       # how far off this face's skin colour counts as foreign
    "occ_luma": 2.6,         # ...or off its brightness
    "occ_ref_chroma": 2.6,   # how far off its OWN learned appearance here
    "occ_ref_luma": 2.6,
    "occ_min_frac": 0.010,   # smallest occluder, as a fraction of the face area
    "occ_out_frac": 0.08,    # how much of the blob must lie outside the face
    # Working resolution for occlusion. The decision is Gaussian-blurred to a
    # twentieth of the crop before use, so computing it at full 512 bought
    # nothing and cost 36.7 ms/frame (and 3.15 MB of appearance model per
    # track). Fixed at 192 it is 4.6 ms and 0.44 MB, independent of crop size
    # - which also means the model stays valid when the adaptive crop size
    # changes as the face moves toward or away from the camera.
    "occ_work": 192,
    "occ_rim": 0.055,        # width of the face-edge band, as a fraction of the crop
    "occ_rim_frac": 0.02,    # how much of the blob must reach that band
    # A blob this large (as a fraction of the face) that is both off this
    # face's skin AND off its own learned appearance is taken as an occluder
    # even if it never reaches the face edge - which is the case for food held
    # right at the lips. Safe only because the appearance model already vouches
    # for the lips and the mouth cavity, so they are not novel and cannot
    # reach this test; it is therefore gated on the model being well warmed.
    "occ_big_frac": 0.045,
    "occ_ref_ema": 0.10,     # learning rate of the per-pixel appearance model
    "occ_ref_warm": 12,      # frames before the model is trusted to veto
    "occ_open": 0.45,        # how fast a hole OPENS (was effectively 0.10)
    "occ_close": 0.22,       # ...and closes
    "grain_strength": 0.90,  # fraction of the measured grain deficit restored
    "grain_max": 9.0,        # hard cap on added sigma (8-bit levels)
    "grain_chroma": 0.35,    # chroma grain relative to luma grain
    "grain_ema": 0.25,       # temporal smoothing of the sigma estimate
    "mblur_strength": 0.85,  # fraction of the measured sharpness excess removed
    "mblur_max": 0.055,      # blur length cap as a fraction of crop size
    "mblur_ema": 0.30,
    # SDOS-090 face-motion key frames (see core_pipeline._FaceMotionKeys). adaptive_key = 0 restores v11.2.89 exactly.
    "adaptive_key": 1.0,            # 1 = extra key frames (detect + swap) while a face moves or changes fast
    "adaptive_key_move": 0.08,      # travel since the last swap, in face widths, that makes an extra one
    "adaptive_key_change": 0.20,    # accumulated change of look since the last swap that makes an extra one
    "adaptive_key_min_gap": 1,      # frames; never closer than this
    "adaptive_key_max_frac": 0.60,  # extra key frames never exceed this share of all frames (runtime cap)
    # SDOS-091 occlusion net (FaceFusion's XSeg face occluder). occ_net = 0 restores v11.2.90 exactly.
    "occ_colour": 1.0,              # the colour-novelty occluder (v11.2.81) on; 0 = off (tests)
    "contact_calm": 0.08,           # v11.2.97: the occluder net is ignored within this share of the crop of another face
                                    #   touching this one (the face-against-face rule decides there); 0 = v11.2.96
    "contact_full": 0.15,           #   ...and over the whole crop once that zone covers this share of it (a kiss)
    "occ_net": 1.0,                 # 1 = a trained occlusion model cuts what is in front of the mouth and chin (food, a cup, a hand)
    "occ_net_blur": 3.0,            # smoothing of the model's answer (pixels of its 256-px input)
    "occ_net_hull_erode": 0.08,     # only inside the face's own landmark outline, this far in (fraction of the crop): the outline of the swap is not touched
    "occ_net_core_y": 0.58,         # a cut piece must reach the mouth/chin zone: below this row...
    "occ_net_core_x": 0.17,         # ...and within this distance of the centre line (fraction of the crop)...
    "occ_net_core_min": 0.002,
    "own_pixels": 1.0,              # v11.2.95: each pixel belongs to one face (0 = the v11.2.94 rival cut)
    "own_grow": 0.02,               #   the other face's outline grown by this share of the crop before it is cut out
    "own_feather": 0.03,            #   softness of that cut (was rival_feather 0.09)
    "own_near": 0.12,               #   a face behind keeps only its own outline within this share of the crop of the front face
    "occ_net_zone": 1.0,
    "occ_novel": 1.0,               # v11.2.95: objects arriving at the face's edge (forehead, cheek edge, jaw) are cut
    "occ_novel_grow": 0.20,         #   ring: from the shrunk outline out to the outline grown by this share of the crop (0.10 left 7.7 % of a cup at the forehead painted; 0.20: 0.6 %)
    "occ_novel_ref": 0.75,          #   a pixel counts as 'was face' when the memory says at least this
    "occ_novel_min": 0.003,         #   smallest new piece, share of the crop            # v11.2.95: 1 = occluders anywhere on the face (0 = mouth / chin only, as v11.2.91-94)
    "occ_net_face_min": 0.004,      # v11.2.95: smallest occluder piece away from the mouth, share of the crop
    # v11.2.92 (SDOS-092) extreme side views: there the 5-point alignment puts the fixed oval half in front of the
    # face (air, background) and leaves the back cheek out. From profile_lo to profile_hi (nose offset from the eye
    # midpoint, in eye distances: 0.25 at 18 deg of yaw, 0.45 at 45 deg, 0.76+ from 55 deg on the AFLW2000 photos)
    # the mask moves over to the visible face's own outline (occlusion net's face map x landmark outline).
    "profile_mask": 1.0,
    "profile_lo": 0.45,
    "profile_hi": 0.70,
    "profile_feather": 0.025,
    "side_eye": 0.0,        # v11.2.94 test, OFF: the visible eye's point gets better but the drawn eye does not (SDOS-094 §2)
    # v11.2.92 (SDOS-092) one mouth while eating: when the occlusion net cuts something at the mouth that reaches
    # OUTSIDE the lips (food, a spoon, a cup; an open mouth, teeth or a tongue stay inside them), the footage's own
    # lips are shown with the object, so there is one set of lips around it instead of the swapped lips next to the
    # real ones. eat_lo/eat_hi: share of the crop cut outside the lips for the rule to start / be complete.
    "eat_mouth": 1.0,
    "eat_lo": 0.002,
    "eat_hi": 0.008,
    "eat_attack": 0.6,
    "eat_release": 0.12,
    "eat_lip_pad": 0.035,
    # v11.2.93: change of a face's coarse look between two frames above which its mouth / side-view / occlusion
    # state is dropped (a missed shot change: 60+; the fastest turn measured: 6).
    "look_jump": 20.0,      # ...by at least this share of the crop. Hair over the cheeks, beards and shadows are left alone.
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
    # Relative identity admission (SDOS-078). A slot that has locked onto a
    # person used to accept a later detection only if it matched that one stored
    # frame at >= 0.38 (>= 0.40 after a gap) - an absolute wall. Measured on the
    # real assign(): a returning person is re-bound ~100% of the time at 0.45,
    # ~30% at 0.40, and 0% at 0.37 or below, for ANY gap length, forever - the
    # lock is only refreshed by faces that clear 0.38, so it can never move
    # toward the appearance that is failing. With two or more identified people
    # the right question is not "is this above an absolute number" but "which
    # person is this, relative to the others", which is easy: different people
    # sit near 0 against each other's locks.
    "trk_rel_floor": 0.25,    # never admit below this, however clear the margin
    "trk_rel_margin": 0.15,   # must beat EVERY other identified slot by this much
    "trk_rel_confirm": 2,     # consecutive, spatially consistent detections required
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


def _face_stats(aligned_bgr):
    """This face's own skin statistics, and every pixel as a normalised residual.

    The sampling zone is forehead and outer cheeks only - never the eyes, brows
    or mouth, which are legitimately not skin and would poison the reference.

    The variance floors are PROPORTIONAL to the face's own brightness, not
    absolute. With absolute floors the normalisation stops being scale
    invariant: dimming a shot scales the measured spread but not the floor, so
    every learned residual shifts and the whole face reads as novel. Measured
    with absolute floors, a 45% light drop produced a 14.5% false cut on every
    frame afterwards.
    """
    h, w = aligned_bgr.shape[:2]
    if h != w:
        return None
    ycc = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    yy, cr, cb = ycc[:, :, 0], ycc[:, :, 1], ycc[:, :, 2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ys = (ys + 0.5) / h
    xs = (xs + 0.5) / w
    inface = canonical_template(h) > 0.5
    zone = inface & (((ys > 0.14) & (ys < 0.28))
                     | ((ys > 0.44) & (ys < 0.62) & ((xs < 0.33) | (xs > 0.67))))
    if int(zone.sum()) < 100:
        return None
    mcr = float(np.median(cr[zone]))
    mcb = float(np.median(cb[zone]))
    my = float(np.median(yy[zone]))
    fl = max(2.0, 0.06 * my)
    scr = float(np.std(cr[zone])) + fl
    scb = float(np.std(cb[zone])) + fl
    sy = float(np.std(yy[zone])) + fl * 1.4
    nrm = np.dstack([(yy - my) / sy, (cr - mcr) / scr, (cb - mcb) / scb]).astype(np.float32)
    return nrm, inface, ys


def _occ_work(aligned_bgr):
    """Downscale to the occlusion working resolution (no-op if already small)."""
    h = aligned_bgr.shape[0]
    wrk = int(_P.get("occ_work", 192) or 192)
    if h <= wrk:
        return aligned_bgr
    return cv2.resize(aligned_bgr, (wrk, wrk), interpolation=cv2.INTER_AREA)


def occluder_keep(aligned_bgr, ref=None, nref=0):
    """1 on the face, 0 on an object held in front of it.

    The previous implementation could not fire AT ALL. It searched a box of
    rows 0.58-0.66 by columns 0.36-0.64 - 2.24% of the crop - and then
    required a connected component of at least 4% of the crop to accept it.
    The minimum area was larger than the entire search window, so the test
    was arithmetically unsatisfiable at every crop size and the function
    returned all-ones on every frame of every job. Verified against ice cream,
    a spoon and a hand: 0.00% removed in every case.

    It had been narrowed to that box, and given that floor, to stop it cutting
    the lower lip and the chin - because its only evidence was "this pixel is
    off the face's median chroma", and lips ARE off it. The replacement keeps
    the lip and chin by using evidence that actually separates them from an
    object, and so does not need to be blinded:

      1. NOT SKIN - off this face's own chroma or luma, in units of the face's
         own spread, so one threshold works on any skin tone under any light.
      2. NOT ITS OWN FACE - differs from the per-pixel appearance model this
         identity has built up in aligned space (see TrackState.occ_ref). A
         beard, glasses, a birthmark or a scar match their own history; ice
         cream in front of them does not. This is what lets the search cover
         the whole face instead of a 2% box, and it is also what catches food
         held OVER a beard, which a location-only exemption cannot.
      3. INTRUDING - the blob reaches outside the face oval or touches the
         crop border. This is the geometric fact that separates the lips from
         a held object: an object is attached to a hand, an arm, a cone, so
         its foreign region continues outside the face. The lips are an
         interior island and never touch the boundary. It is also what keeps
         an open mouth: the cavity is interior too.
      4. BIG ENOUGH - as a fraction of the face area, not of the crop.

    All four must hold. Nothing is cut during the first occ_ref_warm frames,
    while the appearance model is still forming: cutting then would punch a
    hole in a beard before the model can vouch for it.
    """
    try:
        H, W = aligned_bgr.shape[:2]
        small = _occ_work(aligned_bgr)
        h, w = small.shape[:2]
        st = _face_stats(small)
        if st is None:
            return np.ones((H, W), np.float32)
        nrm, inface, ys = st

        if int(nref) < int(_P.get("occ_ref_warm", 12)):
            return np.ones((H, W), np.float32)

        notskin = ((np.sqrt(nrm[:, :, 1] ** 2 + nrm[:, :, 2] ** 2)
                    > float(_P.get("occ_chroma", 2.6)))
                   | (np.abs(nrm[:, :, 0]) > float(_P.get("occ_luma", 2.6))))
        # Eyes and brows are not skin and never an occluder.
        notskin &= ~((ys > 0.24) & (ys < 0.46))

        novel = np.ones((h, w), bool)
        if ref is not None and getattr(ref, "shape", None) == nrm.shape:
            d_ch = np.sqrt((nrm[:, :, 1] - ref[:, :, 1]) ** 2
                           + (nrm[:, :, 2] - ref[:, :, 2]) ** 2)
            d_lu = np.abs(nrm[:, :, 0] - ref[:, :, 0])
            novel = ((d_ch > float(_P.get("occ_ref_chroma", 2.6)))
                     | (d_lu > float(_P.get("occ_ref_luma", 2.6))))

        k = max(3, (h // 64) | 1)
        cand = cv2.morphologyEx((notskin & novel).astype(np.uint8), cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        if int(cand.max()) == 0:
            return np.ones((H, W), np.float32)

        face_area = float(inface.sum())
        outside = ~inface
        border = np.zeros((h, w), bool)
        b = max(2, h // 64)
        border[:b, :] = border[-b:, :] = border[:, :b] = border[:, -b:] = True
        # A band just inside the edge of the face region. Requiring a blob to
        # extend fully OUTSIDE the oval was too brittle: the part of a held
        # object that bridges to the hand is often itself close enough to skin
        # colour not to register, leaving the detected blob entirely inside
        # the oval with nothing outside it to measure. Reaching the EDGE of
        # the face is the robust version of the same fact, and the lips and
        # the mouth cavity sit far inside it - measured, the lip blob stays
        # about 0.16 of the crop clear of this band.
        rim_px = max(2, int(round(h * float(_P.get("occ_rim", 0.055)))))
        rim = inface & ~cv2.erode(
            inface.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rim_px * 2 + 1,) * 2)
        ).astype(bool)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        occ = np.zeros((h, w), np.uint8)
        min_frac = float(_P.get("occ_min_frac", 0.010))
        out_frac = float(_P.get("occ_out_frac", 0.08))
        rim_frac = float(_P.get("occ_rim_frac", 0.02))
        for i in range(1, n):
            comp = labels == i
            a_in = float((comp & inface).sum())
            if a_in / max(1.0, face_area) < min_frac:
                continue
            a_out = float((comp & outside).sum())
            a_tot = max(1.0, a_in + a_out)
            a_rim = float((comp & rim).sum())
            big = (a_in / max(1.0, face_area) >= float(_P.get("occ_big_frac", 0.045))
                   and int(nref) >= 2 * int(_P.get("occ_ref_warm", 12)))
            if (bool((comp & border).any())
                    or a_out / a_tot > out_frac
                    or a_rim / a_tot > rim_frac
                    or big):
                occ[comp] = 255
        if int(occ.max()) == 0:
            return np.ones((H, W), np.float32)
        kf = max(3, (h // 20) | 1)
        soft = cv2.GaussianBlur(occ, (kf, kf), 0).astype(np.float32) / 255.0
        keep = np.clip(1.0 - soft, 0.0, 1.0).astype(np.float32)
        if (h, w) != (H, W):
            keep = cv2.resize(keep, (W, H), interpolation=cv2.INTER_LINEAR)
        return keep
    except Exception as e:
        log.debug("occluder detection skipped: %s", e)
        return np.ones(aligned_bgr.shape[:2], np.float32)


def update_face_ref(ref, nref, aligned_bgr, keep):
    """Learn this face's per-pixel appearance - but never under an occluder.

    Learning where the mask was cut would teach the model the ice cream, and
    the model would then vouch for it on the next frame.
    """
    try:
        small = _occ_work(aligned_bgr)
        st = _face_stats(small)
        if st is None:
            return ref, nref
        nrm = st[0]
        k = keep
        if k.shape[:2] != nrm.shape[:2]:
            k = cv2.resize(k, (nrm.shape[1], nrm.shape[0]), interpolation=cv2.INTER_AREA)
        if ref is None or getattr(ref, "shape", None) != nrm.shape:
            return nrm.copy(), 1
        a = float(_P.get("occ_ref_ema", 0.10)) * (k > 0.9).astype(np.float32)[:, :, None]
        return (ref * (1.0 - a) + nrm * a).astype(np.float32), int(nref) + 1
    except Exception as e:
        log.debug("appearance model update skipped: %s", e)
        return ref, nref


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
                 "pose_ok", "occ_keep", "grain_ema", "mblur_ema",
                 "occ_ref", "occ_nref", "occn_keep", "prof_w", "eat_w", "look_sig", "occ_map")

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
        # Per-pixel model of what THIS face looks like in aligned space, held
        # as NORMALISED residuals against each frame's own skin statistics.
        # The aligned crop is pose-normalised, so a beard, glasses, a
        # birthmark or a scar sits at the same crop coordinates on every frame
        # and matches its own history; an object held in front of the face
        # does not. Residuals rather than absolute YCrCb is what lets it
        # survive a lighting change or a cut - measured, storing absolute
        # values made a 45% light drop read the whole beard as novel and cut
        # 14.5% of the face on every frame afterwards.
        self.occ_ref = None
        self.occ_nref = 0
        self.occn_keep = None
        self.prof_w = None
        self.eat_w = 0.0
        self.look_sig = None
        self.occ_map = None
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
        # Defensive: this used to assume alpha_ema could never be None, which
        # was true until a reset path set it to None and took a whole job down
        # with a TypeError. A smoothing state is never worth crashing a render
        # over - fall back to the construction default and carry on.
        cur = self.alpha_ema
        if cur is None:
            cur = 1.0
            log.debug("alpha_ema was None; recovered to 1.0")
        try:
            cur = float(cur)
        except Exception:
            cur = 1.0
        if t >= 0.94:
            # Snap-up / fast catch-up — never cap below full during stable track.
            self.alpha_ema = max(cur, float(cur * 0.45 + t * 0.55), t)
        elif t < cur:
            # Intentional fade-out (exit / miss budget) — keep slower EMA.
            self.alpha_ema = float(cur * 0.72 + t * 0.28)
        else:
            self.alpha_ema = float(cur * 0.45 + t * 0.55)
        return self.alpha_ema

    def reset_for_cut(self):
        """Forget WHERE this person was and what they looked like - not WHO.

        At a hard cut nothing about the previous shot is evidence about this
        one: the position, the velocity, the cached aligned crop, the colour
        and mask EMAs and the measured grain all describe a frame that no
        longer exists. Carrying them is what pastes the last shot's face onto
        the first frames of the next one, and what makes the geometry glide
        from the old position to the new one over the next half second.

        identity is deliberately KEPT - id_lock, emb, slot_gender. The person
        we were told to swap is still the person we were told to swap; we have
        only lost track of where they are. Clearing identity here would hand a
        virgin slot to whoever the detector happens to see first in the new
        shot, which is the "wrong person bound to the slot" defect this engine
        has been fixed for twice.
        """
        self.bbox = None
        self.obs_bbox = None
        self.kps = None
        self.lmk = None
        self.ab_bbox = None
        self.ab_kps = None
        self.last_hit_bbox = None
        self.last_hit_kps = None
        self.occn_keep = None
        self.prof_w = None
        self.eat_w = 0.0
        self.look_sig = None
        self.occ_map = None
        self.vel_bbox = np.zeros(4, np.float32)
        self.vel_kps = None
        self.kps_scale = None
        # Appearance state measured on the old shot's light.
        self.mask_ema = None
        self.hull_ema = None
        self.occ_keep = None
        self.cm_dmean = None
        self.cm_ratio = None
        self.grain_ema = None
        self.mblur_ema = None
        # 1.0, NOT None. __init__ sets this to 1.0 and every other assignment
        # in this class is a float; smooth_alpha does `float(self.alpha_ema)`
        # with no guard because the field was never None in the first place.
        # Nulling it here broke that invariant and crashed the job on the
        # first frame after a shot cut:
        #   TypeError: float() argument must be ... not 'NoneType'
        # The opacity EMA is not stale evidence about the old shot - it is a
        # smoothing state whose correct post-cut value is "fully opaque",
        # exactly as at construction.
        self.alpha_ema = 1.0
        # The cached aligned crop IS the old shot's face. This is the one that
        # produces the reported "the pasted face holds" on a scene change.
        self.fake = None
        self.fake_corr = None
        self.fake_size = 0
        self.fake_frame = -10**9
        # Treat the person as not currently seen, so the next real detection is
        # a re-acquisition rather than a continuation, but do not kill the
        # track outright - its identity is still valid.
        self.missed = max(int(self.missed), 1)
        self.hits = 0
        self.confirm_hits = 0
        self._confirm_ease = 0
        self._paste_frozen = False
        self._reacquired = True
        self.crossing = False
        self.flip_votes = 0
        self.pose_ok = False
        self.last_face = None
        self.det_score = 0.0

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


# SDOS-075 attribution: how often the occluder gate actually cut anything.
# Without this, "is the new occluder gate causing the reverts?" can only be
# answered by guessing.
OCC_STATS = {"frames": 0, "cut_frames": 0, "cut_area_sum": 0.0}

# --------------------------------------------------------------------------
# Occlusion net (SDOS-091): a trained face-occlusion segmenter decides what is
# in front of the mouth and chin.
#
# The colour/novelty gate above could not tell food from lips on real faces
# (SDOS-083..087: 8-76 % of a scoop found depending on its colour, and on real
# footage it cut 82-91 % of all face-frames). This is FaceFusion's "face
# occluder" (XSeg, DeepFaceLab lineage, 256x256 BGR in, 1 = face / 0 =
# occluded out), trained on real occlusions. Measured on 12 real faces: every
# pixel of a realistic scoop at the mouth, the lips or a cheek is found, for
# all six colours tested, and nothing is cut on a clean face (normal, dim,
# dark+noise, side light, blur). Two rules keep the rest of the swap exactly
# as it was:
#   * only INSIDE the face's own landmark outline (eroded), so the edge of the
#     swap (hairline, jaw, ears) is not re-shaped;
#   * only pieces that reach the mouth/chin zone, so hair falling over a cheek
#     (which the model rightly calls an occluder) does not cut the face.
# --------------------------------------------------------------------------
OCCNET_STATS = {"frames": 0, "cut_frames": 0, "cut_area_sum": 0.0, "ms": 0.0}
_OCCNET = {"sess": None, "name": None, "error": None}


def occ_net_load(path, threads=2):
    """Load the occlusion model; never raises. Returns True when ready."""
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(threads))
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        shp = sess.get_inputs()[0].shape
        if len(shp) != 4 or int(shp[1]) != 256 or int(shp[3]) != 3:
            raise RuntimeError("unexpected input shape %s" % (shp,))
        _OCCNET["sess"] = sess
        _OCCNET["name"] = os.path.basename(str(path))
        _OCCNET["error"] = None
        return True
    except Exception as e:
        _OCCNET["sess"] = None
        _OCCNET["error"] = "%s: %s" % (type(e).__name__, e)
        return False


_OCCNET_RAW = {"key": None, "m": None}
PROFILE_STATS = {"frames": 0, "w_sum": 0.0}
SIDE_EYE_STATS = {"faces": 0}


def occ_net_raw(aligned_bgr):
    """The occlusion net's own face map (256x256, 1 = face) for an aligned crop; None without the net.
    The last result is kept, so the profile mask and the mouth/chin rule share one forward pass per crop."""
    sess = _OCCNET.get("sess")
    if sess is None or aligned_bgr is None:
        return None
    key = (aligned_bgr.shape, aligned_bgr.__array_interface__["data"][0], int(aligned_bgr[::7, ::5].sum()))
    if _OCCNET_RAW["key"] == key:
        return _OCCNET_RAW["m"]
    W = 256
    x = cv2.resize(aligned_bgr, (W, W), interpolation=cv2.INTER_LINEAR).astype(np.float32)[None] / 255.0
    m = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, :, :, 0]
    m = np.clip(m, 0.0, 1.0).astype(np.float32)
    _OCCNET_RAW["key"], _OCCNET_RAW["m"] = key, m
    return m


def profile_weight(kps):
    """0 for any pose up to about 45 deg of yaw, rising to 1 at about 55-60 deg (calibrated on AFLW2000's true yaw).

    The nose's sideways offset from the eye midpoint, roll-corrected, in eye distances. Frontal 0.0-0.25,
    36-45 deg 0.34-0.51, 55-71 deg 0.76-1.48. Raised cosine between profile_lo and profile_hi."""
    if kps is None or float(_P.get("profile_mask", 0.0) or 0.0) <= 0.0:
        return 0.0
    try:
        k = np.asarray(kps, np.float32).reshape(-1, 2)
        if k.shape[0] < 3:
            return 0.0
        le, re, nose = k[0], k[1], k[2]
        ev = re - le
        ed = float(np.linalg.norm(ev)) + 1e-6
        roll = float(np.arctan2(ev[1], ev[0]))
        c, s = math.cos(-roll), math.sin(-roll)
        rel = nose - (le + re) * 0.5
        h = abs(float(rel[0] * c - rel[1] * s)) / ed
        lo, hi = float(_P.get("profile_lo", 0.45)), float(_P.get("profile_hi", 0.70))
        if h <= lo:
            return 0.0
        if h >= hi:
            return 1.0
        return float(0.5 - 0.5 * math.cos(math.pi * (h - lo) / (hi - lo)))
    except Exception:
        return 0.0


def with_forehead(lmk106):
    """106 landmarks plus an estimated forehead (v11.2.95): the 106-point outline stops at the eyebrows, so the other
    person's forehead and hairline were not protected in contact. The jaw outline (points 0-32) is mirrored across the
    eye line and shortened to 70 % - the usual forehead estimate."""
    if lmk106 is None:
        return None
    try:
        L = np.asarray(lmk106, np.float32).reshape(-1, 2)
        if L.shape[0] < 97:
            return L
        e1, e2 = L[33:43].mean(0), L[87:97].mean(0)
        d = e2 - e1
        n = np.array([-d[1], d[0]], np.float32) / (float(np.linalg.norm(d)) + 1e-6)   # normal to the eye line
        c = (e1 + e2) * 0.5
        jaw = L[0:33]
        off = (jaw - c) @ n                                  # signed distance from the eye line (jaw side)
        if float(np.median(off)) < 0:
            n, off = -n, -off
        top = jaw - n[None, :] * (off[:, None] * (1.0 + 0.7))
        return np.vstack([L, top[off > 0]]).astype(np.float32)
    except Exception:
        return lmk106


def side_eye_kps(kps, lmk106):
    """v11.2.94 (SDOS-094): in a side view, place the VISIBLE eye where the 106-point model puts it.

    On AFLW2000's side views (50-71 deg, true 3-D landmarks) the detector's 5-point eye was 1.0-2.9 % of the face
    size off for the visible eye, the 106-point eye contour's centre 0.2-1.3 % (better on 5 of 6). For the hidden
    eye neither is reliably better, so it is left as detected. Same ramp as the side-view mask (profile_weight);
    the landmark eye is ignored when it disagrees by more than half an eye distance (a failed landmark fit).
    Returns new kps (a copy) or the input unchanged."""
    s_eye = float(_P.get("side_eye", 1.0) or 0.0)
    if kps is None or lmk106 is None or s_eye <= 0.0:
        return kps
    try:
        k = np.asarray(kps, np.float32).reshape(-1, 2)
        L = np.asarray(lmk106, np.float32).reshape(-1, 2)
        if k.shape[0] < 5 or L.shape[0] < 97:
            return kps
        w = profile_weight(k) * s_eye
        if w <= 0.0:
            return kps
        ed = float(np.linalg.norm(k[1] - k[0])) + 1e-6
        near = 0 if np.linalg.norm(k[0] - k[2]) > np.linalg.norm(k[1] - k[2]) else 1
        cands = [L[33:43].mean(0), L[87:97].mean(0)]
        e = min(cands, key=lambda p: float(np.linalg.norm(p - k[near])))
        if float(np.linalg.norm(e - k[near])) > 0.5 * max(ed, 1.0) and float(np.linalg.norm(e - k[near])) > 6.0:
            return kps
        out = k.copy()
        out[near] = k[near] * (1.0 - w) + e * w
        SIDE_EYE_STATS["faces"] += 1
        return out.astype(np.float32)
    except Exception:
        return kps


def profile_mask(aligned_bgr, pts_aligned, size):
    """Paint weight for an extreme side view: the visible face's own outline, None when it cannot be built.

    Face map of the occlusion net (it segments the face on profiles, not just frontal faces) times the landmark
    outline dilated a little (so a neck, a hand or another face next to it is never taken for this face), with a
    short feather and the template's crop-border roll-off. Without the net: the landmark outline alone."""
    try:
        hull = None
        if pts_aligned is not None:
            hm = np.zeros((size, size), np.uint8)
            cv2.fillConvexPoly(hm, cv2.convexHull(np.round(np.asarray(pts_aligned, np.float32)).astype(np.int32)), 255)
            d = int(max(1, round(size * 0.04)))
            hm = cv2.dilate(hm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)))
            hull = hm.astype(np.float32) / 255.0
        m = occ_net_raw(aligned_bgr)
        if m is not None:
            seg = cv2.resize(m, (size, size), interpolation=cv2.INTER_LINEAR)
            seg = np.clip((seg - 0.35) / 0.30, 0.0, 1.0)
            pm = seg if hull is None else seg * hull
        elif hull is not None:
            pm = hull
        else:
            return None
        k = int(max(3, round(size * float(_P.get("profile_feather", 0.025)) * 2))) | 1
        pm = cv2.GaussianBlur(pm.astype(np.float32), (k, k), 0)
        b = float(_P.get("mask_border", 0.08) or 0.0)
        if b > 0.0:
            g = (np.arange(size, dtype=np.float32) + 0.5) / size
            e = np.minimum(g, 1.0 - g)
            pm *= np.clip(np.minimum(e[:, None], e[None, :]) / b, 0.0, 1.0)
        return np.clip(pm, 0.0, 1.0).astype(np.float32)
    except Exception as e:
        log.debug("profile mask skipped: %s", e)
        return None


EAT_STATS = {"frames": 0, "w_sum": 0.0}
LOOK_STATS = {"jumps": 0}


def _look_sig(crop):
    """Coarse look of an aligned face crop (32x32 grey of its blurred centre), comparable between crop sizes."""
    n = crop.shape[0]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    a, b = int(n * 0.22), int(n * 0.80)
    g = cv2.GaussianBlur(g[a:b, a:b], (0, 0), max(0.8, n / 85.0))
    return cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
_LIPS = list(range(52, 72))      # insightface 2d106det: outer and inner lip contour
NOVEL_STATS = {"frames": 0, "cut_frames": 0}
OWN_STATS = {"frames": 0}
CONTACT_STATS = {"frames": 0, "full": 0}


def occ_net_novel(track, pts_aligned, size, M=None, frame_wh=None):
    """v11.2.95 (SDOS-095): objects at the EDGE of the face (forehead above the brows, cheek edge, jaw) - where the
    occlusion net's own face/not-face boundary already sits on every clean face (hairline, ears, jaw), so a plain
    threshold there cut 5.5 % of every clean face. Here only CHANGE is cut: each face keeps a running memory of the
    net's face map in its aligned crop, and in the ring between the shrunk and the grown landmark outline a piece that
    WAS face and now is not (an object arriving) is cut on the frame it appears. The memory never learns pixels it is
    cutting, so a hand that stays stays cut; it restarts at a cut / look jump. Returns a keep map (1 = paint) or None."""
    if track is None or pts_aligned is None:
        return None
    m = _OCCNET_RAW.get("m")
    if m is None:
        return None
    try:
        W = m.shape[0]
        mb = cv2.GaussianBlur(m, (0, 0), float(_P.get("occ_net_blur", 3.0)))
        hm = np.zeros((W, W), np.uint8)
        pts = np.asarray(pts_aligned, np.float32) * (W / float(size))
        cv2.fillConvexPoly(hm, cv2.convexHull(np.round(pts).astype(np.int32)), 1)
        gi = int(round(float(_P.get("occ_novel_grow", 0.10)) * W)) | 1
        ei = int(round(float(_P.get("occ_net_hull_erode", 0.08)) * W)) | 1
        grown = cv2.dilate(hm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gi, gi))).astype(bool)
        inner = cv2.erode(hm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ei, ei))).astype(bool)
        ring = grown & ~inner
        if M is not None and frame_wh is not None:
            IM = cv2.invertAffineTransform(np.asarray(M, np.float32).reshape(2, 3))
            g = (np.arange(W, dtype=np.float32) + 0.5) * (size / float(W)) - 0.5
            gx, gy = np.meshgrid(g, g)
            ix = IM[0, 0] * gx + IM[0, 1] * gy + IM[0, 2]
            iy = IM[1, 0] * gx + IM[1, 1] * gy + IM[1, 2]
            fw, fh = float(frame_wh[0]), float(frame_wh[1])
            ring &= ~cv2.dilate(((ix < 0.5) | (iy < 0.5) | (ix > fw - 1.5) | (iy > fh - 1.5)).astype(np.uint8),
                                np.ones((9, 9), np.uint8)).astype(bool)
        ref = getattr(track, "occ_map", None)
        keep = np.ones((W, W), np.float32)
        cutmask = np.zeros((W, W), bool)
        if ref is not None and ref.shape == mb.shape:
            novel = (mb < 0.5) & (ref > float(_P.get("occ_novel_ref", 0.75))) & ring
            novel = cv2.morphologyEx(novel.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
            if novel.any():
                need = float(_P.get("occ_novel_min", 0.003)) * W * W
                n, lab = cv2.connectedComponents(novel.astype(np.uint8), connectivity=8)
                for c in range(1, n):
                    comp = lab == c
                    if float(comp.sum()) >= need:
                        keep[comp] = 0.0
                        cutmask |= comp
            # learn slowly, never from what is being cut
            upd = ~cutmask
            ref = ref.copy()
            ref[upd] = ref[upd] * 0.9 + mb[upd] * 0.1
        else:
            ref = mb.copy()
        track.occ_map = ref.astype(np.float32)
        NOVEL_STATS["frames"] += 1
        if cutmask.any():
            NOVEL_STATS["cut_frames"] += 1
        keep = cv2.GaussianBlur(keep, (0, 0), 2.0)
        if size != W:
            keep = cv2.resize(keep, (size, size), interpolation=cv2.INTER_LINEAR)
        return np.clip(keep, 0.0, 1.0).astype(np.float32)
    except Exception as e:
        log.debug("novel occluder skipped: %s", e)
        return None


def _eating_mouth(mask, keep, pts_aligned, size, track=None):
    """Show the footage's own lips while an object is at the mouth (see eat_mouth). Returns the new mask."""
    try:
        pts = np.asarray(pts_aligned, np.float32).reshape(-1, 2)
        if pts.shape[0] < 72:
            return mask
        lips = pts[_LIPS]
        lm = np.zeros((size, size), np.uint8)
        cv2.fillConvexPoly(lm, cv2.convexHull(np.round(lips).astype(np.int32)), 1)
        pad = int(max(1, round(size * float(_P.get("eat_lip_pad", 0.035)))))
        lips_pad = cv2.dilate(lm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1)))
        mw = float(lips[:, 0].max() - lips[:, 0].min())
        zr = int(max(3, round(mw * 0.6)))
        zone = cv2.dilate(lm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * zr + 1, 2 * zr + 1)))
        cut = (keep < 0.5) & (zone > 0) & (lips_pad == 0)
        frac = float(cut.sum()) / float(size * size)
        lo, hi = float(_P.get("eat_lo", 0.002)), float(_P.get("eat_hi", 0.008))
        e = 0.0 if frac <= lo else 1.0 if frac >= hi else 0.5 - 0.5 * math.cos(math.pi * (frac - lo) / (hi - lo))
        if track is not None:
            prev = float(getattr(track, "eat_w", 0.0) or 0.0)
            a = float(_P.get("eat_attack", 0.6)) if e > prev else float(_P.get("eat_release", 0.12))
            e = prev + (e - prev) * a
            track.eat_w = e
        if e < 0.01:
            e = 0.0
        if e <= 0.0:
            return mask
        k = int(max(3, round(size * 0.05))) | 1
        region = cv2.GaussianBlur(lips_pad.astype(np.float32), (k, k), 0)
        EAT_STATS["frames"] += 1
        EAT_STATS["w_sum"] += e
        return (mask * (1.0 - e * np.clip(region, 0.0, 1.0))).astype(np.float32)
    except Exception as ex:
        log.debug("eating mouth skipped: %s", ex)
        return mask


def occ_net_keep(aligned_bgr, pts_aligned, size, M=None, frame_wh=None):
    """1 = paint the swap, <1 = something is in front of the mouth/chin here. None when the net is unavailable or there is no outline."""
    sess = _OCCNET.get("sess")
    if sess is None or aligned_bgr is None or pts_aligned is None:
        return None
    try:
        t0 = time.perf_counter()
        W = 256
        m = occ_net_raw(aligned_bgr)
        if m is None:
            return None
        m = (np.clip(cv2.GaussianBlur(m, (0, 0), float(_P.get("occ_net_blur", 3.0))), 0.5, 1.0) - 0.5) * 2.0
        hull = np.zeros((W, W), np.uint8)
        pts = np.asarray(pts_aligned, np.float32) * (W / float(size))
        cv2.fillConvexPoly(hull, cv2.convexHull(np.round(pts).astype(np.int32)), 1)
        _he = float(_P.get("occ_net_hull_erode", 0.08))
        if _he >= 0.0:
            e = int(round(_he * W)) | 1
            hull = cv2.erode(hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e))).astype(bool)
        else:   # v11.2.95: negative = grow the outline (forehead above the brows, where the swap still paints)
            e = int(round(-_he * W)) | 1
            hull = cv2.dilate(hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e))).astype(bool)
        keep = np.ones((W, W), np.float32)
        occ = (m < 0.5) & hull
        # Where the crop runs off the picture it is filled with stretched edge pixels, which the model reads as an
        # occluder. Nothing that touches that part may be cut (a face at the frame edge stays as it was).
        outside = None
        if M is not None and frame_wh is not None:
            try:
                IM = cv2.invertAffineTransform(np.asarray(M, np.float32).reshape(2, 3))
                g = (np.arange(W, dtype=np.float32) + 0.5) * (size / float(W)) - 0.5
                gx, gy = np.meshgrid(g, g)
                ix = IM[0, 0] * gx + IM[0, 1] * gy + IM[0, 2]
                iy = IM[1, 0] * gx + IM[1, 1] * gy + IM[1, 2]
                fw, fh = float(frame_wh[0]), float(frame_wh[1])
                outside = (ix < 0.5) | (iy < 0.5) | (ix > fw - 1.5) | (iy > fh - 1.5)
                if bool(outside.any()):
                    outside = cv2.dilate(outside.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
                else:
                    outside = None
            except Exception:
                outside = None
        if bool(occ.any()):
            yy, xx = np.mgrid[0:W, 0:W]
            cx = float(_P.get("occ_net_core_x", 0.17))
            core = (np.abs(xx / float(W) - 0.5) < cx) & (yy / float(W) >= float(_P.get("occ_net_core_y", 0.58)))
            need = float(_P.get("occ_net_core_min", 0.002)) * W * W
            # v11.2.95 (SDOS-095): occ_net_zone 1 = anything in front of the face anywhere inside its outline (a hand
            # over an eye, a cup at the forehead, hair across a cheek), not only at the mouth / chin; a piece must
            # cover at least occ_net_face_min of the crop.
            anywhere = float(_P.get("occ_net_zone", 0.0) or 0.0) >= 1.0
            need_any = float(_P.get("occ_net_face_min", 0.004)) * W * W
            n, lab = cv2.connectedComponents(occ.astype(np.uint8), connectivity=8)
            for c in range(1, n):
                comp = lab == c
                if outside is not None and bool((comp & outside).any()):
                    continue
                if float((comp & core).sum()) >= need or (anywhere and float(comp.sum()) >= need_any):
                    keep[comp] = m[comp]
        if size != W:
            keep = cv2.resize(keep, (size, size), interpolation=cv2.INTER_LINEAR)
        OCCNET_STATS["ms"] += (time.perf_counter() - t0) * 1000.0
        return np.clip(keep, 0.0, 1.0).astype(np.float32)
    except Exception as e:
        log.debug("occlusion net skipped: %s", e)
        return None

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
                   aligned_target=None, rivals=None, frame_wh=None):
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

        # v11.2.93: a face whose look jumps from one frame to the next (a shot change the cut detector missed: 60+
        # grey levels of the coarse face; a fast turn reaches 6) must not inherit the previous shot's mouth, side-view
        # or occlusion state - that blended the previous shot into the new face for a few frames.
        if track is not None and aligned_target is not None:
            try:
                sig = _look_sig(aligned_target)
                prev_sig = getattr(track, "look_sig", None)
                if prev_sig is not None and prev_sig.shape == sig.shape and \
                        float(np.abs(sig - prev_sig).mean()) > float(_P.get("look_jump", 20.0)):
                    track.eat_w = 0.0
                    track.prof_w = None
                    track.occn_keep = None
                    track.occ_map = None
                    LOOK_STATS["jumps"] += 1
                track.look_sig = sig
            except Exception:
                pass

        # Extreme side view (SDOS-092): move over to the face's own outline. 0 below ~45 deg = mask unchanged.
        pw = profile_weight(getattr(face, "kps", None))
        if track is not None:
            prev_pw = getattr(track, "prof_w", None)
            pw = pw if prev_pw is None else 0.6 * float(prev_pw) + 0.4 * pw
            track.prof_w = pw
        if pw < 0.005:
            pw = 0.0
        if pw > 0.0 and aligned_target is not None:
            pm = profile_mask(aligned_target, pts, size)
            if pm is not None:
                mask = mask * (1.0 - pw) + pm * pw
                try:
                    PROFILE_STATS["frames"] += 1
                    PROFILE_STATS["w_sum"] += pw
                except Exception:
                    pass

        # v11.2.97 (SDOS-097): where another face touches this one, the face-against-face rule alone decides who owns
        # each pixel; the occluder net saw the other face's lips / cheek as an "object" and cut a slightly different
        # piece of this face every frame - the flicker between original and swap in a full kiss
        _rival_zone = None
        _cz = float(_P.get("contact_calm", 0.0) or 0.0)
        if rivals and _cz > 0.0:
            _zw = int(max(1, round(size * _cz))) * 2 + 1
            _rz = np.zeros((size, size), np.uint8)
            for rv in rivals:
                lm = rv.get("lmk") if isinstance(rv, dict) else rv
                rp_ = _landmarks_to_aligned(with_forehead(lm), M, size)
                if rp_ is not None:
                    cv2.fillConvexPoly(_rz, cv2.convexHull(np.round(rp_).astype(np.int32)), 1)
            if _rz.any():
                _rz = cv2.dilate(_rz, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_zw, _zw)))
                _fk = int(max(3, round(size * 0.03))) | 1
                _rival_zone = np.clip(cv2.GaussianBlur(_rz.astype(np.float32), (_fk, _fk), 0) * 1.5, 0.0, 1.0)
        if rivals and float(_P.get("own_pixels", 0.0) or 0.0) >= 1.0:
            # v11.2.95 (SDOS-095): every pixel belongs to ONE face. A face in front is cut out of this one along its
            # own tight outline at full strength; a face BEHIND this one is cut out of this face wherever it lies
            # outside this face's own outline (this face's soft oval edge used to paint onto the other cheek).
            gw = int(max(1, round(size * float(_P.get("own_grow", 0.02))))) * 2 + 1
            fk = int(max(3, round(size * float(_P.get("own_feather", 0.03))))) | 1
            nr = float(_P.get("own_near", 0.12) or 0.0)
            near = int(round(size * nr)) * 2 + 1 if nr > 0 else 0
            own = None
            if pts is not None:
                om = np.zeros((size, size), np.uint8)
                cv2.fillConvexPoly(om, cv2.convexHull(np.round(with_forehead(pts)).astype(np.int32)), 1)
                ow = int(max(1, round(size * 0.03))) * 2 + 1
                own = cv2.dilate(om, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ow, ow))).astype(bool)
            block = np.zeros((size, size), np.float32)
            for rv in rivals:
                lm, front = (rv.get("lmk"), bool(rv.get("front", True))) if isinstance(rv, dict) else (rv, True)
                rp = _landmarks_to_aligned(with_forehead(lm), M, size)
                if rp is None:
                    continue
                rm = np.zeros((size, size), np.uint8)
                cv2.fillConvexPoly(rm, cv2.convexHull(np.round(rp).astype(np.int32)), 1)
                rm = cv2.dilate(rm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gw, gw))).astype(bool)
                if not front:
                    if own is None:
                        continue
                    rm &= ~own
                elif own is not None and near > 0:
                    # This face is BEHIND: next to the front face it keeps only its own outline (its soft oval used
                    # to run past its jaw onto the front person's neck, ear and hair, which the face hull misses).
                    zone = cv2.dilate(rm.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (near, near)))
                    rm |= zone.astype(bool) & ~own
                block[rm] = 1.0
            if float(block.max()) > 0.0:
                block = cv2.GaussianBlur(block, (fk, fk), 0)
                mask = mask * (1.0 - float(_P.get("rival_cut", 1.0)) * np.clip(block, 0.0, 1.0))
                OWN_STATS["frames"] += 1
        elif rivals:
            rivals = [rv.get("lmk") if isinstance(rv, dict) else rv for rv in rivals
                      if not isinstance(rv, dict) or rv.get("front", True)]
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
        if aligned_target is not None and float(_P.get("occ_colour", 1.0) or 0.0) > 0.0:
            keep = occluder_keep(aligned_target,
                                 ref=getattr(track, "occ_ref", None) if track is not None else None,
                                 nref=getattr(track, "occ_nref", 0) if track is not None else 0)
            if track is not None:
                prev = getattr(track, "occ_keep", None)
                if prev is None or getattr(prev, "shape", None) != keep.shape:
                    track.occ_keep = np.ones_like(keep)
                else:
                    # The hole used to open at 0.10 per frame, which is 22
                    # frames to reach 90% of it - most of a second of the new
                    # face being painted over the spoon before the hole was
                    # even open, and for food that moves to and from the mouth
                    # every second it never opened at all. Opening is now the
                    # FAST direction (occ_open) because the detector it is
                    # smoothing no longer fires on lips, an open mouth, a
                    # beard, glasses, a shadow or a lighting change; closing
                    # stays slower, as cheap insurance against a blink.
                    o = float(_P.get("occ_open", 0.45))
                    c = float(_P.get("occ_close", 0.22))
                    a = np.where(keep < track.occ_keep, o, c).astype(np.float32)
                    track.occ_keep = (track.occ_keep * (1.0 - a)
                                      + keep * a).astype(np.float32)
                keep = track.occ_keep
                try:
                    OCC_STATS["frames"] += 1
                    _cut = float(1.0 - float(keep.mean()))
                    if _cut > 0.002:
                        OCC_STATS["cut_frames"] += 1
                        OCC_STATS["cut_area_sum"] += _cut
                except Exception:
                    pass
                # Learn AFTER deciding, from the decision, so the model is
                # never taught the occluder it just rejected.
                track.occ_ref, track.occ_nref = update_face_ref(
                    track.occ_ref, track.occ_nref, aligned_target, keep)
            mask = mask * keep
        if aligned_target is not None and pts is not None and float(_P.get("occ_net", 0.0) or 0.0) > 0.0:
            nk = occ_net_keep(aligned_target, pts, size, M=M, frame_wh=frame_wh)
            # (not in a side view: there the mask already IS the net's face outline - SDOS-092 - and the edge rule only
            # added small false cuts as that outline moved; measured 36 of 120 frames on a slow side-view clip)
            if nk is not None and float(_P.get("occ_novel", 0.0) or 0.0) > 0.0 and pw < 0.5:
                nv = occ_net_novel(track, pts, size, M=M, frame_wh=frame_wh)
                if nv is not None:
                    nk = np.minimum(nk, nv)
            if nk is not None and _rival_zone is not None:
                if float(_rival_zone.mean()) >= float(_P.get("contact_full", 0.15)):
                    # heavy contact (a kiss): most of this crop is the OTHER face, and the net - trained on one face per
                    # crop - takes this face's own visible part for "not face", a little differently every frame
                    nk = np.ones_like(nk)
                    CONTACT_STATS["full"] += 1
                else:
                    nk = (1.0 - (1.0 - nk) * (1.0 - _rival_zone)).astype(np.float32)
                CONTACT_STATS["frames"] += 1
            if nk is not None:
                if track is not None:
                    prev = getattr(track, "occn_keep", None)
                    if prev is not None and getattr(prev, "shape", None) == nk.shape:
                        # a cut follows at once; giving the face back takes a few frames, so one missed frame does not flash
                        nk = np.minimum(nk, 0.5 * prev + 0.5 * nk).astype(np.float32)
                    track.occn_keep = nk
                try:
                    OCCNET_STATS["frames"] += 1
                    _c = float((nk < 0.5).mean())
                    if _c > 0.001:
                        OCCNET_STATS["cut_frames"] += 1
                        OCCNET_STATS["cut_area_sum"] += _c
                except Exception:
                    pass
                mask = mask * nk
                if float(_P.get("eat_mouth", 0.0) or 0.0) > 0.0 and not bool(getattr(face, "predicted", False)):
                    # only on a real detection: a held / predicted face is not where its mouth was measured
                    mask = _eating_mouth(mask, nk, pts, size, track)
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
                               rivals=rivals,
                               frame_wh=(int(orig.shape[1]), int(orig.shape[0])))
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
        # Relative-admission bookkeeping (SDOS-078). _rel_pend[slot] is the last
        # detection that was a candidate for rescue: (bbox, consecutive count,
        # call number). A rescue needs trk_rel_confirm consecutive, spatially
        # consistent detections, so a one-frame look-alike never binds.
        self._rel_pend = {}
        self._call_n = 0
        # Why faces were NOT bound, per slot. Without this a job that binds
        # nothing for minutes is indistinguishable from one with nobody there.
        self.diag = {s: {"id_reject": 0, "rel_rescued": 0, "rel_waiting": 0,
                         "rel_no_margin": 0, "rel_below_floor": 0}
                     for s in slots}

    def _rel_admit(self, i, j, s, f, sim, sim_all, has_id_all, slots):
        """Admit a face the absolute gate rejected, if it is clearly THIS person.

        Returns True to admit. See _P["trk_rel_floor"] for why this exists.
        Deliberately conservative on every axis - this is the gate that decides
        whether a bystander gets the swap, and a wrong-person swap is worse than
        a missed frame."""
        d = self.diag.setdefault(s, {"id_reject": 0, "rel_rescued": 0,
                                     "rel_waiting": 0, "rel_no_margin": 0,
                                     "rel_below_floor": 0})
        floor = float(_P.get("trk_rel_floor", 0.25))
        margin = float(_P.get("trk_rel_margin", 0.15))
        need = max(1, int(_P.get("trk_rel_confirm", 2)))
        if sim < floor:
            d["rel_below_floor"] += 1
            # NOTE: deliberately does NOT clear _rel_pend. Faces are evaluated
            # one after another within a call, so a different face failing the
            # floor says nothing about the candidate being confirmed - clearing
            # here reset the real candidate's count on every call (the other
            # person is usually evaluated first) and it never reached
            # trk_rel_confirm. Staleness is handled by the consecutive-call
            # check below instead.
            return False
        others = [sim_all[i2][j] for i2 in range(len(slots))
                  if i2 != i and has_id_all[i2]]
        if not others:
            # Nothing to be RELATIVE to - with a single identified slot there is
            # no margin to measure, so keep the absolute rule.
            return False
        if sim - max(others) < margin:
            d["rel_no_margin"] += 1
            return False
        try:
            if self._belongs_to_other(s, f):
                return False
        except Exception:
            pass
        bb = np.asarray(f.bbox, np.float32).reshape(-1)[:4]
        pend = self._rel_pend.get(s)
        if (pend is not None and pend[2] == self._call_n - 1
                and _iou(pend[0], bb) >= 0.25):
            n = pend[1] + 1
        else:
            n = 1
        self._rel_pend[s] = (bb, n, self._call_n)
        if n < need:
            d["rel_waiting"] += 1
            return False
        d["rel_rescued"] += 1
        return True

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
        best = None
        if lock is not None:
            best = float(np.dot(emb, lock))
        elif track is not None and track.emb is not None:
            best = float(np.dot(emb, track.emb))
        # The Detect-frame reference is the identity the USER chose. It used to
        # be ignored the moment the slot locked onto its first bound frame -
        # a single frame, at whatever pose and light that happened to be - so
        # the stored lock silently outranked the one reference known to be a
        # good, deliberate picture of this person. Keep it as a second
        # prototype: the face matches if it matches EITHER.
        if ref is not None:
            try:
                r = float(_ref_cos(emb, ref))
            except Exception:
                r = float(np.dot(emb, np.asarray(ref, np.float32)))
            best = r if best is None else max(best, r)
        return -1.0 if best is None else best

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
    def mark_cut(self):
        """A hard cut happened. Reset every track's geometry, keep identities."""
        n = 0
        self._rel_pend.clear()
        for tr in self.tracks.values():
            if tr is None:
                continue
            try:
                tr.reset_for_cut()
                n += 1
            except Exception as e:
                log.debug("cut reset failed on slot %s: %s", getattr(tr, "slot", "?"), e)
        return n

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

        self._call_n += 1
        # (slot index, face index) pairs the relative-admission rule let through
        # the absolute gate. The post-pairing check further down has ITS OWN
        # absolute floor (0.32) and has to be told about these, or it quietly
        # throws away exactly the faces the rescue just admitted.
        rescued = set()
        # Every slot's similarity to every face, computed up front. The main
        # loop below visits slot 0 first, when slot 1's row has not been filled
        # in yet, and the relative-admission rule needs all of them.
        sim_all = [[self._identity(f, self.tracks[s_], refs.get(s_) if refs else None)
                    for f in faces] for s_ in slots]
        has_id_all = [
            (getattr(self.tracks[s_], "id_lock", None) is not None
             or self.tracks[s_].emb is not None
             or (refs.get(s_) if refs else None) is not None)
            for s_ in slots]

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
                    _abs_ok = True
                    if long_gap:
                        if sim < 0.40:
                            _abs_ok = False
                    elif sim < 0.38 and not (spatial >= 0.55 and sim >= 0.22):
                        _abs_ok = False

                    if not _abs_ok:
                        # The absolute wall rejected it. Before dropping a face
                        # that may simply be this person at a different pose or
                        # light, ask whether it is clearly THEM relative to the
                        # other people - see _rel_admit.
                        if not self._rel_admit(i, j, s, f, sim, sim_all,
                                               has_id_all, slots):
                            self.diag[s]["id_reject"] += 1
                            continue
                        rescued.add((i, j))
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
            elif (getattr(tr, "id_lock", None) is not None and lock_sim < 0.32
                    and (i, j) not in rescued):
                # Below the absolute floor. This used to be an unconditional
                # "not this slot's person". The gate above admits faces down to
                # 0.22 when they sit where the track expects its person (the
                # spatial rescue) and this line then discarded them, so that
                # rescue never worked - but it also happened to be the only
                # thing stopping a bystander standing in the tracked person's
                # spot at 0.22-0.32 from taking the slot. Removing the line
                # outright was measured to raise the bystander false-bind rate
                # from 0.4% to 2.0% (and 0% to 43% for a sustained look-alike).
                # So replace the blind floor with the real question - is this
                # clearly THEM relative to the other people, seen consistently -
                # which is exactly what _rel_admit asks.
                if not self._rel_admit(i, j, s, f, lock_sim, sim_all,
                                       has_id_all, slots):
                    self.diag[s]["post_reject"] = self.diag[s].get("post_reject", 0) + 1
                    tr.predict(dt_frames)
                    continue
                self.diag[s]["rel_rescued_post"] = self.diag[s].get("rel_rescued_post", 0) + 1

            # v11.2.0: fail-closed if landmarks missing (core usually pre-filters
            # with _kps_reliable; this is defense-in-depth for HairGate confirm).
            _kps = getattr(f, "kps", None)
            _kps_ok = _kps is not None and len(_kps) >= 5
            tr.update(f, update_embedding=good, dt_frames=dt_frames, kps_ok=_kps_ok)
            if changed:
                tr.reset_appearance()
            result[s] = f
            # This slot is now TRACKING this face. The confirmation count in
            # _rel_admit exists to vet a NEW acquisition; a face the slot has
            # just been bound to, still at the same place, must not have to
            # re-earn trust each time its similarity wobbles across a
            # threshold. Without this, a face hovering around 0.32 was bound on
            # the frames above it, sent through the rescue on the frames below
            # it, and the frames above never advanced the rescue's
            # "consecutive" chain - measured recall 60-75% at a similarity of
            # 0.33 against ~99% once the chain is carried.
            try:
                self._rel_pend[s] = (np.asarray(f.bbox, np.float32).reshape(-1)[:4],
                                     max(1, int(_P.get("trk_rel_confirm", 2))),
                                     self._call_n)
            except Exception:
                pass

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


# =====================================================================================================================
# PATCH REMOVAL (v11.2.96, SDOS-096)
#
# The user paints over a mark (tattoo, mole, dark spot, any shape) on ONE frame; it is removed on every frame of the job.
#
#   fill        exemplar inpainting (Criminisi et al. 2004, "Region filling and object removal by exemplar-based image
#               inpainting"): the hole is filled from its edge inwards with real patches of the skin around it, each
#               chosen to agree with the pixels already known - texture is genuine skin, and lines that run through
#               the hole (a lip line, a crease, the edge of a shadow) are continued instead of smeared.
#   video       the fill is made ONCE, on the anchor frame (the frame the user painted). On every other frame that
#               content moves with the skin (it is warped by the tracked transform), and only the change of lighting
#               measured on the skin around the hole is added, smoothly (push-pull). Fixed content = no shimmer from
#               frame to frame; the lighting follows the scene; motion blur is matched by softening the content.
#   tracking    the mark and the skin around it are followed from the anchor frame forwards and backwards: optical-flow
#               points (forward-backward checked, RANSAC affine) for the motion, then an ECC alignment of the anchor's
#               patch against the current frame, which removes drift (always aligned to the ANCHOR, never chained).
#               When the area leaves the picture or is hidden, ORB matching of the anchor's patch finds it again.
#   objects     anything in front of the mark (a hand, a cup) does not look like the anchor's patch there; those
#               pixels are left untouched, so an object passing over the spot is never painted with skin.
# =====================================================================================================================

PATCH_CFG = {
    "max_patches": 10,         # marks per job (v11.2.100: was 4 - the config's PATCH_MAX only gated the settings, this one the engine)
    "halo_px": 3,              # the painted hole is grown by this (v11.2.102: 3 - a mark's soft edge is covered too)
    "peek_band": 0.0,          # v11.2.97 edge repair outside the hole; v11.2.98: OFF (nothing outside the painted area changes)
    "bridge_max": 45,          # gaps of up to this many frames between followed frames are bridged (v11.2.102: was 3); a bridged
                               #   pose is "weak": it paints only where the mark itself is found (find_mark)
    "hold_max": 15,            # a followed run's last pose is held this many frames further (weak) - the end of a run is not a loss
    "find_budget_s": 90.0,     # v11.2.104: one search for the painted AREA (the painted frame is not in this video) stops after this many seconds ...
    "find_total_s": 240.0,     #   ... and all of a job's searches together after this many (a 4-mark job spent 946 s on searches that found nothing)
    "redetect_now": True,      # v11.2.103: a step whose followed / predicted pose does not fit looks for the patch at once (near the prediction, then
                               #   in the whole frame), not after two lost frames
    "tm_radius": 160,          # px around the predicted pose searched by that local template match (at the frame's own scale)
    "blur_ok": True,           # a re-detected pose with 85 % of the structure matched and a lighting-free score of 0.72 is accepted although the
                               #   plain correlation is lower (motion blur)
    "smooth_err": 1.0,         # v11.2.103: px the smoothing may bend the followed path (the window shrinks where the motion curves)
    "find_mark": True,         # v11.2.102: on every frame the mark itself is looked for (it differs from the skin around it) near where
                               #   the track puts it; the paint is centred on what is found and covers all of it
    "rescue": True,            #   ... and when the track loses the patch (blur, turn, light) the mark found near its predicted place
                               #   keeps it followed
    "opaque": True,            # the paste is opaque (no partial transparency): only strong evidence of an object holds it back
    "clone_texture": True,     # at the anchor: the fill gets the fine texture of clean skin beside the mark (copied, not synthetic)
    "max_views": 20,           # painted positions in all (several per mark) (v11.2.100: was 12)
    "grow_band": 0.0,          # dark rim around the painted area joins it (v11.2.97; OFF: on a face it ate creases and shadows
                               #   beside the mark - the per-frame mark finder covers the mark's own rim instead)
    "follow_margin": 45,       # v11.2.101: a painted position is followed from its anchor to the NEXT painted position and this many
                               #   frames past it (not through the whole video): 10 positions cost ~2-3 x one, not 10 x
    "cover": 1.5,              # v11.2.101: stronger cover - the fill is matched UP to the frame's own skin on the stroke's
    "cover_max": 10.0,         #   margin (it is never darker than that skin; at most this many levels; 1.5 = over by half)
    "lift": 1.0,               # v11.2.98: no faint dark blob may remain under the mark: the fill is lifted (inside the
    "lift_max": 6.0,           #   painted area only, very smoothly, by at most this many levels) to the level around it
    "feather_px": 1.5,         # soft edge of the fill
    "light_sigma": 4.0,        # lighting change is measured on the ring at this smoothness
    "gate_thr": 22.0,          # object gate: colour distance (0-255) to the anchor's appearance, 2 px tolerance
    "track_ncc_min": 0.55,     # the tracked area must still look like the anchor's (else: lost, nothing painted)
    "anchor_window_s": 1.5,    # the anchor frame is searched within this time of the app's frame first
    "smooth": (1.0, 2.0, 3.0, 2.0, 1.0),   # centred smoothing of the per-frame transform
    "detail_track": True,      # v11.2.99: the tracker compares a lighting-free view (a shadow moving across the mark
    "detail_sigma": 8.0,       #   lost it on a third of the frames); scale (px) of the light change it ignores
    "detail_thr": 4.0,         #   inlier tolerance in that view (% of local brightness)
    "detail_tol": 2,           #   and in space (px): skin bends a little, an affine track cannot follow thin strokes exactly
}

PATCH_STATS = {"frames_found": 0, "frames_shifted": 0, "frames_rescued": 0, "frames_weak_skipped": 0,
               "patches": 0, "anchored": 0, "frames_tracked": 0, "frames_filled": 0, "frames_lost": 0,
               "frames_gated": 0, "frames_refused": 0, "frames_bridged": 0, "frames_peek": 0, "views": 0,
               "fill_ms": 0.0, "track_ms": 0.0}


def _pr_detail(g, inside=None):
    """v11.2.99: a lighting-free view of a gray patch - log brightness minus its smooth version, x100 (so in % of the
    local brightness). A shadow edge or a light change across the patch is smooth and multiplies the brightness: it
    cancels here. The mark, its edges, pores and texture stay. inside: the valid pixels (border handling)."""
    s = float(PATCH_CFG.get("detail_sigma", 8.0))
    lg = np.log(g.astype(np.float32) + 8.0)
    if inside is None:
        lo = cv2.GaussianBlur(lg, (0, 0), s)
    else:
        v = inside.astype(np.float32)
        lo = cv2.GaussianBlur(lg * v, (0, 0), s) / np.maximum(cv2.GaussianBlur(v, (0, 0), s), 1e-3)
    return (lg - lo) * 100.0


def _pr_odd(n):
    n = int(round(n))
    return n if n % 2 else n + 1


def _pr_disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_pr_odd(2 * r + 1),) * 2)


# --------------------------------------------------------------------------------------------------- smooth fills
def _pr_lowpass(img, valid, sigma):
    """Normalised convolution: blur of the VALID pixels only."""
    v = valid.astype(np.float32)
    k = _pr_odd(max(3, sigma * 3))
    num = cv2.GaussianBlur(img * (v[..., None] if img.ndim == 3 else v), (k, k), sigma)
    den = cv2.GaussianBlur(v, (k, k), sigma)
    return num / (np.maximum(den, 1e-4)[..., None] if img.ndim == 3 else np.maximum(den, 1e-4)), den


def _pr_pushpull(img, w):
    """Fill where w == 0 from where w == 1 (pull-push pyramid + a little relaxation): smooth, seamless."""
    squeeze = img.ndim == 2
    if squeeze:
        img = img[..., None]
    w = np.clip(w.astype(np.float32), 0.0, 1.0)
    pyr = [(img * w[..., None], w)]
    while True:
        p, q = pyr[-1]
        if q.shape[0] <= 2 or q.shape[1] <= 2 or float(q.min()) > 1e-3:
            break
        nh, nw = (q.shape[0] + 1) // 2, (q.shape[1] + 1) // 2
        p2 = cv2.resize(p, (nw, nh), interpolation=cv2.INTER_AREA)
        q2 = cv2.resize(q, (nw, nh), interpolation=cv2.INTER_AREA)
        pyr.append((p2.reshape(nh, nw, -1), q2))
    p, q = pyr[-1]
    est = p / np.maximum(q, 1e-6)[..., None]
    if float(q.min()) <= 1e-6:
        est[q <= 1e-6] = p.reshape(-1, p.shape[-1]).sum(0) / max(float(q.sum()), 1e-6)
    for p, q in reversed(pyr[:-1]):
        up = cv2.resize(est, (q.shape[1], q.shape[0]), interpolation=cv2.INTER_LINEAR).reshape(q.shape[0], q.shape[1], -1)
        qn = np.clip(q * 2.0, 0.0, 1.0)
        est = p / np.maximum(q, 1e-6)[..., None] * qn[..., None] + up * (1.0 - qn)[..., None]
    hole = w < 0.5
    if hole.any():
        for _ in range(10):
            b = cv2.blur(est, (5, 5)).reshape(est.shape)
            est[hole] = b[hole]
    return est[..., 0] if squeeze else est


# --------------------------------------------------------------------------------------------------- exemplar fill
def _pr_exemplar(img, hole, srcmask=None):
    """Fill ``hole`` of float32 BGR ``img`` with patches of the known area (structure first, nearest sources).

    srcmask: where source patches may come from (default: everything not near the hole)."""
    H, W = hole.shape
    ys, xs = np.nonzero(hole)
    if len(ys) == 0:
        return img.copy()
    size = max(int(np.ptp(ys)), int(np.ptp(xs))) + 1
    B = int(np.clip(_pr_odd(size / 5.0), 9, 19))
    r = B // 2
    out = img.copy()
    known = ~hole
    ok = known & ~cv2.dilate(hole.astype(np.uint8), _pr_disk(2)).astype(bool)
    if srcmask is not None:
        ok &= srcmask
    # a source window is valid only if all of it is allowed
    bad = cv2.integral((~ok).astype(np.uint8)).astype(np.int64)
    Ws, Hs = W - B + 1, H - B + 1
    if Ws < 1 or Hs < 1:
        return out
    badwin = (bad[B:B + Hs, B:B + Ws] - bad[0:Hs, B:B + Ws] - bad[B:B + Hs, 0:Ws] + bad[0:Hs, 0:Ws]) > 0
    if badwin.all():
        return out
    # structure term: edges in the known area (smoothed), so lines are carried into the hole first
    g = cv2.GaussianBlur(cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    gm = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    gm = gm / (float(np.percentile(gm[known], 95)) + 1e-3)
    # guidance: the smooth fill says what colour belongs where in the hole; a source patch must agree with it
    # (weakly) as well as with the pixels already known (strongly) - an eye is never copied onto a cheek
    v0 = ~hole
    lowg, deng = _pr_lowpass(img, v0, 3.0)
    guide = _pr_pushpull(lowg, (v0 & (deng > 0.25)).astype(np.float32))
    R = int(max(3 * B, 1.0 * size))                      # search radius around the target
    yy, xx = np.mgrid[0:Hs, 0:Ws].astype(np.float32)
    lam = 0.15 * B * B * 3.0 / float(R * R)              # gentle preference for nearby sources (same lighting)
    unk = hole.copy()
    guard = 0
    while unk.any() and guard < 4000:
        guard += 1
        kn = (~unk).astype(np.float32)
        conf = cv2.boxFilter(kn, -1, (B, B), normalize=True)
        front = unk & (cv2.dilate(kn.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        if not front.any():
            front = unk
        pri = conf * (0.25 + np.minimum(cv2.dilate(gm * kn, np.ones((3, 3), np.uint8)), 1.5))
        pri[~front] = -1.0
        py, px = np.unravel_index(int(np.argmax(pri)), pri.shape)
        y0, x0 = int(np.clip(py - r, 0, H - B)), int(np.clip(px - r, 0, W - B))
        kn_t = ~unk[y0:y0 + B, x0:x0 + B]
        tpl = np.where(kn_t[..., None], out[y0:y0 + B, x0:x0 + B], guide[y0:y0 + B, x0:x0 + B]).astype(np.float32)
        m = np.where(kn_t, 1.0, 0.45).astype(np.float32)   # SQDIFF weight = mask^2: known 1, guidance 0.2
        sy0, sy1 = max(0, y0 - R), min(Hs, y0 + R + 1)
        sx0, sx1 = max(0, x0 - R), min(Ws, x0 + R + 1)
        cand = ~badwin[sy0:sy1, sx0:sx1]
        if not cand.any():                                 # nothing valid nearby: search everywhere
            sy0, sy1, sx0, sx1 = 0, Hs, 0, Ws
            cand = ~badwin
        area = out[sy0:sy1 + B - 1, sx0:sx1 + B - 1]
        cost = cv2.matchTemplate(area, tpl, cv2.TM_SQDIFF, mask=np.repeat(m[..., None], 3, axis=2))
        cost = cost / max(float((m * m).sum()), 1.0)
        cost = cost + lam * ((yy[sy0:sy1, sx0:sx1] - y0) ** 2 + (xx[sy0:sy1, sx0:sx1] - x0) ** 2)
        cost[~cand] = np.inf
        j = int(np.argmin(cost))
        sy, sx = np.unravel_index(j, cost.shape)
        sy, sx = sy + sy0, sx + sx0
        src = img[sy:sy + B, sx:sx + B].copy()
        mk = kn_t
        if mk.sum() >= 6:                                  # small colour correction towards the target's known part
            src += (tpl[mk].mean(0) - src[mk].mean(0)) * 0.6
        wr = unk[y0:y0 + B, x0:x0 + B]
        blk = out[y0:y0 + B, x0:x0 + B]
        blk[wr] = src[wr]
        unk[y0:y0 + B, x0:x0 + B] = False
    return out


def _pr_seams(filled, hole):
    """Even out the patch seams inside the hole: keep the texture, pull the low frequencies towards a smooth fill
    from the hole's edge."""
    lo_in = cv2.GaussianBlur(filled, (0, 0), 3.0)
    edge = ~hole
    lo_ref = _pr_pushpull(_pr_lowpass(filled, edge, 3.0)[0], edge.astype(np.float32))
    corr = cv2.GaussianBlur(lo_ref - lo_in, (0, 0), 2.0)
    out = filled.copy()
    out[hole] = filled[hole] + 0.5 * corr[hole]
    return out


def _pr_grow(img, hole):
    """Join to the hole the dark rim of the mark the finger missed (only darker-than-skin pixels touching it)."""
    ys, xs = np.nonzero(hole)
    size = max(int(np.ptp(ys)), int(np.ptp(xs))) + 1
    bw = int(round(size * PATCH_CFG["grow_band"])) + 3
    band = cv2.dilate(hole.astype(np.uint8), _pr_disk(bw)).astype(bool) & ~hole
    valid = ~hole
    low, den = _pr_lowpass(img, valid, 3.0)
    lf = _pr_pushpull(low, (valid & (den > 0.25)).astype(np.float32))
    d = (img - lf).mean(axis=2)
    ring = cv2.dilate(hole.astype(np.uint8), _pr_disk(max(6, bw * 2))).astype(bool) & ~hole
    sig = float(np.median(np.abs(d[ring] - np.median(d[ring])))) * 1.4826 + 0.5 if ring.any() else 3.0
    dark = band & (d < -max(10.0, 3.5 * sig))
    if not dark.any():
        return hole
    n, lab = cv2.connectedComponents((dark | hole).astype(np.uint8), connectivity=8)
    touch = np.unique(lab[hole])
    add = np.isin(lab, touch[touch > 0]) & dark
    return hole | cv2.dilate(add.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def _pr_alpha(hole):
    """1 inside the painted area, a soft edge INSIDE it, 0 outside (v11.2.98: nothing outside the painted area changes)."""
    a = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), PATCH_CFG["feather_px"] * 0.7)
    return np.clip((a - 0.35) / 0.4, 0.0, 1.0) * hole.astype(np.float32)


def _pr_lift(around, filled, hole, core=None):
    """v11.2.98: no faint darkness may remain where the mark was. Inside the painted area, the fill over the mark's
    CORE (the pixels that were dark) is compared with the fill around it in the painted area and the untouched skin
    just outside; where the core is darker, it is lifted to that level - smoothly (a level, not detail), by at most
    lift_max levels. The painted area's overall brightness is left as it is; nothing outside it changes."""
    k = float(PATCH_CFG.get("lift", 0.0) or 0.0)
    if k <= 0 or not hole.any():
        return filled
    ys, xs = np.nonzero(hole)
    size = max(int(np.ptp(ys)), int(np.ptp(xs))) + 1
    q = 4 if size >= 64 else (2 if size >= 32 else 1)          # a smooth level: solved at reduced size for big marks
    if q > 1:
        H0, W0 = hole.shape
        sm = (max(4, W0 // q), max(4, H0 // q))
        dn = lambda x, interp=cv2.INTER_AREA: cv2.resize(x.astype(np.float32), sm, interpolation=interp)
        a2, f2 = dn(around), dn(filled)
        h2 = dn(hole.astype(np.float32)) > 0.5
        c2 = None if core is None else dn(core.astype(np.float32)) > 0.3
        lifted = _pr_lift_at(a2, f2, h2, c2, k)
        lift = cv2.resize((lifted - f2).mean(axis=2), (W0, H0), interpolation=cv2.INTER_LINEAR)
        return filled + (lift * hole)[..., None]
    return _pr_lift_at(around, filled, hole, core, k)


def _pr_lift_at(around, filled, hole, core, k):
    ys, xs = np.nonzero(hole)
    if len(ys) == 0:
        return filled
    size = max(int(np.ptp(ys)), int(np.ptp(xs))) + 1
    sig = max(1.5, size / 8.0)
    ag = around.mean(axis=2) if around.ndim == 3 else around
    fg = filled.mean(axis=2)
    if core is None:                                     # the mark: pixels clearly darker than the fill made for them
        core = hole & (ag < fg - 6.0)
    core = cv2.dilate(core.astype(np.uint8), _pr_disk(2)).astype(bool) & hole
    if core.sum() < 6:
        return filled
    outside = ~cv2.dilate(hole.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ring = cv2.dilate(hole.astype(np.uint8), _pr_disk(max(3, size // 6))).astype(bool) & outside
    refv = (hole & ~core) | ring                         # the fill around the mark + the skin just outside
    if refv.sum() < 20:
        return filled
    src = np.where(hole, fg, ag).astype(np.float32)
    lv, den = _pr_lowpass(src, refv, sig)
    ref = _pr_pushpull(lv, (refv & (den > 0.2)).astype(np.float32))
    sh = max(1.2, sig * 0.4)                             # the spot's own shape (not one level for the whole core)
    hv, _ = _pr_lowpass(fg, core, sh)                    # the fill's level over the core
    d = np.where(core, ref - hv, 0.0).astype(np.float32)
    w = cv2.GaussianBlur(core.astype(np.float32), (0, 0), sh)
    d = cv2.GaussianBlur(d, (0, 0), sh) / np.maximum(w, 1e-3)
    lift = np.clip(d, 0.0, float(PATCH_CFG.get("lift_max", 8.0))) * k * np.clip(w * 1.5, 0.0, 1.0) * hole
    LIFT_STATS["n"] += 1
    LIFT_STATS["sum"] += float(lift[core].mean())
    return filled + lift[..., None]


LIFT_STATS = {"n": 0, "sum": 0.0}


def _pr_cover(roi, res, hole, core, ok):
    """v11.2.100: stronger cover. The stroke covers plain skin around the mark, and in the CURRENT frame that skin is
    there to be seen: it is the best reference for the level the fill should have. Over those margin pixels (inside the
    painted area, 2 px from its edge, away from the mark's core, not in front of an object) the fill is compared with
    the frame; where the fill is darker (a faint ghost of the mark, lighting that did not quite follow) its level is
    brought up to the frame's, smoothly and also over the mark itself. Brightening only, at most cover_max levels,
    fading to nothing at the painted area's edge; nothing outside the painted area changes."""
    k = float(PATCH_CFG.get("cover", 0.0) or 0.0)
    if k <= 0 or not hole.any():
        return res
    ys, xs = np.nonzero(hole)
    size = max(int(np.ptp(ys)), int(np.ptp(xs))) + 1
    Hf, Wf = hole.shape
    y0, y1 = max(0, int(ys.min()) - 6), min(Hf, int(ys.max()) + 7)          # work on the painted area's own box (the ROI is
    x0, x1 = max(0, int(xs.min()) - 6), min(Wf, int(xs.max()) + 7)          #  several times larger): same result, a fraction of the time
    if (y1 - y0) * (x1 - x0) < 0.6 * Hf * Wf:
        out = res.copy()
        out[y0:y1, x0:x1] = _pr_cover(roi[y0:y1, x0:x1], res[y0:y1, x0:x1], hole[y0:y1, x0:x1],
                                      None if core is None else core[y0:y1, x0:x1], ok[y0:y1, x0:x1])
        return out
    H, W = hole.shape
    inner = cv2.erode(hole.astype(np.uint8), _pr_disk(2)).astype(bool)
    if core is not None and core.any():
        inner &= ~cv2.dilate(core.astype(np.uint8), _pr_disk(3)).astype(bool)
    mv = inner & ok
    if mv.sum() < 30:
        return res
    e = (roi - res).mean(axis=2)                          # > 0: the fill is darker than the frame's own skin
    med = float(np.median(e[mv]))
    mad = float(np.median(np.abs(e[mv] - med))) * 1.4826 + 0.5
    mv &= np.abs(e - med) < max(6.0, 3.0 * mad)         # specks, hairs, a misplaced edge are not level
    if mv.sum() < 30:
        return res
    q = 4 if size >= 64 else (2 if size >= 32 else 1)
    sig = max(1.5, size / 8.0)
    if q > 1:
        sm = (max(4, W // q), max(4, H // q))
        v_lo = cv2.resize(mv.astype(np.float32), sm, interpolation=cv2.INTER_AREA)
        e_lo = cv2.resize(e * mv, sm, interpolation=cv2.INTER_AREA) / np.maximum(v_lo, 1e-3)
        low, den = _pr_lowpass(e_lo.astype(np.float32), v_lo > 0.5, sig / q + 0.5)
        fld = _pr_pushpull(low, ((v_lo > 0.5) & (den > 0.2)).astype(np.float32))
        fld = cv2.resize(fld.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        low, den = _pr_lowpass(e * mv, mv, sig)
        fld = _pr_pushpull(low, (mv & (den > 0.2)).astype(np.float32))
    ramp = np.clip((cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 3) - 0.5) / 5.0, 0.0, 1.0)   # full 5.5 px in
    add = np.clip(fld, 0.0, float(PATCH_CFG.get("cover_max", 8.0))) * k * ramp * hole
    COVER_STATS["n"] += 1
    COVER_STATS["sum"] += float(add[hole].mean())
    return res + add[..., None]


COVER_STATS = {"n": 0, "sum": 0.0}


# Optional learned inpainter (set by the pipeline): fn(crop_bgr_uint8, mask_bool) -> filled crop_bgr_uint8, or None.
# v11.2.96 uses LaMa (Suvorov et al. 2022, "Resolution-robust Large Mask Inpainting with Fourier Convolutions",
# Apache-2.0; ONNX export from the OpenCV model zoo). It runs once per mark (anchor frame / preview), never per frame.
PATCH_INPAINTER = None


def _pr_model_fill(img, hole):
    """Fill with the learned inpainter on a crop around the hole (about 2.6x the mark). None if unavailable."""
    fn = PATCH_INPAINTER
    if fn is None:
        return None
    H, W = hole.shape
    ys, xs = np.nonzero(hole)
    size = max(int(np.ptp(ys)), int(np.ptp(xs))) + 1
    side = int(max(160, 2.6 * size))
    ch, cw = min(H, side), min(W, side)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    y0 = int(np.clip(cy - ch // 2, 0, H - ch))
    x0 = int(np.clip(cx - cw // 2, 0, W - cw))
    crop = np.clip(img[y0:y0 + ch, x0:x0 + cw] + 0.5, 0, 255).astype(np.uint8)
    mk = hole[y0:y0 + ch, x0:x0 + cw]
    if mk.sum() < hole.sum():                              # hole bigger than the crop: give up, use the fallback
        return None
    try:
        res = fn(crop, mk)
    except Exception as e:
        logging.warning("patch inpainter failed (%s) - exemplar fill instead", e)
        return None
    if res is None or res.shape != crop.shape:
        return None
    out = img.copy()
    blk = out[y0:y0 + ch, x0:x0 + cw]
    blk[mk] = res.astype(np.float32)[mk]
    s = 512.0 / max(ch, cw)
    if s < 0.9:                                            # the model saw it smaller: add back the skin's fine grain
        ring = cv2.dilate(hole.astype(np.uint8), _pr_disk(max(6, size // 4))).astype(bool) & ~hole
        hp = (img - cv2.GaussianBlur(img, (0, 0), 1.0)).mean(axis=2)[ring]
        sig = float(np.median(np.abs(hp - np.median(hp)))) * 1.4826 if hp.size > 20 else 0.0
        grain = cv2.GaussianBlur(np.random.default_rng(5).normal(0, 1, hole.shape).astype(np.float32), (0, 0), 0.6)
        grain *= sig / (float(grain.std()) + 1e-6) * math.sqrt(max(0.0, 1.0 - s * s))
        out[hole] += grain[hole][:, None]
    return out


_PR_EVID = {"v": 1.0}          # how much like the mark the predicted core looks (share of its anchor contrast), set by _pr_find_mark


def _pr_find_mark(g, hole, core, pol, con, area_exp, reach, zone_r=3):
    """v11.2.102: where is the MARK ITSELF in this picture? A mole, a tattoo, a birthmark differs from the skin around it by
    about ``con`` levels (darker when pol = +1, lighter when -1). Around the PREDICTED place (``hole``, searching ``reach`` px
    further) the skin level is taken from OUTSIDE the search area (a smooth interpolation of it); what departs from it with the
    mark's polarity by half of ``con`` or more, in pieces that touch the mark's own predicted core, are the mark's strong
    pixels (seeds), and the mark is those plus its soft edge (candidate pixels a few px from a seed at 30 % of ``con``).
    Works on a crop around the predicted place. -> (mask, (x, y) centroid, area, seeds) in ``g``'s coordinates, or None."""
    Hg, Wg = hole.shape
    ys, xs = np.nonzero(hole)
    if len(ys) < 4:
        return None
    size = max(int(np.ptp(ys)), int(np.ptp(xs))) + 1
    pad = int(reach) + 12
    cy0, cy1 = max(0, int(ys.min()) - pad), min(Hg, int(ys.max()) + 1 + pad)
    cx0, cx1 = max(0, int(xs.min()) - pad), min(Wg, int(xs.max()) + 1 + pad)
    hl = hole[cy0:cy1, cx0:cx1]
    gc = g[cy0:cy1, cx0:cx1].astype(np.float32)
    cr = None if core is None else core[cy0:cy1, cx0:cx1]
    dist = cv2.distanceTransform((~hl).astype(np.uint8), cv2.DIST_L2, 3)
    srch = dist <= reach
    outer = (dist > reach) & (dist <= reach + 6)
    if outer.sum() < 40:
        return None
    q = 2 if size >= 48 else 1                                  # the skin's level is smooth: work at half size on big marks
    if q > 1:
        sm = (max(4, gc.shape[1] // q), max(4, gc.shape[0] // q))
        o_lo = cv2.resize(outer.astype(np.float32), sm, interpolation=cv2.INTER_AREA) > 0.5
        g_lo = cv2.resize(gc, sm, interpolation=cv2.INTER_AREA)
        lo, den = _pr_lowpass(g_lo, o_lo, max(2.0, size / (6.0 * q)))
        sl = cv2.resize(_pr_pushpull(lo, (o_lo & (den > 0.15)).astype(np.float32)).astype(np.float32),
                        (gc.shape[1], gc.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        lo, den = _pr_lowpass(gc, outer, max(3.0, size / 6.0))
        sl = _pr_pushpull(lo, (outer & (den > 0.15)).astype(np.float32))
    c = cv2.GaussianBlur((pol * (sl - gc)).astype(np.float32), (0, 0), 0.8)
    med = float(np.median(c[outer]))
    if cr is not None and int(cr.sum()) >= 4:
        _PR_EVID["v"] = float(np.percentile(c[cr] - med, 75)) / max(con, 1e-3)
    # the threshold follows the picture's NOISE (its fine residual), not the scene's structure (shading, a nostril, the lips)
    fine = gc - cv2.GaussianBlur(gc, (0, 0), 1.5)
    sn = float(np.median(np.abs(fine[outer] - float(np.median(fine[outer]))))) * 1.4826
    thr = max(0.30 * con, 3.0 * sn + 1.5)
    cand = ((c - med) > thr) & srch
    if not cand.any():
        return None
    seed = ((c - med) > max(0.5 * con, thr)) & srch
    n, lab, st, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8), connectivity=8)
    zsrc = cr if cr is not None and int(cr.sum()) >= 4 else hl
    zone = cv2.distanceTransform((~zsrc).astype(np.uint8), cv2.DIST_L2, 3) <= zone_r
    amin = max(4.0, 0.03 * area_exp)
    pieces = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= amin and zone[lab == k].any()]
    if zone_r > 12 and len(pieces) > 1:                         # a wide search: the piece nearest the expected size
        pieces = [min(pieces, key=lambda k: abs(math.log(max(1.0, st[k, cv2.CC_STAT_AREA]) / max(1.0, area_exp))))]
    if not pieces:
        return None
    seeds = np.isin(lab, pieces)
    rg = max(1, int(round(0.25 * math.sqrt(float(seeds.sum()) / math.pi))))             # the soft edge: a quarter of the seeds' radius
    keep = cand & cv2.dilate(seeds.astype(np.uint8), _pr_disk(rg)).astype(bool)
    keep |= seeds
    area = int(keep.sum())
    if area < 0.12 * area_exp or area > 2.4 * area_exp:          # (a blob 3x the mark is the scene's structure, not the mark)
        return None
    y2, x2 = np.nonzero(keep)
    full = np.zeros((Hg, Wg), bool)
    full[cy0:cy1, cx0:cx1] = keep
    fseed = np.zeros((Hg, Wg), bool)
    fseed[cy0:cy1, cx0:cx1] = seeds
    return full, (float(x2.mean()) + cx0, float(y2.mean()) + cy0), area, fseed


def _pr_texture_clone(img, filled, hole):
    """v11.2.102: the fill's fine texture is the SKIN'S OWN. The inpainter returns the right colour and shading but a smoother
    skin than the skin around it. Its fine detail is replaced by the fine detail of a patch of clean skin beside the mark (same
    shape as the hole, taken from the best of a ring of candidate places: wholly plain skin, similar texture strength), copied
    pixel for pixel. Zero-mean detail only, so the level and shading of the fill are untouched. Made once, at the anchor; it
    is then carried with the skin."""
    if not PATCH_CFG.get("clone_texture", False) or hole.sum() < 30:
        return filled
    ys, xs = np.nonzero(hole)
    size = max(int(np.ptp(ys)), int(np.ptp(xs))) + 1
    H, W = hole.shape
    sig = 1.2
    hp = img - cv2.GaussianBlur(img, (0, 0), sig)
    known = ~cv2.dilate(hole.astype(np.uint8), _pr_disk(3)).astype(bool)
    g = img.mean(axis=2)
    lo, den = _pr_lowpass(g, known, max(3.0, size / 6.0))
    sl = _pr_pushpull(lo, (known & (den > 0.15)).astype(np.float32))
    an = np.abs(g - sl)
    ring = cv2.dilate(hole.astype(np.uint8), _pr_disk(max(8, size // 3))).astype(bool) & known
    if ring.sum() < 60:
        return filled
    s_an = float(np.median(an[ring])) * 1.4826 + 0.5
    bad = (an > max(4.0 * s_an, 8.0)) | ~known
    hp_g = hp.mean(axis=2)
    ref_sd = float(np.std(hp_g[ring]))
    best, best_sc = None, 1e9
    for rr in (1.0, 1.4, 1.9):
        for k in range(14):
            ang = 2.0 * math.pi * k / 14.0 + 0.3 * rr
            dx, dy = int(round(rr * (size + 10) * math.cos(ang))), int(round(rr * (size + 10) * math.sin(ang)))
            y2, x2 = ys + dy, xs + dx
            if y2.min() < 0 or x2.min() < 0 or y2.max() >= H or x2.max() >= W:
                continue
            if float(bad[y2, x2].mean()) > 0.02:
                continue
            seg = hp_g[y2, x2]
            if float(np.abs(seg).max()) > 5.0 * max(ref_sd, 1.0):        # a crease, an edge, a hair: not plain texture
                continue
            sd = float(np.std(seg))
            sc = abs(sd - ref_sd) / max(ref_sd, 1e-3) + 0.05 * rr
            if sc < best_sc:
                best, best_sc = (dy, dx), sc
    if best is None:
        return filled
    dy, dx = best
    out = filled.copy()
    lo_f = cv2.GaussianBlur(filled, (0, 0), sig)
    cap = 4.0 * max(ref_sd, 1.0)
    out[ys, xs] = lo_f[ys, xs] + np.clip(hp[ys + dy, xs + dx], -cap, cap)
    return out


def fill_still(img, hole):
    """Remove the mark under ``hole`` (bool) in float32 BGR ``img`` (a region of interest).
    -> (out, alpha, hole used, filled content)."""
    c = PATCH_CFG
    hole = hole.astype(bool)
    if c["halo_px"] > 0:
        hole = cv2.dilate(hole.astype(np.uint8), _pr_disk(c["halo_px"])).astype(bool)
    if float(PATCH_CFG.get("grow_band", 0.0) or 0.0) > 0:
        hole = _pr_grow(img, hole)
    filled = _pr_model_fill(img, hole)
    if filled is None:
        filled = _pr_seams(_pr_exemplar(img, hole), hole)
    filled = _pr_texture_clone(img, filled, hole)
    # (no lift here: the fill itself leaves no darkness - measured within +/-0.5 L of clean skin - and a lift on it
    #  overshot to a light spot; the lift acts per frame in PatchTrack.apply, where the faint darkness builds up)
    a = _pr_alpha(hole)
    return img * (1.0 - a[..., None]) + filled * a[..., None], a, hole, filled


def _pr_roi(hole, shape, extra=1.0):
    ys, xs = np.nonzero(hole)
    size = max(int(np.ptp(ys)), int(np.ptp(xs))) + 1
    pad = int(max(40, extra * (1.5 * size + 30)))
    H, W = shape[:2]
    return max(0, ys.min() - pad), min(H, ys.max() + 1 + pad), max(0, xs.min() - pad), min(W, xs.max() + 1 + pad)


def remove_patch_image(img_bgr, mask):
    """Preview: remove the painted mark(s) on a still frame. img_bgr uint8; mask: painted where > 127 (any size,
    resized to the image). -> (out uint8, info list)."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m.max(axis=2)
    m = m > 127 if m.dtype != bool else m
    if m.shape[:2] != img_bgr.shape[:2]:
        m = cv2.resize(m.astype(np.uint8) * 255, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
    out = img_bgr.copy()
    n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    info = []
    for k in range(1, n):
        comp = lab == k
        if comp.sum() < 4:
            continue
        y0, y1, x0, x1 = _pr_roi(comp, img_bgr.shape)
        roi = out[y0:y1, x0:x1].astype(np.float32)
        res, a, used, _ = fill_still(roi, comp[y0:y1, x0:x1])
        out[y0:y1, x0:x1] = np.clip(res + 0.5, 0, 255).astype(np.uint8)
        info.append({"area_px": int(used.sum()), "box": [int(x0), int(y0), int(x1), int(y1)]})
    return out, info


# --------------------------------------------------------------------------------------------------- tracking
def _pr_gray(f):
    return f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


def _pr_ncc(a, b, m=None):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    if m is not None:
        a, b = a[m], b[m]
    if a.size < 16:
        return -1.0
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum())) + 1e-6
    return float((a * b).sum()) / d


def _pr_inv(A):
    return cv2.invertAffineTransform(np.asarray(A, np.float64))


def _pr_compose(A, B):
    """A after B (both 2x3)."""
    return (np.vstack([A, [0, 0, 1]]) @ np.vstack([B, [0, 0, 1]]))[:2]


class PatchTrack:
    """One mark: its anchor frame, its hole and fill (anchor coordinates); follows the area through the frames."""

    def __init__(self, anchor_idx, anchor_bgr, hole, make_fill=True):
        t0 = cv2.getTickCount()
        self.anchor_idx = int(anchor_idx)
        Hf, Wf = hole.shape
        fy0, fy1, fx0, fx1 = _pr_roi(hole, hole.shape)
        roi = anchor_bgr[fy0:fy1, fx0:fx1].astype(np.float32)
        if make_fill:
            _, _, used, filled = fill_still(roi, hole[fy0:fy1, fx0:fx1])
        else:                                                  # (a finder only: no fill needed)
            used = hole[fy0:fy1, fx0:fx1].copy()
            filled = roi
        self.off = (fx0, fy0)
        self.hole_roi = used                                   # bool, fill-ROI coords
        self.fill_roi = filled.astype(np.float32)              # anchor content with the mark removed
        self.orig_roi = roi                                    # anchor as it was (object gate, lighting)
        # the mark's core: pixels the removal really changes (the stroke around it is plain skin)
        chg = np.abs(cv2.GaussianBlur(roi, (0, 0), 1.5) - cv2.GaussianBlur(filled, (0, 0), 1.5)).max(axis=2)   # (not the texture)
        self.core_roi = used & (chg > max(10.0, float(np.percentile(chg[used], 60)) if used.any() else 10.0))
        # what the mark looks like against the skin around it (v11.2.102): polarity and strength, in grey levels
        self.core_area = float(self.core_roi.sum())
        self.mark_pol, self.mark_con = 1.0, 20.0
        self.mark_bgr = None
        if self.core_area >= 6:
            dd = (filled.mean(axis=2) - roi.mean(axis=2))[self.core_roi]
            self.mark_pol = 1.0 if float(np.median(dd)) >= 0 else -1.0
            # how strongly the mark's CENTRE differs from the skin (the core's outer part is the soft edge): a high percentile
            self.mark_con = float(max(8.0, np.percentile(self.mark_pol * dd, 80)))
            strong = self.core_roi & ((self.mark_pol * (filled.mean(axis=2) - roi.mean(axis=2))) >= 0.6 * self.mark_con)
            self.mark_bgr = roi[strong].mean(axis=0) if int(strong.sum()) >= 4 else roi[self.core_roi].mean(axis=0)
        self.weak = set()                                      # frames whose pose is a bridge / hold / rescue: paints only where the mark is found
        yh_, xh_ = np.nonzero(used)
        self._hbox = (int(xh_.min()), int(yh_.min()), int(xh_.max()), int(yh_.max())) if len(yh_) else (0, 0, 1, 1)
        full = np.zeros((Hf, Wf), bool)
        full[fy0:fy1, fx0:fx1] = used
        ys, xs = np.nonzero(full)
        self.size = int(max(np.ptp(ys), np.ptp(xs))) + 1
        pad = int(max(30, 1.0 * self.size + 24))               # tracking box: the mark and the skin around it
        self.box = (max(0, ys.min() - pad), min(Hf, ys.max() + 1 + pad), max(0, xs.min() - pad), min(Wf, xs.max() + 1 + pad))
        y0, y1, x0, x1 = self.box
        self.anchor_gray = _pr_gray(anchor_bgr)
        self.tmpl = cv2.GaussianBlur(self.anchor_gray[y0:y1, x0:x1], (5, 5), 1.0).astype(np.float32)
        self._det = bool(PATCH_CFG.get("detail_track", False))
        self.tmpl_c = _pr_detail(self.tmpl) if self._det else self.tmpl     # what _check / _ecc compare against
        tol = int(PATCH_CFG.get("detail_tol", 0) or 0) if self._det else 0
        self._tlo = cv2.erode(self.tmpl_c, np.ones((2 * tol + 1,) * 2, np.uint8)) if tol > 0 else None
        self._thi = cv2.dilate(self.tmpl_c, np.ones((2 * tol + 1,) * 2, np.uint8)) if tol > 0 else None
        gx, gy = cv2.Sobel(self.tmpl_c, cv2.CV_32F, 1, 0), cv2.Sobel(self.tmpl_c, cv2.CV_32F, 0, 1)
        self._tgrad = gx * gx + gy * gy                        # where the patch's structure is (the mark's edges)
        gx, gy = cv2.Sobel(self.tmpl, cv2.CV_32F, 1, 0), cv2.Sobel(self.tmpl, cv2.CV_32F, 0, 1)
        self._tgrad_g = gx * gx + gy * gy                      # (the same in the plain view)
        self.A = {}
        self.conf = {}
        self.orb = cv2.ORB_create(nfeatures=800, scaleFactor=1.2, nlevels=6, edgeThreshold=15, patchSize=21)
        m = np.zeros(self.anchor_gray.shape, np.uint8)
        m[y0:y1, x0:x1] = 255
        self.kp, self.des = self.orb.detectAndCompute(self.anchor_gray, m)
        PATCH_STATS["fill_ms"] += (cv2.getTickCount() - t0) * 1000.0 / cv2.getTickFrequency()

    # -- alignment helpers --------------------------------------------------------------------------------------
    def _points(self, gray, A):
        y0, y1, x0, x1 = self.box
        quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
        q = (quad @ A[:, :2].T + A[:, 2]).astype(np.int32)
        m = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(m, q, 255)
        p = cv2.goodFeaturesToTrack(gray, maxCorners=160, qualityLevel=0.003, minDistance=4, mask=m, blockSize=5)
        if p is None or len(p) < 16:
            gy, gx = np.mgrid[y0:y1:7, x0:x1:7]
            g = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64)
            p = (g @ A[:, :2].T + A[:, 2]).astype(np.float32)
            inside = (p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0] < gray.shape[1]) & (p[:, 1] < gray.shape[0])
            p = p[inside]
        return p.reshape(-1, 1, 2).astype(np.float32)

    def _boxwarp(self, gray, A, border=cv2.BORDER_REFLECT):
        y0, y1, x0, x1 = self.box
        M = _pr_compose(A, np.array([[1, 0, x0], [0, 1, y0]], np.float64))     # box coords -> frame
        cur = cv2.warpAffine(gray, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=border)
        return cur, M

    def _ecc(self, gray, A0, inl=None):
        """Refine the anchor->frame affine by aligning the anchor's patch to the frame (anchored: no drift).
        inl: pixels of the box known to match (an object in front is left out of the fit)."""
        y0, y1, x0, x1 = self.box
        cur, M = self._boxwarp(gray, A0)
        cur = cv2.GaussianBlur(cur, (5, 5), 1.0).astype(np.float32)
        if self._det:
            cur = _pr_detail(cur)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            msk = None if inl is None else inl.astype(np.uint8)
            cc, warp = cv2.findTransformECC(self.tmpl_c, cur, warp, cv2.MOTION_AFFINE,
                                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4), msk, 3)
        except cv2.error:
            return A0, -1.0
        w, h = x1 - x0, y1 - y0
        if not np.isfinite(warp).all() or abs(warp[0, 2]) > 0.3 * w or abs(warp[1, 2]) > 0.3 * h:
            return A0, -1.0
        T = np.array([[1, 0, x0], [0, 1, y0]], np.float64)
        return _pr_compose(_pr_compose(M, warp.astype(np.float64)), _pr_inv(T)), float(cc)

    def _check(self, gray, A, plain=False):
        """Does the frame, aligned by A, show the anchor's patch? Robust to an object over part of it and to the
        area being partly out of the picture: only pixels that match count, but enough of them must match.
        plain: compare the plain brightness even when the lighting-free view is on (v11.2.99).
        -> (score, matching pixels of the box)."""
        det = self._det and not plain
        y0, y1, x0, x1 = self.box
        cur, M = self._boxwarp(gray, A, cv2.BORDER_CONSTANT)
        inside = cv2.warpAffine(np.full(gray.shape, 255, np.uint8), M, (x1 - x0, y1 - y0),
                                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
        inside = cv2.erode(inside.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if inside.mean() < 0.12:
            return -1.0, None
        c = cv2.GaussianBlur(cur, (5, 5), 1.0).astype(np.float32)
        t = self.tmpl_c if det else self.tmpl
        if det:                                               # v11.2.99: lighting-free view (a shadow cancels)
            c = _pr_detail(c, cv2.dilate(inside.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        ci, ti = c[inside], t[inside]
        st = float(ti.std()) + 1e-3
        tmin = float(PATCH_CFG.get("detail_thr", 4.0)) if det else 6.0
        # brightness (contrast) of the frame vs the anchor, robustly (an object over part of the area must not skew it)
        if det:
            k = 1.0
        else:
            k = float(np.median(ci / np.maximum(ti, 1.0)))
            k = float(np.clip(k, 0.5, 2.0))
        thr = max(tmin, 0.45 * st * k)
        r0 = np.abs(ci - k * ti) < thr
        if r0.sum() > 50:
            k = float(np.clip((ci[r0] * ti[r0]).sum() / max(1e-6, float((ti[r0] * ti[r0]).sum())), 0.5, 2.0))
        if det and self._tlo is not None:                     # v11.2.99: a little spatial tolerance (bending skin)
            res = np.maximum(k * self._tlo - c, 0.0) + np.maximum(c - k * self._thi, 0.0)
        else:
            res = np.abs(c - k * t)
        thr = max(tmin, 0.45 * st * k)
        inl = inside & (cv2.dilate((res < thr).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (res < 2.5 * thr)
        frac = float(inl.sum()) / max(1.0, float(inside.sum()))
        self._last_frac = frac
        if frac < 0.40 or inl.sum() < 150:
            return -1.0, None
        ncc = _pr_ncc(c, t, inl)
        self._last_all = _pr_ncc(c, t, inside)                  # the plain score over everything in the picture
        G = self._tgrad if det else self._tgrad_g
        self._last_cov = float(G[inl].sum()) / max(1e-6, float(G[inside].sum()))   # share of the patch's structure matched
        # an object in front makes ONE compact region of disagreement; a wrong place makes it everywhere, scattered
        bad = (inside & ~inl).astype(np.uint8)
        nb = int(bad.sum())
        if nb < 30:
            self._last_compact = 1.0
        else:
            bad = cv2.morphologyEx(bad, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            n_, lab_, st_, _ = cv2.connectedComponentsWithStats(bad, connectivity=8)
            self._last_compact = float(st_[1:, cv2.CC_STAT_AREA].max()) / max(1.0, float(st_[1:, cv2.CC_STAT_AREA].sum())) if n_ > 1 else 1.0
        return ncc * min(1.0, frac / 0.6), inl

    def _redetect_tm(self, gray, around=None, radius=160.0):
        """Find the anchor's patch by normalised template matching at half size, a few scales and turns (works on
        smooth skin where there are too few corners for ORB).
        around (v11.2.103): a pose; the search is then limited to ``radius`` px around where it puts the patch, and the turns and
        scales tried are centred on its own (a local search: cheap, and a repeat of the skin's texture elsewhere cannot win)."""
        y0, y1, x0, x1 = self.box
        t = cv2.resize(self.tmpl, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        ox0 = oy0 = 0
        gsrc = gray
        a0, s0 = 0.0, 1.0
        if around is not None:
            cx_a, cy_a = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            pc = np.asarray(around, np.float64)[:, :2] @ np.array([cx_a, cy_a]) + np.asarray(around, np.float64)[:, 2]
            half = int(radius + 0.5 * max(x1 - x0, y1 - y0))
            gx0, gy0 = int(max(0, pc[0] - half)), int(max(0, pc[1] - half))
            gx1, gy1 = int(min(gray.shape[1], pc[0] + half)), int(min(gray.shape[0], pc[1] + half))
            if gx1 - gx0 < 24 or gy1 - gy0 < 24:
                return None
            gsrc = gray[gy0:gy1, gx0:gx1]
            ox0, oy0 = gx0, gy0
            a0 = math.degrees(math.atan2(float(around[0][1]), float(around[0][0])))      # (cv2.getRotationMatrix2D's sign)
            s0 = math.sqrt(abs(float(np.linalg.det(np.asarray(around, np.float64)[:, :2]))))
        g = cv2.GaussianBlur(cv2.resize(gsrc, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), (3, 3), 0.8).astype(np.float32)
        th, tw = t.shape
        best = (-2.0, None)
        scales = (0.9, 1.0, 1.1) if around is None else (s0 * 0.92, s0, s0 * 1.08)
        angles = (-8.0, 0.0, 8.0) if around is None else (a0 - 8.0, a0, a0 + 8.0)
        for scl in scales:
            for ang in angles:
                R = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), ang, scl)
                tt = cv2.warpAffine(t, R, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                cw, ch = int(tw * 0.7), int(th * 0.7)                      # the middle (no reflected border)
                ox, oy = (tw - cw) // 2, (th - ch) // 2
                core = tt[oy:oy + ch, ox:ox + cw]
                if g.shape[0] < ch or g.shape[1] < cw or float(core.std()) < 1.0:
                    continue
                r = cv2.matchTemplate(g, core, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(r)
                if mx > best[0]:
                    # pose: anchor box centre -> found centre, with this turn and scale
                    cx_a, cy_a = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                    cx_f, cy_f = (loc[0] + cw / 2.0) * 2.0 + ox0, (loc[1] + ch / 2.0) * 2.0 + oy0
                    A = cv2.getRotationMatrix2D((cx_a, cy_a), ang, scl)
                    A[0, 2] += cx_f - cx_a
                    A[1, 2] += cy_f - cy_a
                    best = (mx, A.astype(np.float64))
        return best[1] if best[0] >= 0.5 else None

    def _redetect(self, gray):
        if self.des is None or len(self.kp) < 8:
            return None
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or len(kp) < 8:
            return None
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(self.des, des, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        if len(good) < 8:
            return None
        src = np.float32([self.kp[g.queryIdx].pt for g in good])
        dst = np.float32([kp[g.trainIdx].pt for g in good])
        A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if A is None or inl is None or int(inl.sum()) < 8:
            return None
        return A.astype(np.float64)

    def _step(self, gray, prev, pts, apts, lastA, velA, lost, strict_all=False):
        """One frame. Candidates: flow (LK points -> RANSAC affine, points that disagree with the motion model left
        out - an object crossing the area must not drag the pose), the motion model's prediction, and (when lost)
        ORB re-detection. Each is aligned by ECC and scored; the best acceptable one wins.
        -> (A or None, score, pts, apts)."""
        pred = None
        if lastA is not None:
            pred = lastA if velA is None else _pr_compose(velA, lastA)
            for _ in range(min(lost, 12)):
                if velA is not None:
                    pred = _pr_compose(velA, pred)
        flow, fpts, fapts = None, None, None
        if prev is not None and pts is not None and len(pts) >= 6:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None, winSize=(21, 21), maxLevel=3)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, nxt, None, winSize=(21, 21), maxLevel=3)
            fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
            if pred is not None and ok.sum() >= 6:
                exp = cv2.transform(apts, pred).reshape(-1, 2)
                dev = np.linalg.norm(nxt.reshape(-1, 2) - exp, axis=1)
                tol = max(6.0, 0.15 * self.size) * (1 + min(lost, 4))
                ok &= dev < tol
            if ok.sum() >= 6:
                An, inl = cv2.estimateAffine2D(apts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2),
                                               method=cv2.RANSAC, ransacReprojThreshold=2.0)
                if An is not None and inl is not None and int(inl.sum()) >= 6:
                    flow = An.astype(np.float64)
                    sel = np.nonzero(ok)[0][inl.ravel() > 0]
                    fpts, fapts = nxt[sel], apts[sel]
        cands = []
        if flow is not None:
            cands.append(("flow", flow))
        if pred is not None:
            cands.append(("pred", pred))
        best = self._pick(gray, cands, lost, strict_all)
        # v11.2.103: when the followed pose and the prediction do not fit (a fast move: the body part is further than the
        # constant-velocity guess, and blurred), look for the patch NOW - near where it was first, then in the whole frame -
        # instead of waiting two lost frames (the mark showed through the paint on every one of them)
        if best is None and PATCH_CFG.get("redetect_now", True):
            near_ = self._redetect_tm(gray, around=(lastA if pred is None else pred), radius=float(PATCH_CFG.get("tm_radius", 160)))
            if near_ is not None:
                best = self._pick(gray, [("tm", near_)], lost, strict_all)
        if best is None and (not cands or lost >= 2 or PATCH_CFG.get("redetect_now", True)):
            far = [("orb", r) for r in (self._redetect(gray), self._redetect_tm(gray)) if r is not None]
            if far:
                best = self._pick(gray, far, lost, strict_all)
        if best is None:
            return None, -1.0, None, None
        q, A1, kind = best
        if kind == "flow":
            return A1, q, fpts, fapts
        return A1, q, None, None

    def _pick(self, gray, cands, lost, strict_all=False):
        """Align each candidate pose by ECC, score it, return the best acceptable (q, A, kind) or None."""
        best = None
        for kind, A0 in cands:
            sc0, inl0 = self._check(gray, A0)
            if inl0 is None and self._det:
                sc0, inl0 = self._check(gray, A0, plain=True)
            if inl0 is None and kind != "orb":
                continue
            Ar, cc = self._ecc(gray, A0, inl0 if (inl0 is not None and inl0.mean() < 0.9) else None)
            for A1 in ((Ar, A0) if cc > 0.5 else (A0,)):
                # v11.2.99: the lighting-free view first (a shadow moving over the mark), then the plain view (an
                # object over part of it: in the lighting-free view only its two edges disagree, not one region)
                q = None
                for plain in ((False, True) if self._det else (False,)):
                    sc, inl = self._check(gray, A1, plain=plain)
                    if inl is None:
                        continue
                    al, cov = self._last_all, self._last_cov
                    # re-detected anywhere in the frame, or predicted across a gap: it must match the patch tightly
                    strict = strict_all or kind in ("orb", "tm") or (kind == "pred" and lost >= 1) or (lost >= 3 and kind != "flow")
                    whole = (al >= 0.8 and cov >= 0.7) if strict else (al >= PATCH_CFG["track_ncc_min"] and cov >= 0.5)
                    if not whole and strict and not strict_all and kind in ("orb", "tm") and PATCH_CFG.get("blur_ok", True):
                        # motion blur lowers the plain correlation of a sharp anchor against a smeared frame, but nearly all of its
                        # structure still lines up (coverage) and the lighting-free score stays high
                        whole = al >= 0.58 and cov >= 0.85 and sc >= 0.72
                    part = (not strict_all and kind != "orb" and sc >= 0.85 and cov >= 0.3 and lost <= 10
                            and self._last_compact >= 0.6)                         # partly covered by one object
                    if whole or part:
                        q = sc * (0.5 + 0.5 * cov)
                        break
                if q is None:
                    continue
                if best is None or q > best[0]:
                    best = (q, A1, kind)
                break
        return best

    def track(self, gray_at, idxs, tick=None):
        """Follow the area over the frame indices idxs (moving away from the anchor, the anchor first). After a
        gap, the area found again is also followed back into the gap. tick() is called once per frame; if it returns
        True the following stops (progress reporting and cancelling, v11.2.101)."""
        t0 = cv2.getTickCount()
        prev, pts, apts, lastA, velA = None, None, None, None, None
        lost = 0
        for p, idx in enumerate(idxs):
            if tick is not None and tick():
                break
            gray = gray_at(idx)
            if idx == self.anchor_idx:
                A = np.eye(2, 3, dtype=np.float64)
                self.A[idx], self.conf[idx] = A, 1.0
                pts = self._points(gray, A)
                apts = pts.copy()
                prev, lastA = gray, A
                continue
            if idx in self.A and lost == 0:                    # (already followed by a back-fill)
                newA = self.A[idx]
                velA = _pr_compose(newA, _pr_inv(lastA)) if lastA is not None else None
                lastA = newA
                pts = self._points(gray, newA)
                apts = cv2.transform(pts, _pr_inv(newA)).astype(np.float32)
                prev = gray
                continue
            A, sc, pts, apts = self._step(gray, prev, pts, apts, lastA, velA, lost)
            if A is None and lost >= 2 and lost % 2 == 0:
                r = self._redetect(gray)
                if r is not None:
                    A, sc, pts, apts = self._step(gray, None, None, None, r, None, 0)
            if A is None and lastA is not None and lost <= 45 and PATCH_CFG.get("rescue", False):
                pred = lastA if velA is None else _pr_compose(velA, lastA)
                for _ in range(min(lost, 12)):
                    if velA is not None:
                        pred = _pr_compose(velA, pred)
                r = self._rescue(gray, pred)
                if r is not None:                                   # followed by the mark itself: weak, paints only where it is seen
                    A, sc, pts, apts = r, 0.45, None, None
                    self.weak.add(idx)
                    PATCH_STATS["frames_rescued"] += 1
            if A is not None:
                self.A[idx], self.conf[idx] = A, sc
                PATCH_STATS["frames_tracked"] += 1
                if lastA is not None and lost == 0:
                    velA = _pr_compose(A, _pr_inv(lastA))
                else:
                    velA = None
                if lost > 0:                                   # found again: walk back into the gap
                    self._backfill(gray_at, idxs[max(0, p - lost):p][::-1], gray, A)
                lastA = A
                lost = 0
                if pts is None or len(pts) < 40:
                    pts = self._points(gray, A)
                    apts = cv2.transform(pts, _pr_inv(A)).astype(np.float32)
            else:
                lost += 1
                PATCH_STATS["frames_lost"] += 1
                pts = None
                if lost > 45:
                    lastA, velA = None, None
            prev = gray
        PATCH_STATS["track_ms"] += (cv2.getTickCount() - t0) * 1000.0 / cv2.getTickFrequency()

    def _backfill(self, gray_at, gap, gray0, A0):
        prev, lastA, velA = gray0, A0, None
        pts = self._points(gray0, A0)
        apts = cv2.transform(pts, _pr_inv(A0)).astype(np.float32)
        for idx in gap:
            if idx in self.A:
                break
            g = gray_at(idx)
            A, sc, pts, apts = self._step(g, prev, pts, apts, lastA, velA, 0, strict_all=True)
            if A is None:
                break
            self.A[idx], self.conf[idx] = A, sc
            PATCH_STATS["frames_tracked"] += 1
            velA = _pr_compose(A, _pr_inv(lastA))
            lastA = A
            if pts is None or len(pts) < 40:
                pts = self._points(g, A)
                apts = cv2.transform(pts, _pr_inv(A)).astype(np.float32)
            prev = g

    def bridge(self):
        """Fill gaps of up to bridge_max frames between two followed frames whose poses agree, and hold the pose of a run's
        last (first) frame for hold_max frames into a longer gap or past the ends. These poses are WEAK (v11.2.102): the paint
        is applied on them only where the mark itself is found (find_mark), so a bridge across a blurred or turning stretch
        keeps the mark covered and a wrong guess paints nothing."""
        gmax = int(PATCH_CFG.get("bridge_max", 0) or 0)
        hmax = int(PATCH_CFG.get("hold_max", 0) or 0)
        if (gmax <= 0 and hmax <= 0) or not self.A:
            return
        ks = sorted(self.A)
        cx, cy = (self.box[2] + self.box[3]) / 2.0, (self.box[0] + self.box[1]) / 2.0
        c = np.array([cx, cy, 1.0])
        tol = 0.25 * self.size + 10.0
        new = {}
        strong_skip = set()
        for a, b in zip(ks[:-1], ks[1:]):
            g = b - a - 1
            if g < 1:
                continue
            Aa, Ab = self.A[a], self.A[b]
            if gmax > 0 and g <= gmax and np.linalg.norm(Aa @ c - Ab @ c) <= tol + 40.0 * (g + 1):
                ends_weak = a in self.weak or b in self.weak     # (a rescued end is a guess: what lies between is too)
                for j in range(1, g + 1):
                    w = j / float(g + 1)
                    new[a + j] = ((1.0 - w) * Aa + w * Ab, 0.55 if g <= 6 else 0.4)
                    if g > 6 or ends_weak:                       # (a short bridge between two followed frames is as good as one)
                        strong_skip.add(a + j)
                    PATCH_STATS["frames_bridged"] += 1
            elif hmax > 0:                                       # a longer gap: hold both ends' poses into it
                for j in range(1, min(hmax, g // 2 + 1) + 1):
                    new.setdefault(a + j, (Aa, 0.3))
                    new.setdefault(b - j, (Ab, 0.3))
        if hmax > 0:                                             # past the ends of the followed frames
            for j in range(1, hmax + 1):
                new.setdefault(ks[-1] + j, (self.A[ks[-1]], 0.3))
                if ks[0] - j >= 0:
                    new.setdefault(ks[0] - j, (self.A[ks[0]], 0.3))
        for k, (Ak, cf) in new.items():
            if k not in self.A:
                self.A[k], self.conf[k] = Ak, cf
                if k in strong_skip or cf < 0.5:
                    self.weak.add(k)

    def _rescue(self, gray, pred):
        """The patch's texture could not be matched on this frame (blur, a turn, a change of light), but the MARK may still be
        seen near where the motion puts it: find it (as find_mark does) and place the pose on it. -> pose or None."""
        if not PATCH_CFG.get("rescue", False) or self.core_area < 6:
            return None
        Hg, Wg = gray.shape[:2]
        h0, w0 = self.hole_roi.shape
        M = _pr_compose(pred, np.array([[1, 0, self.off[0]], [0, 1, self.off[1]]], np.float64))
        ys, xs = np.nonzero(self.hole_roi)
        corners = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]], np.float64)
        q = corners @ M[:, :2].T + M[:, 2]
        scale = math.sqrt(abs(float(np.linalg.det(pred[:, :2]))))
        reach = int(min(70, np.clip(0.5 * self.size * scale + 6, 6, 45) * 1.8))
        px0, py0 = int(max(0, np.floor(q[:, 0].min()) - reach - 10)), int(max(0, np.floor(q[:, 1].min()) - reach - 10))
        px1, py1 = int(min(Wg, np.ceil(q[:, 0].max()) + reach + 10)), int(min(Hg, np.ceil(q[:, 1].max()) + reach + 10))
        if px1 - px0 < 16 or py1 - py0 < 16:
            return None
        Mc = M.copy()
        Mc[:, 2] -= (px0, py0)
        sz = (px1 - px0, py1 - py0)
        hole_c = cv2.warpAffine(self.hole_roi.astype(np.uint8) * 255, Mc, sz, flags=cv2.INTER_LINEAR) > 127
        core_c = cv2.warpAffine(self.core_roi.astype(np.uint8) * 255, Mc, sz, flags=cv2.INTER_LINEAR) > 127
        if hole_c.sum() < 4 or core_c.sum() < 4:
            return None
        r = _pr_find_mark(gray[py0:py1, px0:px1].astype(np.float32), hole_c, core_c, self.mark_pol, self.mark_con,
                          self.core_area * scale * scale, reach)
        if r is None:
            return None
        _, cen, area, _s = r
        if not (0.5 <= area / max(1.0, float(core_c.sum())) <= 2.0):
            return None
        yy, xx = np.nonzero(core_c)
        A2 = pred.copy()
        A2[:, 2] += (cen[0] - float(xx.mean()), cen[1] - float(yy.mean()))
        return A2

    def smooth(self):
        """Centred smoothing of the transforms (removes jitter). Symmetric only: near the end of a run the window
        shrinks on both sides, so the smoothing never lags behind a moving mark."""
        self.bridge()
        k = np.array(PATCH_CFG["smooth"], np.float64)
        r = len(k) // 2
        out = {}
        tol = 0.25 * self.size + 10.0                         # a neighbour this far off is not averaged in
        cx, cy = (self.box[2] + self.box[3]) / 2.0, (self.box[0] + self.box[1]) / 2.0
        ctr = {i: A @ np.array([cx, cy, 1.0]) for i, A in self.A.items()}
        # v11.2.103: the window also shrinks where the motion CURVES (a fast move that turns round): smoothing a curve with a window of
        # variance v moves it by about half its acceleration x v - 16 px at the turn of a 70 px/frame swing, and the paint sat off the mark
        curv = {}
        for i in self.A:
            if (i - 1) in self.A and (i + 1) in self.A:
                curv[i] = float(np.linalg.norm(ctr[i - 1][:2] - 2.0 * ctr[i][:2] + ctr[i + 1][:2]))
        budget = float(PATCH_CFG.get("smooth_err", 1.0))

        def _var(rr_):
            w_ = k[r - rr_:r + rr_ + 1]
            return float((w_ * (np.arange(-rr_, rr_ + 1) ** 2)).sum() / w_.sum())
        for i in self.A:
            rr = 0
            while (rr < r and (i - rr - 1) in self.A and (i + rr + 1) in self.A
                   and np.linalg.norm(ctr[i - rr - 1] - ctr[i]) < tol + 30 * (rr + 1)
                   and np.linalg.norm(ctr[i + rr + 1] - ctr[i]) < tol + 30 * (rr + 1)):
                rr += 1
            while rr > 0:                                         # ...down to the widest window whose bending of the path stays within budget
                a_loc = max([curv.get(i + j, 0.0) for j in range(-rr, rr + 1)] + [0.0])
                if 0.5 * a_loc * _var(rr) <= budget:
                    break
                rr -= 1
            acc, ws = np.zeros((2, 3)), 0.0
            for j in range(-rr, rr + 1):
                acc += k[j + r] * self.A[i + j]
                ws += k[j + r]
            out[i] = acc / ws
        self.A = out

    def apply(self, frame, idx):
        A = self.A.get(idx)
        if A is None:
            return frame
        t0 = cv2.getTickCount()
        H, W = frame.shape[:2]
        h0, w0 = self.hole_roi.shape
        M = _pr_compose(A, np.array([[1, 0, self.off[0]], [0, 1, self.off[1]]], np.float64))   # fill ROI -> frame
        quad = np.array([[0, 0], [w0, 0], [w0, h0], [0, h0]], np.float64) @ M[:, :2].T + M[:, 2]
        x0, y0 = int(max(0, np.floor(quad[:, 0].min()))), int(max(0, np.floor(quad[:, 1].min())))
        x1, y1 = int(min(W, np.ceil(quad[:, 0].max()))), int(min(H, np.ceil(quad[:, 1].max())))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return frame
        # v11.2.102: work on the painted area and the skin around it, not the whole (much larger) fill window
        hb = self._hbox
        qh = np.array([[hb[0], hb[1]], [hb[2], hb[1]], [hb[2], hb[3]], [hb[0], hb[3]]], np.float64) @ M[:, :2].T + M[:, 2]
        sc0 = math.sqrt(abs(float(np.linalg.det(A[:, :2]))))
        pad2 = int(max(36, 0.9 * self.size * sc0 + 16))
        x0, y0 = max(x0, int(np.floor(qh[:, 0].min())) - pad2), max(y0, int(np.floor(qh[:, 1].min())) - pad2)
        x1, y1 = min(x1, int(np.ceil(qh[:, 0].max())) + pad2), min(y1, int(np.ceil(qh[:, 1].max())) + pad2)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return frame
        Mr = M.copy()
        Mr[:, 2] -= (x0, y0)                                   # fill ROI -> this ROI
        sz = (x1 - x0, y1 - y0)
        roi = frame[y0:y1, x0:x1].astype(np.float32)

        def _warp_all(Mw):
            ins = cv2.warpAffine(np.full((h0, w0), 255, np.uint8), Mw, sz, flags=cv2.INTER_NEAREST) > 0
            hl = (cv2.warpAffine(self.hole_roi.astype(np.uint8) * 255, Mw, sz, flags=cv2.INTER_LINEAR) > 127) & ins
            cr = (cv2.warpAffine(self.core_roi.astype(np.uint8) * 255, Mw, sz, flags=cv2.INTER_LINEAR) > 127) & ins
            return ins, hl, cr

        inside, hole, core = _warp_all(Mr)
        if hole.sum() < 4:
            return frame
        # v11.2.102: look for the MARK ITSELF in this frame, near where the pose puts it. It is found by what it is (it differs
        # from the skin around it, with the anchor's polarity and about its size), not by matching the anchor's texture - so a turn,
        # blur or a change of light that spoils the match does not hide it. The paint is centred on what is found and covers all of it.
        found, unsure = None, False
        scale = math.sqrt(abs(float(np.linalg.det(A[:, :2]))))
        if PATCH_CFG.get("find_mark", False) and self.core_area >= 6:
            reach = int(np.clip(0.5 * self.size * scale + 6, 6, 45))
            if idx in self.weak:
                reach = int(min(70, reach * 1.8))
            gcur_ = roi.mean(axis=2)
            found = _pr_find_mark(gcur_, hole, core, self.mark_pol, self.mark_con, self.core_area * scale * scale, reach)
            evid0, wide, unsure = _PR_EVID["v"], False, False
            if found is None and idx not in self.weak:
                # the mark is not where the pose says: look further (the pose may be off), and where it is not found at all and
                # the predicted core does not look like the mark, paint NOTHING - a patch of skin on the wrong place is worse
                found = _pr_find_mark(gcur_, hole, core, self.mark_pol, self.mark_con, self.core_area * scale * scale,
                                      int(min(80, 2 * reach + 10)), zone_r=int(reach))
                wide = found is not None
                unsure = found is None and evid0 < 0.12
            if found is not None:
                mnow, cen, area, mstrong = found
                yy_, xx_ = np.nonzero(core)
                if len(yy_) >= 4:
                    sh = np.array([cen[0] - float(xx_.mean()), cen[1] - float(yy_.mean())])
                    ar = area / max(1.0, float(core.sum()))
                    lim = (0.6 if (idx in self.weak or wide) else 0.3) * self.size * scale + 4.0   # (a pose error, not a different feature)
                    if wide:
                        lim = 1.3 * self.size * scale + 4.0
                    if 0.35 <= ar <= 2.0 and 1.0 <= float(np.hypot(*sh)) <= lim:
                        Mr = Mr.copy()
                        Mr[:, 2] += sh                          # centre the pose on the mark that was found
                        inside, hole, core = _warp_all(Mr)
                        PATCH_STATS["frames_shifted"] += 1
                PATCH_STATS["frames_found"] += 1
                # ...all of it: the mark's own strong pixels (not a faint shadow joined to it), a little margin, and never further
                # than 4 px from where the pose puts the painted area
                near = cv2.dilate(core.astype(np.uint8), _pr_disk(max(4, int(0.2 * self.size * scale)))).astype(bool)
                # ...and only pixels of the mark's own colour: on the line from the skin around to the mark's ink / pigment (blur and
                # edges mix the two), not a hand, a cup or a dark object of another hue passing over it
                lm = np.ones(near.shape, bool)
                if self.mark_bgr is not None:
                    ringc = cv2.dilate(hole.astype(np.uint8), _pr_disk(6)).astype(bool) & ~hole & inside
                    if int(ringc.sum()) >= 30:
                        skin_c = np.median(roi[ringc], axis=0)
                        u = np.asarray(self.mark_bgr, np.float32) - skin_c
                        nu = float(np.dot(u, u))
                        if nu > 400.0:
                            nyx = np.nonzero(near)                    # (only where it matters: the pixels near the mark)
                            dv = roi[nyx] - skin_c
                            t = (dv @ u) / nu
                            perp = np.sqrt(np.maximum(0.0, np.sum(dv * dv, axis=1) - t * t * nu))
                            lm = np.zeros(near.shape, bool)
                            lm[nyx] = (t > 0.1) & (t < 1.3) & (perp < 30.0)
                if (idx in self.weak or wide) and self.mark_bgr is not None:
                    n_all, n_ok = int((mstrong & near).sum()), int((mstrong & near & lm).sum())
                    if n_all >= 4 and n_ok < 0.6 * n_all:       # not the mark's colour: structure of the scene, not the mark
                        PATCH_STATS["frames_weak_skipped"] += 1
                        return frame
                near = near & lm
                found = (found[0] & near, found[1], found[2], mstrong & near)        # only what is at the mark itself
                hole = hole | (cv2.dilate(mstrong.astype(np.uint8) & near.astype(np.uint8), _pr_disk(2)).astype(bool)
                               & cv2.dilate(hole.astype(np.uint8), _pr_disk(4)).astype(bool) & inside)
            elif idx in self.weak:                              # a bridged / held pose paints only where the mark is seen
                PATCH_STATS["frames_weak_skipped"] += 1
                return frame
        fill = cv2.warpAffine(self.fill_roi, Mr, sz, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        orig = cv2.warpAffine(self.orig_roi, Mr, sz, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # object gate: is this still the anchor's mark / skin? (2 px tolerance; lighting compensated)
        ring = cv2.dilate(hole.astype(np.uint8), _pr_disk(7)).astype(bool) & ~hole & inside
        if ring.sum() > 20:
            gain = (roi[ring].mean(0) + 1.0) / (orig[ring].mean(0) + 1.0)
        else:
            gain = np.ones(3, np.float32)
        ref = orig * gain
        lo = cv2.erode(ref, np.ones((5, 5), np.uint8))
        hi = cv2.dilate(ref, np.ones((5, 5), np.uint8))
        d = (np.maximum(lo - roi, 0) + np.maximum(roi - hi, 0)).max(axis=2)
        d = cv2.GaussianBlur(d, (0, 0), 1.5)
        thr = PATCH_CFG["gate_thr"]
        keep = np.clip((d - thr) / (0.6 * thr), 0.0, 1.0)
        if PATCH_CFG.get("opaque", False):
            # v11.2.102: the paste is OPAQUE. What the mark itself looks like is never an "object" (a turned or blurred mark
            # no longer lines up with the anchor's picture of it), and only strong evidence of something else in front -
            # a hand, a cup - holds the paint back at all; a half-way verdict no longer leaves the mark showing through.
            if found is not None:
                keep = keep * (1.0 - cv2.GaussianBlur(cv2.dilate(found[0].astype(np.uint8), _pr_disk(2)).astype(np.float32), (0, 0), 1.0))
            keep = np.clip((keep - 0.55) / 0.35, 0.0, 1.0)
        pb = int(min(8, max(3, round(float(PATCH_CFG.get("peek_band", 0.0)) * self.size * scale))))
        zone = cv2.dilate(hole.astype(np.uint8), _pr_disk(max(4, pb + 2))).astype(bool)   # (the edge repair's band too)
        keep[~zone] = 0.0
        if float(keep[hole].mean()) > 0.02:
            PATCH_STATS["frames_gated"] += 1
        # nothing like the mark was found where the pose puts it, the core does not look like it, and no object explains that:
        # the pose is wrong here - paint NOTHING (a patch of skin on the wrong place is worse than a frame without the patch)
        if unsure and (PATCH_CFG.get("opaque", False) or float(keep[core].mean() if core.any() else 0.0) < 0.3):
            PATCH_STATS["frames_refused"] += 1
            return frame
        # last safeguard: the mark's own core must still look like the mark. If it does not (covered by an object,
        # or the pose is off and the core lands on plain skin), nothing of this patch is painted on this frame -
        # the skin part of the stroke would otherwise copy the anchor's skin onto some other part of the body.
        want = float(self.core_roi.sum()) * abs(float(np.linalg.det(A[:, :2])))
        if want >= 12 and float(core.sum()) < 0.3 * want:          # the mark is (mostly) outside the picture
            PATCH_STATS["frames_refused"] += 1
            return frame
        if core.sum() >= 12 and float((keep[core] < 0.5).mean()) < (0.15 if PATCH_CFG.get("opaque", False) else 0.3):
            PATCH_STATS["frames_refused"] += 1                  # (v11.2.102: only when the object covers nearly all of the core)
            return frame
        # lighting: (current - anchor) on the clean ring, spread smoothly over the hole
        valid = ~hole & inside & (keep < 0.5)
        delta = roi - orig
        if valid.sum() > 20:                                    # misaligned edges give big local deltas: not light
            dm = delta.mean(axis=2)
            med = float(np.median(dm[valid]))
            mad = float(np.median(np.abs(dm[valid] - med))) * 1.4826 + 0.5
            valid &= np.abs(dm - med) < max(8.0, 3.0 * mad)
        q = 4 if min(sz) >= 64 else 1                           # light is smooth: solve it at 1/4 size
        if q > 1:
            sm = (max(1, sz[0] // q), max(1, sz[1] // q))
            dv = cv2.resize(valid.astype(np.float32), sm, interpolation=cv2.INTER_AREA)
            dd = cv2.resize(delta * valid[..., None], sm, interpolation=cv2.INTER_AREA) / np.maximum(dv, 1e-3)[..., None]
            low, den = _pr_lowpass(dd, dv > 0.5, PATCH_CFG["light_sigma"] / q + 0.5)
            dls = _pr_pushpull(low, ((dv > 0.5) & (den > 0.2)).astype(np.float32))
            dl = cv2.resize(dls, sz, interpolation=cv2.INTER_LINEAR)
        else:
            low, den = _pr_lowpass(delta, valid, PATCH_CFG["light_sigma"])
            dl = _pr_pushpull(low, (valid & (den > 0.2)).astype(np.float32))
        # blur: soften the anchor's content when this frame is softer than the anchor (motion blur, focus)
        band = cv2.dilate(hole.astype(np.uint8), _pr_disk(10)).astype(bool) & valid
        g = 1.0
        if band.sum() > 30:
            hp_c = float((roi - cv2.GaussianBlur(roi, (0, 0), 1.5))[band].std())
            hp_a = float((orig - cv2.GaussianBlur(orig, (0, 0), 1.5))[band].std())
            g = float(np.clip(hp_c / max(hp_a, 1e-3), 0.15, 1.15))
        if g < 0.95:
            lf = cv2.GaussianBlur(fill, (0, 0), 1.5)
            fill = lf + (fill - lf) * g
        res = fill + dl
        core_now = core & hole
        res = _pr_cover(roi, res, hole & (keep < 0.5), core_now, keep < 0.5)
        res = _pr_lift(np.where(valid[..., None], roi, res), res, hole & (keep < 0.5), core_now & (keep < 0.5))
        a = _pr_alpha(hole) * (1.0 - keep)
        # edge repair (v11.2.97): dark mark pixels just outside the hole (skin bent a little, a pixel of tracking
        # error) are covered with the relit skin of the painted frame - the "opening" at the mark's edge
        if float(PATCH_CFG.get("peek_band", 0.0)) > 0:
            band = cv2.dilate(hole.astype(np.uint8), _pr_disk(pb)).astype(bool) & ~hole & inside & (keep < 0.5)
            if band.sum() > 10:
                gcur = roi.mean(axis=2)
                gres = res.mean(axis=2)
                dd = gcur - gres
                med = float(np.median(dd[band]))
                mad = float(np.median(np.abs(dd[band] - med))) * 1.4826 + 0.5
                peek = band & (dd < med - max(8.0, 3.5 * mad))
                if peek.sum() >= 3:
                    peek = cv2.dilate(peek.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & band
                    ap = cv2.GaussianBlur(peek.astype(np.float32), (0, 0), 0.8)
                    a = np.maximum(a, np.clip(ap * 1.6, 0.0, 1.0) * (1.0 - keep))
                    PATCH_STATS["frames_peek"] += 1
        out = frame.copy()
        out[y0:y1, x0:x1] = np.clip(roi * (1.0 - a[..., None]) + res * a[..., None] + 0.5, 0, 255).astype(np.uint8)
        PATCH_STATS["frames_filled"] += 1
        PATCH_STATS["fill_ms"] += (cv2.getTickCount() - t0) * 1000.0 / cv2.getTickFrequency()
        return out


# --------------------------------------------------------------------------------------------------- planning
def _pr_fit_ref(ref, rmask, size_wh):
    """The app's frame (any size; possibly not rotated like the decoded video) -> candidates at the video's size."""
    W, H = size_wh
    out = []
    for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        r = ref if rot is None else cv2.rotate(ref, rot)
        m = rmask if rot is None else cv2.rotate(rmask, rot)
        if abs(r.shape[1] / float(r.shape[0]) - W / float(H)) > 0.04 * (W / float(H)):
            continue
        out.append((cv2.resize(r, (W, H), interpolation=cv2.INTER_AREA if r.shape[1] > W else cv2.INTER_LINEAR),
                    cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)))
    return out


def _pr_thumb(gray):
    w = 160
    h = max(8, int(round(gray.shape[0] * w / float(gray.shape[1]))))
    return cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)


_PR_LOC = {}          # v11.2.104: why the last anchor search succeeded or failed (for the job log): numbers only


def locate_anchor(ref, rmask, size_wh, thumbs, gray_at, hint, window):
    """Find the frame the user painted on, and the painted mask in that frame's pixels.
    thumbs: per-frame _pr_thumb; gray_at(i): full gray frame; hint: expected index. -> (idx, hole bool, score) or None."""
    best = (-2.0, None, None)
    n = len(thumbs)
    for r, m in _pr_fit_ref(ref, rmask, size_wh):
        t = _pr_thumb(_pr_gray(r))
        order = list(range(max(0, hint - window), min(n, hint + window + 1)))
        rest = [i for i in range(n) if i not in set(order)]
        for pass_ in (order, rest):
            for i in pass_:
                sc = _pr_ncc(t, thumbs[i])
                if sc > best[0]:
                    best = (sc, i, (r, m))
            if best[0] >= 0.85:
                break
    _PR_LOC.clear()
    _PR_LOC["frame_ncc"], _PR_LOC["frame_idx"] = float(best[0]), best[1]
    _PR_LOC["ref_wh"] = tuple(int(v) for v in ref.shape[1::-1])
    _PR_LOC["video_wh"] = tuple(int(v) for v in size_wh)
    _PR_LOC["candidates"] = len(_pr_fit_ref(ref, rmask, size_wh))
    if best[1] is None:
        return None
    sc, i, (r, m) = best
    g = gray_at(i)
    rg = cv2.GaussianBlur(_pr_gray(r), (5, 5), 1.0).astype(np.float32)
    fg = cv2.GaussianBlur(g, (5, 5), 1.0).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    try:                                        # the app's frame vs the decoded frame: small shifts / scaling
        half = (rg.shape[1] // 2, rg.shape[0] // 2)
        _, warp = cv2.findTransformECC(cv2.resize(rg, half), cv2.resize(fg, half), warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 3)
        warp[:, 2] *= 2.0
    except cv2.error:
        warp = np.eye(2, 3, dtype=np.float32)
    hole = cv2.warpAffine((m > 127).astype(np.uint8) * 255, warp.astype(np.float64), (g.shape[1], g.shape[0]),
                          flags=cv2.INTER_LINEAR) > 127
    fine = _pr_ncc(cv2.warpAffine(rg, warp, (g.shape[1], g.shape[0])), fg)
    _PR_LOC["fine"] = float(fine)
    if not hole.any() or fine < 0.6:
        return None
    return i, hole, fine


def find_patch(ref, rmask, size_wh, n, gray_at, stride=3, budget_s=None, stop=None):
    """The painted frame is not in this video (another part of a split job, or the clip was cut differently): look
    for the painted area itself in the frames (ORB + template matching against the app's frame, strictly checked).
    -> (idx, hole bool, score) or None."""
    best = (0.0, None, None)
    t_start = time.time()
    seen, near_best, timed_out = 0, 0.0, False
    for r, m in _pr_fit_ref(ref, rmask, size_wh):
        hole = m > 127
        if not hole.any():
            continue
        fnd = PatchTrack(-1, r, hole, make_fill=False)
        for i in range(0, n, stride):
            if (budget_s is not None and time.time() - t_start > budget_s) or (stop is not None and stop()):
                timed_out = True
                break
            seen += 1
            g = gray_at(i)
            for A0 in (fnd._redetect(g), fnd._redetect_tm(g)):
                if A0 is None:
                    continue
                Ar, cc = fnd._ecc(g, A0)
                A1 = Ar if cc > 0.5 else A0
                sc, inl = fnd._check(g, A1)
                if inl is None:
                    continue
                al, cov = fnd._last_all, fnd._last_cov
                near_best = max(near_best, min(al, cov))
                if al >= 0.8 and cov >= 0.7 and al * cov > best[0]:
                    best = (al * cov, i, (A1, hole))
            if best[0] >= 0.95:
                break
        if timed_out:
            break
    _PR_LOC["area_samples"], _PR_LOC["area_best"] = seen, float(near_best)
    _PR_LOC["area_seconds"], _PR_LOC["area_timed_out"] = float(time.time() - t_start), bool(timed_out)
    if best[1] is None:
        return None
    A1, hole = best[2]
    g = gray_at(best[1])
    h = cv2.warpAffine(hole.astype(np.uint8) * 255, A1, (g.shape[1], g.shape[0]), flags=cv2.INTER_LINEAR) > 127
    return (best[1], h, best[0]) if h.any() else None


class PatchGroup:
    """One mark painted at one or more positions (v11.2.97). Each position is followed by its own PatchTrack; on every
    frame the view nearest in time that is followed there is used (a view whose confidence is clearly lower gives
    way), and if that view refuses the frame, the next one is tried."""

    def __init__(self, tracks):
        self.tracks = list(tracks)

    @property
    def A(self):
        keys = set()
        for t in self.tracks:
            keys.update(t.A.keys())
        return {k: None for k in keys}

    def apply(self, frame, idx):
        cands = [t for t in self.tracks if idx in t.A]
        if not cands:
            return frame
        best = max(float(t.conf.get(idx, 0.0)) for t in cands)
        near = sorted((t for t in cands if float(t.conf.get(idx, 0.0)) >= best - 0.15),
                      key=lambda t: abs(idx - t.anchor_idx))
        rest = [t for t in cands if t not in near]
        for t in near + rest:
            out = t.apply(frame, idx)
            if out is not frame:
                return out
        return frame


def plan_tracks(specs, size_wh, n, thumbs, gray_at, bgr_at, log=None, progress=None, stop=None):
    """specs: [(ref_bgr, mask_uint8, hint_index, window[, group])] - several entries with the same group are views of
    one mark at different positions. -> list of PatchGroup ready to apply (by frame index).
    progress(text): called about once a second while following (v11.2.101: the job no longer sits silent);
    stop(): True cancels (the result is then an empty list)."""
    groups = {}
    order = []
    dropped = max(0, len(specs) - int(PATCH_CFG.get("max_views", 20)))
    for sp in specs[:int(PATCH_CFG.get("max_views", 20))]:
        gk = sp[4] if len(sp) > 4 and sp[4] is not None else ("_v%d" % len(order))
        if gk not in groups:
            if len(groups) >= PATCH_CFG["max_patches"]:
                dropped += 1
                continue
            groups[gk] = []
            order.append(gk)
        groups[gk].append(sp[:4])
    if dropped and log:                                      # never silent: the job log says what was left out
        log("WARNING: %d painted position(s) were left out - the engine takes at most %d marks and %d positions per job"
            % (dropped, int(PATCH_CFG["max_patches"]), int(PATCH_CFG.get("max_views", 20))))
    # 1) find every painted position in this video (cheap), 2) follow each through its own stretch
    located = []
    deadline = time.time() + float(PATCH_CFG.get("find_total_s", 240.0))      # (searches for the painted AREA share this time)
    for m, gk in enumerate(order):
        if stop is not None and stop():
            return []
        located.append(_locate_views(groups[gk], m + 1, size_wh, n, thumbs, gray_at, log, stop, deadline))
    total = max(1, sum(f["hi"] - f["lo"] + 1 for fl in located for f in fl))
    st = {"done": 0, "t0": time.time(), "last": 0.0, "mark": 0, "marks": len(order)}

    def tick():
        if stop is not None and stop():
            return True
        st["done"] += 1
        now = time.time()
        if progress is not None and now - st["last"] >= 1.0:
            st["last"] = now
            frac = min(1.0, st["done"] / float(total))
            eta = (now - st["t0"]) * (1.0 - frac) / max(frac, 1e-3)
            progress("Patch removal: following mark %d of %d · %d%% · about %s left"
                     % (st["mark"], st["marks"], int(100 * frac), ("%d min" % int(eta / 60 + 0.5)) if eta >= 90 else ("%d s" % int(eta))))
        return False

    out = []
    for m, gk in enumerate(order):
        st["mark"] = m + 1
        PATCH_STATS["patches"] += 1
        tracks = _follow_views(located[m], m + 1, n, gray_at, bgr_at, log, tick, stop)
        if stop is not None and stop():
            return []
        if tracks:
            out.append(PatchGroup(tracks))
            if log and len(tracks) > 1:
                cov = set()
                for t in tracks:
                    cov.update(t.A.keys())
                log("patch %d: %d positions, together followed on %d of %d frames" % (m + 1, len(tracks), len(cov), n))
    return out


def _locate_views(views, mark_no, size_wh, n, thumbs, gray_at, log, stop=None, deadline=None):
    """Where is each painted position of this mark in this video? -> [{anchor, hole, score, tag, lo, hi}], by anchor frame,
    with the stretch each one is to be followed over: from the previous painted position (and follow_margin frames
    before it) to the next one (and follow_margin frames past it); the first goes back to frame 0, the last to the end."""
    found = []
    for v, (ref, rmask, hint, window) in enumerate(views):
        if stop is not None and stop():
            return []
        tag = "%d" % mark_no if len(views) == 1 else "%d.%d" % (mark_no, v + 1)
        PATCH_STATS["views"] += 1
        hint = int(np.clip(hint, 0, max(0, n - 1)))
        loc = locate_anchor(ref, rmask, size_wh, thumbs, gray_at, hint, window)
        d0 = dict(_PR_LOC)
        if loc is None:
            left_s = None if deadline is None else deadline - time.time()
            budget = float(PATCH_CFG.get("find_budget_s", 90.0))
            if left_s is not None:
                budget = min(budget, max(0.0, left_s))
            if budget >= 3.0:
                loc = find_patch(ref, rmask, size_wh, n, gray_at, budget_s=budget, stop=stop)
            elif log:
                log("patch %s: the time for searching the painted area is used up - not searched" % tag)
            if loc is not None and log:
                log("patch %s: the painted frame is not in this video; the painted area was found on frame %d" % (tag, loc[0]))
        if loc is None:
            if log:
                # numbers only: how close the best frame came (needs 0.85 on the 160 px picture and 0.60 after alignment), and how the
                # search for the area itself went (needs 0.8 and 0.7)
                d1 = dict(_PR_LOC)
                log("patch %s: neither the painted frame nor the painted area was found in this video - skipped "
                    "(app frame %s px, video %s px, %s size match(es); best whole-frame match %.2f at frame %s, after alignment %s; "
                    "area search: %d frames tried in %.0f s%s, best %.2f)"
                    % (tag, "x".join(str(v) for v in d0.get("ref_wh", ("?",))), "x".join(str(v) for v in d0.get("video_wh", ("?",))),
                       d0.get("candidates", "?"), float(d0.get("frame_ncc", -1.0)), d0.get("frame_idx", "-"),
                       ("%.2f" % d0["fine"]) if "fine" in d0 else "n/a",
                       int(d1.get("area_samples", 0)) if "area_samples" in d1 else 0, float(d1.get("area_seconds", 0.0)),
                       " (stopped: time budget)" if d1.get("area_timed_out") else "", float(d1.get("area_best", 0.0))))
            continue
        i, hole, sc = loc
        if any(f["anchor"] == i for f in found):                # (two positions painted on the same frame)
            continue
        found.append({"anchor": i, "hole": hole, "score": sc, "tag": tag})
    found.sort(key=lambda f: f["anchor"])
    mg = int(PATCH_CFG.get("follow_margin", 45))
    for j, f in enumerate(found):
        f["lo"] = 0 if j == 0 else max(0, found[j - 1]["anchor"] - mg)
        f["hi"] = n - 1 if j == len(found) - 1 else min(n - 1, found[j + 1]["anchor"] + mg)
    return found


def _follow_views(found, mark_no, n, gray_at, bgr_at, log, tick, stop=None):
    tracks = []
    for f in found:                                              # ascending anchors: one pass over the video serves bgr_at
        if stop is not None and stop():
            break
        i = f["anchor"]
        PATCH_STATS["anchored"] += 1
        tr = PatchTrack(i, bgr_at(i), f["hole"])
        tr.track(gray_at, list(range(i, f["hi"] + 1)), tick)
        tr.track(gray_at, list(range(i, f["lo"] - 1, -1)), tick)
        tr.smooth()
        if log:
            log("patch %s: anchored on frame %d (match %.2f), %d px, followed on %d of %d frames"
                % (f["tag"], i, f["score"], int(tr.hole_roi.sum()), len(tr.A), f["hi"] - f["lo"] + 1 if len(found) > 1 else n))
        tracks.append(tr)
    return tracks
