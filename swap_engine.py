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
    "TrackState",
    "MultiFaceTracker",
    "PredictedFace",
    "swap_and_composite",
    "ENGINE_VERSION",
]

ENGINE_VERSION = "aequus-1.0.4-softstable"

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

    k = int(max(3, round(size * _P["mask_feather"]))) | 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    _TEMPLATE_CACHE[size] = mask
    return mask


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
                 "flip_votes", "crossing", "last_face", "det_score", "last_hit_bbox")

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
        return (np.asarray(base, np.float32) + self.vel_bbox).astype(np.float32)

    def predict(self):
        """Advance geometry one frame with constant velocity. Used when the
        detector was skipped or missed, so the swap can still run instead of
        the frame reverting to the original face."""
        self.missed += 1
        damp = 0.85 ** min(self.missed, 12)
        if self.bbox is not None:
            self.bbox = (self.bbox + self.vel_bbox * damp).astype(np.float32)
        if self.obs_bbox is not None:
            self.obs_bbox = (self.obs_bbox + self.vel_bbox * damp).astype(np.float32)
        if self.kps is not None and self.vel_kps is not None:
            self.kps = (self.kps + self.vel_kps * damp).astype(np.float32)
        return self.bbox

    def update(self, face, update_embedding=True):
        a = float(_P["trk_alpha"])
        bb = np.asarray(face.bbox, np.float32).reshape(4).copy()
        if self.obs_bbox is None:
            self.vel_bbox = np.zeros(4, np.float32)
        else:
            self.vel_bbox = (self.vel_bbox * 0.5 + (bb - self.obs_bbox) * 0.5).astype(np.float32)
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
                raw_speed = float(np.mean(np.abs(kp - self.kps)))
                cutoff = float(_P["kps_min_cutoff"]) + float(_P["kps_beta"]) * raw_speed
                r = 2.0 * math.pi * cutoff
                a = r / (r + 1.0)
                sm = self.kps * (1.0 - a) + kp * a
                v = sm - self.kps
                self.vel_kps = (v if self.vel_kps is None
                                else self.vel_kps * 0.6 + v * 0.4)
                self.kps = sm.astype(np.float32)
            else:
                self.kps = kp
                self.vel_kps = np.zeros_like(kp)

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
        if self.cm_dmean is None or self.missed > 8:
            self.cm_dmean = dmean.astype(np.float32).copy()
            self.cm_ratio = ratio.astype(np.float32).copy()
        else:
            max_step = float(_P.get("cm_delta_clamp", 5.0) or 5.0)
            prev = self.cm_dmean
            clamped = np.clip(dmean, prev - max_step, prev + max_step)
            self.cm_dmean = (prev * (1.0 - a) + clamped * a).astype(np.float32)
            self.cm_ratio = (self.cm_ratio * (1.0 - a) + ratio * a).astype(np.float32)
        return self.cm_dmean, self.cm_ratio

    def reset_appearance(self):
        self.mask_ema = None
        self.cm_dmean = None
        self.cm_ratio = None


class PredictedFace:
    """Minimal duck-typed stand-in matching insightface's Face attributes."""

    __slots__ = ("bbox", "kps", "landmark_2d_106", "normed_embedding",
                 "det_score", "predicted", "_track", "_occlusion_guard")

    def __init__(self, bbox, kps, lmk=None, emb=None, det_score=0.5):
        self.bbox = np.asarray(bbox, np.float32).reshape(4).copy()
        self.kps = np.asarray(kps, np.float32).copy()
        self.landmark_2d_106 = None if lmk is None else np.asarray(lmk, np.float32).copy()
        self.normed_embedding = emb
        self.det_score = float(det_score)
        self.predicted = True
        self._track = None
        self._occlusion_guard = False


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
    def build_mask(self, face, M, size, track=None, occlusion_guard=False,
                   aligned_target=None):
        mask = canonical_template(size).copy()

        pts = _landmarks_to_aligned(getattr(face, "landmark_2d_106", None), M, size)
        h = hull_mask(pts, size)
        if h is not None:
            floor = float(_P["hull_floor"])
            mask = mask * (floor + (1.0 - floor) * h)

        if occlusion_guard and aligned_target is not None:
            mask = mask * skin_confidence(aligned_target, mask)

        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        if track is not None:
            mask = track.smooth_mask(mask)
        return mask

    # -- colour ----------------------------------------------------------
    @staticmethod
    def _masked_stats(lab: np.ndarray, w: np.ndarray):
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

        # In-place on f_lab — no full LAB duplicate (CPU bandwidth).
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

        warp_face = cv2.warpAffine(fake_bgr, local_M, (rw, rh),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        warp_mask = cv2.warpAffine(mask, local_M, (rw, rh),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warp_mask = np.clip(warp_mask * float(alpha), 0.0, 1.0)
        if float(warp_mask.max()) <= 0.004:
            return img

        out = img if img.flags.writeable else img.copy()
        m = warp_mask[..., None]
        roi = out[y1:y2, x1:x2].astype(np.float32)
        fac = warp_face.astype(np.float32)
        out[y1:y2, x1:x2] = (fac * m + roi * (1.0 - m)).astype(np.uint8)
        return out

    # -- one-shot -----------------------------------------------------------
    def run(self, work, orig, face, src_face, *, track=None, alpha=1.0,
            colour_strength=1.0, occlusion_guard=False):
        """Swap `face` in `work` with `src_face`. Returns (image, ok)."""
        fake, M = self._raw_swap(work, face, src_face)
        if fake is None:
            return work, False

        size = int(fake.shape[0])
        aligned_target = cv2.warpAffine(orig, M, (size, size),
                                        borderMode=cv2.BORDER_REPLICATE)
        mask = self.build_mask(face, M, size, track=track,
                               occlusion_guard=occlusion_guard,
                               aligned_target=aligned_target)
        fake = self.colour_match(fake, aligned_target, mask, track=track,
                                 strength=colour_strength)
        out = self.paste_back(work, fake, mask, M, alpha=alpha)
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

    Slots <= 4 and detections <= 10 in practice, so an exhaustive search over
    permutations is both optimal and fast. Unlike the previous greedy
    slot-by-slot loop, this cannot hand face #1's source to face #2 just
    because face #1 happened to be iterated first — which is exactly the
    hug/kiss mix-up.
    """
    n = len(cost)
    if n == 0:
        return {}
    m = len(cost[0])
    if m == 0:
        return {}
    forbidden = forbidden or set()
    best, best_cost = {}, float("inf")
    idx = list(range(n))
    for r in range(min(n, m), 0, -1):
        found = False
        for rows in permutations(idx, r):
            for cols in permutations(range(m), r):
                total, ok = 0.0, True
                for a, b in zip(rows, cols):
                    if (a, b) in forbidden or cost[a][b] >= 1e6:
                        ok = False
                        break
                    total += cost[a][b]
                if ok and total < best_cost:
                    best_cost = total
                    best = dict(zip(rows, cols))
                    found = True
        if found:
            break
    return best


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
    def assign(self, faces, refs):
        """faces: list of detections. refs: {slot: reference embedding}.

        Returns {slot: face} for slots that got a detection this frame. Slots
        that missed are advanced by prediction, and the caller can still swap
        them via TrackState.predicted_face().
        """
        self._mark_crossings()
        slots = sorted(self.tracks.keys())
        if not faces:
            for t in self.tracks.values():
                t.predict()
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
                tr.predict()
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
                    tr.predict()
                    continue
                if sim_tab[i][j] < _P["trk_flip_margin"] + 0.30:
                    tr.flip_votes += 1
                    if tr.flip_votes < _P["trk_flip_frames"]:
                        # Keep the previous binding for now; geometry carries it.
                        tr.predict()
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

            tr.update(f, update_embedding=good)
            if changed:
                tr.reset_appearance()
            result[s] = f

        for s in slots:
            if self.tracks[s].missed > _P["trk_max_missed"]:
                keep_emb = self.tracks[s].emb
                self.tracks[s] = TrackState(s)
                self.tracks[s].emb = keep_emb
        return result

    def bind(self, slot, face, update_embedding=True):
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
        tr.update(face, update_embedding=update_embedding)
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
            if tr.missed == 0 or not tr.established:
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

