"""Swamitech Phoenix v11.2.7 "EyesOpen" configuration (the pipeline now looks at the pixels)."""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Tuple

ENV_DEFAULTS = {
    "GRADIO_ANALYTICS_ENABLED": "False",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "GRADIO_TEMP_DIR": "/tmp/gradio",
}
for _k, _v in ENV_DEFAULTS.items():
    os.environ.setdefault(_k, _v)

DET_MAX_W = 720
DET_SIZE = (640, 640)
# CPU-only speed tiers: smaller detector input for the common single-face path.
DET_SIZE_BALANCED = (512, 512)
DET_SIZE_BEST = (640, 640)
FACE_MODEL_NAME = os.environ.get("PHOENIX_FACE_MODEL", "buffalo_l").strip() or "buffalo_l"
ENABLE_HI_DET_PROBE = os.environ.get("PHOENIX_HI_DET_PROBE", "0").strip().lower() in ("1", "true", "yes", "on")

# Detector acceptance threshold. InsightFace defaults to 0.5; profile and
# distant faces routinely score 0.3-0.5 and are therefore never returned at the
# default. A missed detection used to trigger a revert to the original face,
# which is far more visible than a weak detection the tracker then rejects.
DET_THRESH = 0.32

# ========================= CPU EFFICIENCY TUNING ===========================
#
# These settings are deliberately conservative. Speed depends on the selected
# output resolution, source FPS, model provider and enhancer; benchmark before
# changing them for a particular Space.
#
# INPUT_MAX_W: historical decode cap. v11.0.2 reader resizes once to the
# chosen output size (ow×oh) — an extra pre-downscale to this width caused
# crush→upscale on 720p+ outputs. Kept for docs/compat; pipeline no longer
# applies it as an intermediate scale.
INPUT_MAX_W = 720

# Maximum concurrent Python frame workers. Keep at 1 on HF CPU so ORT/OpenCV/
# x264 native threads are not thrashing. Override with PHOENIX_VIDEO_WORKERS≤2.
VIDEO_WORKERS = 1

# NOTE (v11.1.0): swap_n / SKIP_N no longer decide how many frames get a face.
# EVERY output frame is composited; these now only decide how often the swap
# NETWORK runs. Between network runs the cached aligned result is re-projected
# onto the current frame with the current frame's own geometry, mask,
# background and lighting, so raising the skip interval costs expression
# freshness, not face presence or placement.
#
# DET_SKIP_INTERVAL: default detector cadence ONLY when det_n/det_int is Auto.
# Do NOT use this as a hard floor over explicit preset/user values (that bug
# forced re-detect every ≥3 keyframes even when Stable set det_n=1 → flicker).
# Set to 1 = detect every keyframe when Auto (least flicker, more CPU).
# Higher values (e.g. 3) are a speed default for Auto on constrained CPU.
DET_SKIP_INTERVAL = 1

# How long a tracked identity may be held/faded once real detection stops
# (occlusion, a fast pan, a genuine multi-frame detector miss) before the
# render loop gives up and shows the original frame instead. Longer trades
# more exposure to a stale/drifting held position if the subject genuinely
# left (the frame-edge/containment checks in _run_job_body still catch that
# case regardless of this value, and a sustained content-visibility failure
# like occlusion is separately routed back to the short floor - see
# _ACTIVE_REJECT_STREAK in core_pipeline.py) for far fewer needless reverts
# during ordinary, brief tracking gaps that HoldThrough's bbox-overlap
# rescues cannot help with (those need SOME detected box to compare
# against; a genuine multi-frame miss has none).
# v11.2.7: 3.0 -> 1.0. Three seconds was set in v11.2.5 to stop needless
# reverts, on the reasoning that a revert is the worse artifact. That was
# measured against reverts and never against the opposite failure, and the
# opposite failure is what got reported: at 3.0s a subject who turns away
# keeps a held face painted over the back of their head for practically the
# whole turn - measured on t_extended_lookaway, 106 of a 120-frame turn-away
# were still being painted.
#
# Two things make the shorter window safe now in a way it was not in v11.2.5.
# Suppression FADES rather than cuts (PASTE_FADE_SEC below), so reaching the
# end of the window costs a soft fade, not the hard flash of the real face
# that the long window was reacting to. And an empty detector return is no
# longer conflated with positive evidence of absence (see _vis_marks in
# core_pipeline), so this window is now only ever spent on genuine "we cannot
# see anything" gaps rather than on frames the content gate already judged.
#
# 2.0s is now only the FADE LENGTH for a gap, not the whole hold budget: a
# run of empty detector returns longer than BLIND_AFTER_SEC below opens a
# blind span and drops to the short cadence taper regardless of this value.
# So this can be generous, and being generous is what keeps a brief gap from
# dimming - at 2.0s an 18-frame motion-blur dropout sits at 91% opacity
# instead of 64%, and the fade's per-frame slope (2/taper, which is what
# frame-to-frame area change tracks) is halved.
REACQUIRE_GRACE_SEC = 2.0

# How long the composited face takes to fade out when a suppression path
# fires, instead of cutting to the untouched original in a single frame.
#
# This is the fix for the flicker/revert family at its shared root. The
# pipeline already had two smooth fades (TrackState.smooth_alpha's EMA and
# _geom_for_frame's taper), but EVERY path that decides "do not paint this
# frame" bypassed both and emitted the pristine original immediately - so a
# gate flipping for one or two frames showed as a hard flash of the real
# face (flicker), and the same flip sustained showed as a revert. Fading
# instead of cutting makes a brief flip cost a few percent of opacity rather
# than a full-strength flash, while a sustained one still ends up fully
# reverted, just smoothly.
#
# Only the DOWNWARD direction is slewed. Coming back up stays instant, which
# preserves v11.2.1 SolidFace's "when the gates say yes, sit at full
# strength" behaviour - a face appearing is never itself a flash of the
# original, so it needs no ramp.
# How long a run of "the detector returned nothing" may last before it stops
# being treated as a gap to ride through and starts being treated as evidence
# that the face is gone.
#
# This is the one number separating two cases that look identical from the
# detector's side: a face hidden by motion blur for a few frames, which must
# be ridden through or it flickers, and a subject who has turned away or left,
# which must be dropped or their real face gets a swap pasted over it. Only
# duration tells them apart. 0.75s clears the longest dropout in this repo's
# fixtures (t_modes' 18 frames, 0.6s) and bounds a genuine absence well inside
# a second.
#
# Set deliberately SHORTER than REACQUIRE_GRACE_SEC: that one is now the fade
# length for gaps this one has not yet declared blind.
BLIND_AFTER_SEC = 0.75

PASTE_FADE_SEC = 0.5
FACE_ROI_PAD = 0.22
FACE_EMA_ALPHA = 0.40
COLOR_MATCH_SCALE = 0.25
# Reference cadence table (Auto swap_n). Best/Ultra stay at 1 so Stable /
# HQ presets never silently skip every other AI swap frame.
SKIP_N = {"Fast": 6, "Balanced": 4, "Optimized": 5, "Best": 1, "Ultra": 1}

RES: Dict[str, Tuple[int, int]] = {
    "540p (Fastest)": (960, 540),
    "640p (Fast)": (1136, 640),
    "680p": (1208, 680),
    "720p (HD)": (1280, 720),
    "900p (HD+)": (1600, 900),
    "1080p (Full HD)": (1920, 1080),
}

RETAIN_SEC = 10800  # completed server outputs retained for 3 hours
ORPHAN_SEC = 86400
SESSION_TTL_SEC = 86400
SESSION_DIR = "/tmp/swamitech_session"
SESSION_JSON = "/tmp/swamitech_session.json"

AUTO_SAVE_ENABLED = True
AUTO_SAVE_DIR = "/tmp/.swamitech_autosave"
AUTO_SAVE_RETAIN_SEC = 10800
HF_OUTPUT_REPO = os.environ.get("SWAMITECH_HF_OUTPUT_REPO", "").strip()
HF_UPLOAD_ENABLED = bool(os.environ.get("HF_TOKEN")) and bool(HF_OUTPUT_REPO)

# Authoritative server-side result storage. Attach an HF Storage Bucket to /data
# for persistence across Space restarts; the app writes completed outputs to
# this directory before marking a job done. Override with an environment variable
# when using a different mounted persistent volume.
PERSISTENT_OUTPUT_DIR = os.environ.get("PHOENIX_PERSISTENT_OUTPUT_DIR", "/data/phoenix_outputs").strip() or "/data/phoenix_outputs"
PERSISTENT_JOB_STATE_DIR = os.environ.get("PHOENIX_PERSISTENT_JOB_STATE_DIR", "/data/phoenix_jobs").strip() or "/data/phoenix_jobs"
SERVER_OUTPUT_TTL_SEC = 10800  # exactly 3 hours after successful completion

DUR = [10, 20, 30, 60, 90, 120, 150, 180, 240, 300, 360]
FPS = [15, 24, 30, 40, 50, 60]

VERSION = "v11.2.7"
BUILD = "EyesOpen · a face is only pasted where a face can be seen"
VERSION_FULL = f"{VERSION} ({BUILD})"

@dataclass
class Settings:
    det_max_w: int = DET_MAX_W
    det_size: Tuple[int, int] = DET_SIZE
    version: str = VERSION_FULL

SETTINGS = Settings()


# ---------------------------------------------------------------------------
# v11 engine tunables (swap_engine.AlignedCompositor / MultiFaceTracker).
# These are the dials worth touching if output needs adjusting; everything is
# applied at import time via swap_engine.configure(**ENGINE_TUNABLES).
# ---------------------------------------------------------------------------
ENGINE_TUNABLES = {
    # --- mask shape, defined in the pose-normalised aligned crop -----------
    # Because the crop is pose-normalised, one shape fits every head angle.
    # That is what makes it flicker-free: nothing about it varies per frame.
    "mask_rx":          0.435,   # half-width  (fraction of crop). Larger = more coverage
    "mask_ry":          0.495,   # half-height. Raise if the chin/forehead is clipped
    "mask_feather":     0.16,    # edge softness. Raise if you can see the seam
    "hull_dilate":      0.085,   # landmark-hull dilation
    "hull_floor":       0.30,    # hull can never remove more than 70% of the template
    "mask_ema":         0.36,    # SoftStable: keep 0.36 (do not raise like ProStable 0.40)

    # --- colour transfer ---------------------------------------------------
    # cm_std_lo is the important one. The v10.9.3 bug was effectively a hard
    # 0.20 contrast ratio, which flattened the face into the brown patch.
    # Never set cm_std_lo below ~0.65.
    # SoftStable: gentler luma / tighter shift / delta clamp — brightness only.
    "cm_strength_ab":   0.85,    # chroma follows the scene strongly
    "cm_strength_l":    0.42,    # CinemaQA: steadier than SoftStable 0.45
    "cm_std_lo":        0.72,    # floor on the contrast ratio
    "cm_std_hi":        1.45,
    "cm_ema":           0.15,    # CinemaQA: steadier than SoftStable 0.18
    "cm_max_shift":     20.0,    # SoftStable: was 26 — tighter LAB mean cap
    "cm_delta_clamp":   4.0,     # CinemaQA: tighter than SoftStable 5.0

    # --- occlusion guard (hug / kiss / hand across the face) ---------------
    "occl_min_keep":    0.45,    # SolidFace: center stays opaque under marginal occl
    # The guard is a CONTINUOUS weight, not an on/off flag. A flag changed the
    # mask silhouette - and so the colour statistics weighted by that mask - in
    # a single frame every time it flipped, moving outline and brightness at
    # once. occl_ramp is how fast it may slew per frame; lower = smoother.
    "occl_ramp":        0.25,
    # Two faces in contact have essentially identical chroma, so the skin
    # confidence term cannot see the other person's cheek inside this face's
    # aligned crop. Their own landmark hull can be projected in and subtracted.
    # rival_cut 0 disables it; raise toward 1.0 to trim contact harder.
    "rival_cut":        0.85,
    "rival_feather":    0.09,
    # The 106-point hull is the only pose-DEPENDENT term in an otherwise
    # pose-normalised mask, so it is what makes the silhouette breathe on a yaw
    # turn. Lower = steadier outline on profile turns, slower to follow a real
    # change in face shape.
    "hull_ema":         0.22,

    # --- multi-face association -------------------------------------------
    "trk_gate_new":     0.30,    # identity similarity needed to CREATE a binding
    "trk_gate_hold":    0.10,    # ...and to KEEP an established one (deliberately low)
    "trk_lock_hits":    4,       # frames before a track counts as established
    "trk_cross_iou":    0.12,    # tracks this close are "crossing" -> embeddings frozen
    "trk_flip_margin":  0.14,    # identity margin needed to justify a label flip
    "trk_flip_frames":  5,       # ...sustained for this many consecutive frames
    "trk_max_missed":   24,      # carry-through budget before the track gives up
    "trk_alpha":        0.42,    # bbox smoothing on update
    # One-Euro landmark smoothing (keys must exist in swap_engine._P to apply)
    "kps_min_cutoff":   0.08,
    "kps_beta":         0.040,
}
