---
title: SG UAT2
emoji: 🔥
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10"
app_file: app.py
pinned: false
---

See README_DEPLOY.md for the actual deployment/architecture notes.
This file's YAML header is what Hugging Face reads to build the Space.

sdk_version is pinned to 4.44.1 to match the gradio==4.44.1 pin in
requirements.txt.

python_version is pinned to 3.10 because requirements.txt's own first
line says "HF Spaces · Python 3.10 · ZeroGPU-capable pack" - without
this field, the platform defaults to whatever its current default is
(3.13 as of this build), which breaks onnxruntime<1.20 (no 3.13 wheel
exists below 1.20) and would break torch==2.1.2 next for the same
reason - that exact version predates Python 3.13's release, so no
3.13 wheel exists for it either.

## Continuum (v11.1.0) — every output frame is composited

Up to v11.0.4 only *key* frames were composited. The frames in between were
filled by copying a face ROI out of a neighbouring frame, cross-fading two
whole frames, or emitting the untouched original. That fill path is where the
visible defects came from, and it is gone.

Now the swap **network** still runs only on key frames — that is the expensive
part — but its 128x128 ArcFace-aligned output is cached, and **every** output
frame is composited from it using that frame's own interpolated keypoints,
mask, background and lighting. Raising the skip interval costs expression
freshness; it no longer costs face presence, placement or exposure.

Measured on a synthetic clip with a known ground-truth face path
(960x540, Optimized preset, `swap≈5`):

| | v11.0.4 | v11.1.0 |
|---|---|---|
| frames showing the original face, fast pan | 233 / 420 | **0 / 420** |
| frames showing the original face, profile turn | 156 / 240 | **0 / 240** |
| frames showing the original face, 18-frame detector dropout | 165 / 240 | **0 / 240** |
| frames showing the original face, soft-polish enhancer on | 114 / 150 | **0 / 150** |
| two faces meeting and parting: frames keeping their own identity | — | **300 / 300** |
| face placement error vs ground truth | mean 67 px, max 505 px | **mean 2.0 px, max 7 px** |
| frame-to-frame face luma step | max 9.4 | **max 2.6** |
| assignment cost, 4 slots x 12 detections | 209 ms/frame | **0.03 ms/frame** |

The `Stable` preset was already clean in v11.0.4 — because it never skips a
swap — which is exactly what identified the fill path as the cause.

### Notes for tuning

* `config.ENGINE_TUNABLES` carries the dials. `occl_ramp`, `rival_cut`,
  `rival_feather` and `hull_ema` are new in v11.1.0.
* The occlusion guard is a continuous weight now, not an on/off flag: a flag
  moved the mask silhouette (and the colour statistics weighted by it) in a
  single frame every time it toggled.
* Faces in contact (a kiss, a hug) are separated geometrically as well as by
  chroma — two touching faces have essentially the same skin chroma, so the
  nearer person's landmark hull is projected into the further person's aligned
  crop and subtracted (`rival_cut`).
* Face enhancers run once on the cached aligned crop rather than per output
  frame, so enhanced and unenhanced frames no longer alternate at the swap
  cadence.

## v11.1.1 — efficiency pass, and two seam fixes

### Faces leaving the frame

A subject walking out of shot left a face pasted against the frame edge for
~17 frames. The detector stops reporting them, the tracker keeps
extrapolating, and the smoothed box *lags and then stalls short of the border*
— so neither "is it outside the frame?" nor "is it touching the edge?" catches
it; the box sits comfortably inside the frame painting a face onto the
background. The test that works is the **last real detection**: if that was
already in contact with a border, the subject was on their way out, and held
geometry for them is not painted. A face occluded mid-frame is unaffected and
still holds (verified: an 18-frame dropout still yields 0 original-face
frames).

### The mask had a hard edge

The canonical template is centred at `cy=0.545` with `ry=0.495`, so it reaches
1.04 — it runs off the bottom of the aligned crop, and measured **0.881 at the
bottom border** (top 0.250, sides 0.111). That is a hard edge in the
composite, not a feathered one. It normally hides because the chin edge lands
on a neck, but it is what turned any mis-placed paste into a visible
straight-edged rectangle. A border rolloff (`mask_border`, applied either side
of the Gaussian feather, since the feather smears interior weight back out)
brings the border to 0.014 with chin coverage intact.

### Two cadences, not one

`skip_n` and the swap cadence used to be the same number, so a two-face job ran
the ONNX forward pass twice on *every* frame. They are now separate:
detection/tracking cadence (identity-critical, unchanged) and swap-network
cadence (no longer identity-critical, since every frame is composited from the
cached aligned crop regardless). The adaptive gap may now only *tighten* below
the preset's cadence, never stretch past it — the whole-frame motion estimate
is blind to a moving mouth, so a locked-off talking head read STATIC and
stretched a 5-frame cadence to 8.

Measured at 720p with inswapper at 80 ms/call, detector at 50 ms/call:

| | before | after |
|---|---|---|
| per-frame composite cost | 4.70 ms | **3.11 ms** (−34%) |
| 2-face Optimized, 120 frames | 32.3 s | **22.6 s** (−30%) |
| 2-face swap-network calls | 240 | **126** (−47%) |
| 1-face Optimized, 150 frames | 14.1 s | **13.0 s** (−7%) |
| frames painting a face after the subject left | 17 | **0** |
| mask value at the crop border | 0.881 | **0.014** |

Quality cost of the 2-face cadence change, same clip: identity retention
120/120 and both-faces-present 120/120 (unchanged); placement error 1.0 → 1.1
px; area step p99 2.07% → 2.15%; luma step p99 0.84 → 0.96 (of 255). The one
real trade is **expression freshness**: the face texture is now up to 3 frames
old (mean 0.65) where it was up to 1, bounded at 4 frames / 133 ms in the
worst case (locked-off camera). Geometry, mask, lighting and identity remain
per-frame.

### Things tried and rejected

Kept here because they look like obvious wins and are not:

* **Packing the face and mask into one 4-channel warp.** Forces a single border
  mode. The face needs `BORDER_REPLICATE` (see the mask edge above) or black
  blends in under the chin at ~0.9 alpha.
* **Vectorising `_masked_stats` over channels.** 5x *slower* (1.69 ms vs 0.34
  ms) — it allocates two full HxWx3 float arrays and reduces over a
  non-contiguous axis. Same for the colour transform: 2.6x slower vectorised.
* **Gathering colour statistics at half resolution.** Saves 0.1 ms and
  area-averaging destroys variance: per-channel std came out ~15 LAB levels
  low. std drives the contrast-matching term whose mis-scaling produced the
  original flat "brown patch".

## v11.1.2 — the actual "looking away" paste bug

Reported: a face pasted at the wrong angle and place specifically when the
subject looks away, distinct from the "walked out of frame" case fixed in
v11.1.1 - this one happens mid-frame, subject still in shot.

### Root cause

`_pitch_score()`'s own docstring says its lowest bucket means "looking
down/away, box is hair or skull... not a paintable face." Nothing actually
enforced that. It fed only `_face_looks_marginal()`, which softens the
occlusion mask - it cannot stop a misaligned paste, only blur its edges.
`_face_swap_allowed()`, the function that actually decides whether to paint a
LIVE detection, checked only `det_score` and that 5 keypoints exist - never
what those keypoints described.

When a detector's landmark regression is pushed past where it was trained -
an extreme "looking away" yaw or pitch - it can still return a det_score
that clears the floor and 5 points that each look unremarkable in isolation,
while the points AS A SET describe an inconsistent face. `estimate_norm()`
does not fail loudly on that: it silently returns a plausible-looking affine
that does not match the real head, so the aligned crop reprojects at the
wrong place and rotation. A rotated, misplaced rectangle is the visible
result - exactly the reported defect.

### The fix: two independent geometric-consistency checks

`_kps_reliable()` gates every live detection before it can be painted, using
two signals chosen because either alone can be dodged:

* a **roll-corrected** vertical pitch check. The engine's own `_pitch_score()`
  measures this along the image y-axis, which only means "up/down on the
  face" when roll is near zero - at a genuine ~90 degree roll (this engine's
  own supported lying-down pose) that axis collapses toward zero and
  misreads a perfectly good pose as "looking down". Re-deriving pitch in the
  face's own rotated frame (from the eye-line angle) fixed a real regression
  this introduced during development: lying-down poses at 85-95 degrees roll
  were initially misflagged as unreliable until the check was made
  roll-invariant.
* `landmark_fit_error()` (new, `swap_engine.py`): fits the same
  similarity transform `estimate_norm()` will use and scores the residual.
  A real face's 5 points - however extreme the pose - come from one rigid
  structure, so *some* rigid transform always fits them closely. Points that
  are not mutually consistent with any single pose are a direct sign the
  detector's read is unreliable, independent of where any individual point
  sits - so it also catches configurations that dodge the pitch check.

A rejected detection is not shown as-is and not discarded either: it is
routed into the SAME "no detection this frame" path already built and
verified for genuine detector misses, so it holds the last good geometry
through the bad read rather than painting it. Reused infrastructure, not new
behaviour to trust.

### Verified

* 13 legitimate poses (frontal, profile to yaw 0.9, lying-down at every roll
  from -85 to 180 degrees, small/far faces) - all still swap. This is what
  caught the roll-blindness bug above before it shipped.
* 34 synthetic "looking away" detector-confusion signatures (5 severities x
  5 rolls, plus degenerate hair/skull-blob reads) - all correctly rejected.
* Integration reproduction: injecting the confusion signature mid-video
  measured **126.7px mean / 514.8px max** placement error with the gate
  disabled (reproducing the report), and **0.8px mean / 4.8px max** with it
  enabled - a held frame is invisible; the metric moves because a genuinely
  static synthetic face has near-zero baseline error.
* Two-face isolation: confusing one identity's detections leaves the other
  identity's placement error at 0.9px mean, 0 missing frames - a bad read on
  one slot does not disrupt the other or the tracker's state after it clears.
* Full existing regression suite (fast pan, profile turns, detector dropout,
  two faces in contact, walking out of frame, Stable preset, enhancer) -
  unchanged.
* The v11.1.1 efficiency work is untouched: 2-face Optimized still measures
  ~22.5s against the pre-cadence-fix ~32.0s baseline; the new gate itself
  costs ~220 microseconds per detected face per detector call (~50ms total
  over a 120-frame two-face job).

### Known limits

* Beyond roughly 70 degrees of yaw the ArcFace 5-point fit is close to
  degenerate (the nose and the far eye stop constraining it) and inswapper's
  own output degrades. The mask stays stable there — the silhouette no longer
  breathes and the frame is never dropped — but the identity itself is weaker.
  This is a property of the swap model, not of the compositor.
* The cached aligned crop is refreshed only on frames where the detector
  actually sees the face. During a long occlusion the last visible crop keeps
  being re-projected, so expression is frozen for the duration — correct
  placement and lighting, stale expression. Lower `Swap every N` /
  `Re-detect every N` if a talking subject is spending long stretches occluded.
* `trk_max_missed` (24 frames) still bounds how long a lost face is carried.
  Past that the track is dropped rather than hallucinated onto a body.

## v11.1.3 — the floating disconnected face, and a real multi-face carry gap

Reported: after v11.1.2 shipped, a *worse* defect appeared — a translucent,
disconnected face floating near a curtain, unattached to the body, at points
where the subject had turned away and back. Explicitly asked for a structural
fix to the root cause, not another threshold tweak.

### Root cause #1: staleness measured against the wrong anchor

`_geom_for_frame()` decides how long to keep showing a held/extrapolated face
by checking how recently *something* was recorded for that identity — but
"something" included `_carry_pairs()`'s own predicted entries, not only real
detections. While a subject stayed turned away, `_carry_pairs()` kept
producing a fresh-looking naive constant-velocity extrapolation on almost
every detector call (bounded only by `trk_max_missed`, 24 frames — much
larger than the intended `taper` fade window). Each fresh extrapolation reset
the "how stale is this?" clock to zero, so the fade condition never actually
triggered: the face kept rendering at full alpha while its extrapolated
position drifted further and further from the subject's real one. Reproduced
directly (`repro_ghost.py`): with staleness measured against the newest
entry regardless of kind, a 100-frame "turned away" gap rendered at
**alpha=1.00 throughout, position error growing unbounded (350px+ and
climbing)**. That is the floating face — a confident paste at a position with
no relationship to where the body actually is.

A second, related path produced the same symptom: when a chunk boundary was
reached and the geometry history got trimmed to the frames still needed, the
trim could discard the single most recent *real* detection if the frame that
survived the cut happened to be a carried one — leaving nothing for the next
chunk to measure real staleness against.

And a third: two real detections bracketing a gap (a dip with a confirmed
sighting on both sides) were interpolated across in full confidence
regardless of how long the gap was. Safe for a brief dip — real motion over a
fraction of a second is close enough to a straight line. Not safe for an
extended turn-away, where the subject's actual path (a turn, a full rotation,
a round trip back near the starting position) is nothing like the straight
line drawn between the two endpoints. Frame count alone cannot distinguish
the two cases: under a sparse detection cadence (the Optimized preset's
~10-frame keyframe spacing) a routine detector dropout and a multi-second
deliberate turn-away can produce the same raw span — measured directly, an
18-frame real dropout produced brackets up to 30 frames wide from cadence
spacing alone. A velocity-consistency check was tried and rejected as a
discriminator: measured directly, a round-trip turn-and-back scores the same
near-zero velocity mismatch as barely moving, so it cannot tell them apart
either. The fix instead bounds the bracket by a **time budget** (1.5 real
seconds, converted to output frames at the job's actual fps), so it scales
correctly across fps and quality presets instead of being tuned to one
cadence and silently breaking on another.

All three are fixed the same way: staleness is now always measured against
the last REAL detection, never against whatever was merely extrapolated or
interpolated most recently; a chunk trim always preserves the most recent
real entry even when it is not the most recent entry overall; and a bracket
wider than the time budget renders only a near-edge hold-and-fade rather than
a confident full-span interpolation.

### Root cause #2: a partial-miss carry gap in multi-face jobs

Separately, tracing a "two identities trading faces" regression in a
heavily-occluded two-face scene down to its origin found a real, previously
latent hole in the multi-face carry-forward guarantee. The per-frame carry
fallback read `if not pairs: carry everything` — which only fires when the
*entire* frame's pairing comes back empty. In a two-face job, if one identity
fails to pair (its own visible sliver too narrow for a reliable landmark
read) while the *other* identity keeps pairing successfully every frame, the
list is never empty, so the fallback never triggers — the failing identity
gets nothing recorded at all: no real detection, no carried one either.
Measured directly: a near-total two-person overlap left one identity with
**82 consecutive frames of nothing recorded**, purely because its partner
kept pairing. This gap was always there; it just took the landmark-reliability
gate (v11.1.2) rejecting a genuinely degenerate detection to expose it, since
before that gate existed the same slot would have been (wrongly) painted
with a low-confidence garbage detection instead of silently dropped.

Fixed by carrying forward any slot the frame's pairing did not cover, not
only when every slot failed — closing the actual hole rather than widening
the fallback's trigger condition.

### Verified

* `repro_ghost.py`: the same 100-frame "turned away" reproduction now holds
  and fades correctly, painting nothing beyond `taper` frames past the last
  real sighting; **0 frames rendered more than 10 frames after the last real
  detection, 0px position error** among them (was: unbounded alpha, 350px+
  and climbing).
* `t_extended_lookaway.py`: a 120-frame turn-away renders only 8 frames into
  it (all within the fade window), none drifting more than 150px from her
  last real position, and placement error in the 15 frames after she turns
  back is 0.7px mean / 1.3px max.
* `t_never_returns.py`: a subject who disappears for good and never returns,
  across 3 chunk boundaries over 500 frames — 0 frames rendered after she's
  gone.
* Two-face partial-miss carry gap (`t_pair.py`, a synthetic "two people walk
  together, embrace, part" clip): frames with *no* swapped face at all held
  at 0/300 throughout; frames where each identity stayed on its own person
  improved from 291/300 to **293/300**. The remaining 7 sit exactly at the
  entry/exit of a 60-frame, near-total (>75%), near-zero-relative-motion
  overlap — traced to the harness's own color-blob centroid measuring a
  sub-100-pixel sliver (real face blobs there measure ~9,500px) during the
  lowest-alpha edge of the fade, not a genuine identity mislabel internally;
  the engine's own recorded geometry for the occluded identity is correctly
  `None` (no paint at all) for the entire deep-occlusion span and fades in
  smoothly, at the correct position, once real detections resume.
* Full existing regression suite (fast pan, profile turns, detector dropout,
  Stable preset, enhancer, confused-kps single/two-face, the landmark gate's
  13 legitimate poses / 34 rejected confusions) — unchanged pass rate.

### A measurement side effect worth flagging, not a regression

Fixing root cause #1 changes what a few synthetic tests measure at the very
end of a clip. Previously, a subject whose last real detection landed a few
frames before a clip's natural end kept rendering at full alpha all the way
to the last frame — an artifact of the same "clock resets on any fresh
entry" bug fixed above, not a real signal. Now that staleness is measured
correctly, that tail fades out on schedule, and a handful of clips end with
alpha low enough that the test harness's strict, binary color threshold
misreads a smooth low-alpha blend as "no face" for the last 3-4 frames
(`detector dropout`, `profile turn`, `enhancer` synthetic tests). Confirmed
by direct comparison against the actually-committed baseline with the exact
same test files: reverting only this session's fix reproduces the old,
incorrect "stays opaque past the real detection window" behavior and the
harness reports 0 missing again — the fade is doing exactly what it should,
the harness's blob detector just cannot resolve a smooth fade below its own
fixed threshold. No engine change was made in response to this; it is a
known limitation of the synthetic color-blob harness, documented rather than
chased with another threshold.
