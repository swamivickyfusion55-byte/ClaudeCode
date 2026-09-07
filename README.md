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

## v11.1.4 — the paste that goes wrong-but-confident, not gone

Reported: after v11.1.3, most of the flickering/ghosting reports were
resolved, but a real clip still showed a hard-edged, warped patch across the
cheek/jaw that stayed frozen in almost the same screen position for ~15
frames while the head, hair and hand kept moving underneath, clearing once
the motion settled. Framed by the user as "the face moves out of frame and
comes back" — the actual clip never left the frame; a fast head/hair motion
produced the same failure signature the exit/re-entry case does.

### Root cause #1: a stale stabilization anchor survives any gap

Two independent per-frame smoothing layers exist in the single-face path.
`TrackState`'s own EMA is one; a second, separate pass —
`_stabilize_face_geometry()`, blending a live detection toward
`prev_kps_state` / `prev_bbox_state` when it looks "close enough" — decides
what geometry the compositor actually uses. Neither is reset by anything
short of the detector finding literally nothing: `prev_kps_state` /
`prev_bbox_state` sit untouched through any gap (out of frame, a hard
occlusion, a rejected read) and, on return, get blended against whatever the
identity's position was BEFORE the gap — treating a jump across unconstrained
motion as ordinary frame-to-frame jitter, exactly the failure fixed for
`_geom_for_frame` in v11.1.3, one stage earlier in the pipeline. The same
staleness existed across a chunk boundary too: the next chunk unconditionally
re-seeded `prev_bbox_state` from the last known bbox regardless of how old it
already was when the chunk cut.

Fixed by invalidating both whenever a real gap (the detector finding nothing
for at least one frame) just ended, and by only carrying the last bbox into a
new chunk when the subject was still being seen right up to the cut.

### Root cause #2: a self-consistent detection can still be the wrong one

The v11.1.2 landmark-reliability gate (`_kps_reliable`) catches keypoints
that are not mutually consistent with any rigid pose — a detector regression
pushed somewhere impossible. It cannot catch keypoints that fit a pose just
fine but are the WRONG pose for this identity right now: hair sweeping
across part of the face mid-turn can pull the landmark regression onto a
plausible-looking configuration that simply is not where the eyes and mouth
actually are. That reading passes every existing gate — self-consistent,
correct identity (embedding similarity survives partial occlusion), roughly
the right bounding box — and nothing compared it against where this
identity's own recent, established motion said it should be. When hair
drapes over the same region for several consecutive frames, the detector can
report a similarly-wrong reading each time, painting a confidently-placed
but wrong patch that looks frozen relative to the real, continuing motion —
matching the clip exactly.

Added a check to the single-face live-detection path: compare a new
detection's keypoints against the track's own motion-compensated expected
position (established velocity, not a static last position, so genuine fast
motion is not mistaken for this) before it reaches `tracker.bind()`. A
reading that deviates past a tolerance — which widens with elapsed frames,
since a longer gap between detector calls under a sparse cadence naturally
means more legitimate displacement — is treated as unreliable and routed
into the existing hold/carry-and-fade path instead of accepted as ground
truth.

### A veto that cannot lock up forever

The first version of this check rejected outright, with no bound. Measured
directly: injecting the same failure into an ordinary fast horizontal sweep
(no hair, no gate, just normal motion under a sparse cadence) took an
end-to-end regression test from 4/420 frames without a swapped face to
**293/420**. The mechanism: rejecting a detection skipped updating the
track, so its position and velocity stayed frozen at whatever they were
before the first rejection; every later frame — including perfectly good
real motion — was then compared against that same frozen expectation, the
gap could only widen, and the swap silently stopped for the rest of the job.
The veto, not the hair, was what permanently lost the face.

Fixed by treating a veto exactly like a genuine detector miss — advancing
the track's own decaying-confidence extrapolation and its `missed` counter
on every veto — and by bounding how many consecutive detector calls the veto
may override before conceding, reusing `trk_flip_frames` (5), the same
constant the multi-face tracker already uses for "how long may a competing
signal override the established one before conceding." Not a new tuned
number; the same hysteresis idiom applied where the single-face path was
missing it.

### Verified

* `t_e2e.py`, `t_final.py`, `t_modes.py`, `t_exit.py`, `t_confused_kps.py`,
  `t_confused_2face.py`, `t_pair.py`, `verify_gate.py`, `repro_ghost.py`,
  `verify_geom_cases.py`, `t_extended_lookaway.py`, `t_never_returns.py` —
  identical results to the v11.1.3 baseline; the fast-sweep lockup above is
  fixed (back to 4/420) and does not recur.
* A synthetic reproduction of the reported failure — a detector that keeps
  finding the right bounding box but reports keypoints translated ~70px off
  (self-consistent, so every prior gate accepts it) for 3 consecutive
  detector calls, matching the ~15-raw-frame duration and Optimized preset's
  cadence in the reported clip: **14.1px mean / 143.5px max** placement
  error without the fix, **1.0px mean / 4.8px max** with it.
* The same reproduction stretched to an intentionally extreme 14 consecutive
  detector calls (far beyond anything in the reported clip) to check the
  bounded-veto behaves safely rather than perfectly: 72.6px mean / 102.1px
  max — worse than the realistic case once the veto concedes, but bounded,
  and critically does not reproduce the 293/420 lockup. A real, sustained
  mismatch this long is expected to look imperfect for the frames beyond the
  veto's budget, not silently drop the face for the remainder of the video.
* A synthetic re-entry test (face undetectable for 40 frames, reappearing
  ~40px from where it left, well inside the old blending gate's radius):
  1.0px mean / 2.1px max placement error across the 15 frames after return.

### Honest caveat

None of this could be run against the real face detector or the reported
clip itself — this environment has no model weights. Every number above
comes from the synthetic harness used throughout this project: a detector
stand-in that reports exactly the keypoints a test script injects. The
mechanism (a self-consistent-but-wrong reading surviving every existing
gate, then getting compared against nothing) is grounded directly in the
code paths involved and matches the clip's visual signature (frozen,
hard-edged, resolves once motion settles), but confirming it end-to-end
needs a test against the actual clip on the next deploy.

## v11.1.5 — the v11.1.4 veto's own expectation was wrong

Reported: after v11.1.4 shipped, the same class of defect kept appearing —
a hard-edged mismatch during fast head/hair motion, and separately, in a
second clip, what looked like a whole extra face floating near a pillow
while the real one moved normally above it, with no second person ever in
the source video.

### Root cause: the veto compared against a deliberately-too-cautious guess

v11.1.4's motion-consistency check computed "where should this identity's
keypoints be right now" as `tr.kps + tr.vel_kps * dt`, using `tr.kps` -
the same field `TrackState.predict()` advances. `predict()`'s damping
(`0.85 ** missed`, capped) is deliberate for what IT is for: a cautious,
under-committing guess to paste while a face is genuinely lost, so a stale
guess doesn't confidently drift forever. Reusing that same damped value as
the veto's "expected" position was the wrong tool for a different job: it
under-advances ON PURPOSE, so a perfectly real, constant-velocity motion -
someone simply moving faster, or leaning toward the camera - falls further
behind the damped estimate every single veto cycle. The measured deviation
then grows without bound even though nothing is actually wrong, and the
veto never gets a chance to agree with reality again. Two visible failure
modes came from the exact same bug:

* A synthetic ordinary fast horizontal sweep (no confusion injected at all,
  just normal motion under a sparse cadence) went from 4/420 frames without
  a swapped face to **293/420** - the veto, not any real defect, was
  rejecting good detections indefinitely once it started disagreeing.
* The "extra face" in the second clip: once the veto starts rejecting a
  real, moving face because the damped estimate has fallen behind, the live
  detection never gets pasted (correctly showing her real, unswapped face)
  while the compositor keeps painting the stale, held geometry from before
  the mismatch started - visually, her real face plus a second, wrongly
  positioned paste of the same identity. Not a duplicate-rendering bug and
  not two people; one identity, held at the wrong place, next to her own
  unmodified face showing through where the live detection was rejected.

A second, compounding bug: the same check compared a detection right after
a GENUINE gap (out of frame, occluded, turned away) against the identity's
pre-gap position - which carries no information about where the subject
actually is after a real absence - and rejected legitimate returns outright.
Measured directly: this alone took the v11.1.4 "reappear after leaving
frame" fix's placement error from 1.0px back to a full miss.

### The fix

* The veto's expectation is now projected from `last_hit_kps` (the actual
  last real detection) over the FULL elapsed time (`missed + this
  interval`), undamped - an accurate constant-velocity projection, not a
  cautious extrapolation borrowed from a mechanism built to under-commit.
* Its tolerance now scales with how much the identity's own tracked
  velocity says it is ACTUALLY moving, not with elapsed frames directly -
  scaling by elapsed frames alone let the budget balloon past 200px on a
  150px-wide face after just a 10-frame cadence gap, loose enough to accept
  almost anything exactly when a sustained confusion or occlusion had
  already widened the cadence.
* The check is skipped outright on the first detection after a genuine gap,
  and its own streak counter resets there too - consistency is only
  meaningful between two hits that were never separated by a real absence.

### Verified

* The ordinary fast-sweep regression above: fixed for continuous real
  motion. An intentionally extreme, discontinuous 10x step-change in
  velocity (0.6 to 6 px/frame with no ramp - not representative of real
  head motion, which accelerates smoothly) still costs a bounded, self-
  correcting ~51/420 frames while the estimate catches up; it does not lock
  up for the rest of the job the way the damped version did.
* `t_reentry.py` (the v11.1.4 leaving-frame/return fix): restored to
  1.0px mean / 2.1px max, matching its original result exactly.
* The realistic hair-confusion reproduction (3 consecutive detector calls,
  matching the ~15-raw-frame duration seen in the reported clip): 1.0px
  mean / 4.8px max, unchanged from v11.1.4's result for this case.
* Full existing regression suite (`t_final`, `t_modes`, `t_exit`,
  `t_confused_kps`, `t_confused_2face`, `t_pair`, `verify_gate`,
  `repro_ghost`, `verify_geom_cases`, `t_extended_lookaway`,
  `t_never_returns`) - unchanged from the v11.1.4 baseline.

### Honest caveat

The "extra face" explanation is inferred from the code paths and the
measured fast-sweep mechanism, not from re-running the actual reported clip
- this environment has no model weights to do that. It is the most direct
explanation that fits every observation (one identity, her real face
unmodified where the live detection was rejected, the stale paste drifting
closer to correct over several frames as the tracker's estimate caught up)
without requiring a second, unrelated bug, but confirming it needs a test
against the actual clip on the next deploy.

## v11.1.6 — the actual structural fix: one geometry, not two

Asked directly for a structural analysis after the issue persisted through
three consecutive round-by-round fixes (v11.1.4, v11.1.5, and the veto
tuning within it). Each round fixed the specific failure just reported and
each round was followed by a new, related one - a pattern that means the
individual fixes were treating symptoms of one design flaw, not the flaw
itself.

### The actual root cause

The single-face path had **two independent, separately-maintained "smoothed
face position" systems** live at once, fed similar but not identical inputs,
updated in a different order, with no synchronization between them:

1. **`TrackState`** (`swap_engine.py`) - `tr.bbox` / `tr.kps`, EMA and
   One-Euro-filter smoothed, updated by `tracker.bind()` from the RAW
   detection.
2. **`_stabilize_face_geometry`** (`core_pipeline.py`) - a second, separate
   blend toward `prev_kps_state` / `prev_bbox_state`, running AFTER
   `tracker.bind()` and mutating the detection's geometry IN PLACE - this
   second value, not TrackState's, was what actually got composited.

Every defect chased across v11.1.3-v11.1.5 traces back to these two
systems disagreeing:

* The floating/frozen-face reports (v11.1.3, v11.1.4) were `prev_kps_state`
  / `prev_bbox_state` surviving a gap that `TrackState` itself handles
  correctly, then blending a fresh, correct detection toward stale data.
* The v11.1.4 motion-consistency veto compared a live detection against
  `TrackState`'s history (`tr.kps`) - a DIFFERENT position than whatever
  `_stabilize_face_geometry` had actually painted last frame. A check that
  passes is meaningless if it is not checking the thing that got rendered.
* Fixing the veto's own internals (v11.1.5, twice) kept working around
  symptoms of that same disconnect - a reappearance the veto now handled
  correctly could still be repainted through a stale `_stabilize_face_geometry`
  blend one line later, and vice versa.

Two systems tracking the same fact, on different schedules, are not more
stable than one - they are a standing opportunity for exactly this kind of
report to keep recurring in a new shape each time only one of the two gets
patched.

### The fix

`_stabilize_face_geometry` is removed from the single-face path entirely,
along with its state (`prev_kps_state`, `prev_bbox_state`, and the
chunk-boundary reseeding that existed only to keep it working across a
chunk cut). An accepted real detection is now painted with its own raw
geometry - **exactly how the multi-face path has always worked**, which
never had a second smoothing layer and has been comparatively stable
throughout this entire engagement. `_kps_reliable()` (the roll-corrected
pitch + landmark-fit-error check from v11.1.2) is the sole gate on whether
a detection is geometrically sane; it tests a detection against itself, not
against history, so it cannot go stale.

The one thing that check cannot catch - keypoints that are individually
self-consistent but describe the wrong pose for THIS identity right now
(hair pulling the landmark read sideways) - is still worth catching, so the
v11.1.5 motion-consistency veto is kept, but now reads and writes only
`TrackState` fields: the same, single state `_pairs_for_frame`'s selection
step and `tracker.bind()` both already use. There is now exactly one
position for this check, the bind() call right after it, and the compositor
to ever agree or disagree about - not two.

### Verified

Full regression suite, including every test written across v11.1.3-v11.1.5:
* `t_reentry` (leaving frame and returning): 1.0px mean / 2.1px max -
  identical to its best-ever result, now achieved with no special-cased
  "reset on gap" logic at all, because there is no second, separately-aged
  piece of state left to need resetting.
* Realistic hair-confusion reproduction (3 detector calls, matching the
  reported clip's duration): 1.0px mean / 4.8px max - unchanged from
  v11.1.5.
* The ordinary fast-sweep regression the veto itself risks: back to the
  original 4/420 baseline exactly (previously 51/420 even after the v11.1.5
  fix, since that fix's own veto - correct in isolation - was still only
  ever checked against `TrackState`, one of the two disagreeing systems;
  with the other one gone, the same veto no longer has anything to
  disagree with).
* `t_final`, `t_modes`, `t_exit`, `t_confused_kps`, `t_confused_2face`,
  `t_pair`, `verify_gate`, `repro_ghost`, `verify_geom_cases`,
  `t_extended_lookaway`, `t_never_returns` - unchanged.
* The intentionally extreme, discontinuous 10x-instant-speed-jump stress
  case (not representative of real head motion) still costs the same
  bounded, self-correcting ~51/420 frames documented in v11.1.5 while the
  veto's estimate catches up - accepted then and unchanged now, since it
  is a property of the veto's own motion model, not of the two-systems bug
  this round fixes.

### Honest caveat

Unchanged from v11.1.5: none of this has been run against the real face
detector or the reported clips - this environment has no model weights.
The structural diagnosis (two independently-updated position trackers,
one driving what renders, one driving what the veto judges) is grounded
directly in the code paths every prior round's fix touched, and explains
why each fix in isolation kept being followed by a new, related failure
rather than silence. Confirming it needs a test against real footage,
specifically footage combining a leaving-frame/return moment WITH a
fast hair/hand occlusion in the same clip, since that combination is what
exercised both halves of the old disagreement at once.

## v11.1.7 — the swap could be handed to ANY face-shaped thing in frame

The v11.1.6 structural fix (one geometry system instead of two) was real
and stayed in - the same "extra face" signature still reported afterward
had a second, independent cause that v11.1.6 never touched, in a part of
the pipeline none of the v11.1.3-v11.1.6 rounds had looked at: single-face
candidate SELECTION, not geometry smoothing or motion consistency.

### Root cause

`_pairs_for_frame`'s single-face path scores every detected candidate
against the locked identity and the last known position. When the best
score fails to clear a minimum bar, `best` stays `None` - and the code
unconditionally fell through to an `_open_score` fallback that picks
whichever detected candidate looks most frontal, confident, and large,
with **no identity or position check at all**:

```python
if best is None:
    best = max(faces, key=_open_score)   # front/det/area only
```

That fallback is correct for the ONE case it was written for: true cold
start, before any reference embedding has ever been established, when
there is nothing yet to check identity against. It is wrong every other
time it fires - and it fires unconditionally whenever the scored path's
confidence dips below the bar, identity already locked or not. A moment
where the tracked face is at a hard angle or motion-blurred (a real,
ordinary thing that happens on any video) drops its own score below the
bar; if the detector ALSO reports anything else remotely face-shaped
elsewhere in frame that same moment - a false-positive on a hand, a
shadow, a pillow crease - this fallback hands it the swap with zero
regard for whether it is actually the tracked person. Verified directly
against the exact function: a synthetic frame with only a false-positive
detection (valid-looking keypoints, an unrelated random embedding, no
relation to the locked identity) scored 0.0 against the locked identity,
failed the bar, and the old code selected it anyway via `_open_score`.

This produced exactly the reported symptom and nothing else: her real,
unmodified face keeps showing normally (nothing was ever swapped onto
it this frame), while the swap gets confidently painted onto the
unrelated region the detector also reported - one identity, in the wrong
place, next to itself unmodified. Not a duplicate render, not a geometry
or motion-consistency bug (which is why v11.1.3 through v11.1.6, all
aimed at geometry and motion, never touched it).

### The fix

`_open_score` now fires only on true cold start - `ref0 is None`, meaning
no identity has ever been locked yet. Once an identity exists and nothing
this frame clears minimum confidence against it, that is treated as
`return []`: the existing "no reliable detection this frame" case, which
already holds the last good geometry and fades rather than painting
anything - the same standard applied everywhere else in this engine, now
applied here too. The narrower no-`prev_bbox` branch (identity score
alone, before the first successful bind) got the same minimum-confidence
floor for the same reason.

### Verified

* Direct reproduction at the function level: a spurious-only detection
  (no real face reported that call) scores 0.0 against the locked
  identity; the old code selected it via `_open_score`, the new code
  returns no pair.
* Full existing regression suite (`t_final`, `t_e2e`, `t_modes`, `t_exit`,
  `t_confused_kps`, `t_confused_2face`, `t_pair`, `t_reentry`,
  `t_hair_confusion`, `verify_gate`, `repro_ghost`, `verify_geom_cases`,
  `t_extended_lookaway`, `t_never_returns`) - unchanged from the v11.1.6
  baseline; this fix only changes behavior in the specific case it targets
  (a scored candidate failing the bar with an identity already locked),
  which none of the existing tests happen to construct.

### Honest caveat

Same as every round: not verified against the real detector or the
reported clips - no model weights in this environment. This is the first
fix in this whole engagement that targets candidate SELECTION rather than
geometry or motion, found by reading the one code path in the single-face
pipeline that had not yet been examined after geometry smoothing (v11.1.6)
and motion consistency (v11.1.4/v11.1.5) were both ruled out by the clip
still showing the same symptom afterward. If it recurs a third time after
this, the next place to look is upstream of the compositor entirely: what
the actual face detector reports for the specific frames involved, which
requires the real model and cannot be narrowed further from this
environment.

## v11.1.8 — the detector cadence itself was not fps-aware

The user's own hypothesis, checked directly rather than assumed: every
reported clip has been ~24fps, and the reports keep clustering around
rapid movement. Worth checking on its own merits regardless of whether it
explained the specific clips already fixed in v11.1.6/v11.1.7.

### What was actually wrong

`SKIP_N` (`{"Fast": 6, "Balanced": 4, "Optimized": 5, ...}`) and the
"Optimized" tier's cadence are a RAW FRAME COUNT - "detect every 5
frames" - with no reference to how much real time 5 frames spans. Verified
directly: an identical 60-frame synthetic clip produced exactly 10
detector calls whether encoded at 24fps or 30fps - the same number of
calls, spread over 2.5 real seconds at 24fps versus 2.0 real seconds at
30fps. **25% more real time between detector calls at 24fps for the
identical nominal quality setting**, and therefore 25% more opportunity
for genuine motion to invalidate the linear velocity/interpolation math
used everywhere between two real detections - none of which is itself
fps-aware. `taper` and `max_bracket_frames` were already converted to a
real-time budget for exactly this class of problem in v11.1.3; this is
the same fix one level earlier, at how often the detector is asked to
look at all, not just how long a gap between two of its answers may be
trusted.

It also means every regression test in this project, all the way back to
v11.1.0, ran at the synthetic harness's default of 30fps and so could not
possibly have caught this - a real, verifiable blind spot in how
everything up to this point was validated, independent of whether it
explains any specific reported clip.

### The fix

The table-driven cadence (`SKIP_N["Optimized"]` and the "Auto" `swap_n`
path for other quality tiers) is now rescaled by the ratio of the job's
actual fps to 30 - the fps this table was tuned against and the synthetic
harness's own default. An explicit numeric override (a user literally
typing a frame count into Swap-every-N) is left untouched, since that
number means exactly what it says regardless of fps. Verified directly:
the same 60-frame clip now produces 12 detector calls at 24fps and 10 at
30fps - a constant ~167ms between calls at both, instead of 208ms vs
167ms before.

### Verified

* Direct mechanism check: detector-call real-time spacing is now constant
  across 24/30/60fps on an identical clip (was 25%/50% looser at 24fps
  before, relative to 30fps/60fps respectively).
* Full existing regression suite, run at its 30fps default (the reference
  fps this fix rescales against, so `_fps_scaled(n) == n` there) -
  byte-for-byte unchanged from the v11.1.7 baseline, confirming this
  costs nothing at the fps every prior test in this project has run at.

### Honest caveat

This one is weaker than prior rounds' verification, and worth saying
plainly rather than overstating: I could not build a clean synthetic
before/after demonstration of the fps fix's benefit under genuine rapid
movement specifically - the test harness's video writer and the
pipeline's frame-count accounting disagreed with each other once pushed
to extreme synthetic velocities, in a way traced to the harness's own
plumbing rather than the server code this project ships, and not worth
half-fixing under this round's time budget rather than reporting
honestly. What IS directly verified is the underlying mechanism this fix
targets (detector cadence is now fps-invariant in real time, provably, at
the exact numbers involved) and that it changes nothing at the reference
fps. Whether it measurably improves the reported clips specifically can
only be confirmed by testing this build against footage at 24fps with
rapid movement, on the real detector, which this environment cannot do.

## v11.1.9–v11.2.1 — ReentrySafe, HairGate, CinemaQA, SolidFace (external round)

Between this project's own v11.1.8 and this entry, the user tested a
build modified by a different assistant against their real footage and
reported it fixed the standing defect list (looking away, under-20%
visible, leaving frame, no detection, fast movement). That build is now
the baseline this project continues from. It was not developed under
this project's own README convention, so — reconstructed directly from
its code and comments, since no changelog entry existed for it — here is
what it actually changed, for the record:

* **ReentrySafe** (`TrackState._paste_frozen`, `_frame_wh`): once a
  track's bbox is mostly outside the real frame bounds (containment
  ratio < 0.70, checked in `predict()` against the actual last-known
  frame size), the track freezes — stops extrapolating and stops
  offering itself for paste — instead of coasting on constant-velocity
  prediction into empty space. This replaces a plain frame-count budget
  with a geometric one for the specific "face genuinely left the shot"
  case.
* **HairGate** (`confirm_hits`, the hair/skull rejection added to
  `_kps_reliable()`, the post-gap `det_score >= 0.32` filter): on
  reacquiring a track after a freeze or a real detector gap, at least two
  consecutive reliable frames are now required before paste resumes, and
  `_kps_reliable()` fails closed on missing keypoints (was fail-open) and
  added a geometric hair/skull-blob rejection. Together these stop the
  very first, often-unreliable detection after a gap (frequently a
  motion-blurred hair/skull read) from being painted.
* **CinemaQA**: reacquiring a track now wipes that slot's entire geometry
  timeline (not just predicted stubs) so `_geom_for_frame()` cannot
  linear-interpolate a "ghost glide" between a pre-exit and a post-return
  real position across a short gap; `max_bracket_frames` was shortened
  from 1.5s to 0.55s for the same reason; colour-transfer constants
  (`cm_strength_l`, `cm_ema`, `cm_delta_clamp`) were tightened for a
  steadier result.
* **SolidFace** (`smooth_alpha()`, `_render_alpha()`, quality-tier alpha,
  `occl_min_keep`): composite opacity now snaps up to full strength as
  soon as the gates say yes, instead of a slow symmetric EMA that let the
  replacement sit half-transparent — with the real face bleeding through
  — for several frames every time; the soft pose-based alpha fade (0.94)
  and the per-quality-tier alpha discount (Fast was 0.92) were both
  removed in favour of full-strength paste whenever painting is allowed;
  `occl_min_keep` (the mask-opacity floor kept under partial occlusion)
  was raised so the center of the face stays more solid under marginal
  occlusion instead of thinning out.

None of this conflicts with v11.1.0–v11.1.8's structural fixes (single
geometry source of truth, the motion-consistency veto, the fps-aware
detector cadence, the cold-start-only `_open_score` fallback) — those
were left intact and are unaffected. It does fully remove v11.2.0's
content-based occlusion gate (`_visible_face_fraction` /
`_face_anchor_samples` / `_bootstrap_visible_ref`) in favour of the
purely geometric hair/skull check above; the two approaches were not
combined.

## v11.2.2 — the SolidFace round over-corrected: paste now absent through most of the clip

Direct user report against real footage, immediately after the v11.2.1
sync above: the new face renders weak-to-absent through most of the
video, with only occasional partial blending. Root-caused to FOUR of the
v11.1.9–v11.2.1 changes, not to anything from this project's own
v11.1.0–v11.1.8 work (unchanged and re-verified — the existing regression
suite plus a direct before/after against the pre-sync v11.2.0 baseline
caught three of the four directly, contrary to this project's usual
"cannot verify without real footage" caveat: **on a build with all
`_kps_reliable`/HairGate/ReentrySafe machinery in place, most of what
went wrong here is a geometry/tracking bug that a synthetic detector
reproduces exactly, not a real-detector-only content problem**):

1. **`_predicted_miss_budget()` was capped at a flat 10 frames**,
   regardless of detector cadence, "to prefer skip over arm/body paste
   after exit." `missed` advances by frames actually elapsed since the
   last detector call, not by call count — on Fast/Optimized cadence
   (`SKIP_N` 5-6 frames between calls), two consecutive ordinary
   (non-exit) detector misses in a row already exhausts a 10-frame
   budget, dropping the paste to the original face for the rest of the
   clip until the next clean hit.
2. **The new hair/skull rejection in `_kps_reliable()` rejected any face
   whose eye line sat past 45% down the detection box.** Reproduced
   directly with the synthetic harness: a profile-turn clip whose
   detector box narrows as it turns (width shrinks, height does not) hit
   `landmark_fit_error > 0.15` — HairGate's OWN tightened threshold — on
   13 of 28 detector calls, purely because a similarity transform cannot
   rescale width and height independently to match the canonical
   template's fixed aspect ratio as the box distorts; nothing about the
   face itself became less reliable. Direct before/after against the
   pre-sync v11.2.0 baseline: 0 of 28 calls rejected at the old 0.20
   threshold, 13 of 28 at HairGate's 0.15, and the resulting clip went
   from showing the swap on 237/240 frames to 0/240 — every rejection
   fed straight into bug 3 below.
3. **A reacquire event, once triggered, could never be confirmed away on
   real footage with a sparse detector cadence.** `TrackState.update()`'s
   "did the identity jump elsewhere" check compared each new detection's
   raw position against the last real hit, with a threshold that does
   not scale with elapsed time. At Fast/Optimized cadence (5-6 frames
   between detector calls) any genuine, ordinary motion covers a
   meaningful fraction of the box between two real detections purely
   because of the gap, not because the identity changed — indistinguishable
   from a real re-acquisition under the old check. Caught directly by this
   project's own `t_final.py`: constant 10px/frame motion at a 10-frame
   detector cadence was measured as **zero** learned velocity, because
   every single update re-triggered "reacquire" and reset it.
4. **A confirmed reacquire wiped a slot's ENTIRE geometry timeline, not
   just the entries near the gap.** This pipeline detects a whole chunk
   ahead of rendering (see `_record_geometry`'s own docstring), so
   `_geom_hist[slot] = []` at the moment of reacquire does not just
   prevent bridging the actual gap — it also destroys every already-
   recorded, already-valid entry for every EARLIER frame in the same
   chunk that has not been rendered yet. Measured directly: a single
   reacquire recovering from an 18-frame detector dropout, on an
   otherwise perfectly ordinary clip, wiped 90 already-good frames of
   history and left the first 115 of 240 output frames showing the
   original face — for a track that only ever had one 18-frame real gap
   in it.

Bugs 2 and 3 compound in exactly the pattern the user reported: bug 2
(and, on real footage, ordinary cadence noise) makes reacquire trigger
far more than intended, bug 3 means a triggered reacquire on sparse
cadence essentially never clears, and bug 4 turns each such event into a
much larger visible hole than the 1-2 frames it was meant to cost.

### The fix

* `_predicted_miss_budget()` no longer applies its own 10-frame cap; it
  trusts `trk_max_missed` (default 24) again. This does not reopen the
  arm/body-paste-after-exit bug the cap was reacting to: that case is now
  caught independently and unconditionally by ReentrySafe's
  `_paste_frozen` (a geometric containment check against the real frame
  bounds, evaluated in `_face_swap_allowed()` before the miss budget is
  ever consulted) — the frame-count budget's remaining job is only
  "how long to ride out an ordinary detector miss," which does not need
  to be nearly this short.
* `_kps_reliable()`'s `landmark_fit_error` threshold reverts 0.15 → 0.20
  (this project's own previously-measured number). The hair/skull
  eye-line threshold is also loosened from 0.45 to 0.65 (reject only when
  the eye line sits in the bottom third of the box). The pre-existing
  `vert < 0.12` check remains the primary signal for genuine hair/skull
  confusion — both of these are defensive second/third checks, not the
  only signal, so loosening them does not remove hair/skull protection,
  it removes the false positives they were producing on ordinary poses
  and on any box whose aspect ratio isn't the canonical template's.
* `TrackState.update()`'s reacquire check now compares the new detection
  against this identity's own velocity-projected expected position
  (`last_hit_bbox` centre plus `vel_bbox * elapsed`), the same idiom
  already used and validated by the motion-consistency veto in
  `core_pipeline.py`, instead of the raw last-hit position. Continuous
  motion over a sparse cadence no longer looks like a jump; an actual
  position discontinuity still does.
* The reacquire geometry-timeline wipe (`_scrub_reacquire_timelines()`
  and the matching branch in `_record_geometry()`) now only drops entries
  within a bounded, fps-derived window of the current frame
  (`round(out_fps * 0.55) + 5`, the same order of magnitude as
  `max_bracket_frames` itself) instead of the whole list. This still
  prevents the "ghost glide" a full pre-exit/post-return bracket could
  produce across the actual gap, without erasing history far enough back
  to be irrelevant to any bracket the interpolator could actually form.
  (A first attempt used a flat 120-frame window on the reasoning that it
  was "generous but much smaller than a whole clip" — measured directly
  and found to degenerate to the same full wipe whenever the reacquire
  happens within the first ~120 frames of a clip, which is exactly the
  case that was reported.)

### Verified

* `t_final.py`: the velocity-learning regression (bug 3) now passes —
  10px/frame motion at every tested detector cadence (1, 5, 10 frames)
  is learned as ~9.7px/frame, not zero.
* Direct before/after against the pre-sync v11.2.0 baseline, same
  synthetic clips: profile-turn test back to 3/240 original-face frames
  (baseline: 3/240; before this fix: 240/240, 0 swap calls in the whole
  clip). Detector-dropout (18-frame) test improved from 119/240 to
  32/240 original-face frames (baseline: 4/240) — the remaining gap
  versus baseline is the bounded, intentional cost of the geometry-wipe
  window itself (it still clears entries near the gap on purpose), not
  an uncontrolled regression.
* Full existing regression suite (`t_e2e`, `t_exit`, `t_confused_kps`,
  `t_confused_2face`, `t_pair`, `t_reentry`, `t_hair_confusion`,
  `verify_gate`, `repro_ghost`, `verify_geom_cases`, `t_extended_lookaway`,
  `t_never_returns`) unchanged in outcome.

### Honest caveat

Bug 1 (the miss-budget cap) and the two threshold loosenings for bug 2
are still verified by re-deriving the numbers from the code and the
user's report rather than a synthetic reproduction, the same limitation
logged for the v11.1.8 fps fix and the v11.2.0 occlusion gate. Bugs 3 and
4, by contrast, were reproduced and fixed against this project's own
synthetic harness directly — this sandbox still has no access to the
user's real footage or a real detector, but "cannot verify without real
footage" is not a blanket truth for this round the way it was for prior
ones. If the new face is still weak in specific spots after this build,
the next most likely remaining lever is the occlusion-guard mask opacity
path (`build_mask()` / `skin_confidence()` / `_face_looks_marginal()` in
`swap_engine.py` and `core_pipeline.py`), which was left untouched this
round since it moved opacity *up*, not down, in the v11.2.1 sync and so
is a less likely source of the reported weakness.

## v11.2.3 — rapid movement could deadlock a track in the frozen state permanently

Direct user report, immediately after v11.2.2 shipped: the new face still
reverts to the original at times during rapid movement and open mouth.
Reproduced directly with this project's own `t_fps_rapid.py` (a face
oscillating sinusoidally up to ~55px/frame, an intentionally extreme
stress test rather than typical footage) — and found something much
worse than "at times": **176 of 200 frames (88%) showed no swap at all**,
with only 2 real swap-network invocations in the whole clip and every
recorded reacquire event's own velocity estimate stuck at exactly zero,
never recovering. The pre-sync v11.2.0 baseline, run through the same
test, already showed a real (pre-existing, not new) weakness here -
101-119/200 missing - but nothing close to this.

### Root cause: two of ReentrySafe/HairGate's own mechanisms zero a
### track's velocity right when sustained rapid motion needs it most

`TrackState.update()`'s reacquire path (v11.1.9 ReentrySafe) explicitly
zeroed both `vel_bbox` and `vel_kps` on every reacquire snap ("we don't
know this identity's motion yet, don't guess"). That is the right call
for a track that was genuinely lost and is starting fresh. It is the
wrong call for the much more common way a reacquire actually fires during
sustained rapid motion: the subject never stopped moving, they just
covered more distance between two sparse-cadence detector calls than a
CONSTANT-velocity projection expected - exactly what `core_pipeline.py`'s
existing motion-consistency veto also has to reason about.

The zeroed velocity then poisoned two downstream checks that both assume
"whatever is already in vel_kps/vel_bbox is a real prior worth trusting":

1. The motion-consistency veto in `core_pipeline.py` projects an expected
   position from `last_hit_kps + vel_kps * elapsed`. With `vel_kps`
   forced to zero, the expected position stayed pinned at the reacquire
   snap point no matter how far the subject had genuinely moved since -
   so the very next real, correct detection during continued fast motion
   read as a huge deviation and got vetoed. A vetoed frame never reaches
   `update()`, so `vel_kps` never gets a chance to become non-zero -
   deadlock.
2. `vel_kps`'s own EMA (`v if self.vel_kps is None else vel_kps*0.6 +
   v*0.4`) already distinguishes "no estimate yet" (`None`) from "a real
   prior" - but the reacquire path set it to `zeros_like(kp)`, not
   `None`, so the very first genuine velocity reading after a reacquire
   was itself damped 60% toward that false zero. `vel_bbox`'s own EMA has
   no such distinction at all (always blends unconditionally), so the
   same zeroing damped it there too, on every reacquire, without
   exception.

### The fix

* The motion-consistency veto now skips itself when the track has no
  reliable velocity basis yet (`vel_kps` is `None` or all-zero) — the
  same "nothing to compare against, trust the candidate" principle the
  veto already applies right after a real detector gap (`_after_real_gap`),
  extended to cover a just-reacquired track for the same reason.
* `vel_kps` is reset to `None` on reacquire instead of an explicit zero
  vector, restoring the same fresh-start (undamped) treatment a genuine
  first-time establishment already gets from the existing EMA.
* The reacquire path's own `missed >= 8` OR-clause (forcing a re-confirm
  purely on elapsed frame count, regardless of position) is raised to
  match `trk_max_missed` (24) - the same bar `_predicted_miss_budget()`
  and this class' own missed-tracking already use for "how long may a
  gap be trusted." The flat, unrelated "8" could fire well before that on
  nothing but a couple of veto-rejected frames, forcing an unnecessary
  reacquire-and-reconfirm cycle at the exact moment a good detection
  arrived to end the gap.
* Tried and **reverted**: keeping the pre-gap `vel_bbox` instead of
  zeroing it, on the reasoning that the subject likely kept moving.
  Measured directly and found worse (176/200 again) - at a genuine
  direction reversal (a sine wave's peak/trough, exactly where a
  reacquire is likeliest), the pre-gap velocity points the WRONG way, so
  projecting forward with it overshoots further than assuming no
  velocity at all. Reverted to zero for `vel_bbox` specifically; noted
  here so a future round does not re-attempt the same fix and re-measure
  the same regression.

### Verified

* Same `t_fps_rapid.py` stress test: 176/200 → 141/200 missing frames,
  swap-network calls 2 → 10, and — the qualitative change that matters
  most — the failure mode changed from a permanent, whole-clip deadlock
  (never recovers after the first reacquire) to intermittent recovery
  (the track successfully re-establishes multiple times through the
  clip, even though sustained sinusoidal motion still occasionally
  re-triggers it). The remaining gap versus the pre-sync baseline's
  101-119/200 is a pre-existing constant-velocity-model limitation this
  round did not introduce and does not attempt to fully close (see honest
  caveat).
* Full existing regression suite (`t_final`, `t_e2e`, `t_exit`,
  `t_confused_kps`, `t_confused_2face`, `t_pair`, `t_reentry`,
  `t_hair_confusion`, `verify_gate`, `repro_ghost`, `verify_geom_cases`,
  `t_extended_lookaway`, `t_modes`, `t_never_returns`) unchanged in
  outcome.

### Honest caveat

**"Open mouth" specifically was investigated and NOT reproduced.** A
direct synthetic check — moving the ArcFace mouth-corner keypoints down
and outward by up to 40px at typical crop scale, simulating a wide mouth
open, and feeding that through the real `landmark_fit_error()` — found
the fit error DECREASES as the mouth opens in that model, staying well
under every threshold this project uses. That does not mean open-mouth
reverts are not real; it means the mechanism, if it is a mechanism this
project's synthetic tools can represent at all, was not found this
round. The more likely explanation, unverified: an open mouth is often
accompanied by head motion (talking, laughing, turning to react to
someone), and what actually gets reported as "open mouth reverts" may be
the rapid-motion deadlock this entry fixes, triggered by the head motion
that happens to co-occur with the expression, not by the mouth shape
itself. If open-mouth reverts persist on their own, with the head
otherwise still, that would be strong evidence against this explanation
and worth a dedicated follow-up with an actual clip showing it in
isolation - this sandbox has no way to manufacture that case blind.

The residual rapid-motion gap versus baseline (141 vs ~110/200 on the
synthetic stress test) is a real, pre-existing limitation of this
pipeline's motion model: `vel_bbox`/`vel_kps` are a single constant-
velocity estimate, smoothed by a fixed-weight EMA, with no
representation of acceleration or direction reversal. Sustained,
continuously-accelerating motion (a mathematical sine wave; a person
swinging their head rhythmically) will keep finding the edges of that
model no matter how the surrounding thresholds are tuned. Closing that
gap fully would mean adding an actual acceleration term or a proper
filter (a per-axis alpha-beta or Kalman filter in place of the flat EMA)
to `TrackState`, not another threshold adjustment - a larger, riskier
change than this round's time budget and lack of real-footage validation
justify attempting blind. Typical real "rapid movement" (a single fast
head turn, a quick gesture) is a brief, transient event rather than a
sustained periodic oscillation, so the practical impact of this residual
gap should be substantially smaller than the worst-case synthetic number
above suggests - but that is reasoning from the mechanism, not a
measurement against the user's own footage, which this sandbox cannot
run.

## v11.2.5 — flicker and reverts are one bug: suppression was instantaneous

Direct user report against real footage, on the externally-produced v11.2.4
"HoldThrough" build: the new face still reverts to the original, and a
second clip also showed flicker. Asked explicitly for a deep fix rather
than another threshold.

### The structural root cause

The pipeline already had two smooth fades — `TrackState.smooth_alpha()`'s
opacity EMA, and `_geom_for_frame()`'s taper. Neither was ever reached by
the decision that matters. Every one of the **seven** paths in the render
loop that concludes "do not paint this frame" bypasses both and emits the
pristine original immediately (`core_pipeline.py`, the Phase 2b loop):

1. `not records` → put the untouched frame
2. `_touches_frame_edge(hit_bbox)` → `continue`
3. `_touches_frame_edge(bbox)` → `continue`
4. `_frame_containment < 0.60` → `continue`
5. taper alpha `<= 0.02` → `continue`
6. `_reuse_one` returns not-ok → `continue`
7. `not any_ok` → `out = frm`

So the fade applied only to "paint, but weaker". "Do not paint" was always
instantaneous and total: opacity went from ~1.0 to exactly 0 between two
adjacent output frames.

That single fact produces **both** reported symptoms, which are the same
bug separated only by duration:

* a gate flipping for one to three frames = a full-strength flash of the
  real face = **flicker**;
* the same flip sustained past `taper` = **revert to original**.

It also explains the shape of this project's history. Ten rounds
(v11.1.4 → v11.2.4) each changed *which* gate fires and *when* — thresholds,
vetoes, bypasses, hold budgets. Not one changed the fact that firing is a
hard cut, so each round moved the trigger and the symptom reappeared in a
new shape.

### The fix

Suppression is now continuous at the output stage. Each slot carries a
`_paint_alpha` that is slewed toward whatever this frame wants, over
`PASTE_FADE_SEC` (0.5s, `config.py`), with `_held_rec` keeping the last
record that actually composited so there is frozen — never extrapolated —
geometry to fade out from.

Three properties make this safe rather than another trade:

* **Downward only.** Coming back up stays instant, preserving v11.2.1
  SolidFace's "when the gates say yes, sit at full strength", and leaving
  every existing "is the face present" measurement untouched.
* **Departure evidence still cuts instantly.** Paths 2–4 above are positive
  evidence the subject has *left* the frame, not that this frame's read is
  unreliable. Fading a face out over half a second onto background someone
  has already walked off is the v11.1.1 defect, so those keep the old
  behaviour — `t_exit` is unchanged at 0 frames painted after departure.
* **It is invisible except where the old code stepped.** The taper already
  drives alpha to ~0.02 before `records` empties, so the slew starts from a
  low value in the ordinary fade-out case and changes nothing. It only
  produces a visible ramp where opacity was *high* the frame before — which
  is exactly the hard step being fixed.

### Two blind spots in how this project has always tested

Both are the same class as the fps blind spot found in v11.1.8 — a
condition every test shared, so no test could see past it.

* **`det_score` is always 0.92.** `harness.FakeFace` hardcodes it. The
  shipped code branches on det_score at **0.18, 0.20, 0.28 and 0.32**, and
  on `landmark_fit_error` at 0.20 / 0.32 — and a perfect 0.92 read with a
  canonical keypoint arrangement clears every one of them unconditionally.
  Not one test in this project's history has ever exercised those
  thresholds. `noisy_det.py` (scratchpad) now models a detector whose
  confidence random-walks through that band, with landmark jitter and hard
  misses.
* **`t_fps_rapid.py` carries the harness resolution mismatch.** It renders
  1280×720 but never sets `H.OW/H.OH`, so the fake encoder reconstructs the
  byte stream at the default 960×540. The **176/200 and 141/200 figures
  quoted in the v11.2.3 entry above came from this uncorrected test and are
  not sound.** Re-run with the resolution matched (`t_rapid_fixed.py`), the
  same clip shows **0/200 missing frames** — but a placement error of
  **mean 283px, max 681px**.

### The rapid-motion defect, now located and measured (NOT yet fixed)

That 283px is the more likely explanation for what gets reported as
"reverts to original" on fast movement: the face is painted on every frame,
but far enough from the head that the real face is visible underneath it.

Instrumented directly on that clip: the pipeline made **42 detector calls
but only 11 became real anchors**, and those 11 land in tight clusters
around the sine's turning points — `[4,8,12, 44,48, 84,88, 124,128,
164,168]`, i.e. only where the subject is momentarily slow. Detections are
rejected *while the subject is moving*. That leaves real anchors ~36 frames
apart, forcing **61%** of rendered frames into `_geom_for_frame`'s frozen
"span exceeded `max_bracket`" branch, which does not interpolate at all.

The mechanism is a logic inversion in the motion-consistency veto: its
budget is `_face_w * 0.35 + 0.80 * _est_speed`, and `_est_speed` comes from
the track's own velocity estimate — so when that estimate is stale or points
the wrong way (which a direction reversal produces twice per oscillation)
the budget collapses to the static floor. **Uncertainty about velocity makes
the veto stricter, when it should make it more permissive.** Each veto then
skips the update that would have corrected the velocity. v11.2.3 fixed the
all-zero case; the stale and wrong-direction cases remain.

Left unfixed deliberately, with the measurement recorded at the veto site:
the veto exists to catch a hair/occlusion read that is self-consistent but
wrong (v11.1.4), and that case *also* passes `_kps_reliable`, so it cannot
be told from genuine fast motion by any single-frame test. Separating them
needs coherence across consecutive rejected reads — real motion keeps
travelling, a hair confusion clusters at one spot — which is a real design
change and wants validation against footage this sandbox does not have.

### Two regressions found in the v11.2.4 HoldThrough build itself

Both verified against the **unmodified** upload, before any change here:

* **The occlusion gate is fully defeated.** A sustained hand occlusion went
  from 90/90 frames correctly suppressed to **0/90** — the swap is painted
  onto the occluder on every frame. HoldThrough rescues a
  geometrically-implausible detection on bbox *overlap alone*, and covering
  the lower face barely moves the box while collapsing its extent.
* **The v11.1.4 hair-confusion defect is reopened.** `t_hair_confusion`
  measures **14.7px mean / 151.8px max** placement error, against 1.7px /
  10.5px on v11.2.3 — a 15× worse worst case. HoldThrough's `_same_head_now`
  IoU bypass lets exactly the read the veto was built to catch straight
  through.

`_same_extent()` is added for the first of these and wired into
`_face_swap_allowed` both as a condition on the overlap rescue and as a
check on the accept path: a candidate must still be about the same **size**
as the identity's last good box. Height is the discriminator, with a very
permissive width band, because a profile turn narrows width while its height
holds (that asymmetry is what made an over-tight fit threshold reject
profiles in v11.2.2) whereas an occluder collapses height. Verified at unit
level: a box cut to 30% of its height is now rejected, where before it was
accepted — it passes `_kps_reliable` because keypoints derived from a
collapsed box are still mutually self-consistent, which is precisely why
self-consistency cannot substitute for visibility.

`t_occlusion` still reports 0/90 afterwards, and that is *not* this gate. It
is the held-geometry path plus the grace window deliberately holding the
last good face through the occlusion — which is what "HoldThrough" means and
what "never show the original face" asks for. **These two goals genuinely
conflict** ("never show the original" vs "never paste onto an occluder") and
which one wins is a product decision, surfaced rather than decided here.

### Also tried and reverted

Tightening the **detector** cadence on motion (`det_mult` 0.75 / 0.50 on
MEDIUM / HIGH, mirroring `_adaptive_swap_gap`'s tighten-only idiom, since
the detector cadence could previously only ever *stretch*). Measured: the
frozen-branch share stayed at 61% and placement error did not improve,
because the anchors are not sparse from being scheduled too rarely — they
are sparse because the detections that do happen are rejected. Reverted
rather than left in, since it costs real detector calls on a CPU-bound
Space and bought nothing measurable. Noted at the site so a later round does
not re-measure the same null result.

### Verified

* `t_exit` 0 frames painted after departure, `verify_gate` 0 pose
  regressions / 0 bad-pose misses, `t_final` no failures, `verify_geom_cases`
  unchanged — the four that most constrain this change.
* `t_e2e` **improved**: 0/420 original-face frames, placement error mean
  1.4px / max 6.6px.
* `t_modes` 0/240 original-face frames on all four scenarios; `t_pair`
  0/300 missing and 300/300 identity retention; `t_pair2`,
  `t_confused_2face`, `t_confused_kps`, `t_reentry`,
  `t_extended_lookaway` all in line with baseline.
* New: `t_flicker.py` (presence transitions, longest gap, area-step
  oscillation — the existing suite measured only binary presence and could
  not see flicker at all), `t_flicker_dropouts.py`, `t_flicker_noisy.py`,
  `noisy_det.py`, `t_rapid_fixed.py`, `diag_geom_branch.py`,
  `diag_anchor_spacing.py`.

### Honest caveat

Unchanged from every prior round: no model weights and no access to the
reported footage, so none of this is confirmed against the actual clips.
What is different this time is that the primary fix does not depend on
having guessed the trigger correctly — it changes what happens *when any
suppression fires*, whichever one it is, so it applies to triggers this
sandbox cannot reproduce. The rapid-motion finding above, by contrast, is a
direct measurement on a synthetic clip and is the most likely remaining
cause of what is being reported; it is located precisely and left unfixed
on purpose rather than guessed at.
