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
