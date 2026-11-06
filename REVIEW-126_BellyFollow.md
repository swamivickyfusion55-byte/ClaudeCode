# Review of Phoenix v11.2.126 "BellyFollow" (Grok) — kiss/contact occlusion and belly clean

**Status: static review, with one stand-in measurement. The 126 pack's own code was NOT run** (the sandbox refused to run the
uploaded code; see "How this was checked"). No refined pack is attached yet. "Kids occlusion" is read as **kiss occlusion**
(faces touching or overlapping).

The pack is v11.2.118 LockFinder plus unlabelled changes 119-126: `ENGINE_VERSION` still says `aequus-1.2.118-lockfinder`, and the
pack has no SDOS or notes for 119-126, so none of these changes comes with test evidence.

## How this was checked
- **Read**: every changed file diffed against our v11.2.118 pack (config, core_pipeline, phoenix_api_adapter, clean_belly,
  belly_clear, swap_engine).
- **Measured (stand-in)**: the new dark-mark setting, applied to *our own* v11.2.118 `belly_clear.py` (not the upload), on the
  synthetic belly clip with no tattoo (`verification/patchremoval/review126_dark_thresh.py`).
- **Not done**: running the 126 engine, the kiss clips, the belly suite on the 126 files. Nothing seen on real footage.

## Findings

| # | Area | Sev. | Finding | Evidence |
|---|---|---|---|---|
| 1 | belly | **High** | Dark-mark threshold halved and the x2 noise factor dropped: `dark_min` 12 -> 6 and `max(dark_min, 3*noise)` instead of `max(12, 3*noise*2)`. Clean skin gets "marks", so every key frame has something to fill (the `KEY_MIN_PX = 30` test always passes) and a clean belly is filled with LaMa on every job. | Measured, stand-in: clean skin, 10 frames — **118 setting 25 px/frame (0/10 frames over 200 px); 126 setting 707 px/frame (10/10 over 200 px)**. Same with side light (791), a skin fold (749), video noise sigma 4 + JPEG (442). |
| 2 | belly | Med | `belly_shift` = `null`, `""` or NaN becomes the **maximum shift up** (+0.12): `clamp_scale` returns 1.0 for junk *before* range-clamping, and the range clamp then turns 1.0 into +0.12. For a shift the safe default is 0. | Read: `belly_clear.clamp_scale`, `phoenix_api_adapter` (`st.get("belly_shift", 0.0)` passes a present `null` through). |
| 3 | belly | Med | The new "nose below the hips" and "knees must be past the hips" checks in `region_polygon` **never run**. `detect_all()` and `canonical()` keep only 4 landmarks (shoulders, hips); `lm[NOSE]`, `lm[L_KNEE]` raise `KeyError`, which the surrounding `try/except: pass` swallows. The wrong-person / sitting-body cases they were meant to stop are unguarded. | Read. |
| 4 | belly | Med | The preview no longer outlines the band: `preview_frame` returns `(out, out.copy(), info)`. The app's "Show outline" now shows two identical pictures. (We removed none of this; 117 drew the polygon.) | Read: end of `clean_belly.preview_frame`. |
| 5 | belly | Med | **Woman-only gate** (`female_body`): a man's belly is not cleaned (`belly clean runs only for a woman`). Gender comes from InsightFace, which the engine's own comments say "often calls the man a woman, and the woman a man for a few frames"; a wrong tag on the sampled frame skips the key frame. It also fails *open* (cleans) when no face is found, e.g. head covered or pixelated. It loads the full face model (`cp._load()`) and runs face detection per body per sampled frame inside the belly pass, and in the preview. This is a **product decision**, not a bug fix; it needs the user's yes. | Read. |
| 6 | belly | Med | `as_paint_stroke`: closes marks with an ellipse up to 23 px wide, so separate marks join into one blob; the "on skin" test dilates the skin by the same kernel, so the paint can reach ~11 px over clothing (waistband, shorts) wherever the band touches it. A bigger fill area than the mark means more smudge. | Read; not measured. |
| 7 | kiss | **High** | **Identity wall switched off near any other face.** 118: `touching` = box IoU > 0.02 or centre distance < 0.55, **and** the face is where the slot's track expects it (`spatial >= 0.20`). 126: IoU > 0 or distance < **1.15** (box-widths), and the position test is gone. Any face within ~1 face-width of another is admitted to any slot that has an identity, at **any** similarity. People standing side by side are inside 1.15. A bystander or a false detection next to the pair can take a slot when its real face is hidden. | Read. 118's own comment: removing the floor outright raised the bystander false-bind rate 0.4 % -> 2.0 % (0 % -> 43 % for a sustained look-alike). |
| 8 | kiss | Med | The post-bind check also skips `_rel_admit` when "kissing" (same loosened test) and `lock_sim >= 0.16`. Together with 7 the identity has almost no guard in a close pair. | Read. |
| 9 | kiss | Med | The mouth/chin occluder ("EAT" rule) is skipped whenever a rival face is near (`... and not _rival_near`). An object (hand, glass, food) in front of a kissing pair's mouth then relies on the new chroma split alone. | Read. |
| 10 | kiss | Low-Med | `_kiss_skin_split` splits the overlap by skin chroma (Cr/Cb norm >= 3.5, margin 2.0) and centroid closeness. Skin tones of two people close together are often within that margin; the tie falls back to closeness, which is the old behaviour. Gain unproven; no clip, no number. | Read. |
| 11 | kiss | Low-Med | `core_pipeline` keeps a *weak* second detection if it overlaps any tracker anchor by IoU >= 0.12 even when it overlaps a reliable face by >= 0.35: a duplicate box on the same face can be kept and swapped twice. | Read. |
| 12 | both | Low-Med | `_match_skin_tone` in `fill_still` applies to **every** patch fill, not only belly: it lifts any darker area inside the patch to the ring's median, which also removes real shading. | Read; not measured. |
| 13 | belly | Info | The app (1.2.37) has no `belly_shift` control, so the shift is unreachable from the UI. | Read. |

## What looks good in 126 (keep)
- `shift` plumbing, range-clamped on both the job and preview path (apart from #2).
- Several pose people (`num_poses = 2`) with the body that passes the band test used, instead of only the first.
- The belly group forces the body-locked fill (`skin_hair`) and keeps it for up to 18 lost frames: closes the "belly mark drops when pose is lost" gap. Needs a retest for drift over a hand or object.
- The wide-detection dedupe for a single face (`separate`) is a sound idea.

## Proposed fixes for the refined pack (v11.2.127)
1. Restore the dark threshold to 118's (`dark_min` 12, x2 noise) and test clean, shaded, fold and noisy skin at 0-25 px/frame.
2. `belly_shift`: junk -> 0 (own default per parameter); null handled in the adapter.
3. Keep nose and knees in the landmark dict (canonical keeps them) so the checks run; test with a sitting / head-below-hips pose.
4. Preview: draw the band outline again; keep the info dict.
5. `as_paint_stroke`: a smaller close kernel, no skin dilation past the band, per-mark close so far-apart marks do not merge; clothing overlap measured on the shorts clip.
6. Gender gate: leave **off by default** (setting `belly_woman_only`, default false) until you confirm; no face-model load in the belly pass.
7. Kiss: restore `spatial >= 0.20` and the 0.55 distance for `touching` (keep IoU > 0 for the overlap case), and restore `_rel_admit` for non-overlapping faces; keep the skin split only if the kiss clips show a gain (REVERTS, FACE-EVENTS, bystander false-bind rate).
8. `core_pipeline`: the `owned` keep requires IoU < 0.35 with every reliable face.
9. `_match_skin_tone`: belly group only.
10. Version labels, `ENGINE_VERSION`, SDOS-127 with the full evidence.

## Limits
Synthetic media only (a posed test picture with painted skin and a tattoo; synthetic kiss clips). Nothing seen on a real body, and
no real footage was analysed.
