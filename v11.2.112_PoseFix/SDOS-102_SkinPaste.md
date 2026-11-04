# SDOS-102 — SkinPaste: the mark is found by itself on every frame; opaque skin paste in the skin's own colour and texture

**Pack:** v11.2.102 "SkinPaste". Engine `aequus-1.2.102-skinpaste`. **App: no change** (1.2.29 stays).
Replaces v11.2.101 "PatchSimple".

**Reported:**
1. "The patch remover is not able to identify the body part and remove the patch even when the position of the body changes."
2. "The removal is still not clean. Configure it to pick the skin colour of the outside area to be pasted, including its texture. This should be clear skin pasting and not transparent."
3. "Go back to the earliest versions where it worked. I told you to make refinements. Please see what worked there."

All tests use synthetic clips (procedural skin, public photos); no user footage.

---

## 1. What I did about "go back to the earliest versions"

I ran the earliest engines on new, harder clips (the body part turns, swings, blurs, changes light), next to the current one:

| Clip (150 frames) | v11.2.97 (earliest with positions) | v11.2.101 | **v11.2.102** |
|---|---|---|---|
| mole, hard | 69.1 % left, **104** frames missed | 0.2 % / 0 | **0.3 % / 0** |
| birthmark, hard | 52.2 % / **77** | 1.5 % / 0 | **0.9 % / 0** |
| tattoo, hard | 40.4 % / **44** | -0.2 % / 0 | **-0.5 % / 0** |
| astronaut mole, hard | 23.6 % / **35** | 19.4 % / **29** | **-12.8 % / 0** (a little lighter than the clean clip) |

(left = share of the mark's contrast still visible, averaged over the frames where it is in the picture; missed = frames where more than half is left.)

So the earliest versions look right **only while the body part stays put**; once it moves, they miss most frames. That is the "not able to identify the body part" you describe. v11.2.96 stopped with an error on the same clips (not investigated: v11.2.97 replaced it).

**Kept from the earliest working behaviour (the look):** the inpainter fills the painted area once; the fill is carried on the tracked position; the lighting of the frame is applied to it; the painted area plus 2-3 px halo only; nothing outside it changes; the simple, fast per-frame blend of v11.2.98.
**Not taken back:** v11.2.97's two "edge bands" (`peek_band`, `grow_band`). I tried `grow_band` again: it painted 719 px of creases and skin outside the mark. The mark's own edge is now covered by finding the mark itself (section 2). Both stay at 0 in `PATCH_CFG`.

Every refinement below has its own switch in `PATCH_CFG` (`find_mark`, `rescue`, `opaque`, `clone_texture`); with all four `False` the paint is the v11.2.101 behaviour.

## 2. Why it missed and looked transparent — and the change

**Cause.** The engine located the mark by comparing the picture of the painted frame with the current one (2 px tolerance). When the body part turns, blurs or the light changes, the comparison fails:
- the mark was *lost* (no paint), or
- the part of the mark that no longer lined up was read as an **object in front** of it and the paint was held back there (`alpha = 1 - keep`). That is the half-transparent paste with mark fragments showing through.

**Now the mark is found by what it is, on every frame** (`find_mark`, function `_pr_find_mark`):
- the skin level around the predicted place is taken from *outside* the search area (a smooth interpolation of it);
- what differs from that skin with the mark's own polarity (darker for a mole, lighter for a scar) by half of the mark's contrast, in pieces that touch the predicted core, are the mark's strong pixels; its soft edge is the pixels a few px from those at 30 % of the contrast;
- the threshold follows the picture's noise, not the scene's structure; the search reaches half the mark's size plus 6 px (up to 45; wider on frames where the pose is a guess);
- the paint is **centred on what was found** and covers all of it (strong pixels, a margin, never more than 4 px from where the pose puts the painted area);
- only pixels of the mark's own colour are covered (on the line from the surrounding skin to the mark's ink / pigment): a hand, a cup or a dark object of another hue passing over it is not painted.

**Poses on every frame.** A gap of up to 45 frames between two followed frames is bridged; a run's last pose is held 15 frames further; a frame the tracker loses is "rescued" by finding the mark near where the motion puts it. These poses are **weak**: they paint **only where the mark itself is found**. A bridge that touches a rescued pose is weak too. So a wrong guess paints nothing.

**Nothing where there is no mark.** On the first test of this build a fast swing took the mark out of the picture and the held pose painted skin over clean skin (593 px changed far from the mark). Fixed: a found blob must be 0.12-2.4 x the mark's size (it was up to 3.5), a weak or widely searched frame needs 60 % of the found pixels in the mark's colour, and a frame where the mark is not found and the place does not look like it is left alone even if the object gate would "explain" it. After the fix: 0 px changed away from the mark on those clips.

## 3. Opaque skin paste

- The object gate is now **binary**: what the mark itself looks like is never an "object" (a turned or blurred mark no longer lines up with the anchor's picture), and only strong evidence of something else in front holds the paint back at all. A half-way verdict no longer leaves the mark showing through.
- The safeguard that refuses a frame needs the object to cover nearly all of the mark (85 %, was 70 %).
- **Colour:** unchanged and taken from outside the painted area: the fill's tone comes from the ring of skin around the mark, the frame's lighting is applied to it, and the stroke-margin cover (v11.2.101) brings it up to the frame's own skin level.
- **Texture** (`clone_texture`): the inpainter returns the right colour but a smoother skin than the skin around it. Its fine detail is replaced, once at the anchor, by the fine detail of a patch of clean skin beside the mark (same shape, best of 42 places on three rings, wholly plain skin, similar texture strength; creases, edges and hairs are never chosen), copied pixel for pixel, zero-mean so the level and shading of the fill are untouched. It then moves with the skin.

## 4. Results (synthetic, per frame, painted area only)

| Clip | left | missed frames | error (L) | outside px |
|---|---|---|---|---|
| hard mole / birthmark / tattoo | 0.3 % / 0.9 % / -0.5 % | 0 / 0 / 0 | 0.7 / 1.0 / 1.5 | 0 / 0 / 0 |
| hard astronaut mole | -12.8 % | 0 | 10.6 | 124 (v11.2.101: 117) |
| procedural mole | 5.4 % | 6 | 3.2 | 0 |
| procedural tattoo | 3.8 % | 3 | 3.6 | 0 |
| astronaut mole (fast swing) | 6.4 % | 6 | 8.7 | 0 |
| astronaut edge | 11.4 % | 6 | 16.4 | 0 |

- The 6 missed frames of the swing clips are the frames of the clip's swing: the mark moves about 140 px per frame there and leaves the picture; it is not followed on those frames and nothing is painted.
- An object over the mark stays untouched: 0.0 % changed on the hard clips; 2.0 % (worst 3.3 %) on the procedural tattoo (same as v11.2.101).
- Speed (one mark, 2 cores, clips running in parallel so noisy): mole 17-35 ms per frame, tattoo 90-132 ms (v11.2.101: 20 and 135).
- Texture (unit test, noisy synthetic skin): fine-texture strength of the fill 0.17 of the skin's before, **0.67 of 0.94 (71 %)** after; level change -0.01.

**Full jobs from the extracted zip** (the real render path, final encode; 720p, Balanced; scored against the clean clips on the mark):

| Job | Result |
|---|---|
| hard mole | removed on 150 of 150 frames; 3.5 % of the mark left after the final encode; missed on 0 |
| hard astronaut mole + a face swap in the same job | faces painted on 150 of 150 frames; removed on 150; **-10.5 %** left (lighter than clean); missed on 0; an object left untouched on 17 frames |
| 10 marks on one frame | all 10 anchored and followed; removed on 1420 frames of 1500; 5.8 % left on average; 1 frame missed; "mark itself found" on 1424 |
| kiss, the detector made blind for 40 frames of contact | both faces painted on 120 of 120 frames; `REVERTS=0`, `contact_hold=474/56` (same as v11.2.99 / 11.2.101) |

(The hard mole and 10-mark clips contain no face, so those two jobs say "no faces were swapped"; Patch Removal ran as normal.)

## 5. Limits

- **Synthetic tests only.** Real moles, tattoos and skin are not reproduced; the finder was tuned on precision (89-99 % of found pixels are the mark) and recall on these clips.
- The hard astronaut clip ends 12.8 % *lighter* than the clean clip on the mark: the cover step brightens, and an opaque paste on a mark over a strongly textured suit shows it. If a patch looks too light on real footage, set `PATCH_CFG["cover"]` to 1.0.
- A mark that moves extremely fast (the swing clips: about 140 px per frame) or leaves the picture is not followed; on those frames nothing is painted.
- On a bridged or held frame the paint appears only if the finder sees the mark. A very blurred mark that the finder misses stays visible on that frame.
- The painted area is still the area you painted (plus 2-3 px and the finder's margin). A mark that reaches far beyond your stroke is not covered beyond that.
- The texture is copied from one patch of skin, once; it moves with the skin and does not change with the frame.
- Texture is the fill's own where no plain skin beside the mark is available (a mark on a lip, an eyebrow, a nostril).

## 6. Verification (all from the zip's own files)

- `unit102.py`: all checks pass (finder: found 10 px off, plain skin gives nothing, shadow edge, blur, big blob and wrong polarity refused; strong and weak poses; opaque paste of a blurred mark; an object of another colour untouched; texture; bridges and holds).
- Earlier suites on the pack: `unit101` 21/21, `unit98` 11/11, `unit97` 18/18, `unit96` 18/18. Two tests changed meaning on purpose: `unit97` (a long gap is now bridged as weak poses; a jump is not interpolated) and `unit101` (a position's followed range may extend by `hold_max` frames, weak).
- Clip suite above: standard and hard clips.
- The job log's `PATCH-REMOVAL:` line now also says on how many frames the mark itself was found, the paint re-centred on it, the mark followed by itself, and a bridged or held pose left unpainted because the mark was not seen.

## 7. Install

- **Space:** upload the 8 files (or "Push to all Spaces", which skips busy Spaces). Log: `Phoenix v11.2.102`, `swap_engine aequus-1.2.102-skinpaste`.
- **App:** nothing to install; 1.2.29 works with this Space.
- If a mark still shows on a real job, send the `PATCH-REMOVAL:` line of the Space log (text only). The counts tell which path was taken.
