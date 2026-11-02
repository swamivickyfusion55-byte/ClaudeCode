# SDOS-099 — PatchBlend: no residue under the paint, the skin's own texture, no revert during a kiss

**Pack:** v11.2.99 "PatchBlend" (v11.2.98 PatchStrong + this). Engine `aequus-1.2.99-patchblend`.
**App:** no change. 1.2.27 stays.

**Reported:**
- "Make the paint over the patch completely blended with the skin colour, texture and look around it. No residue of the underlying patch. It still shows some residue underneath the paint, though much cleaner than before."
- "For a fraction of a second, during kissing, the face reverts to the original."

All tests use synthetic clips (procedural skin, public photos); no user footage.

---

## 1. Where the residue came from

The earlier test clips move the skin rigidly under steady light. On those, v11.2.98 covers the mark on every frame where it removes it, and leaves about 1 level of darkness. That was not what you were seeing.

**New test clips, closer to real footage:**
- the skin bends a little (non-rigid, up to 3 px);
- a soft shadow edge sweeps across the mark;
- the clip is encoded at phone quality (CRF 24).

On these clips v11.2.98 failed badly:

| Clip | Frames where the mark showed | Mark left on average |
|---|---|---|
| mole, moving shadow | **50 of 150** | 36 % |
| birthmark, moving shadow | **74 of 150** | 46 % |
| tattoo, bending skin | 6 of 150 | 8.4 % |

I split the cause: **bending alone was handled well** (0 frames missed). **The moving shadow was the cause:**
- The tracker compared brightness, so a shadow over part of the mark made the right place look wrong. It lost the mark, and on those frames the mark showed through the paint.
- The fill's lighting was carried over from the painted frame. Under a shadow edge it came out a little too light on one side and too dark on the other.

This is very likely the residue you saw: real footage has light changing across the skin all the time.

## 2. What changed (Patch Removal)

**Tracking that ignores light.**
- The tracker now compares a lighting-free view of the mark: log brightness minus its smooth version. A shadow or light change cancels out; the mark and the skin's texture stay.
- It allows 2 px of bending, so a tattoo's thin strokes are not lost when the skin flexes.
- When an object passes over the mark, it falls back to the plain comparison. In the lighting-free view an object looks like two edges, not one covered region, and the rule for "partly covered" did not recognise it.

**Relighting that follows a shadow.**
- On every frame, the fill is relit by a local light field: how much brighter or darker each part of the area is now than on the painted frame.
- It is measured on the skin around the mark and on the plain skin inside your stroke, and bridged across the mark itself. The bridge carries a shadow edge straight across instead of blurring both sides together.
- An object in the ring around the mark is left out: it changes colour, or changes too much, so it is not light.
- The object gate uses the same light. A shadowed mark is never taken for an object in front of it.

**The skin's own texture (the "paint" look).**
- The inpainted fill came out smoother than the skin around it. On the photo clips it had 50–80 % of the surrounding fine texture.
- The fill now gets the grain of the skin around it, in two bands (fine grain, and pores and mottling), up to the skin's level. Inside the painted area only.
- The grain is made once and moves with the skin, so it does not boil from frame to frame.

**The faint-darkness lift of v11.2.98** stays on. It now judges the level with this frame's lighting divided out, so a shadow gradient over the mark is no longer read as residue. Under a 40 % shadow edge, the error is 1.88 levels with the lift and 1.89 without it.

**Unchanged:** nothing outside the painted area (+2 px halo) changes.

## 3. Results (synthetic clips, removal per frame)

**Mark left on average / frames where it showed / error against the clean clip (lightness 0–255):**

| Clip | v11.2.98 | **v11.2.99** |
|---|---|---|
| mole, moving shadow, bending | 36.3 % / 50 / 15.5 | **−1.6 % / 0 / 1.2** |
| birthmark, moving shadow, bending | 46.2 % / 74 / 10.3 | **−3.1 % / 0 / 1.1** |
| tattoo, moving shadow, bending | 8.4 % / 6 / 7.5 | **−3.3 % / 0 / 2.6** |
| cheek mole (photo), moving shadow, bending | −0.9 % / 0 / 6.4 | **−1.1 % / 0 / 6.2** |
| procedural mole | 10.4 % / 13 / 5.0 | **5.5 % / 6 / 3.2** |
| procedural birthmark | 16.9 % / 21 / 4.0 | **8.6 % / 10 / 2.6** |
| procedural tattoo | 4.7 % / 6 / 3.9 | **4.4 % / 6 / 3.8** |
| cheek mole (photo) | −1.1 % / 6 / 9.8 | **−0.0 % / 6 / 9.3** |
| jaw-line mole (photo; mark reaches past the paint) | 8.7 % / 6 / 16.2 | 9.8 % / 6 / 16.5 |
| mole, two positions across a cut | — | **2.5 % / 1 / 7.0** |
| tattoo, two positions across a cut | — | **1.5 % / 2 / 2.2** |

- The remaining missed frames on the procedural clips are where the mark is out of the picture or covered (the clip's swing at frames 92–112).
- An object over the mark is still left untouched (0.0–0.2 % of its pixels changed).

**The darkness left inside the mark** (removed frames, + = darker than clean):

| Clip | v11.2.98 | v11.2.99 |
|---|---|---|
| mole, moving shadow | +4.93 (on the 100 frames it removed) | **−0.42** (all 150) |
| cheek mole (photo) | −1.34 | **−0.54** |
| procedural mole / birthmark | +1.08 / +1.08 | +1.06 / +1.05 |

**Texture of the preview fill against the skin around it** (1.00 = the same):

| Clip | v11.2.98 | v11.2.99 |
|---|---|---|
| cheek mole (photo) | 0.61 | **0.98** |
| cheek mole, moving shadow | 0.50 | **0.98** |
| procedural mole | 0.79 | **0.92** |

## 4. The brief revert during a kiss

**What happens.**
- In deep contact the detector often loses both faces for a moment.
- Once nothing has been found for more than 0.75 s, v11.2.98 treats the stretch as "the faces are gone". It shortens the hold to a few frames and the original shows until a face is detected again.

**Reproduced.** On the synthetic kiss clip, with the detector made to find nothing (including its close-up re-checks) for 40 frames of contact:
- v11.2.98: both faces went back to the original for 8–12 frames (about 0.3 s) in the middle of the kiss. That matches what you saw.
- The job log reported `REVERTS=4 blind=24 faint=9`.

**The fix.** A face lost while touching another face is held, not dropped:
- **full strength for 1 s** (`CONTACT_FLAT_SEC`), then the usual fade;
- a gap with a real sighting on both sides is **bridged up to 2 s** (`CONTACT_HOLD_SEC`);
- the "faces are gone" rule never shortens it;
- the held face stays at its last real position (no drifting guess);
- the swap is not run again on the held face, because that crop is cut slightly off the moving head. The nearest real crop is reused.
- Faces that are not touching are handled exactly as before.

**Result on the same test:**
- 0 frames reverted, 0 faint (v11.2.98: 12 and 5 frames below half strength for the two faces);
- log `REVERTS=0 faint=0 contact_hold=474/56`.

The kiss without a dropout is unchanged: `REVERTS=0 blind=0`, and frame-to-frame change on the faces matches v11.2.98 (mean 5.20 against 5.17 for one face, 4.17 against 4.13 for the other; one frame where the faces part reads 12 against 9.6).

## 5. Limits

- **Synthetic tests only.** The moving-shadow clips are a model of real light, not real light.
- **Kiss hold:**
  - While the faces are held, the face is the last real one (its expression does not change for up to a second).
  - The frame-to-frame change on the held faces is a little higher than with detection (about 5 against 4).
  - One frame at the start of a hold shows a step in the face behind, where the outline of the two faces is redrawn (11–13 against 3–6 on other frames).
  - If your kiss lasts longer than 2 s with no face found at all, the hold ends and the faces fade out.
- **Tattoos under strong bending:** the 2 px tolerance covers ordinary skin movement, not a fold.
- **Jaw-line clip:** the part of the mark beyond the paint stays, as in v11.2.98. Paint a little past the edge.
- **A hard-edged shadow** (direct sun) across the mark is followed less sharply than a soft one.
- **Speed:** relighting and texture add about 15–30 ms per frame per mark. The big tattoo (150 × 90 px) is about SPEED.

## 6. Verification

- `verification/patchremoval/unit99.py`: 18 checks, all pass. They cover:
  - the lighting-free view;
  - the light field under a shadow edge and with an object in the ring;
  - tracking under a moving shadow (all 130 frames; plain view 94);
  - the gate and the relit fill under a 40 % shadow (1.9 levels against 3.1 for v11.2.98);
  - nothing outside the painted area changes;
  - the lift under shading;
  - texture matching (adds grain to a smooth fill, nothing to a textured one, nothing outside, no level shift);
  - the contact hold (old behaviour without contact, full strength and bridging with contact).
- Earlier suites on this pack: unit98 all pass, unit97 16/16, unit96 18/18.
- New test material:
  - `make_patchclip.py OUT warp`: the bending, moving-shadow, CRF 24 clips;
  - `bench_pipeline.py DROP_CONTACT=both DROP_RUN=lo:hi:per`: the detector made to lose the faces in contact, including its close-up re-checks.
- **Full jobs from the zip** (the real render path):
  - mole under a moving shadow on bending skin: removed on 150 of 150 frames, 1.9 % of the mark left after the final encode, missed on none;
  - cheek mole with a face swap in the same job: faces on 150 of 150 frames; the mark removed on every frame where it is in the picture except the 6 of the clip's out-of-picture swing (−3.2 % left: very slightly lighter);
  - the kiss with the detector losing both faces for 40 frames of contact: faces painted on 120 of 120 frames, `REVERTS=0 faint=0 contact_hold=474/56`, no frame below half strength.

## 7. Install

- Upload the 8 files, or use "Push to all Spaces".
- Log lines: `Phoenix v11.2.99`, `swap_engine aequus-1.2.99-patchblend`.
- The job summary line now also shows `contact_hold=held/reused`.
- If a revert still shows in a kiss, send me that line (`REVERTS=… blind=… contact_hold=…`). It tells which path was taken. Text only, no footage.
- No app change: keep 1.2.27.
