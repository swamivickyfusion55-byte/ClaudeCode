# SDOS-140 — BellyTattoo: belly clean covers the whole belly (tattoos and marks)

**Pack:** v11.2.140 "BellyTattoo". Engine `aequus-1.2.140-bellytattoo`. **App:** 1.2.44, no change needed (same job setting
`clean_belly`, same preview route `phoenix_clean_belly_preview`; it ignores keys it does not know, such as `person_gender`).

**Asked:** "I have problem with belly tattoos only" / "Please create only for belly tattoos and marks".

**Built on v11.2.118 LockFinder**, our last tested pack, not on Grok's v11.2.139: the sandbox refused to run the 139 code, and
untested code is not shipped. Changes 119-139 are therefore **not** in this pack (among them the kiss/contact changes, which
REVIEW-126 found risky, and the hair-patch box). Only `belly_clear.py`, `clean_belly.py` and the version labels differ from 118.

All tests use synthetic media (MediaPipe's own pose test photo with synthetic skin, tattoos, a mole and a navel painted on); no
user footage.

---

## 1. What was wrong

- **118 (ours):** the band ran from just below the navel to a little below the hip line (0.22 to -0.03 of the way from the hip
  joints to the shoulders; Length could lift it to 0.40 at most). A tattoo above or beside the navel was never looked at: on the
  new test clip, **0 %** of the upper-belly tattoo, 0 % of the tattoo beside the navel and 0 % of a mole were found.
- **139 (Grok):** belly clean was rewritten into a hair-patch finder (dark strands in a fence below the navel, one cluster, a
  box). It does not look for tattoos at all.

## 2. The change (`belly_clear.py`)

1. **The whole belly.** The band runs from **0.56** of the way up (just under the chest; the underbust is about 0.62) down to
   **0.10** above the hip-joint line. Length / Breadth (50-150 %) still scale it, but it never goes above 0.62 (the chest) or
   below 0.10.
2. **The floor (0.10, the bikini line) is a hard limit.** It runs parallel to the hip line, so it tilts with the pelvis. Marks
   stop 4 px above it, so even the fill's 3 px soft edge stays above it: nothing below the floor is changed. A tattoo lower than
   that is not touched (by design).
3. **The navel, wherever it is.** 118 looked for it only at the band's top edge. Now it is placed on the body: a compact,
   filled dark spot on the middle line (within 0.15 of the hip width), 0.18-0.46 of the way up. It is found on the marks' solid
   core, so a tattoo stroke touching the navel does not hide it, and a mark grown next to it does not grow over it. Only the one
   most navel-like spot is kept; a mole elsewhere on the middle line is still a mark.
4. **Ink must be ink.** With the band reaching the waist's sides, shaded skin there fell just outside the skin-colour model
   (measured 3.7-4.0 against the cut at 3.5) while being no darker than the skin around it (0-2 levels), and was taken for ink.
   A non-skin piece now counts only when it is at least 8 levels darker than its ring of skin, or at least 6 from the skin's
   colour (coloured inks). Measured on the clip: the tattoos are 89-104 levels darker and 12-14 from the skin's colour.
5. **Darker-skin marks (moles, pigment) not on the body's outline.** They must lie inside the skin, away from clothing, the
   background and the band's edge (2.5 % of the band's size, at least 4 px).
6. **Strokes running into cloth, only where cloth meets the belly at the bottom.** 112 rescues thin tattoo strokes that touch a
   waistband or shorts (the cloth is thick, the stroke thin). With the band now reaching the waist's sides, the same rule
   rescued the dark gap between the body and the background, and bits of the background (measured: 260, 41 and 36 px on one
   frame; the engine then painted them on every frame, beside the body). A thin piece is now rescued only when it is in the
   band's lower part (0.15 of the torso above the floor), away from the band's sides (0.15 of its width), does not lie along the
   cloth's edge (at most 60 % of it within 5 px of the cloth) and passes the ink test of point 4.

Unchanged: the pose model (MediaPipe Tasks 0.10.35), the skin model, clothing handling (waistband, shorts), the key-frame
choice, the Patch Removal engine that carries the fill through the video, the preview route and its outputs.

## 3. Verification

`verification/patchremoval/make_bellytattoo.py` makes the clip (800 x 680, 90 frames, moving: shift, zoom, small turn): an
upper-belly tattoo (a rose-like design, t 0.40-0.50), a tattoo beside the navel (a script stroke and three dots, 0.42 of the hip
width to the side), a lower-belly tattoo (a wave and a ring, t 0.12-0.20), a brown mole, the navel at t 0.27, and black shorts.
A second clip puts the navel at t 0.36, where a tattoo leaf touches it. `unit140.py` runs the checks.

### A. The mark finder (true landmarks, frames 0 / 22 / 45 / 67)

| Check | v11.2.118 | **v11.2.140** |
|---|---|---|
| upper-belly tattoo found | 0 % | **100 %** (navel 0.36 clip: worst 93 %) |
| tattoo beside the navel found | 0 % | **100 %** |
| lower-belly tattoo found | 67-70 % | **88-90 %** |
| brown mole found | 0 % | **100 %** |
| navel marked | 0 px | **0 px** (also with the tattoo leaf touching it) |
| black shorts marked | 6 of 100,718 px | **0 px** |
| marked below the floor | 321-338 px | **0 px** |
| marked on the chest (above 0.62) | 0 px | **0 px** |
| clean skin (the same frames without marks) | 55-66 px per frame | **0 px** |

18 of 18 checks pass on the 140 pack (both clips).

**A dark waistband above the floor** (a third clip: a black band across the body at t 0.11-0.15, and a tattoo stroke running
down into it), frames 0 / 22 / 45 / 67:

| Check | Result |
|---|---|
| the stroke running into the waistband found | worst **95.2 %** |
| the waistband itself marked | **1 of 8,234 px** |
| the other tattoos and the mole | 100 % / 100 % / 100 % |
| navel marked; the same frames without marks | 0 px; 0 px |

### B. The job path (`build_clean_belly`, the real pose model on the moving clip)

| Check | Result |
|---|---|
| positions found | 3 key frames (19, 48, 63) |
| upper-belly tattoo, tattoo beside the navel, mole | **100 %** on all three key frames |
| lower-belly tattoo | 88.4 % and 92.5 %; **84.9 % on frame 19** (below the 85 % bar: see below) |
| navel marked | 0 px on all three |
| marked below the true floor (8 px of pose error allowed) | 0 px on all three |

**The one check below its bar:** on frame 19 the clip's motion puts the lower tattoo's small ring on the body's outline. Its
outer half touches the background there, so it joins the background piece and is not taken (147 of its 289 px). The wave beside
it is found completely (682 of 682 px). A tattoo stroke that runs into the body's outline is a limit (section 4).

### C. The preview route (`phoenix_clean_belly_preview`, as the app calls it; real pose model, LaMa fill)

| Check | Result |
|---|---|
| body found, marks | yes, 5,970 px, 15 s on this machine (mostly the LaMa fill) |
| upper-belly tattoo left | **0 %** of its contrast (-0.1 %) |
| tattoo beside the navel left | **0 %** (-0.2 %) |
| mole left | **1.4 %** |
| lower-belly tattoo left | **11.8 %** |
| navel changed | 0.0 levels |
| changed below the detected pose's floor (the line the code guards) | **0 px** (before the strict floor: 23 px, up to 3 px below: the fill's halo) |
| changed below the true floor (8 px of pose error allowed) | 0 px; the lowest change is 5.5 px above it |
| changed outside the band (beyond 4 px) | 0 px |
| out-of-range Length / junk Breadth | brought into range (150 %, 100 %) |

**43 of 44 checks pass** (A 18 of 18, waistband 5 of 5, B 9 of 10, C 11 of 11), run from the pack's final files.

### D. The whole video (the job's own positions through the real Patch Removal engine, 90 moving frames)

`build_belly_gt.py` takes the job's key-frame positions; `video_test.py` runs the engine on the clip (group "belly", as in the
job) and compares every output frame with the same clip without marks.

| | v11.2.118 on its own (lower-belly) clip | **v11.2.140 on the whole-belly clip** |
|---|---|---|
| marks still visible (mean over 90 frames) | 1.5 % | **0.6 %** |
| worst frame | 2.7 % | **0.9 %** |
| frames where a mark was not removed (over 50 % left) | 0 of 90 | **0 of 90** |
| difference from the clean clip on the marks (incl. codec noise) | 5.8 L | **2.1 L** |
| frame-to-frame shimmer on the marks | 3.06 L | **1.67 L** |
| changed more than 12 levels farther than 25 px from any mark (worst frame) | 216 px | **0 px** (475 px before fix 6) |
| below the true floor (worst frame, more than 12 levels) | - | 14 px, of which 2 px more than 8 px below (pose error and the engine's carried fill) |
| the navel | - | mean change 2.0-2.6 levels per frame; the worst single pixel 14 (re-encoding noise on untouched sharp edges: 7-8) |

Speed on this machine: planning 43 s (pose on 12 sampled frames, the fill), then 64 ms per frame.

## 4. Limits

- **Not tested on a real body.** The test body is MediaPipe's own pose photo with synthetic skin and synthetic marks.
- **The head must be in the frame**, with both shoulders and both hips, in a front view (as before): that is how the pose model
  finds the body. Frames without it get no band; the log says how often.
- **A tattoo below the floor** (lower than 0.10 above the hip joints, the bikini line) is not removed, by design. Paint it by
  hand with Patch Removal if needed. Measured on the v11.2.117 test clip, whose tattoo runs from 0.03 to 0.17 (a third of it at
  or below the floor): 118 found 97.7 % of it, **140 finds 75.2 %** (95.8 % of its part above the floor).
- **A tattoo stroke that runs into the body's outline** (the background showing beside it) joins the background and is not
  taken; the rest of the tattoo is (section 3B, frame 19).
- **A tattoo stroke running into cloth** is rescued only at the band's lower middle (a waistband, shorts). At the band's sides,
  or where the cloth is a top above, the stroke's end at the cloth stays (fix 6).
- A tattoo that crosses the band's outline keeps its part outside; raise Length or Breadth until the preview's outline holds it.
- Black jewellery or a dark piercing on the belly can be taken for ink. The navel itself is left alone.
- Very pale skin under a strong blue light may fall outside the skin model (as in 117).
- One person (the most prominent).

## 5. Install

- **Space:** upload the pack's files (the same file list as v11.2.118; `belly_clear.py` and `clean_belly.py` changed). No change
  to `requirements.txt` or `packages.txt`, so no rebuild of the dependencies.
- **This replaces v11.2.139 on the Space** (see the top: changes 119-139 are not in it).
- Log at start: `Phoenix v11.2.140`, `swap_engine aequus-1.2.140-bellytattoo`.
- App: 1.2.44 as it is. Its belly text still says "below the navel to the waistline"; the Space now covers the whole belly.
