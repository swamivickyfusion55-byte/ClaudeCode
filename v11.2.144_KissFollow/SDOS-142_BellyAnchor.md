# SDOS-142 — BellyAnchor: belly marks in a pentagon, the video takes what the preview took, the cover held on the body

**Pack:** v11.2.142 "BellyAnchor" (v11.2.141 ContactOrder + this). Engine `aequus-1.2.142-bellyanchor`.
**App:** 1.2.45 (pentagon editor in Belly clean; sends the pentagon and the preview's marks with the job).

**Asked:** "Auto-detect any mark, tattoo or other dark patch below the navel and on the lower belly; outline it with pentagonal
lines on Belly preview, allow adjusting it; on Belly preview match it with the surrounding skin to remove the dark patch; on running
the swap, cover the patch throughout the video. The cover needs to be body-landmark based (navel, shoulders, hips) so that it
doesn't hang around and flicker."

**Boundary (unchanged):** the belly band ends 0.10 above the hip line, parallel to it (the bikini line). Nothing below it is
found, filled or changed; a pentagon corner dragged lower is put back on that line. This clean is for tattoos and marks on the belly.

All tests use synthetic clips (`make_bellytattoo.py`: a drawn body with tattoos, a mole and a navel, moving and turning; `motion=fast`
for 3x the amplitude at twice the speed) with the real pose model (MediaPipe Pose Tasks 0.10.35) and the real fill (LaMa). No
real body or user footage was used, and the app was not run on a phone.

---

## 1. What is new

### 1.1 The pentagon (`belly_clear.py`, `clean_belly.py`, adapter, app 1.2.45)

- **Body coordinates.** A point on the belly is held as (u, t): u across the body in hip widths from the middle line (the hip joints
  at u = -0.5 / +0.5), t up the body, 0 at the hip-joint line and 1 at the shoulder line. A point held in (u, t) moves, turns and
  scales with the hips and shoulders (`body_affine`, `to_body`, `from_body`).
- **Preview belly** finds the marks in the band as before (v11.2.140) and draws a **yellow pentagon with five handles** around
  them: the convex hull of the marks reduced to five corners and grown until it encloses every mark with a margin
  (`auto_pentagon`). No mark: a default pentagon below the navel. The band is still drawn in grey.
- **Adjusting.** In the app: drag a corner, drag inside to move the whole pentagon, pinch to stretch it; then Preview belly again.
  The app sends the corners (`belly_pentagon_px`); the Space keeps them on the belly (`clamp_pentagon_body`: above the floor, below
  the band's top, at most 1.2 hip widths from the middle line) and returns the pentagon in pixels and in body coordinates
  (`pentagon`, `pentagon_body`, `pentagon_source` app / app (body) / auto / default). **Auto pentagon** goes back to the automatic one.
- **Only the marks inside the pentagon are filled**, each piece **whole or not at all**, by its middle. A mark the pentagon's line
  cuts is never half removed (a half mole would be left, and which half would change with the pose's few px from frame to frame).
- **The video takes what the preview took.** The preview also returns each mark it found: where it lies on the body and whether
  it was taken (`marks_body`: [u, t, taken, px]). The app keeps it with the pentagon and sends both with the job (`belly_pentagon`,
  `belly_marks`). On the job's sampled frames each mark is matched to the preview's by its place on the body and gets the
  preview's decision on **every** key frame. A mark the preview did not see is decided by its middle averaged over the frames
  it is seen on. Either way a mark is covered for the whole video or left for the whole video.

  Why: on the test clip a mole sits on the corner of a user's pentagon. Its middle is inside on 6 of the 12 sampled frames and
  outside on the other 6 (the pose's error is 4-7 px), so each frame's own decision flips. Before this, the job's three key
  frames took 22, 36 and 88 px of that mole (of ~100); the preview had left it.

### 1.2 The cover held on the body (`swap_engine.anchor_to_body`, `core_pipeline._patch_prepare`)

With a belly patch, the job's first pass (the one that already reads every frame for Patch Removal) also reads the body:
MediaPipe Pose in video mode on every 2nd frame (`BodyPath`: outlier samples dropped, Gaussian-smoothed, interpolated between
samples, held at most a short while). After the image tracker has followed the belly patch, every frame of its stretch is checked
against the body:

1. **The body's prediction.** B_i = body_i ∘ body_anchor⁻¹ carries the patch from its anchor frame to frame i by the shoulders and hips.
2. **Agreement.** The image tracker's place is compared with the body's, less the steady offset between the two on this track (the
   pose model's own bias on this body, the median). The image is kept where it lies within tol = max(24 px, 8 % of the torso's
   length). Only a **run of 4 frames or more** beyond tol counts as the image having failed (a shorter one is the pose model's noise).
   A confirmed failure is widened over the neighbouring frames that are off by more than max(8 px, 2 × the median image-vs-body
   distance on this track): a tracker that starts to hang drifts off slowly.
3. **Where the image failed or lost the patch,** the body places it (with the image's correction from the agreeing frames around,
   carried at most 30 frames, so there is no jump where the two meet). It is then **found again in the image from the body's
   place**: the tracker's own alignment (ECC, affine), coarse to fine (1/8, 1/4, 1/2 size, then full size on the tracker's
   lighting-free view). The fit is kept when it correlates at 0.70 or better and lands within 1.5 × tol of the body's place;
   otherwise the body's place stands.
4. Frames the image had lost are painted again (confidence 0.6).

Where the image tracker is right (every frame of the normal and fast test clips), nothing changes: the image is exact to the pixel,
the pose model is not (10-15 px on the fast clip).

## 2. How it was measured, and what did not work first

### 2.1 The stress test (`verification/patchremoval/video_test.py`, `VT_BODY`, `VT_FREEZE`, `VT_DROP`, `VT_WINDOW`)

The image tracker did not fail on its own on the synthetic clips, so it was **made to fail** on frames 20-40 of the fast clip:
`VT_FREEZE` holds the position of frame 20 while the body moves on (the cover "hangs"); `VT_DROP` removes the positions (the
tracker lost the patch). Score: the share of the marks' contrast still visible in the finished frames, against the clip's clean twin.

| Case | 141 (image only) | 142, first anchor | 142, + fit from the body's place (full size only) | **142 final** |
|---|---|---|---|---|
| normal clip, all 90 frames | 0.6 %, worst 0.9 % | 0.6 % | 0.6 % | **0.6 %, worst 0.9 %** (anchor: 0 frames placed) |
| fast clip, all 90 frames | 0.8 %, worst 17.3 % | 0.8 % | 0.8 % | **0.8 %, worst 17.3 %** (0 frames placed) |
| frozen 20-40: mean / worst / frames over 25 % | 82.6 % / 100 % / 19 | 21.9 % / 66.2 % / 9 | 21.1 % / 66.2 % / 9 | **1.7 % / 29.4 % / 1** |
| lost 20-40: mean / worst / frames over 25 % | 95.2 % / 100 % / 20 | 25.8 % / 75.8 % / 8 | 24.2 % / 75.8 % / 8 | **0.0 % / 0.3 % / 0** |

(141 frozen / lost and 142 lost are the same clip and frames; the first-anchor and full-size-fit columns are the intermediate versions.)

**What did not work, and why (kept here so it is not tried again):**

- **A tight anchor** (body takes over beyond 6 px / 3.5 %) made the fast clip worse: 6.6 % visible, worst 77 %, against 0.8 % and
  17.3 % with the image alone. On that clip the image was right and the pose was 10-15 px off. Hence the gross-disagreement rule.
- **The fit from the body's place at full size** was accepted on only 2 of 18 (frozen) and 5 of 20 (lost) frames. Instrumented:
  the body's place was 2-14 px off (lost) and up to 28 px off (frozen); the full-size fit converges from about 6 px only (from the true
  place it scores 0.95-1.00, the tracker's own match score there is 0.92-0.99 against 0.27-0.67 at the body's place).
- **A shift-only template search** around the body's place scored 0.24-0.49: the belly patch's box is large (about 550 × 650 px, the
  whole belly with all its marks), and over that area the body's error includes a small turn and scale, not only a shift.
- **Coarse to fine** found all 18 and all 20 frames (final column).

### 2.2 Unit checks (`verification/patchremoval/unit142.py`, no models except the clip): 21 / 21

Body coordinates round trip; the hips at u = ±0.5; a point held in body coordinates follows the body (0.00 px from the true navel
45 frames later); corners dragged below the floor / onto the chest put back; the automatic pentagon encloses every mark; the
default pentagon; a pentagon over one tattoo takes it and leaves the others, every piece whole or not at all; settings parsing
(pentagon, marks); BodyPath smoothing (1.14 px against 2.13 px for the raw samples) and hold. The anchor's rules on a stand-in
track: kept where the image agrees; a hang of 60 px handed to the body within 6 px; lost frames placed and painted; a 3-frame
disagreement ignored; the fit kept or refused by its correlation and reach; no body on the anchor frame leaves the track as it was.
With the tracker's real alignment code on a textured clip: lost frames with the body 15 px off are found to 0.11 px; from 23 px,
3° and 4 % off the coarse-to-fine fit finds the patch to 0.03 px where the full-size fit alone fails.

### 2.3 Preview and job (`preview142.py`, real pose model and LaMa, through the adapter's preview route): 13 / 13

The automatic pentagon (5 corners, px and body coordinates) and every mark inside removed (under 12 % left); a user pentagon over
the upper tattoo removes it and leaves the others at 100 %; returned as given; dragged below the floor it is put back and nothing
below the floor changes; laid on frame 45 by body coordinates it follows the body (2.8 px) and removes the same tattoo; the preview
returns each mark with its decision (the mole on the line: left); the job with the preview's marks takes the upper tattoo on
every key frame, the other tattoos on none, and **leaves the mole on every key frame, as the preview did**; the job with the
pentagon only gives the mole one decision for every key frame.

### 2.4 Through the job's own code (`e2e142.py`)

`clean_belly.build_clean_belly` (the adapter's step), `core_pipeline._patch_specs`, `core_pipeline._patch_prepare` (frames read
as the render reads them, the pose in video mode, BodyPath, the anchor with the job's `gray_at`), `PatchTrack.apply` per frame as
the render does, on the fast clip: 3 key positions; the body on 44 of 45 sampled frames; the anchor ran on all 3 belly tracks
(the image was right everywhere, 0 frames placed); LaMa; **marks left: mean 4.6 %, worst 21.3 % (the clip's last frame), 0 frames
over 25 %**. No error in the log.

**Cost:** the planning pass took 52-56 s for 90 frames (800 × 680, 2 CPUs) with and without the body anchor; the difference is within
run-to-run variation. Pose every 2nd frame; the fit runs only on frames the body places.

## 3. Limits (honest)

- **Synthetic only.** A drawn body, drawn tattoos. The pose model on a real body (loose clothing at the hips, a side view, the
  shoulders out of the frame) is less sure; where it finds no body the anchor is off for those frames and the image tracker works
  alone, as in 141. With no body on the anchor frame the anchor is off for that track (logged).
- **The first frames of a hang** stay with the image until it is 8 px or more off (2 × the pose's median noise on a noisy track):
  1 frame over 25 % in the frozen test (worst 29 %).
- **The cover needs the patch to look like itself.** The fit from the body's place matches the anchor's patch; under a hand or in
  deep shadow it is refused and the body's place stands (10-15 px off on the fast clip).
- **A mark the preview did not see** (e.g. turned away on that frame) is decided by its averaged middle; one right on the line can
  go either way, but the same way for the whole video. Keeping the line clear of marks avoids it.
- **The app** (1.2.45) was compiled (the three changed files together, against 1.2.44; no new errors other than the compile
  harness's own missing org.json and Android types), not built with Gradle or run on a phone. The APK Builder does the real build.

## 4. Files

- Space: `belly_clear.py` (pentagon, body coordinates, `BodyPath`, `mark_pieces`, `parse_marks`, whole pieces), `clean_belly.py`
  (`_pentagon_hits`, preview `marks_body`), `phoenix_api_adapter.py` (preview fields, job `belly_pentagon` / `belly_marks`),
  `swap_engine.py` (`BODY_ANCHOR`, `_body_fit`, `anchor_to_body`), `core_pipeline.py` (pose pass, `BodyPath`, anchor), `config.py`.
- App 1.2.45: `HfSettings.kt` (`bellyPentagon`, `bellyMarks`), `PatchRemoval.kt` (pentagon editor), `PhoenixViewModel.kt`
  (preview and job fields), `build.gradle.kts` (versionCode 165), CHANGELOG, builder README.
- Tests: `verification/patchremoval/unit142.py`, `preview142.py`, `e2e142.py`, `video_test.py` (`VT_BODY`, `VT_FREEZE`, `VT_DROP`,
  `VT_WINDOW`), `make_bellytattoo.py` (`motion=fast`).
