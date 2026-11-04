# SDOS-111 — BellyPreview: belly clean on the Patch Removal frame screen, with a preview and the band's length and breadth

**Pack:** v11.2.111 "BellyPreview" (v11.2.110 + this). Engine `aequus-1.2.111-bellypreview`.
**App 1.2.31** (APK Builder 1.2.31).

**Asked:** "Add an option of preview frames in belly clean, and extend it (to enhance or reduce length and breadth). Add this
(belly clean) as an option in the frame screen of Patch Removal."

All tests use synthetic skin; no user footage.

---

## 1. In the app (1.2.31)

The Patch Removal screen ("Patch Removal / Belly clean" on the main screen) has two modes at the top: **Paint a mark** (as before)
and **Belly clean**. In Belly clean:
- choose the frame with the same slider and frame steps;
- **Belly clean for this video**: on / off (the same setting as the switch in the job options);
- **Length** and **Breadth** of the band, **50-150 %** in 5 % steps (100 % = the band found from the body); **Reset 100 %**;
- **Preview belly**: the frame goes to the Space and comes back cleaned, **with the band outlined**; **Show before / after**,
  **Hide / Show outline**; a line says how many marks were removed (and that the navel was left alone), or that no body was found
  on that frame (the shoulders and hips must be visible, a front view);
- **Use for this video** turns it on from the same screen; the main screen then shows "Belly clean: on (length .. %, breadth .. %)".
- The job sends `belly_length` and `belly_width` with `belly_clear`.

## 2. On the Space (v11.2.111)

- **`phoenix_belly_preview`** (new API route): a frame + settings JSON (`belly_length`, `belly_width`, optional `belly_smooth`)
  -> the frame cleaned, the same with the band outlined, and JSON (`body_found`, `reason`, `band` corners, `marks_px`, `pieces`,
  `navel_left`, `clothing_px`, `length`, `width`, `smooth`, `ms`). The same steps as the job on that frame: the body's pose (MediaPipe,
  still-image mode, kept loaded between previews: about 0.5 s the first time, 14 ms after), the band, its marks removed with Patch
  Removal's still fill (LaMa), the skin evened.
- **Length and breadth**: the band's height is scaled about its middle and its breadth about the body's middle line, x0.5-x1.5
  (values outside are brought into range, junk means x1). Whatever the setting, the band never reaches higher than 0.40 of the
  way from the hips to the shoulders, nor lower than just below the hip line (the lowest part of the waist).
- The job takes `belly_length` / `belly_width` (job log: `BELLY-CLEAR: ... band length x1.30, breadth x0.80`). Each job and each
  preview keeps its own length, breadth and strength: a preview never changes a job that is running.
- **Found by testing the new range:** with the band narrowed or lengthened, the band's outline can cut a waistband so that what is
  left of it inside the band is surrounded by skin, and the waistband was then taken for a mark. Now a thick non-skin piece that
  touches the band's edge is clothing (thin tattoo strokes cut by the edge still count as marks). Over every length and breadth
  from x0.5 to x1.5 (25 settings): the waistband is never taken for a mark, and 100 % of the marks inside the band are found.
  At its default size v11.2.110 was not affected (all its checks pass on the same scene).

## 3. Verification (from the zip's own files)

- `unit111.py` 22/22: length and breadth scaling and their limits; out-of-range values; marks only inside a narrowed band; the
  25-setting sweep (waistband never a mark, marks found); the preview (marks removed to 3.8 % of their contrast, the outlined
  picture differs only on the outline, info complete, no body -> unchanged with the reason); a preview at another strength does not
  change a running job's; the API route (both pictures and ok info, the length / breadth sent are used, junk brought into range,
  an unreadable frame, no pose model, no body); job settings parsed and limited.
- `unit110.py` 26/26 (its waistband now runs on past the band's sides, as real clothing does).
- Earlier suites on the pack: `unit105` 14/14, `unit104` 12/12, `unit103` 12/12, `unit102` 28/28, `unit101` 21/21, `unit98` 11/11,
  `unit97` 18/18, `unit96` 18/18.
- MediaPipe in still-image mode (Space versions): loads, runs, no body on a frame without a person.
- App 1.2.31: compile check 0 errors (kotlinc 2.0.21 + Compose plugin; the new screen, view model and API classes are in the
  output); the builder's static audit 44/44. **Not run on a phone.**

**Full jobs from the extracted zip** (the real render path; this test machine has no MediaPipe, unlike the Space):
- hard mole with a painted mark, `belly_clear` on, length x1.3, breadth x0.8: `BELLY-CLEAR: the pose model could not be loaded (...) - not
  applied`, the painted mark removed as before (150 of 150 frames, 3.5 % left, 0 missed), the job completes;
- the kiss with the detector blinded for 40 frames: both faces painted 120/120, `REVERTS=0`, `FACE-EVENTS none` (unchanged).
- Not run: a full job or a preview with MediaPipe finding a real body (no test footage of a body here).

## 4. Limits

- As in SDOS-110: not tested on a real body here (no test footage of one); one person; front views.
- The preview shows one frame. The video is cleaned with the same steps on its key frames and carried with the body; check the
  `BELLY-CLEAR` and `PATCH-EVENTS` lines of the job.
- A thick, filled tattoo that the band's edge cuts is taken for clothing on that side and stays there; lengthen or widen the band
  so the whole tattoo is inside it.

## 5. Install

- **Space:** upload the 9 files (with `belly_clear.py`). Log: `Phoenix v11.2.111`, `swap_engine aequus-1.2.111-bellypreview`.
- **App:** upload the 6 files of `Phoenix_APK_Builder_1.2.31.zip` to the APK Builder Space and build; install over 1.2.30.
