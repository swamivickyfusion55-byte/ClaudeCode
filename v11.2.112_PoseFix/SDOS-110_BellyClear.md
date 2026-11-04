# SDOS-110 — BellyClear: the lower belly's skin cleared of every mark; review of the "v11.2.109 Abdomen Privacy" pack

**Pack:** v11.2.110 "BellyClear" (built on v11.2.105, not on the reviewed pack). Engine `aequus-1.2.110-bellyclear`.
**App 1.2.30** (APK Builder 1.2.30): a new switch.

**Asked:**
1. "Can we apply the simple logic that all open (uncovered) skin below the navel area, up to the lowest part of the waist, becomes smooth and flawless? I am bothered by any tattoo or mark in that area."
2. "I got it done through ChatGPT, please review." (Phoenix_v11.2.109_Abdomen_Privacy_Updated.zip)
3. "Also enable the front end APK for this (updated APK builder)."
4. "Also see whether the code resolves the face contact issue."

All tests use synthetic skin and synthetic clips; no user footage.

---

## 1. Review of the uploaded pack

It is v11.2.105 plus four new files and edits in four. What it does, and what I found when I ran it:

| Part | What it does | Finding |
|---|---|---|
| `abdomen_smoothing.py` | MediaPipe Pose -> a box from 58 % to 94 % of the way from the shoulders to the hips; an edge-preserving blur, mixed in at 0.35 (at most 0.55), only on skin-coloured pixels | **It does not remove a tattoo.** The skin-colour gate leaves out exactly the ink (tattoo pixels are not skin-coloured), and a light blur does not hide a mole. Its own README says "not tattoo erasure". The box starts above the navel (the navel sits about 73 % of the way down). |
| finishing passes in `core_pipeline.py` | `video_privacy_postprocess.process_video` and then `privacy_postprocess.finish_video` | **Both run** when abdomen smoothing is on: the belly is blurred twice and the finished video is decoded and encoded two more times (an `mp4v` intermediate, then x264): two extra losses of quality and two extra passes of time. |
| adapter defaults | `abdomen_smoothing: True` | **On for every job** unless the app says otherwise (its README says once "default true", once "opt-in"). |
| "face contact" | `contact_anonymise`: blur faces that touch; `face_anonymize`: pixelate all faces | **It does not fix the face swap in a kiss: it blurs the faces** (its README: "This does not fix or modify face swapping"). `contact_anonymise` defaults to true in the adapter but is never passed into the job, so it never runs; `face_anonymize` pixelates every face when sent. |
| v11.2.106 "local deform" (`swap_engine.py`) | places the patch with a second, mark-centred pose | **The patch moves more on a big tattoo**, because those poses skip the smoothing: see the table below. |
| v11.2.108 caches / queues | a decoded-frame cache for planning; frame queues sized by memory | Not measured; the queue change cuts the frame queues to about 10 frames at 1080p (from 48-240). Left out. |

**The patch moving: v11.2.105 against the uploaded engine** (hard tattoo clip: turns, zoom, perspective, bending, shadow; `stick_test.py`):

| | v11.2.105 | uploaded (v11.2.109) |
|---|---|---|
| paint sliding against the body, mean / 95th percentile | 0.76 / 1.76 px | **1.33 / 4.21 px** |
| wobble beyond the body's own motion | 0.20 px | **0.67 px** |
| size flicker of the painted area | 1.3 % | **2.8 %** |
| shimmer on the mark (video test) | 1.23 L | 1.66 L |

On the small moles it is the same as v11.2.105 (it only changes the pose where enough points lie near the mark).

**So:** none of the uploaded pack is used. This pack is built on v11.2.105, and the lower-belly request is done differently (section 2).

## 2. Lower belly: clear skin

**Setting:** `belly_clear` (job settings; app 1.2.30 switch "Lower belly: clear skin"), **off by default**. Strength of the even-skin pass: `BELLY_SMOOTH = 0.5` in `config.py` (0 = remove the marks only).

**What it does** (`belly_clear.py`, run in the Patch Removal planning pass, before the face swap sees a frame):
1. **Where.** MediaPipe Pose (already in `requirements.txt`) on every 3rd frame gives the shoulders and hips. The band runs from **0.22 of the way up from the hip line towards the shoulders** (the navel sits about 0.27 up: the band starts just below it) **down to a little below the hip line** (the lowest part of the waist), and is widened sideways by 0.3 of the hip width on each side (the hip landmarks lie inside the body's outline). Between pose samples the landmarks are interpolated; no band on frames where the body is not found, or in a side view.
2. **Which pixels.** Only skin: a colour model of this person's skin is fitted on the band. A **mark** is
   - anything not skin-coloured that skin surrounds (ink of any colour: tattoos), including thin strokes that run into the waistband;
   - or skin-coloured but clearly darker than the skin around it (moles, pigment).
   **Clothing** (a waistband, shorts: not surrounded by skin, thick) and **the navel** (a compact dark spot at the middle of the band's top) are left alone. Each mark is grown by 4 px into the skin (soft ink edges), never into the clothing.
3. **Removal.** On up to 8 key frames (at least 1.5 s apart, spread over the video) the marks found become **positions of one Patch Removal mark**: filled once (LaMa), then carried with the body and relit on every frame, exactly like a mark painted by hand - with everything of v11.2.102-105 (finding the mark on every frame, opaque skin paste, fast moves).
4. **Even skin.** On every frame the band's skin gets an edge-preserving smoothing mixed in at `BELLY_SMOOTH`, fading out inside the band's edges. Nothing outside the band changes; clothing is not smoothed.

**Job log:** `BELLY-CLEAR key frame ...: band ... px, skin ... px, marks ... px in N piece(s), clothing ... px left alone, navel left alone`; `BELLY-CLEAR: body found on X of Y sampled frames; marks found on K of N key frames (removed as PATCH-EVENTS mark M); even skin 0.50`; `BELLY-CLEAR: even skin applied on F frames`; and the `PATCH-EVENTS` line of that mark.

**Measured** (synthetic belly: procedural skin, a blue-black tattoo of 5 strokes and a ring below the navel, one stroke reaching into a denim waistband, a mole by the waistband, the navel, a mole above the navel outside the band; the body's landmarks given):

| Check | Result |
|---|---|
| tattoo and mole pixels found | **100 %** |
| waistband taken for a mark | 0.16 % of it |
| plain skin taken for a mark | 0.00 % of the band |
| navel inside the band's top | left alone (0 % covered) |
| anything outside the band | 0 px (the mole above the navel stays) |
| clean skin | 0 px found |
| even skin at 0.5 | fine texture 1.34 -> 0.91 (softened, not wiped); level unchanged; clothing unchanged (0.00); outside the band unchanged |
| a 45-frame clip moving and turning, through the pipeline's planning pass (stand-in inpainter) | marks removed to **8.7 %** of their contrast on average (worst frame 9.1 %); waistband not painted (0.00 %); 0 px changed outside the band; even skin on all 45 frames |
| no pose model available | nothing applied, the log says so, the job continues |
| MediaPipe Pose itself (Space versions: numpy 1.26, OpenCV 4.10, MediaPipe 0.10.14) | loads and runs, 17 ms per sampled frame; no person on an empty frame |

**Full jobs from the extracted zip** (the real render path; this test machine has no MediaPipe installed, unlike the Space):
- hard mole with a painted mark and `belly_clear` on: the log says `BELLY-CLEAR: the pose model could not be loaded (...) - not applied`, the painted mark is removed as before (150 of 150 frames, 3.5 % left after the final encode, 0 missed), the job completes;
- `belly_clear` on, no painted mark: the same message, the job completes;
- the kiss with the detector blinded for 40 frames: both faces painted 120/120, `REVERTS=0`, `FACE-EVENTS none` (unchanged).
- Not run: a full job with MediaPipe finding a real body (no test footage of a body here).

**Limits.**
- **Not tested on a real body:** I have no test picture of a real torso, so how well MediaPipe Pose finds the shoulders and hips on your footage is not measured here. Where it does not, nothing is done on that frame (the log's "body found on X of Y" says how often).
- **One person:** MediaPipe Pose follows one person per frame (the most prominent one).
- **Front views:** a side view (hips on top of each other) gets no band.
- The navel's position is estimated from the body's proportions (about 0.27 of the way up from the hips); if the band's top lands on the navel it is recognised and left alone, but a mark right next to the navel may be left too.
- A tattoo running under clothing is removed only where skin shows.
- A mark that the skin model takes for skin (very faint, the same colour and brightness as the skin) stays.
- The even-skin pass changes every frame of the band a little (less fine texture); 0 turns it off.

## 3. Face contact (kisses)

The uploaded pack does not change the face swap; its "contact" mode blurs the faces instead. Nothing new here on contact; v11.2.104's `FACE-EVENTS` lines (in this pack) list, in seconds, when a face was not swapped, faint, or on a predicted geometry only - send them from a job where it happens.

## 4. App 1.2.30

- New switch in the job options, under Motion smoothing: **Lower belly: clear skin** (off by default), with a one-line explanation. It sends `"belly_clear": true`; Spaces older than v11.2.110 ignore it.
- Compile check 0 errors (kotlinc 2.0.21 + Compose plugin), the builder's static audit 44/44. **Not run on a phone.**

## 5. Verification (from the zip's own files)

- `unit110.py` 26/26 (band, marks, navel, clothing, even skin, landmarks, key frames, the moving clip through the planning pass, no pose model).
- Earlier suites on the pack: `unit105` 14/14, `unit104` 12/12, `unit103` 12/12, `unit102` 28/28, `unit101` 21/21, `unit98` 11/11, `unit97` 18/18, `unit96` 18/18.
- API: `belly_clear` absent, `false`, `"off"`, `0` -> off; `true`, `"true"` -> on.

## 6. Install

- **Space:** upload the 9 files (the 8 as before **plus `belly_clear.py`**). Log: `Phoenix v11.2.110`, `swap_engine aequus-1.2.110-bellyclear`.
- **App:** upload the 6 files of `Phoenix_APK_Builder_1.2.30.zip` to the APK Builder Space and build; install over 1.2.29. Turn on **Lower belly: clear skin** for the jobs where you want it.
- Send the `BELLY-CLEAR` and `PATCH-EVENTS` lines (text only) of a job where it does not do what you expect.
