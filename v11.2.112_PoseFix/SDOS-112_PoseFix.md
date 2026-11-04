# SDOS-112 — PoseFix: belly clean works on the Space's MediaPipe (0.10.35)

**Pack:** v11.2.112 "PoseFix" (v11.2.111 + this). Engine `aequus-1.2.112-posefix`. **App:** no change (1.2.31).

**Reported:** "BELLY CLEAN DID NOT WORK". Job log (v11.2.110):
`BELLY-CLEAR: the pose model could not be loaded (AttributeError: module 'mediapipe' has no attribute 'solutions') - not applied`,
then `BELLY-CLEAR: even skin applied on 0 frames`.

All tests use synthetic media, plus MediaPipe's own pose test picture; no user footage.

---

## 1. Cause

- `requirements.txt` had `mediapipe>=0.10.14,<0.11`. The Space resolves that to **MediaPipe 0.10.35** (checked: the same wheel
  downloads here for Python 3.10).
- From 0.10.30, MediaPipe **no longer ships the legacy `mp.solutions` API**. `belly_clear.py` (v11.2.110-111) used
  `mp.solutions.pose`, so the pose model never loaded and belly clean switched itself off on every job and preview.
- My tests had used MediaPipe 0.10.14, which still has `mp.solutions`; the Space's own version was never run. That gap is closed
  now: this pack is tested on 0.10.35 itself.
- 0.10.35's native library also needs two system libraries the Space did not install: `libGLESv2.so.2` and `libEGL.so.1`
  (without them it fails with `OSError: libGLESv2.so.2: cannot open shared object file`).

## 2. The fix

- **`belly_clear.py`:** the pose model now runs through MediaPipe's **Tasks API** (`PoseLandmarker`). It is the same BlazePose
  "full" network as before: on the same picture both versions give the same shoulders and hips, to the pixel.
  - Video mode for the job's frame pass (timestamps from the frame numbers); still-image mode for the app's preview.
  - Where an installed MediaPipe still has `mp.solutions` (older versions), the legacy model is used as before (tested on 0.10.14).
  - **The model file** (`pose_landmarker_full.task`, 9.4 MB, Apache-2.0) is downloaded once from Google's MediaPipe model
    storage into the first writable model folder (`/data/phoenix_models` on the Space, next to LaMa and XSeg). The size and
    sha256 are checked; a broken or wrong download is refused and fetched again next time. If the Space has no internet, upload
    `pose_landmarker_full.task` into the Space's own folder and it is used from there.
- **`packages.txt`:** adds `libgles2` and `libegl1` (no GPU is needed; MediaPipe only has to load them). Without them the job log
  says which two packages to add.
- **`requirements.txt`:** `mediapipe==0.10.35`, the version the Space already resolves to and the one tested here. Nothing
  else changes.

**Found while testing it on 0.10.35, also fixed:**
1. **Head out of the frame.** MediaPipe finds a person by the face. With the head cut off, it still returns shoulders and hips,
   with full confidence, but squeezed together: on the test picture cropped below the chin, the hips came out 40 % of their real
   width. The band would have been narrow and in the wrong place. Now a pose whose hips are narrower than 0.20 of the torso's
   length counts as a side view (or a headless guess) and gives **no band** on that frame. A real front view is about 0.4.
2. **Left and right swapped.** From one frame to the next MediaPipe can take a front view for a back view. A blend between two
   such samples squeezes the band to nothing on the frames between them. The side order is now made the same on every frame.
3. **One-frame jumps.** A sampled pose that jumps more than half a hip width away from the poses around it is dropped. A body
   moving fast and steadily keeps every sample.

**Found by the first run of a body through the whole chain, also fixed** (these were in v11.2.110-111 too; until now no body had
reached them):

4. **Black clothing at the band's bottom taken for skin.** The skin model's first colour gate let neutral black or grey (YCrCb
   Cr about 128) through. With black shorts filling the lower part of the band, the model took them for skin, and the shorts'
   top for dark marks on it. That is 6,200 px of the shorts marked on the test frame, and the paint then could not be followed
   through the clip. The gate now starts at Cr 133, where the usual skin range starts. The shorts are clothing again
   (7 of 5,030 px marked).
5. **Tattoo strokes running into the shorts.** Thin strokes attached to clothing count as ink, but the whole attached piece was
   skipped when any part of it touched the band's edge, and the shorts' ragged top edge always does. Now only the slivers on the
   band's own edge (6 px) are cut off, and each remaining thin piece is judged by itself.
6. **A tattoo taken for the navel.** A V of strokes at the band's top middle passed the navel test, which only checked size and
   shape. The navel must now also be a filled spot (at least 40 % of its box).
   After 4-6: 99 % of the tattoo inside the band is found (it was 41 % with only fix 4).
7. **The mark lost in the video.** The key frames' fill is carried through the video by Patch Removal's tracker. On the moving
   test body the tracker lost the tattoo's strokes on 24 of 90 frames ("held position, mark not seen") and the tattoo showed
   there. Now, on every frame, the even-skin pass first looks for marks again inside that frame's band and fills any it finds
   (OpenCV's Telea fill, a few ms). Thin strokes and small marks come out clean; a large filled tattoo comes out softer there
   than on the key frames. Log: `BELLY-CLEAR: even skin applied on F frames; marks still showing filled frame by frame on C frames`.

**Job log:**
- `BELLY-CLEAR pose model: MediaPipe 0.10.35 (tasks, pose_landmarker_full.task)`
- the `body found on X of Y sampled frames ...` line now ends with `pose MediaPipe 0.10.35 (tasks, ...)`
- when the body was found on fewer than half the sampled frames:
  `BELLY-CLEAR: the body (both shoulders and both hips, a front view, the head in the frame) was not found on N of M sampled frames`
- `BELLY-CLEAR: K sampled pose(s) dropped - the body jumped ...` when that happens.

**Preview:** the JSON adds `"pose"` (the backend). With the head out of the frame the reason is
`shoulders and hips not clear enough (a side view, or the head out of the frame)`.

## 3. Verification

**On MediaPipe 0.10.35 itself** (`unit112.py` 33/33, from the pack's own files; the test body is MediaPipe's own test picture
with the tank top replaced by synthetic skin, a navel and a blue-black tattoo below it, black shorts left as they are;
`make_bellyreal.py`):

| Check | Result |
|---|---|
| backend on 0.10.35 | Tasks; the model downloaded, size and sha256 checked; reused on the next load; a broken file re-downloaded; a bad download refused |
| MediaPipe's test picture (a person standing, front view) | shoulders and hips within 1.1 px of reference; the band found; an empty picture gives no body |
| moving clip with the head in the frame (90 frames: shift, zoom, turn) | body on 30/30 sampled frames in both modes; median error 8 px, worst 15-18 px (hip width 117 px); band on 30/30 |
| same clip, head pixelated (MediaPipe unsure), the job's path | band on 26/30 frames, none far off the body |
| head out of the frame | no band, rather than a squeezed one (hip width / torso 0.17) |
| left/right swapped; one-frame jump | same order restored, the band keeps its breadth between them; the jump dropped, steady fast motion kept |
| missing `libgles2` / `libegl1` | no pose; the error names the packages to add |
| older MediaPipe with `mp.solutions` | the legacy model (also on a real 0.10.14: the same landmarks as Tasks) |
| no MediaPipe at all | no pose, the reason kept, the job goes on |
| black shorts, tattoo strokes running into them, a V of strokes at the band's top middle | shorts under 0.2 % marked; 99.4 % of the tattoo inside the band found; the V not taken for the navel |
| per-frame clean-up alone | the tattoo inside the band filled to 10.6 % of its contrast; nothing outside the band changed; clean skin: 0 px changed |
| preview with the real pose model and the LaMa fill (textured skin) | body found; the tattoo removed to **1.9 %** of its contrast (8 px or more inside the band), **2.6 %** at Length 130 %. Nothing changed more than 3.2 px outside the band. A preview takes about 14 s on this machine, mostly the LaMa fill |

Where a mark crosses the band's outline, the fill's soft edge can reach a few px past it (1-3 px measured). SDOS-110's
"0 px outside" was measured with a stand-in fill that has a hard edge.

Earlier suites on this pack: `unit110` 26/26 (its stand-in pose takes the new timestamp argument; rerun after fix 7),
`unit111` 22/22 (before fix 7),
`unit105` 14/14, `unit104` 12/12, `unit103` 12/12, `unit102` 28/28, `unit101` 21/21, `unit98` 11/11, `unit97` 18/18, `unit96` 18/18.

**Full jobs from the extracted zip** (the real render path, MediaPipe 0.10.35). The moving test body: 90 frames, textured skin,
the face never swapped (detection blanked in the test). Tattoo contrast left in the finished video:

| | before fix 7 | with fix 7 (this pack) |
|---|---|---|
| default band | mean 47.8 %, 40 frames over 25 % | **mean 12.0 %** (10.5 % inside the band), worst frame 35 %, 13 frames over 25 % |
| Length 130 % (whole tattoo inside) | mean 23.4 %, 14 frames over 25 % | **mean 5.6 %**, worst frame 16 %, **0 frames over 25 %** |

- Band flicker went down with fix 7 (95th percentile 13.6 -> 4.9 at Length 130 %).
- The job log: `BELLY-CLEAR pose model: MediaPipe 0.10.35 (tasks, ...)`, `body found on 30 of 30 sampled frames`, marks found on
  2 of 2 key frames, black shorts left alone.
- Painted mark (hard mole clip) with belly clean on, Length 130 %, Breadth 80 %: no body in that clip (`not found on 50 of 50`,
  correct), the painted mark removed on 150 of 150 frames.
- The kiss with the detector blinded for 40 frames: 120/120, `REVERTS=0`, `FACE-EVENTS none` (unchanged).
- Not run: footage of a real bare torso.

## 4. Limits

- **The head must be in the frame**, with both shoulders and both hips, in a front view. That is how MediaPipe finds the body.
  Where it is not, that frame gets no band, and the log's "not found on N of M" line says how often.
- Not tested on a real bare torso: no such footage here. The test body is MediaPipe's own test picture with synthetic skin
  painted on.
- One person (the most prominent).
- A tattoo cut by the band's outline keeps its part outside, and the fill next to the cut can carry a faint shadow of it.
  The preview's outline shows where the band is: raise Length or Breadth until the whole tattoo is inside.
- The skin model needs skin to have some warmth (Cr from 133). Very pale skin under a strongly blue light may fall outside it;
  the log's `skin ... px` on the key frames shows how much of the band was taken for skin.
- The app's no-body message (1.2.31) does not mention the head. The rule above applies anyway.

## 5. Install

- **Space:** upload the same **9 files** as before (with `belly_clear.py`). `packages.txt` and `requirements.txt` change, so the
  Space rebuilds; wait for the rebuild before running a job.
- Log at start: `Phoenix v11.2.112`, `swap_engine aequus-1.2.112-posefix`.
- On the first belly job or preview: `BELLY-CLEAR pose model: MediaPipe 0.10.35 (tasks, pose_landmarker_full.task)`.
- App: no change (1.2.31).
