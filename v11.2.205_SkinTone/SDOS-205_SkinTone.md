# SDOS-205 — SkinTone: brighten or darken the chosen people's skin, whole body

**Pack:** v11.2.205, which is v11.2.204 with the skin tone added. Engine `aequus-1.2.205-skintone`. The face swap is unchanged
(Ultra and Optimized exactly as SDOS-204). **App:** Phoenix Mobile 1.2.51 (APK Builder 1.2.51, versionCode 171) adds the step.

**Asked:**
- "Also add a beautiful step and preview brighten or darken the skin tone for whole body"
- "The skin tone need to be done for selected character(s) only."

All tests use synthetic media: insightface's public sample photo, and two-person scenes layered from cut-outs of it. No user
footage was looked at.

---

## 1. What you get

**In the app: step 04 "Skin tone"**, between Replacement faces and Output & quality.
- One row per replacement face, with that person's face, a slider from **Darker** to **Brighter** (a deep-to-light skin
  track, steps of 5 %), and a chip that reads "Natural", "+25 % brighter" or "15 % darker". Tap the chip to go back to Natural.
- **Preview at N %** tones the frame picked in step 01 on the Space, exactly as the video will be done.
- The result shows **Before | After** with a line to drag (or tap). Each toned person is tagged at their face.
- The setting goes with the job. **All natural** resets it. An older Space ignores it.

**On the Space:** every finished frame (after the face swap) is changed for the chosen people only:
- their face (the swapped face included), neck, arms, hands and legs;
- nothing else: not the other people, not clothes or hair, not the background.

With every slider at Natural (the default) nothing runs and nothing changes: Ultra is exactly SDOS-204's Ultra.

## 2. How it decides what is the chosen person's skin

1. **Skin.** Google MediaPipe's multiclass selfie segmenter (classes: background / hair / body skin / face skin / clothes /
   other) runs once per frame, on a square around the chosen people.
2. **Whose body.** MediaPipe's interactive segmenter (MagicTouch) is given a point on each tracked person's face. It returns that
   person — head, body, arms, hands — as one object. One call per tracked person near the chosen ones.
   - Inside a person's own object no pixel can be background, so there skin is weighed against hair and clothes only. This
     catches shaded arms the skin model took for background.
3. **Where two people's objects overlap** (a hug, a hand on the other person), the way from each face decides:
   - the length of the way to the pixel over the picture;
   - background is very costly to cross, and each step between skin and not-skin costs about a face size, as does a visible
     edge where one limb lies on another;
   - a soft body-shaped territory and each person's own skin colour (from their cheeks) weigh in too.
4. **Faces always belong to their own person.** Another person's face is never changed: 0.00 levels in every test below.
5. **The change itself** works on lightness (Lab):
   - **Brighter** lifts the mid-tones and keeps the highlights: +100 % takes mid-grey skin from 128 to 168.
   - **Darker** scales lightness down, so shadows keep their detail, and deepens the colour a little, like a natural tan:
     -100 % takes 128 to 77.
   - The hue is kept (within 3 degrees).
6. **In a video:**
   - The mask is smoothed over time: a still part keeps its mask, a moving part takes the new one at once.
   - It is fitted to the picture's own edges with a guided filter, and reset at a shot cut.
   - A chosen person lost for a few frames keeps their tone for 12 frames, fading, so it never flashes off and on.

**Who is #1:** the person the job puts replacement face #1 on.
- The job matches people by the Detect references; without references, by each replacement face's gender.
- The Preview matches them the same way. The app sends it the references and the replacement faces.
- Only two people of the same gender with no Detect references can be matched differently. The Preview's message then says how
  it matched them, and capturing each person in Detect (step 02) removes the doubt.

**Models:** both download on first use from Google's model store (22 MB, checksums checked) into the Space's model folder.
Without them the step still works, in a weaker way:
- skin is found by the colour of each person's own face skin;
- ownership comes from the way from each face.

The log says which method ran.

## 3. Verification

**Unit checks:** `verification/skintone/unit205.py`, 30 of 30 pass. They cover:
- the setting (the list, 1-based dicts, out-of-range values, junk);
- the tone curve (direction, black and white kept, no banding, hue kept);
- the colour-only fallback (the chosen person's face and arm change; the other person, a red shirt and the wall do not);
- hold and fade, the shot cut, broken records (the frame is returned unchanged);
- the adapter's matching (references, gender, left to right);
- a job resumed after a restart keeps its skin tone.

The earlier suites pass on this pack: `unit176` 12, `unit176_pipe` 14, `unit203` 18, `unit204` 7.

**Six people at a table** (insightface's sample photo, each person chosen in turn, +60 %):

| | Result |
|---|---|
| The chosen person's face | brighter by 16.1 to 18.8 levels |
| The other five faces | 0.00 levels, for every choice |
| Arms | each arm goes to its own person, including an arm reaching down to the table past another person's head |

On this crowded photo two arms that lie against another person are shared, and get part of the change (the next table
measures this exactly).

**Two people layered with known owners** (`make_pair.py`: a man behind a woman, and two women whose arms meet; 48 frames each,
the front person sliding across the back one; `eval_pair.py`, +60 %). The numbers are the share of the full change each part
received:

| Scene, chosen | Chosen person's skin | Other person's skin | Chosen person's clothes / hair | Background | Frame-to-frame |
|---|---|---|---|---|---|
| Man and woman, the man | 0.86 | 0.047 | 0.08 | 0.001 | 0.002 |
| Man and woman, the woman | 0.78 | 0.002 | 0.14 | 0.001 | 0.009 |
| Two women, the one behind | 0.68 | 0.079 | 0.10 | 0.000 | 0.008 |
| Two women, the one in front | 0.71 | 0.027 | 0.12 | 0.001 | 0.018 |
| **Mean** | **0.76** | **0.039** | **0.11** | **0.001** | |

- The other person gets **4 % on average**. Almost all of it is the one arm where the front woman's hand reaches across and
  touches the other woman's hand. On the closest frame that arm gets about a fifth of the change.
- "Chosen person's clothes / hair" is mostly the edge of the reference skin map itself (made at 256 px for the whole photo),
  not visible tint. On the plain test shapes, clothes and wall measure 0.00.

**Through the real Space app** (`app_check205.py`: `app.py` on Gradio 4.44.1; the face-swap models stubbed; MediaPipe and
scikit-image real; called with `gradio_client` as the phone app calls it). On both scenes:
- `phoenix_skin_preview` is listed and answers with the toned frame, the untouched frame and JSON.
- Brighter #1 on the man-and-woman scene: +14.6 levels on his skin, the woman's skin unchanged (0.00).
- Darker #2: -36.4 levels on her skin, his unchanged (-0.02).
- With the replacement faces sent (#1 a woman's face), #1 is the woman: +13.4 levels, the man +0.01. That is how the job paired
  them.
- On the two-women frame of closest contact the other woman changes by 21 % of the chosen one's change, and 6 % for darker.
- A frame with no face gives a clear message.

**The real job** (`bench_pipeline.py`, the man-and-woman scene, 48 frames, Ultra, two faces swapped, 2 threads):

| | Ultra | Ultra + skin tone on person #1 |
|---|---|---|
| Time | 120.4 s | 142.2 s (**+18 %**; 0.35 s a frame) |
| Faces | swapped on 48 of 48 frames | the same |
| Skin | - | person #1 (the woman, as replacement #1 was put on her) +8.1 levels; the man +0.1; background +0.04 |

The 0.35 s a frame is made up of:
- the skin model, 0.12 s;
- the person model, 0.16 s for the two people;
- the "way" step, 0.06 s on the 28 frames where the two overlapped.

**App 1.2.51** (no Android SDK here):
- The new step (`SkinToneStep.kt`) compiles with Kotlin 2.0.21 and its Compose compiler against the Compose 1.7 API.
- The new view-model code compiles together with the app's real `HfApi.kt` and `HfSettings.kt`.
- `MainActivity.kt` gains no new compile error.
- A deliberate mistake was caught each time, so the check is real.
- Not run on a phone: the APK Builder is the real build.

## 4. Limits

1. **Two people's skin touching** (a hand on the other's arm, cheek to cheek outside the face) can get part of the change:
   4 % on average in the tests, about a fifth on the one contested arm. Faces never. Preview first: it shows exactly what the
   video will do.
2. **Cost:** about 0.35 s a frame with two people near each other (+18 % on Ultra in the test). It runs only when a slider is
   off Natural.
3. **Very small people** (a face under about 40 px in a wide shot) are segmented coarsely: the skin model works at 256 px.
4. **The models come from Google's model store on first use.** If the Space cannot reach it, the colour-only method runs and
   the log says so.
5. **Synthetic tests only.** Still photos layered and moved: no real motion blur, changing light or real hugs.
6. **Who #1 is:** two people of the same gender with no Detect references may be matched differently in the Preview and the
   video. Capture each person in Detect.
7. **People without a replacement face** (bystanders) cannot be chosen, and their skin is not changed.

## 5. Install

**Space:** upload the pack's files. New: `skin_tone.py`. Changed:
- `core_pipeline.py`: the frame hook and the setting;
- `phoenix_api_adapter.py`: the setting and `phoenix_skin_preview`;
- `app.py`: the endpoint;
- `config.py`, `swap_engine.py`: version strings.

The log at start reads `Phoenix v11.2.205`. A job with a skin tone logs `SKIN-TONE: person #1 brighter 50%` at the start, and at
the end `SKIN-TONE: skin: MediaPipe multiclass selfie segmenter; person: MediaPipe MagicTouch; N frames, ... ms a frame ...`.

**App:** upload the APK Builder 1.2.51 pack (every file, including `source.zip`). The page title must read "Phoenix Mobile 1.2.51
Builder". Build, then install over 1.2.50.

SDOS-204 in this pack now compares Optimized only with Ultra, with the final numbers.
