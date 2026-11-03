# SDOS-103 — Stick: the paint stays on the body through fast moves and turns; a log that says when a face is not swapped

**Pack:** v11.2.103 "Stick". Engine `aequus-1.2.103-stick`. **App: no change** (1.2.29 stays).
Replaces v11.2.102 "SkinPaste" (everything of it is kept).

**Reported:**
1. "The pasted paint moves when the body moves; it must stick to the body."
2. "Enhance identification of the body part."
3. "I still see the face disfigured on extreme side turns, in kisses, and reverting to the original in a few frames."
4. "Please render a professional-grade app."

All tests use synthetic clips (procedural skin, public photos, public benchmark side-view photos); no user footage.

---

## 1. The paint did not stick: measured, then fixed

**Measured.** The v11.2.102 tests moved the body part gently. I made harder clips (`make_patchclip.py OUT fast`): the same turn (up to ±32°), zoom, perspective, bending, shadow and phone-quality encoding, but moving 2.5, 4 and 6 times as fast (peak about 45 / 72 / 108 px per frame, with motion blur). On them v11.2.102 lost the body part for stretches, and on every lost frame **the mark showed through** — which looks like the paint "moving" (it comes and goes) instead of sticking:

| Clip (mole, 150 frames) | v11.2.102: mark left / frames missed | **v11.2.103** |
|---|---|---|
| 45 px/frame peak | 18.9 % / **28** | **1.8 % / 2** |
| 72 px/frame peak | 48.1 % / **73** | **-0.9 % / 0** |
| 108 px/frame peak | 61.4 % / **92** | **0.8 % / 1** |

**Two causes, both in the tracker.**
1. **It looked for the patch only after two lost frames.** When the followed pose and the constant-velocity guess did not fit (the body part was further than the guess, and blurred), the step failed and the search of the whole frame started only on the third try. Now a failed step searches **at once**: first a template match near the predicted pose (`tm_radius` 160 px, turns and scales centred on the prediction), then ORB and a whole-frame template match.
2. **Blur made the right answer look wrong.** A sharp anchor against a smeared frame scores a lower plain correlation although nearly all of its structure lines up. A re-detected pose is now also accepted at plain correlation >= 0.58 when 85 % of the structure is matched and the lighting-free score is >= 0.72 (`blur_ok`).
3. **The smoothing cut the curve.** The pose smoothing (5-frame window) moves a curved path by about half its acceleration times the window's variance: **7-20 px off at the turns of a 72 px/frame swing**, so the paint sat beside the mark there. The window now shrinks wherever the motion bends more than `smooth_err` (1 px); on a steady move with jitter it still smooths (spread 0.51 -> 0.27 px in the test).

**How well it sticks now** (`stick_test.py`: distance between what the engine changed and the mark's true centre, and the frame-to-frame change of that distance — the paint sliding against the body):

| Clip | frames painted | offset mean / p95 | slide mean / p95 |
|---|---|---|---|
| mole, 45 px/frame | 149 / 150 (v102: 123) | 0.56 / 1.08 px | 0.80 / 3.74 px |
| mole, 72 px/frame | 150 / 150 (v102: 82) | 0.56 / 1.18 px | 0.72 / 1.54 px |
| mole, 108 px/frame | 149 / 150 (v102: 58) | 0.58 / 1.23 px | 0.55 / 1.07 px |
| mole, turning and swinging (hard) | 150 / 150 | 0.64 / 1.24 px | 0.52 / 1.07 px |

(The mark is about 21 px across; offsets are the centre of the changed area, a few tenths of a pixel of it is measurement noise.)

**Other clips (v11.2.102 -> v11.2.103):** procedural mole 5.4 % / 6 missed -> **1.8 % / 1**; procedural tattoo 3.8 % / 3 -> **1.5 % / 0**; astronaut mole 6.4 % / 6 -> **2.9 % / 1**; astronaut edge 11.4 % / 6 -> **7.9 % / 1**; the hard clips are unchanged (0 missed on mole, birthmark, tattoo, astronaut; jitter clips 0.5 % and 0.1 %, 0 missed; mole across a cut 2.8 %, 0 missed). Fast tattoos (not measured on v11.2.102): 2.0 % / 0, 3.7 % / 1, 5.5 % / 4 missed at 45 / 72 / 108 px per frame. An object over the mark is still left untouched (0.0 % changed on the fast clips).

**Cost:** planning (which runs once, before the render) is 15-25 % longer on hard footage (hard mole 11.2 -> 12.9 s, fast mole 16.4 -> 20.7 s per 150 frames); the per-frame cost while rendering is unchanged (mole 27 ms, tattoo 117 ms).

**Not a defect:** on the astronaut-edge clip one frame (97) reports 343 px "changed away from the mark". That is the frame where the mark is almost out of the picture (17 px visible); the tracker now follows it there and the painted area reaches beyond the sliver that is left.

## 2. Identification of the body part

This is the finder of v11.2.102 (the mark is found by what it is on every frame) on top of the tracker above; the tracker now keeps the pose through fast and blurred moves, so the finder works from a right place instead of a guess. No new model was added. A face-landmark anchor for marks on the face was considered; it needs the face detector to run before the planning step (the planning runs on the footage alone), and is not in this build.

## 3. Faces on extreme side turns, in kisses, and reverting

**Not reproduced.** I ran the pack on 12 synthetic runs: 5 kisses (cheek, near, deep, hide, flip), 4 side-view clips of public benchmark photographs (2 photos, slow and fast sweep), 3 fast turn clips (sharp, mixed, calm). On all 12:
- the face is painted on **every frame** (120/120, 60/60, 70/70);
- `REVERTS=0`, `blind=0`, `faint=0`;
- the side-view outline mask was used on the side-view clips (46-60 face-frames); the kiss hold engaged on the contact clips (`contact_hold` 72-216).

So the disfigured or reverting frames you see are not produced by what my clips contain. I cannot look at your footage, so this build adds the measurement that was missing:

**New job log lines `FACE-EVENTS`** (text only, no picture): per person, the time intervals where
- the tracker believes the person is there but **nothing is painted** (the original face shows): `NOT swapped`;
- the face is painted at **under half strength**: `faint`;
- the face is painted only on a **predicted geometry** for 8 frames or more (no real detection; a bad pose can disfigure here): `painted on a predicted geometry only`.

Example: `FACE-EVENTS person 1: 1.0-1.3 s NOT swapped, 9 fr; 2.0-2.1 s faint, 4 fr; 2.7-3.1 s painted on a predicted geometry only, 12 fr`.
When there is nothing to report the line says so. **Please send me those lines** (with the `Done in ...` line that has `REVERTS=...`) from a job where you see a revert or a disfigured face, and tell me at which time of the video it happens. With the seconds I can say which mechanism let the frame through.

## 4. The app

No app change in this build. "Professional-grade" is wider than a fix; see the message that came with this pack for what I need to know.

## 5. Limits

- Synthetic tests only. A body part that changes shape (a fold, a muscle) or moves faster than about 110 px per frame is not guaranteed to be followed; the fast tattoo clip at 108 px/frame still misses 4 of 150 frames.
- The face problems of section 3 are not fixed in this build, only made measurable.
- Planning is a little slower on hard footage (section 1).

## 6. Verification (from the zip's own files)

- `unit103.py`: smoothing keeps a 72 px/frame swing within 2.5 px and still removes jitter; the local template search finds a patch 40 px from the prediction and nothing when told to look far away; a fast blurred move with turn and zoom is followed within 4 px on 70 of 70 frames (the old way: 52 of 70); the FACE-EVENTS codes and lines.
- Earlier suites on the pack: `unit102`, `unit101` 21/21, `unit98` 11/11, `unit97` 18/18, `unit96` 18/18.
- Clip suite above; `stick_test.py` (new) and the fast clips (`make_patchclip.py OUT fast`, new).
- Face runs: 12 synthetic runs (section 3). The first attempt ran two jobs at once under the same job id and overwrote each other's files (two jobs came out empty); it was discarded and all 12 were rerun with unique ids.

**Full jobs from the extracted zip** (the real render path, final encode, 720p, Balanced; scored against the clean clips on the mark):

| Job | Result |
|---|---|
| hard mole | removed on 150 of 150 frames; 3.5 % of the mark left after the final encode; missed on 0 |
| mole at 72 px/frame | removed on 150 of 150; 4.5 % left; missed on 1 (before: 73 missed of 150 in the engine test) |
| tattoo at 72 px/frame | removed on 150 of 150; 6.0 % left; missed on 0; an object left untouched on 31 frames |
| hard astronaut mole + a face swap in the same job | faces painted on 150 of 150; mark removed on 150; -10.5 % left (a little lighter than clean); missed on 0 |
| 10 marks on one frame | all 10 followed; removed on **1496 of 1500** frames (v11.2.102: 1420); 3.2 % left; missed on 0 |
| kiss, the detector made blind for 40 frames of contact | both faces painted on 120 of 120 frames; `REVERTS=0 faint=0`; `FACE-EVENTS none` |

(The mole, tattoo and 10-mark clips contain no face, so those jobs say "no faces were swapped"; Patch Removal ran as normal.)

## 7. Install

- **Space:** upload the 8 files (or "Push to all Spaces", which skips busy Spaces). Log: `Phoenix v11.2.103`, `swap_engine aequus-1.2.103-stick`.
- **App:** nothing to install; 1.2.29 works with this Space.
- Send the `FACE-EVENTS` lines and the `PATCH-REMOVAL:` line of a job where something still looks wrong (text only).
