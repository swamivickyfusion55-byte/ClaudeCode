# SDOS-207 — SteadyID: no frame of the other person's face in a kiss; skin tone models load; clearer identity log

**Pack:** v11.2.207, which is v11.2.206 with the changes below. Engine `aequus-1.2.207-steadyid`. Masks, occlusion (SDOS-206),
skin tone and the job flow are as in 11.2.206.

**Asked:** "In few jobs the swap was not successful, could you please check and in one frame in a video, the face was affixed
on opposite gender" (two Space logs, v11.2.206). Only the logs were read; no footage.

All tests use synthetic clips made from insightface's public sample photo. No user footage was looked at.

---

## 1. What the logs show

| Job | Painted | Detect pictures | What happened |
|---|---|---|---|
| 316DB9E5 (part 4) | 2892 of 2892 frames | persons 1 and 2 | Fine. Person 2 unpainted on 392 frames, in 4 stretches. |
| C77E081B (part 4) | 1981 of 2343 (85 %) | persons 1 and 2 | Person 2 unpainted on frames 0-971 (the first 32 s) and 1500-1799. |
| 3717965D (part 4, the same video as C77E081B) | 2063 of 2343 (88 %) | none | Person 2 unpainted only on frames 0-84 (up to the first shot cut). |
| 6332D118 (part 10) | 1206 of 2635 (46 %) | person 1 only | 4946 faces refused by the identity check (2524 for person 1, 2422 for person 2), almost all because they looked like neither person's picture or first sighting (similarity under 0.25). Skin tone on, but its models did not load. |

- **The unswapped stretches are the identity check refusing faces**, not the occlusion change of 11.2.206 (that only changes
  the mask of a face already being swapped).
- **C77E081B vs 3717965D is the clearest case.** It is the same video. With a Detect picture, person 2 could only be found by
  that picture. It did not match how they looked in this part, so they stayed original for 32 seconds. Without pictures,
  person 2 was found by elimination at the first shot cut.
- **6332D118:** the log alone cannot tell whether the refused faces were your two people looking very different (another
  scene, light, distance, profile) or other people. 11.2.207 now says it plainly at the end of the job (§2.3).
- **The one frame with the other gender's face** matches a defect measured in 11.2.205/206 on the synthetic smooch clip:
  one frame where the face behind is painted over the face in front (§2.1).

## 2. Changes

### 2.1 Two faces touching: the face going behind is put behind in time (`core_pipeline._front_relations`)

When two faces first touch with boxes of nearly the same size, 11.2.176's rule keeps its first front/behind choice until one
face is 10 % bigger, or until the face in front has lost 8 % for 2 frames.

In the smooch clip, the face going behind shrank steadily (ratios 1.04, 1.01, 0.93) but was still "in front" on frame 34.
It was painted over the other face, and the other person's replacement showed on that face for one frame. Now the choice
switches at once when the face in front has shrunk on each of the last two frames and lost 5 %. Detector jitter goes up and
down and never shrinks twice in a row (unit check: ±6 % jitter, 0 changes). `FRONT_TREND = False` gives 11.2.206.

### 2.2 Skin tone models load on the Space (`packages.txt`)

The Space log shows `MediaPipe skin segmenter unavailable (OSError: libGLESv2.so.2: cannot open shared object file)`. The
skin tone then fell back to the face-colour method.

The Space installs the newest MediaPipe 0.10.x (0.10.3x). It needs the system GL libraries, which the Space did not have. The
0.10.21 I tested with carries its own copies, which is why the tests passed. `packages.txt` now adds `libgles2` and `libegl1`.

Checked: MediaPipe 0.10.35 with these libraries gives the same skin and person masks as 0.10.21, to 0.0000
(`verification/skintone/mp_versions.py`, the sample photo).

### 2.3 A plain warning when the identity check refuses a person (`core_pipeline`, end of job)

When a person was refused at least 60 times and more often than painted, the log now says:

`IDENTITY: person N was refused X times (painted on Y frames): the faces found did not look like person N's Detect picture or first sighting ... If those faces ARE person N, capture person N in Detect from this part, or run it without Detect pictures. If they are other people, leaving them original is correct.`

### 2.4 In the pack, switched off: a Detect picture that never matches no longer blocks (`swap_engine` tracker)

**Planned behaviour:** when a person has a Detect picture that matches no face for `trk_ref_stale_calls` detector calls in a
row, they are found by elimination (gender first) as without a picture. That only applies while a face nobody else has is
on screen, and there are no more faces than people to swap. Another person's picture must not claim the face (0.30). This is
the C77E081B case.

It is **off in this pack** (`trk_ref_stale_calls = 0`, exactly 11.2.206) until its job test is done (harsh-step clip: a
wrong picture for person 2, then the right pictures, 11.2.206 vs 11.2.207). It will be sent with the results.

## 3. Verification

Contact clips, Ultra, the full job, scored against ground truth (`identity_check.py`, `contact_lag.py`):

| Clip | | Identity correct / unclear | Mix-ups | Reverts | Weak frames |
|---|---|---|---|---|---|
| Smooch | 11.2.206 | 168 / 2 | **1** (frame 34) | 0 | 45 |
| | 11.2.207 | **170 / 1** | **0** | 0 | 44 |
| Crossing | 11.2.206 | 192 / 4 | 0 | 0 | 12 |
| | 11.2.207 | **193 / 3** | 0 | 0 | **10** |

Placement and flicker are identical on both clips.

**Not yet run on 11.2.207:** the harsh-step, side-view kiss and hide clips. They follow with the §2.4 test.

**Unit checks on this pack:** `unit176` 12, `unit176_pipe` 14, `unit203` 18, `unit204` 7, `unit205` 30, all pass.
`unit206` passes on the eye and cup clips: the occlusion results are unchanged.

## 4. Limits

1. The identity refusals of 6332D118 are **not fixed by this pack**. Whether those faces were your people cannot be told from
   the log. The new warning says when it happens and what to try.
2. The front/behind fix was measured on synthetic clips. Your frame with the other gender's face may have another cause.
   If it happens again, the job number and the second it happens at, plus the `FRONT/BEHIND` line, will tell.
3. §2.4 is off until tested.

## 5. Install

Upload the pack's files to the Space. Changed:
- `packages.txt`: `libgles2`, `libegl1` (the Space rebuilds its system packages);
- `core_pipeline.py`, `swap_engine.py`, `config.py`.

The log at start reads `Phoenix v11.2.207`. With a skin tone job the end-of-job line should now read
`SKIN-TONE: skin: MediaPipe multiclass selfie segmenter; person: MediaPipe MagicTouch`.
