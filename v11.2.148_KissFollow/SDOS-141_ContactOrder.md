# SDOS-141 — ContactOrder: faces in contact stay bound, separated and painted

**Pack:** v11.2.141 "ContactOrder" (v11.2.140 BellyTattoo + this). Engine `aequus-1.2.141-contactorder`. **App:** 1.2.44, no change
(its M / F letter under each Detect face is now used, see 2.2).

**Asked:** "Do a deep research and refine the face occlusion logic, e.g. kissing, hugging, which keeps the replacement bound to the
face and doesn't revert or flicker or disfigure. Apply all possible logic of separating faces: no revert to the original, skin colour
different, male-female different, but faces need to be separated cut to cut, no mix-up."

**Research:** `reports/Face swap contact identity binding.md` (notes in `research_notes/Face swap contact identity binding/`), built on
the earlier `reports/Occlusion handling in face swap video.md`. All tests use synthetic clips made from InsightFace's public sample
photo; no user footage.

---

## 1. What was measured first (v11.2.140)

**The test bench** (`verification/fastrestore/`), with the real engine, detector, landmarks, swapper and occluder on 2 CPUs:

| Clip | What happens | New? |
|---|---|---|
| smooch, deep, hide, flip | the v11.2.95-99 kiss clips: faces pressed together, A behind B (hide: A only a sliver for 40 frames; flip: the apparent sizes swap) | |
| **cross** | A walks behind B, both carry on: **they end on the other side** (an identity mix-up is caught here) | new, `make_cross.py` |
| **hug** | cheek to cheek, the pair turns half a circle (B stays in front): **sides swapped at the end** | new, `make_cross.py` |

**The scores** (`identity_check.py`, from the finished video and the engine's own paste mask):

- **identity:** each person's face is cut from the output at the person's TRUE 5 points, only their own visible pixels kept, and
  compared by ArcFace with both replacement photos and with their original face. *Mix-up* = the other person's replacement wins by
  0.05. Scored on frames where 70 % or more of the face is visible (ArcFace on a sliver means nothing).
- **painting:** the engine's paste mask over the person's true visible face (the 106-point outline moved into each frame by the
  clip's exact transform), against the same person's fully visible frames. *Revert* = under 40 % of that, *weak* = 40-70 %.

  Two simpler scores were tried first and are not used: a colour-change test (a replacement whose skin tone matches the original
  shows little change, so a fully painted cheek read as "weak"), and ArcFace on the whole crop (half of it was the other person's face).

**v11.2.140:**

| Clip | identity mix-ups | person behind (A): revert / weak frames | person in front (B): revert / weak frames |
|---|---|---|---|
| cross | 0 | 0 / 0 | 0 / **11** (frames 63-72) |
| hug | 0 | 0 / 0 | 0 / **8** (frames 65-72) |
| smooch | 0 | 0 / 0 | 0 / 0 |
| deep | 0 | 0 / 0 | 0 / 0 |
| hide | 0 | 0 / **32** | 0 / 0 |
| flip | 0 | 0 / 0 | 0 / 0 |

- **No mix-up** on any clip, also across the cross and the hug: the identity locks, the crossing lock (no relabel while two faces
  overlap) and the frozen identity during contact hold. The research's extra binding machinery (a frozen snapshot plus an
  embedding bank, summed evidence after separation) has nothing to fix on these clips and was **not** added.
- **No revert** anywhere (the engine's log agrees: `REVERTS=0`).
- **The front person lost paint while the other was hidden behind it** (cross, hug): the hidden person is lost by the detector
  (checked: SCRFD finds no face there at any NMS 0.4-0.6 or score threshold 0.3-0.5 while 20-45 % of it shows), its position is held
  or interpolated, and that guessed position, lying on the front person's face, was ranked **in front**: the one-face-per-pixel rule
  then cut the front person's face where the ghost lay.

## 2. The changes

### 2.1 A face nobody can see is never in front (`core_pipeline._rival_landmarks`)

Each face's geometry on a frame now says whether that frame has a real sighting of it (`seen`: a detection on this very frame, not an
interpolation across a gap, not a contact hold). In the one-face-per-pixel rule a face that is **not seen** is always behind a face
that is: it can no longer cut a visible face. When both are seen, the outline-edge evidence decides as before (v11.2.95).

### 2.2 Male / female from the target person, not the replacement photo (`core_pipeline`, `swap_engine.MultiFaceTracker`)

The tracker places a face that joins late (no Detect-frame reference) in a slot of the same gender. It took that gender from the
**replacement photo**: with a woman's face swapped onto a man (or the other way round) the late-joining person was steered to the wrong
slot. Now:

- the app's **M / F letter** under each Detect face (app 1.2.42+, `settings.person_gender`, tap it if wrong) is the slot's gender;
- without it, the tracker **learns** each person's gender from their own clean frames (identity confirmed, not in contact): 10 votes,
  80 % agreement;
- it stays a soft cue, used only to place a late joiner, **never** a reason to drop a face (InsightFace's gender is per frame and
  unreliable: the research puts single-frame classifiers at 85-96 %, with gaps across groups).

Log: `target people's genders (0=F 1=M): {...} (from the app: [...]) · replacement photos: {...} (not used to place people)` at the
start, `PERSON-GENDER ... learned from votes ...` at the end.

### 2.3 Tried and dropped: following a hidden face by its "visible" points

The research's first recommendation for a face lost in contact: sparse optical flow on its points that are not under the other face,
a robust similarity fit (built and run). **It made things worse:** on the hug the person behind drifted up to **235 px** off its face
(v11.2.140, which interpolates across the gap toward the next real sighting: at most 86 px), and it came back 5-6 frames late. The
points just outside the front person's face outline lie on that person's hair and neck, which move with the front person; a margin
large enough to exclude the whole head leaves almost nothing of the face behind to follow. Not in this pack.

### 2.4 Research items not changed, and why

| Recommendation | Why not now |
|---|---|
| detector NMS 0.4 -> 0.6 / Soft-NMS for two overlapping faces | measured: the hidden face is not found at any setting on these clips; no gain shown |
| frozen snapshot + embedding bank, summed evidence before a re-bind | no mix-up to fix on 6 contact clips (section 1); a change without a failing case is a risk |
| edge-snapped watershed seam between the two faces | the remaining weak case (hide: A a sliver) comes from A's own landmarks being guessed under B, not from the seam |
| skin-colour models to split the two faces | the research: within one face, light varies more than between two people of similar tone; tie-breaker only |
| rigid-point alignment, output quality gates | no disfigured frame found on these clips; kept for a case that shows it |

## 3. Verification (v11.2.141, from the pack's own files)

`unit141.py`: 9 of 9 (the ordering rule with seen / held / contact-held faces and without the new flag; the app's gender values,
JSON and junk; the tracker's gender stores).

**The six contact clips, v11.2.140 -> v11.2.141** (same bench, same scores as section 1):

| Clip | mix-ups | person behind (A): revert / weak | person in front (B): revert / weak | flicker (A / B, x the footage's own) |
|---|---|---|---|---|
| cross | 0 -> 0 | 0 / 0 -> 0 / 0 | 0 / **11 -> 0 / 0** | 1.00 / 1.02 -> 1.00 / 1.00 |
| hug | 0 -> 0 | 0 / 0 -> 0 / 0 | 0 / **8 -> 0 / 0** | 1.01 / 1.03 -> 1.01 / 1.01 |
| smooch | 0 -> 0 | 0 / 0 -> 0 / 0 | 0 / 0 -> 0 / 0 | 1.00 / 0.99 -> 1.00 / 0.99 |
| deep | 0 -> 0 | 0 / 0 -> 0 / 0 | 0 / 0 -> 0 / 0 | 0.99 / 0.99 -> 0.99 / 0.99 |
| hide | 0 -> 0 | 0 / 32 -> 0 / 32 | 0 / 0 -> 0 / 0 | 0.96 / 0.99 -> 0.96 / 0.99 |
| flip | 0 -> 0 | 0 / 0 -> 0 / 0 | 0 / 0 -> 0 / 0 | 1.03 / 1.00 -> 1.03 / 1.00 |

- Identity: on every scoreable frame each person carried their own replacement (ArcFace median 0.74 for the person behind, 0.86-0.87
  for the person in front; the other person's replacement: about 0.00).
- The engine's own log on all six: `FACE PAINTED ON 120/120 FRAMES`, `REVERTS=0`, `FACE-EVENTS none`.
- Gender learned from the people's own frames (no app letters given): both learned **M** on smooch and deep (both target people in
  these clips are men); on hide and flip one person reached 9 of the 10 votes and stayed unknown (no prior is used then). With the
  app's letters (cross, `pg=0,1`) the log shows `from the app: [0, 1]`.
- Speed: unchanged (458-563 s per 120-frame clip on 2 CPUs for both versions).
- The cross and hug runs of v11.2.141 were made just before the end-of-job `PERSON-GENDER` log line was added (a log line only);
  the other four ran on the final files.

## 4. Limits

- **Synthetic clips only**: cut-outs of two faces of a public photo moved over a blurred background. Real kisses have real occlusion
  boundaries, lighting and motion blur; nothing was seen on real footage.
- **A face that is only a sliver** (the hide clip: A 10-20 % visible for 40 frames) is still under-painted on about 30 frames: its
  landmarks are guessed under the other face, so its own outline is wrong there.
- **Gender** is used only to place a person who joins late without a Detect-frame reference; the clips here have no late join, so that
  path was checked by its unit test and the log only.
- **Identity** was not stressed by look-alikes (two similar faces) or by people of very similar build in the same clothes; the clips'
  two people are clearly different.

## 5. Install

- Space: upload the pack's files (the same list as v11.2.140; `core_pipeline.py`, `swap_engine.py`, `phoenix_api_adapter.py`,
  `config.py` changed). No dependency change.
- Built on v11.2.140 (our v11.2.118 line), not on Grok's v11.2.139.
- Log at start: `Phoenix v11.2.141`, `swap_engine aequus-1.2.141-contactorder`.
- App: 1.2.44 as it is. Use **Detect**, check the **M / F** letter under each face and send each face to its person's slot: the job
  then knows each person's gender and identity from the start.
