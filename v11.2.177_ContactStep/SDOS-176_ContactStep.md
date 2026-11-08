# SDOS-176 — ContactStep: on the 11.2.92 face swap, a fast step keeps the face, and in a kiss each face stays on its own person

**Pack:** v11.2.176 "ContactStep", built on the uploaded v11.2.175 ManualPatch (the 11.2.92 face swap and manual mark removal).
Engine `aequus-1.2.176-contactstep`. The 11.2.92 detection, swap, masks and blending are unchanged. Only the tracker's
rules for two people, the front / behind decision and the smoothing of fast motion are new. Manual patch removal is the 175
code, unchanged.

**Asked:** "There is still a loss of swap in few frames while the subject was getting down from stairs, in one step where the
moment was rapid.. Also, during the kiss occlusion also, frames get shaky with revert to original and back.. Can you pls make
it even optimally best." Also: "Where the job is mentioned 2 face, and male, female marked, can't we mandatory paste man's face
to man and woman's face to woman".

Every test uses synthetic clips made from public sample photos: InsightFace's sample photo and two AFLW2000 test images. No
user footage was looked at.

---

## 1. What 11.2.92 / 175 does in the two cases

**Fast step.** A face that moves 40-60 px per frame (one quick step down) is still found by the detector on most frames. The
loss is the **smoothing**: the box filter (0.42) and the 5-point filter (One Euro, min cutoff 0.06) are tuned for a face that
stands still, so on a fast move the painted face trails the real one. Measured on the harsh step clip (180 px in 3 frames):
the paint lagged 19, 34 and 49 px on the three step frames, up to 50 px, and showed as a face sliding off and back. Two more
holes:
- A motion-blurred face that the identity gate rejects is dropped.
- The zoom re-detection was skipped for any box over 220 px, so at 1080p it never ran.

**Kiss.** Who is painted in front was decided on every frame by **box area**. In a side-view kiss both boxes are the same size
within a few percent, so the decision flipped from frame to frame. Each flip cut the other face, which caused the shake and the
revert-and-back. On a stand-in pair whose boxes differ by ±3 % from frame to frame, the 11.2.92 rule changed the face in
front 59 times in 60 frames.

**Side-view kiss, the face behind.** Its identity crop holds half of the partner and reads as the partner (0.80-0.88). Also,
each face's 5 points all sit on its front half, so **the two faces' points are within about 20 px of each other** while their
boxes are clearly apart (overlap 0.13-0.16). The de-duplication and the tracker's "same points" test therefore treated the two
faces as one. The face behind was dropped, its slot ended up on the partner's face, and the partner's real face was left
unpainted (section 4).

## 2. The changes (all in `swap_engine.py` and `core_pipeline.py`)

1. **Two people by elimination** (`MultiFaceTracker._elim_side`, `two_elim: 1`). Two people in the job, two faces found, and
   one face is clearly the other person (0.45 or more, and 0.10 above this face). Then the remaining face is this person, if
   it is not the other person (under 0.40), is nearer where this person was last seen, and is a whole face: its box at least
   0.82 of the last width, and its points not marked unreliable while touching the partner. While touching the partner, the
   box is also compared with the widest this face was identified at in the last 90 frames, so a face sliding slowly behind
   its partner is held, not taken as a sliver. A face turned away, blurred or in profile keeps its own replacement however low
   its own score is.
2. **The face behind, in contact** (`_partner_side`). A face whose crop reads as its partner (by 0.10 or more) is given back
   by geometry: it touches the partner, is not the partner's own detection, and is not a second box on the partner's face.
   *New in this build:* "a second box on the same face" is judged by the **boxes**: overlap 0.40 or more, or one centre inside
   the other box (`_two_boxes`). It is no longer judged by the 5 points, which in profile coincide.
3. **Two people in contact are never merged** (`core_pipeline._contact_apart`). Before the de-duplication, two detections
   that sit where two different tracked people were last seen are kept as two faces when their points agree with that, or
   when their boxes are two faces (as in 2).
4. **Carry in contact** (`contact_carry_max: 240`). A face missed while its partner is seen is carried for up to 240 frames
   instead of 24, moving with the partner (the partner's velocity), not on its own old velocity. When both are lost, the 24
   limit stands. When this person's own face is found elsewhere in the frame (they walked out from behind: a crossing), the
   24 limit stands too, and the slot is picked up again by identity as in 11.2.92.
5. **A steady front / behind** (`_front_relations`). It is decided per pair in this order:
   - identity: the face whose crop reads as its partner is behind;
   - sight: a face that is not seen is never in front of one that is;
   - size: only when the size difference is more than 10 % for 4 frames in a row. Exception: if the first decision was a near
     tie (boxes within 10 %, as when two faces first touch), the first clear difference decides at once.

   The decision holds for up to 60 frames without new evidence. The paint order and the cut (`_rival_landmarks`) follow it.
   With no decision (a frontal pair far apart), the 11.2.92 area rule is used.
6. **Fast snap** (`fast_snap: 1`). When a face moves faster than 4 % of its width per frame, the box and point smoothing ease
   off, and at 10 % they are off. A still face is smoothed exactly as in 11.2.92 (measured jitter 0.65 px, the same). Not
   while two faces cross.
7. **Zoom re-detection on big faces.** For tracked faces it now runs whatever the box size (at 1080p it never ran).
8. **Log lines** (end of every job):
   - `CONTACT: partly hidden faces kept N, kept by elimination (2 people) N, kept apart from the de-duplication N`;
   - `FRONT/BEHIND: decided by identity on N face-pair frames, by sight on N, by size (held) on N; changes N`;
   - `ID-WHY refusals per slot`;
   - `UNPAINTED person N: k frames` with the frame runs.

   "Kept by elimination" counts only faces the identity gate refused.

**Man to man, woman to woman.** Gender is **not** used frame by frame. Read from a side view, a half-hidden face or a blurred
one, it is unreliable (it is read from the same crop that reads as the partner). With two people marked, the same effect comes
from elimination (1) and the face behind (2): once each person is identified from the job's faces, each face keeps its own
replacement through a turn, a blur or a kiss. Use **Detect** and send each face to its own slot.

## 3. Verification (v11.2.176, from the pack's own final files)

**Unit checks** (stand-in faces, no models), run from the pack's final files: 24 of 24 pass
(`verification/fastrestore/unit176.py`, `unit176_pipe.py`). The two newest checks (a slow sliver, a crossing) fail on the
earlier 176 build, so they catch those problems.

- 2 people: a whole face turned away (reads as nobody, ~0.13) is kept by elimination, never swapped  [15 of 15, 0 wrong]
- two_elim 0: the v11.2.92 behaviour (that face is refused)  [0 of 15]
- a sliver (its box 0.75 of its last width) is not taken by elimination  [0 of 15]
- against its partner, a face whose points are marked unreliable (maybe a sliver) is not taken by elimination  [0 of 15]
- away from its partner, a blurred whole face (points unreliable) is kept by elimination - a fast step  [15 of 15, 0 wrong]
- a third face in the frame: elimination does not act  [0 of 15, 0 wrong]
- in contact, a face missed for 60 frames is carried while its partner is seen (v11.2.92: 24)  [60 of 60 (contact_follow 0: 24)]
- ...but when both are lost, the v11.2.92 limit stands (no face kept on a guess for long)  [24 of 60]
- a fast step (60 px a frame): the face's points follow the detection (v11.2.92 trailed it)  [lag [0.0, 0.0, 0.0] px (fast_snap 0: [14.3, 15.2, 15.3] px)]
- a still face with detector jitter is smoothed exactly as in v11.2.92  [frame-to-frame jitter 0.65 px out of 2.18 in (v11.2.92 0.65)]
- against its partner, a face that shrank slowly to 0.76 of its width and reads as nobody is not taken by elimination  [0 of 10 taken]
- a crossing: A comes out on the other side and is its own again within a few frames (not carried with B)  [back on call 31 (A reappears on call 30)]
- boxes the same size within 3 %: the face in front stays the same (v11.2.92 changed it on every frame)  [0 changes in 60 frames (v11.2.92 rule: 59)]
- a face whose crop reads as its partner is behind, whatever the box size  [False]
- ...and that holds through frames without an identity reading (bigger box, 49 frames)  [0 frames flipped]
- a face not seen (carried on a guess) is never in front of one that is seen  [False]
- the face behind is cut by the face in front, never the other way round  [behind cut by 1, front cut by 0]
- no decision made (contact_follow 0): the v11.2.92 rule, the bigger box in front
- first touch a near tie, then the other face clearly bigger: it is in front from that frame on  [[True, True, True, False, False, False, False, False, False, False]]
- ...but once decided on a clear difference, one contrary frame does not change it  [[False, False, False, False, False, False, False, False, False, False, False, False, False]]
- side-view kiss (points of both faces ~15 px apart, boxes apart): both faces kept and each given to its own person  [40 of 40 frames both kept, 40 both right, 0 wrong]
- (without the box test - the points test only - the face behind was merged away or refused)  [0 of 40 both kept, 0 both right]
- two boxes on one face (a cheek box, its centre inside the face's box) are not 'two faces'
- a side-view pair (overlap 0.14, centres outside each other's box) is two faces

**Replay of real detections** (the clips' cached detections through the final pack's pre-tracker steps and tracker, no
reference photos), who holds which person on each of 120 frames:

| Clip | v11.2.175 | v11.2.176 final |
|---|---|---|
| Side-view kiss | A lost on 78 frames; B's slot on A on 43 frames, lost on 19 | **both right on all 120 frames** |
| Harsh step | A lost on 6 frames (the step frames) | **both right on all 120** |
| Hide (A a sliver behind B) | A held 39-82 | **the same** |
| Cross / hug (A walks out the other side) | A picked up again at frames 76 / 78 | 77 / 79 |
| Smooch, deep, flip, sway, cheek | both right on all frames | **the same** |

**The clips, v11.2.175 -> v11.2.176:**

The real job (`bench_pipeline.py`, the Space's own pipeline on CPU, 2 threads, with the paste masks recorded), then
`identity_check.py` and `contact_lag.py` on the output. "Revert" means under 40 % of the usual paint on the person's
visible face. "Weak" means 40-70 %. "Mix-up" means the face reads as the other person's replacement.

| Clip | v11.2.175 | v11.2.176 |
|---|---|---|
| **Harsh step** (180 px in 3 frames) | A worst 50 px off, a 41 px jump; step frames 19 / 34 / 49 px | **worst 4.8 px, jump 2.9 px; step frames 5 / 3 / 3 px** |
| **Step** (30 px a frame) | step frames 4-6 px; worst 16.7 px (frame 84) | **step frames 1-3 px**; worst 18.7 px (the same frame 84: section 5, item 4) |
| **Step, detections dropped on the step frames** | step frames 4-6 px; worst 16.6 px | **step frames 1-3 px**; worst 18.6 px (frame 84) |
| **Side-view kiss**, A behind | revert 12, weak 2, mix-up 4; worst 62 px off | **revert 0, mix-up 0**, weak 12; worst 16 px off |
| **Side-view kiss**, B in front | revert 36, weak 18, mix-up 19; worst 33 px off | **revert 7, weak 15, mix-up 0**; worst 8 px off |
| Side-view kiss: the other person's paint on a face | up to 51 % (A's slot on B, frames 64-73) | **at most 1 %** |
| **Smooch** (frontal, deep) | B: weak 0, mix-up 0; A: weak 45 | B: weak 1, mix-up 1 (frame 34); A: weak 45 (the same) |
| **Hide** (A a sliver) | A weak 30; B clean | an earlier 176 build: A weak 52, fixed since (section 4, item 4); final run pending |
| **Cross** | A revert 0, weak 7; B weak 2 | final run pending |

**Which files each run used.** The step clips and the side-view kiss ran on a build two changes earlier than the final files:
- the near-tie rule in the front / behind decision (section 4, item 3);
- the crossing and sliver fixes (section 4, item 4).

Neither change acts on the step clips, which have no overlapping faces (their log: "FRONT/BEHIND ... 0"). The side-view
kiss replay is the same with the final files (all 120 frames right). The smooch ran with the near-tie rule but without
the sliver and crossing fixes, which do not act there (the replay is unchanged). The final files' own run of all 8 clips was started at delivery
and was not finished; its numbers will follow.

Job log of the side-view kiss (176): `FRONT/BEHIND: decided by identity on 78 face-pair frames, by sight on 0, by size (held)
on 3; changes 0` and `UNPAINTED person 1: 0 frames none`, `UNPAINTED person 2: 0 frames none`.

## 4. What did not work first

1. **The first 176 build** (rules 1-4) was not better on the side-view kiss: B was mixed up on 34 frames. The cause was the
   per-frame area order (1, Kiss), hence rule 5.
2. **Rule 5 alone made the side-view kiss look worse for A.** The face in front was now right (B: mix-up 19 -> 0), but A then
   showed B's replacement on 33 frames. Drawing each slot's paint on the synthetic frames showed why. From frame 66, B's slot
   sat on A's face and B's own face was unpainted. This was already so in 175, hidden there because A's paint lay on top. A
   replay of the clip's real detections showed the cause: the two faces' points 20 px apart, so the de-duplication kept one
   face from frame 69 and the "same points" test refused the face behind. Hence the box test in 2 and 3. Replayed, both
   slots hold their own person on all 120 frames. Before: A's slot was empty on 46 frames, and B's slot was empty on 15 and
   on A's face on 13. On the other nine contact clips the replay is identical with and without it.

3. **The size rule's 4-frame wait on the frontal smooch.** It put the face behind in front on frames 34-37: B was mixed up on
   2 frames and weak on 4, where 175 had none. Measured on the clip's detections: the face going behind is 6-8 % *bigger*
   on the first touching frames, then 12-15 % smaller as it is covered. A first decision on a near tie now follows the first
   clear difference at once.
4. **The contact carry on a crossing and a slow sliver.** Replays of all ten contact clips against 175 showed two more
   regressions:
   - In the cross and hug clips, A walks out from behind B on the other side. The long carry kept A's slot moving with B,
     so A was never picked up again (175: from frames 76 and 78). Now: frames 77 and 79.
   - In the hide clip, A's sliver was taken by elimination on 29 frames where 175 held A. A's weak frames rose from 30 to 52.
     The cause: A's box shrinks slowly (317 -> 244 px), and the width guard compared it only with the frame before.
     Elimination also ran before the normal identity check, so the "identified" widths were never recorded. Now the replay
     holds A on the same frames as 175.

## 5. Limits (honest)

1. **Synthetic tests only.** Cut-out faces from public photos on a blurred background, not real people moving. Real footage has
   hair, hands, light changes and real 3D turns. The numbers show direction and size, not a promise for every clip.
2. **The face in front of a side-view kiss still thins on 7 frames** (175: 36 reverts). Not from the front / behind decision,
   which did not change all clip. Likely the 11.2.92 occlusion masks near the lips, which this build does not touch.
   Unconfirmed.
3. **The face behind is cut where the face in front covers it, as it should be.** In profile the swap covers its cheek and
   eye. On the side-view kiss that shows as "weak" on 12 frames and "unclear" identity (the visible part is small).
4. **The frame just before a sudden step** (frame 84 of the step clip) can be painted up to 19 px ahead (175: 17 px). The
   detector skipped that frame, and the 11.2.92 interpolation between detections leads the step. Unchanged here.
5. **Smooch: 1 frame** (the first touching frame) where the face behind is still in front (175: 0). A quicker switch would
   bring back the frame-to-frame flipping of the side-view kiss.
6. **Elimination needs exactly two people found.** With a third face in the frame it does not act (by design). With only
   one face found, the 11.2.92 rules apply.
7. **A face that slides behind its partner and also moves away from the camera** shrinks for both reasons. Under 0.82 of its
   recent width it is held, not taken: the 11.2.92 behaviour.
8. **Cross / hug:** the person coming out from behind is picked up one frame later than in 175 (frames 77 / 79 against 76 /
   78, replay).

## 6. Install

- Space: upload the pack's files. `core_pipeline.py`, `swap_engine.py`, `config.py` and this document changed from v11.2.175;
  `patch_manual.py` and the other files are the same. No dependency change.
- Log at start: `Phoenix v11.2.176`, `swap_engine aequus-1.2.176-contactstep`.
- Do not restart a Space while a job runs on it.
- Switches (in `swap_engine._P`): `two_elim`, `contact_follow`, `contact_carry_max`, `fast_snap`. 0 restores 11.2.92's rule for
  each.
