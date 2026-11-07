# SDOS-144 — KissFollow: in a kiss each face keeps its own replacement, on its own pixels

**Pack:** v11.2.144 "KissFollow" (v11.2.142 BellyAnchor + this). Engine `aequus-1.2.144-kissfollow`. **App:** 1.2.45, no change.

**Asked:** "Could you fix the face occlusion, this is not solved, I see face swap lagging, flickering, reverting to original during
contact like kissing."

All tests use synthetic clips made from public sample photos (InsightFace's sample photo, two AFLW2000 test images); no user footage
was looked at.

---

## 1. Why v11.2.141 still failed in a kiss

The six contact clips of SDOS-141 are all **frontal**: both people look into the camera while their faces overlap. A real kiss is
mostly **in profile**, and that case was not on the bench. New clip (`make_pkiss.py`): two public AFLW2000 photos turned 55 and 71
degrees towards each other; the heads come together until the lips meet, kiss for 75 frames with a small rhythmic push while the
couple sways (up to 45 px) and tilts (up to 4 degrees), then part. B is nearer the camera (B's face covers A's nose and lips).

v11.2.141 on it showed all three complaints (table in section 4): the person behind reverted to the original face on 56 frames,
was painted on a guessed position for 79 frames (up to 62 px off: the lag), and on 37 frames carried the OTHER person's
replacement over 30 % or more of the face (up to 83 %).

What happens, found by replaying the clip's real detections through the tracker frame by frame:

1. **The detector finds both faces on every frame of the kiss.** The face is not lost; it is thrown away.
2. **In profile a face's identity crop shows the partner.** The partner fills the front half of the ArcFace crop of the face
   behind: during the kiss A's detection scores **0.80-0.88 against B** and 0.05-0.15 against A (B's own: 0.95 against B).
   A's 5 points are where they belong (within 10-20 px).
3. **Two rules meant for two boxes on one face** then removed A: the pipeline's de-duplication (two detections whose embeddings
   agree at 0.55 or more are one face) and, behind it, the tracker's "same person must not fill two slots". From the moment the
   lips met until they parted, A had no detection; the identity gate rejected what was left (84 times).
4. **Front / behind was named the wrong way round.** The one-face-per-pixel rule (v11.2.95) decides who is in front from image
   edges along each face's guessed jaw outline. In profile both guessed outlines run along the one silhouette edge between the
   faces, and A (behind) was named in front on every frame of the kiss: once A was painted again, A's replacement took B's nose
   and lips, and B's own paint was cut back to the cheek.

## 2. The changes

### 2.1 Two people in contact are never merged into one (`core_pipeline._contact_apart`, `_dedupe_faces`)

Before the tracker, two detections are kept as two people when the tracker already follows two people there: each sits where a
different person was last really seen (mutual nearest), its facial points lie nearer that person's last points than the other's,
and the boxes are separate faces (overlap under 0.40). Only the box-overlap test applies to such a pair. A second box on the
partner's own face carries the partner's points and is still merged; boxes on top of each other are still one face.

### 2.2 In contact the tracker gives a face back by geometry (`swap_engine.MultiFaceTracker.assign`)

In contact, a detection whose crop **reads as the other person** (by 0.25 or more over its own person: the profile case above) is
given to a face when the geometry is unambiguous: nearest to where that face was **last really seen**
(not the prediction), no other person last seen nearer to it, its facial points nearer this face's last points than anyone else's,
and no other identified person clearly more like it - unless that person is plainly another detection (one at least as much like
them and nearer to where they were last seen: B's own face, 0.95 against A's crop's 0.85). Such a pair passes the identity floors
(a profile face scores far below them), is not taken for "the same person in two slots", and is not undone as a label flip. The
identity memory is never learned from these frames (unchanged: not while two faces are in contact).

A face that reads as nobody - only a sliver of it shows behind the other - keeps the v11.2.141 rules: its detections carry 5 points
guessed about 40 px off (measured on the hide clip), and it is better held where it was last seen well (section 3).

### 2.3 Front / behind from the faces' own identity (`core_pipeline._mark_cross_identity`, `_order_cues`)

Each face given to a person is marked with how much more its crop is like the OTHER person than like its own. The face behind has
its middle covered by the partner and reads as them (+0.6 to +0.8 in the kiss); the face in front reads as itself (-0.9). When one
face reads as the other by 0.25 or more and the other reads as itself by 0.25 or more, the first is **behind**; the outline cue
decides only when this one-sided signal is absent (both read as themselves: every frontal clip). On frames with no recent
detection of one of them the order this cue set holds for up to 60 frames.

### 2.4 In contact, a missed face moves with its partner (`MultiFaceTracker._contact_motion`, `core_pipeline._follow_partner`)

- Tracker: while two faces are in contact and one is missed, its predicted position moves with its **partner's** motion, not on at
  its own last velocity (when the lips meet the approach stops dead; the old guess carried the missed face on into the partner).
- Paint: on a frame without a detection of it, a face in contact is drawn at its last sighting **moved by the partner's own
  motion** (a similarity fit of the partner's 5 points), and when it is found again later in the same batch the remaining error is
  spread over the gap (no jump). Full strength while its partner is seen, up to `CONTACT_LONG_SEC` (8 s); a partner whose own track
  jumps (scale outside 0.75-1.33, or a move over 3 face widths) is not followed.
- Front / behind no longer flips on every missed frame: a sighting within 2 frames counts as seen (v11.2.141 put a face behind on
  every frame it was not detected; a profile face found and lost by turns flipped front and back).

### 2.5 Switches

`config.py`: `CONTACT_FOLLOW = True`, `CONTACT_LONG_SEC = 8.0`, and the engine's `contact_follow: 1`. Off restores v11.2.141-143
(checked in the unit tests). Job log: `contact_follow=` (frames painted by following the partner), `contact_apart=` (calls where two
people in contact were kept apart), `order_xid=` (frame pairs ordered by the identity cue).

## 3. What did not work first

1. **The first v11.2.144 build** (the partner motion of 2.4 and a re-admission by the last sighting for ANY face in contact) passed
   its unit tests on stand-in faces and changed nothing on the real profile kiss: 84 rejections and 65 reverted frames, as before.
   The stand-ins had the wrong failure (a face that reads as nobody); the real one is a face that reads as its partner, removed
   by the two duplicate rules (1.3). Found by replaying the clip's real detections through the tracker; the stand-in test now
   uses the measured case.
2. **Re-admission by geometry for every low-similarity face in contact** fixed the kiss but made the hide clip worse: the face
   behind, only a sliver, was taken back with points 40 px off instead of being held, and its weak frames rose from 32 to 49.
   Limited to crops that read as the partner (2.2); the hide clip's tracker decisions are then the same as v11.2.141's.
3. **The outline cue for front / behind** (v11.2.95) was kept for frontal pairs, where it was measured right on 89-100 % of
   overlapping frames; in profile it named the face behind as in front on every frame (1.4), hence 2.3.

## 4. Verification (v11.2.144, from the pack's own files)

- `unit144.py` (tracker, stand-in faces): **11 of 11** - a missed face moves with its partner (worst 6 px, v11.2.141: 12 px); a
  profile face whose crop reads as its partner keeps its own face through the kiss (38 of 38 calls, v11.2.141-143: 0); a face that
  reads as nobody keeps the v11.2.141 rules; a stranger beside the pair, the other person standing where the face was, and a stray
  box on the partner's cheek are never given the face; faces apart: the rules never act; `contact_follow 0` = v11.2.141.
- `unit144_pipe.py` (pipeline, stand-in faces): **14 of 14** - two people in contact whose embeddings agree (0.97) are kept apart,
  a second box on one face and boxes on top of each other are still merged; the identity cue orders a profile pair, does not act on
  a frontal pair, holds for 60 frames without a reading and then gives way; `CONTACT_FOLLOW False` = v11.2.141.
- The real job (detector, landmarks, swapper, occluder; 2 CPUs) on 8 synthetic contact clips, every run from the pack's final
  files (checked by checksum). Scores (`identity_check.py`, `contact_lag.py`, and the per-pixel paint ownership from the job's own
  paste masks): *revert* = under 40 % of the person's normal paint over their visible face, *weak* = 40-70 %, *mix-up* = the other
  person's replacement wins the identity test, *lag* = paint centre against the person's true 5 points.

**The profile kiss, v11.2.141 -> v11.2.144:**

| | v11.2.141 | v11.2.144 |
|---|---|---|
| A (behind): reverted / weak frames | **56 / 15** | **0 / 0** |
| A: frames with B's replacement over 30 %+ of A's face | **37** (up to 83 %) | **0** (at most 4 %) |
| A: frames with no paint at all | 27 | 0 |
| A: lag, mean / worst | 24 / **62 px** | 9 / **21 px** |
| B (in front): reverted / weak frames | 33 / 11 | 5 / 23 |
| B: identity correct / mix-up (frames) | 54 / 0 | 111 / 0 |
| B: frames with A's replacement over 5 %+ of B's face | 30 (up to 50 %) | 2 (at most 5 %) |
| flicker (x the footage's own), A / B | 1.05 / 0.98 | 1.02 / 0.97 |

The engine's log (v11.2.144): `REVERTS=0`, `contact_apart=80` (calls where the two were kept apart), `order_xid=62` (frames ordered
by identity). The ArcFace identity score of the face behind is not meaningful in profile (its own similarity median 0.08 in both
versions); one frame (66) is counted a "mix-up" by it although B's paint covers at most 4 % of A's face there - the paint
ownership above is the measure used.

**The earlier contact clips, v11.2.141 -> v11.2.144:**

All six clips of SDOS-141 and a swaying kiss, both versions scored on the same bench (v11.2.141 re-run here; the SDOS-141 numbers
were scored before a frame-alignment fix in the scoring and are not used). Reverted / weak frames, mix-ups 0 everywhere:

| Clip | A (behind) 141 -> 144 | B (in front) 141 -> 144 | flicker A / B 141 -> 144 |
|---|---|---|---|
| smooch | 0 / 0 -> 0 / 0 | 0 / 0 -> 0 / 0 | 1.00 / 0.99 -> 1.00 / 0.99 |
| deep | 0 / 0 -> 0 / 0 | 0 / 0 -> 0 / 0 | 0.99 / 0.99 -> 1.00 / 1.00 |
| hide (A only a sliver for 40 frames) | **0 / 32 -> 0 / 5** | 0 / 0 -> 0 / 0 | 0.96 / 0.99 -> 0.99 / 1.01 |
| flip | 0 / 0 -> 0 / 0 | 0 / 0 -> 0 / 0 | 1.03 / 1.00 -> 1.03 / 0.99 |
| cross | 0 / 0 -> 0 / 0 | 0 / 0 -> 0 / 0 | 1.00 / 1.00 -> 1.00 / 1.00 |
| hug | 0 / 0 -> 0 / **1** | 0 / 0 -> 0 / 0 | 1.01 / 1.01 -> 1.02 / 1.02 |
| sway | 0 / 0 -> 0 / 0 | 0 / 0 -> 0 / 0 | 0.99 / 1.01 -> 0.99 / 1.01 |

- The hug's one weak frame (69) is a threshold case: A's paint share there is 0.14 against 0.15 in v11.2.141 (A 31 % visible, just
  coming out from behind B); the frames around it agree within 0.01.
- Lag where both are measured on the same bench: cross worst 40 px in both, hug worst 86 px in both (the person behind while
  hidden during the turn), deep / smooch identical.
- sway v11.2.141: from a run made earlier in this work on the same clip and bench.
- Speed: 386-564 s per 120-frame clip on 2 CPUs (v11.2.141: 410-560 s); the new tests cost no model call.

## 5. Limits (honest)

- **Synthetic clips only.** The profile kiss is two still photos composited and moved; real kisses add head turns, motion blur,
  hands and hair. Nothing was tested on real footage or on a phone.
- **The person in front, in profile, is still thin in places:** on 13 frames late in the kiss (85-97) B's own paint covers only
  22-38 % of B's face (5 frames under the revert line), where the detector's profile keypoints are rejected and B is drawn from
  held geometry. v11.2.141 had 33 such reverted frames.
- **A small spill at the contact line:** the front face's paint reaches up to 4-6 % into the face behind (in the test, B's edge
  over A's eye), from the front face's guessed outline.
- **The identity cue needs clean identities from before the contact** (the Detect frame, or frames where the two are apart). A
  video that starts mid-kiss has no clean identity for the face behind; then the outline cue decides, and in profile it is wrong.
- **Look-alikes:** two very similar faces would read as each other all the time; the cue is one-sided only when they differ.
- **A face hidden while the pair turns** (hug) is painted where it is guessed to be, up to 86 px off, as in v11.2.141.
- **Belly clean** is unchanged from v11.2.142, including one fault found during this work: in a video with fast motion the belly
  paint can reach up to 31 px below the floor line (0.10 above the hip line) on some frames; it is limited to the line on the
  preview. The per-frame floor cut that fixes it is part of the unfinished v11.2.143 belly work, not this pack.
- Built on our v11.2.142, not on the uploaded v11.2.150: its belly reaches below the hip line, and its changes are untested here.

## 6. Install

- Space: upload the pack's files (the same list as v11.2.142; `core_pipeline.py`, `swap_engine.py`, `config.py` changed). No
  dependency change.
- Built on v11.2.142 (our line), not on the uploaded v11.2.150: see section 5.
- Log at start: `Phoenix v11.2.144`, `swap_engine aequus-1.2.144-kissfollow`.
- App: 1.2.45 as it is. Use **Detect** and send each face to its person's slot: the tracker then knows each person from the start.
