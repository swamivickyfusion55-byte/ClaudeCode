# SDOS-211 — Steady: the swap does not wave where two faces meet, and the occlusion edge shakes less

**Pack:** v11.2.211, built on v11.2.210. Engine `aequus-1.2.211-steady`.

**Asked:** "The swap tends to wave, when the faces come in contact, from the point of their connect to the face, i.e. near the
ear. Also, the occlusion is still not professional, a bit shaky."

No footage was looked at. Every number below is from synthetic clips made from public sample photos (two people in contact,
an object over a face, an ice cream), each also as a copy degraded the way phone footage is (camera noise, blur, H.264 at
crf 30), compared with **11.2.209 and 11.2.210 at Ultra**. Two kinds of run:

- **Edge steadiness** (`mask_steady.py`): the real detector, the real tracker, the real carry step and the real mask builder
  on every frame, without the swap network. Measure: a frame's paint minus the median of its five neighbours, over the pixels
  where the paint is soft (an edge) - the part of an edge's movement that is not smooth motion, what the eye sees as shaking.
  Reported x1000: 1.0 is about a pixel of shake. This is a measurement tool, not the job: it feeds every detection to the
  tracker and has none of the job's gates, so absolute numbers differ from the job's.
- **The whole job** (`bench_pipeline.py`, Ultra) on the same clips.

---

## 1. What was wrong

Eight causes, each found by measurement.

### 1.1 The outline of the face was not filtered

The tracker filters the face's 5 points (One Euro) but kept the 106 face landmarks as the detector gave them. The edge of the
mask, the cut between two faces and the carried position all come from those landmarks, and the detector's landmarks scatter by
1-2 px from frame to frame on a 320-px face: that is the edge that waves, most at the jaw and the ear, where the outline is long
and runs nearly parallel to the other face. On the real detector's readings of the synthetic clips, the landmarks' own jitter
(a frame minus the median of its five neighbours):

| | raw (11.2.210) | 11.2.211 |
|---|---|---|
| frontal kiss, person B | 0.93 px | **0.15 px** |
| frontal kiss, person A | 2.04 px | **1.21 px** |
| side-view kiss, person A | 0.94 px | **0.70 px** |

### 1.2 The carried face's size and turn were measured anew every frame

11.2.210 places a face that is partly behind another by its own visible part. Each frame's size and turn come from a fit to a few
corners and jitter by 0.5-1 % and 0.2 deg: a point near the ear moves 1-3 px. Wobble of the carried points (frontal kiss): **0.90 px**.

### 1.3 The cut along the cheek flickered

When the occlusion model's map shows a thin strip along a face's own edge (the edge of the touching face, hair, a cheek in shade),
the "something is in front" rules cut it, and on the next frame did not. On the crossing clip the cut area of the face in front
alternated between 0.00 and 0.07 of the crop, frame after frame.

### 1.4 A hidden face could be dragged away by the face in front (found on the crossing clip)

While most of a face is hidden, its position comes from the corners of its visible part. Frame-to-frame motion has no picture of
the face to check against: with 60 % of A hidden, the corners it found belonged to the face in front (its hair and ear), moving the
other way. The carry took that for A's own motion ("A was not moving against the face in front, it stays with it") and carried A
along with the front face for 20 frames, **up to 182 px away** from where it was. Its paint, cut by the front face's outline, was
left as a sliver beside the front face's ear and hair. This is the "waving near the ear" in its worst form.

### 1.5 The carry ended too soon and handed the face back with a jump

A face passing behind another was carried on its own motion for 5 frames only, then placed by the job's older rules, which lose
it (they hold the last position and fade): part of A's face was laid on the front face, 105 px from where A was. And when the
detector found the face again on the other side, the position was taken over at once: 20-60 px in one frame.

### 1.6 The carry's filter could overshoot

(Found while testing 11.2.211's own first filter, on a degraded copy.) A reading far from the trend was followed at once, and
its whole jump was also taken for the trend: the next prediction overshot by that jump (placement of the face behind in the
degraded hide clip 12 -> 30 px). Fixed: the trend learns a fraction of what each reading showed, capped.

### 1.7 A carry that could crash without a trace (11.2.210)

On a degraded copy of a clip the carry stopped working for whole stretches. A face that alternated between "stays with the
front face" and "passes behind" reached a state with no stored motion; the step raised an error, the error was caught in the
job, and the carry was off for the chunk. Nothing in the log. Found by running the carry on degraded clips.

### 1.8 What was not the cause, and a measuring error of mine

My side-view test matched the detector's two faces to the two people by their centres, which in a side-view kiss are 10-20 px
apart: for 8-frame stretches the two people were swapped in the test, which looked like the paint jumping to the other face.
That was the test, not the engine (fixed in `mask_steady.py` and `det_err.py`). Ruled out: the side rule of 11.2.210 (its
engagement threshold is 0.45; the pair is at 0.8-1.1 the whole time), the occlusion model's own frame-to-frame changes (a
temporal filter of the face map was tried and made no difference), the rival cut.

## 2. What 11.2.211 changes

| | Change | Switch (`swap_engine._P`; 0 = 11.2.210) |
|---|---|---|
| 1 | **The 106 landmarks are smoothed in the face's own frame.** One Euro per landmark in a 112-px aligned face (so a turn or a move of the head is not smoothed, only the scatter), opening with the point's own speed; re-projected with the already smoothed 5 points. A mouth that opens, a jaw that drops: followed within 4 frames. | `lmk_smooth` 1, `lmk_min_cutoff` 0.08, `lmk_beta` 0.12 |
| 2 | **The carried face's size, turn and centre are filtered** (level and trend each, Holt): steady motion passes without lag, the scatter of the readings is smoothed, a reading far from the trend is followed at once, without taking its jump for the speed. | `hid_filter` 1, `hid_f_scale/turn/pos` 0.35, `hid_f_trend` 0.15 |
| 3 | **A thin strip along the face's own edge is not an object**, where another face touches this one. Applied to the three rules that cut the model's map (mouth, wide, follow). A face on its own keeps 11.2.210's cut: a scoop resting on the chin is a thin piece along the jaw too (with the rule on a lone face, the scoop on the chin was painted over 0.91, 11.2.210: 0.08). | `occ_net_rim` 1, `occ_net_rim_thick` 0.06, `occ_net_rim_hug` 0.12, `occ_net_rim_frac` 0.6, `occ_net_rim_alone` 0 |
| 4 | **A piece cut on the last frame stays cut** while it still overlaps itself (the mouth rule's hold, as the wide rule had). | `occ_net_hold_core` 1 |
| 5 | **Frame-to-frame flow is not used while the carried face is passing behind** (moving against the face in front): only its own motion. In any case it moves the position only, never the learnt speed or the count to "lost". | `hid_pass_noflow` 1, `hid_weak_flow` 1 |
| 6 | A face passing behind another goes on its own way **20 frames** (was 5) before the job's own rules take over. | `hid_lost` 20 |
| 7 | **A face the detector finds elsewhere is handed over in 6 steps**, not at once. | `hid_hand` 1 |
| 8 | **A carried face is cut by the face in front at its outline dilated 0.06 of the crop** (0.085 for any other face): the crescent of a face coming out from behind lost half of its width to the dilation. | `rival_dilate_hid` 0.06 |
| 9 | The carry crash of 11.2.210 (§1.7) fixed; each face's step is guarded on its own (a fault drops that face's carry, nothing else). | - |

Not changed: identity (11.2.209), the front / behind decision, the eating fix of 11.2.210, the object cut of 11.2.206, skin tone,
the swap network and what it is given.

## 3. Verification

### 3.1 Edge steadiness (no swap network), 11.2.209 / 11.2.210 / 11.2.211

x1000, lower = steadier. Person A (behind) / person B (in front), all edge pixels of the paint. `deg` = the degraded copy.
(`verification/steady211/edge_steadiness_209_210_211.txt` also has the seam and the outline bands separately.)

| Clip | 11.2.209 | 11.2.210 | **11.2.211** |
|---|---|---|---|
| Frontal kiss | 8.8 / 1.5 | 11.6 / 1.5 | **7.8 / 1.4** |
| Frontal kiss, deg | 9.6 / 1.5 | 12.5 / 1.5 | **8.6 / 1.5** |
| One face mostly hides the other (hide) | 6.2 / 0.9 | 4.7 / 0.9 | 5.5 / 0.8 |
| hide, deg | 6.5 / 1.0 | 5.6 / 1.0 | 6.4 / 1.1 |
| Crossing | 5.8 / 0.6 | 7.5 / 0.6 | **5.7 / 0.6** |
| Crossing, deg | 8.3 / 0.6 | 9.1 / 0.6 | **7.1 / 0.6** |
| Side-view kiss | 3.3 / 33.8 | 6.1 / 21.7 | 6.3 / 21.6 |
| Side-view kiss, deg | 3.6 / 45.0 | 6.3 / 16.0 | 6.3 / 16.0 |
| Cup at the cheek / deg | - | 0.9 / 1.2 | 0.9 / 1.2 |
| Hand sweeping across / deg | - | 7.9 / 8.0 | 7.9 / 8.0 |

The edge of A away from the contact (the outline beside the ear): frontal kiss 14.7 -> **5.7**, degraded 15.0 -> **7.0**; hide
2.5 -> **1.0**; crossing 3.6 -> **2.8**, degraded 9.3 -> **5.0** (11.2.209: 6.4 / 7.3 / 3.3 / 1.6 / 1.7).

Honest reading: 11.2.210 shook more than 11.2.209 on the clips where it carries the face (it painted more of it, with a
position that wandered); 11.2.211 brings that back to 11.2.209's level or below on the frontal kiss and the crossing, and is
level with 11.2.210 elsewhere. On hide it is **0.8 / 0.8 higher than 11.2.210** (5.5 against 4.7, 6.4 against 5.6): the
narrower cut for a carried face (change 8) costs some seam steadiness there. On the side-view kiss 11.2.209's low number for A
comes from painting less of it (0.31 of its visible face against 0.43).

### 3.2 Where the paint lies, against the true position

Person A, mean placement error of its paint (px) and the mean paint on its visible face (`placement_and_coverage_209_210_211.txt`):

| Clip | 11.2.209 | 11.2.210 | **11.2.211** |
|---|---|---|---|
| Frontal kiss | 29.8 px, 0.39 | 11.1 px, 0.53 | **10.9 px, 0.55** |
| hide | 47.7 px, 0.28 | 18.2 px, 0.41 | **10.1 px, 0.42** |
| hide, deg | 48.4 px, 0.30 | 19.4 px, 0.40 | **12.1 px, 0.41** |
| Crossing | 43.4 px, 0.40 | 32.0 px, 0.46 | **9.0 px, 0.56** |
| Crossing, deg | 49.5 px, 0.39 | 50.0 px, 0.44 | **14.5 px, 0.55** |

### 3.3 The whole job, Ultra, 11.2.209 / 11.2.210 / 11.2.211

Five clips of two people, 120 frames each, scored against the clips' ground truth (`identity_check.py`, `contact_lag.py`;
`full_job_runs_209_210_211.txt`). A is the face that goes behind, B the face in front. "weak" = the replacement is faded
(40-70 % of normal paint over the visible face), "revert" = under 40 %.

| | 11.2.209 | 11.2.210 | **11.2.211** |
|---|---|---|---|
| **Frontal kiss.** A weak / revert frames | 44 / 0 | 0 / 0 | **19** / 0 |
| A paint over its visible face (normal 0.91), frames 40-75: mean / lowest | 0.61 / 0.58 | 0.67 / 0.65 | 0.65 / 0.61 |
| A placement mean / worst, jump | 17.0 / 30.4 px, 5.8 | 10.5 / 14.0 px, 2.4 | **10.2 / 16.0 px**, 2.9 |
| B's identity frames correct / mix-ups | 117 / 1 | 119 / 0 | 119 / 0 |
| **hide.** A weak / revert frames | 3 / 0 | 27 / 0 | **0 / 0** |
| A placement mean / worst, jump | 39.0 / 66.3 px, 8.0 | 25.6 / 41.2 px, 4.9 | **19.3 / 27.0 px**, 4.9 |
| **Crossing.** A weak / revert frames | 8 / 0 | 9 / 0 | 6 / **6** |
| A placement mean / worst, jump | 23.9 / 104.9 px, 5.7 | 17.2 / 104.9 px, 22.9 | **9.0 / 40.9 px**, 8.2 |
| **Side-view kiss.** A / B weak frames | 12 / 15 | 0 / 0 | 0 / 0 |
| B reverted frames | 7 | 0 | 0 |
| Identity frames clearly right, A / B | 39 / 98 | 72 / 115 | 72 / 114 |

The frontal kiss's 19 weak frames are the paint of A over its visible face at 0.61-0.64 where 11.2.210 had 0.65-0.67: the
"weak" line is 0.64. The smoothed landmarks move the cut line between the faces by a few pixels (the line's position moves by up
to 5 px with unrelated small changes of a parameter on this clip; the wobble of the line over time falls from 1.4-1.7 px to 1.1-1.3 px).
11.2.209 had 0.58-0.61 (44 weak frames).

The crossing's 6 reverted frames (68-73) are the emerging crescent of A - 30-50 % of its face, mostly its ear and hair side -
painted at under 40 % of normal. 11.2.210 had no reverted frame there, but it placed A up to 105 px off and laid part of A's
face over the front face (seen in the output); 11.2.211 places A within 9 px on average and lays nothing on the front face. The
crescent is the case left open (§5, 1).

Unchanged from 11.2.210, to the digit: **harsh step** (two people who never touch; 240 of 240 identity frames correct, placement
1.6 / 1.5 px, nothing carried); the same with Detect pictures A and B (120 / 120); a **wrong Detect picture for person B**
(B found by elimination, 68 of 68 frames correct, 0 mix-ups); **eating** (§3.5); **object clips** (§3.6).

### 3.4 The carry alone, against the true position

`hidden_hold.py`: the real detector's readings of both people on every frame, handed to the carry step as the job does. Person
A's 5 points against the clip's true ones, on the frames it is carried:

| Clip | Frames carried | Detector (11.2.209 painted with it): mean / worst | 11.2.210: mean / worst | **11.2.211: mean / worst** | Wobble of the carried points |
|---|---|---|---|---|---|
| Frontal kiss | 65 | 30.7 / 42.4 px | 12.2 / 12.6 px | **12.3 / 12.7 px** | 0.90 -> **0.34 px** |
| hide | 72 | 44.8 / 63.6 px | 14.0 / 17.5 px | **9.5 / 13.0 px** | 0.13 px |
| Crossing | 26 -> 41 | 64.3 / 156.5 px | 15.4 / 83.6 px | **12.3 / 23.6 px** | 0.09 px |

(The detector column is on the frames each version carries: 26 for 11.2.210, 41 for 11.2.211.) Largest change of the carried position in one frame, crossing: 106.9 px (detector) / 106.9 px (11.2.210) / **3.8 px**. With the
faces apart (97 % or more visible) the carry places nothing: 0 of 8, 21 and 48 frames.

### 3.5 Eating, and nothing else moved

All on the final files (checksums compared after the runs: `swap_engine.py` `e6bfc307...`, `core_pipeline.py` `0fe6af09...`,
`config.py` `8a66c779...`).

| Check | Result |
|---|---|
| Eating, whole job, Ultra (`lick_job.py`) | strawberry and vanilla **equal to 11.2.210**: lips shown as the replacement's, scoop on the lips 0.88 / 0.90, between licks 1.00, largest change in a frame 0.11 / 0.09; the scoop painted over 0 % / 0 %, worst 3 % / 0 % |
| Eating, the mask alone, three colours (`lick_mask.py`) | the scoop painted over on the lips 0.01 / 0.00 / 0.00, resting on the chin 0.08 / 0.11 / 0.03: **equal to 11.2.210** (with the strip rule on a lone face it was 0.91) |
| Object clips as whole jobs (`obj_metrics.py`, 11.2.206 / 210 / 211) | hand over the eye: painted over 3.8 % / 0.0 % / 0.1 %; cup at the cheek 1.1 % / 0.2 % / 0.2 %; hand sweeping 1.0 % / 0.7 % / 0.7 %; reverted frames 0 / 0 / 0, 0 / 0 / 0, 0 / 1 / 1 |
| Faces with nothing in front, 36 runs incl. noise, blur, dark, bright, JPEG 25 (`nofalse206.py`, `DEGRADE=1`) | 0.00 % more of any face cut than 11.2.205 |
| Unit suites | `unit211` 23, `unit210` 19, `unit209` 13, `unit206` 4 + 4 (two clips), `unit205` 30, `unit204` 7, `unit203` 18, `unit176` 12, `unit176_pipe` 14: **all pass** (`unit206` and `unit210` switch 211's own switches off too for their "equals the older version" checks) |
| Cost (`cost210.py`, 2 threads, nothing else running) | the face mask 185.2 -> 176.7 ms per face (no change beyond noise); the carry step 0.09 ms per frame with the faces apart, 8.8 ms while a face is carried (11.2.210: 0.07 / 7.8). Against about 3 s per frame of the Ultra swap this does not show. |

`unit211.py` is new: the landmark filter (jitter, no lag in motion and in a turn, a jaw that drops is followed, no detection =
the points' motion, switch off = as detected), the strip and hold rules (a strip is not cut, a spoon-sized blob still is, a
held piece stays, switches off = 11.2.210), the carry filter (steady motion, noise, a reversal), the carry's decisions (flow of
the face in front not used while passing behind, hand-back after `hid_lost`, hand-over in steps, switches off = 11.2.210) and the
cut of a carried face.

## 4. What the log says now

The `TOUCH:` line of the job log is as in 11.2.210 (frames carried, by reference or by last motion, handed back, let go, found
again elsewhere; side-view pairs). `faults` in the carry's statistics counts a face whose step failed and was dropped.

## 5. Limits

1. **The crescent of a face coming out from behind another** (crossing, 30-50 % visible) is painted thinly for about 6 frames
   (§3.3). The visible part is mostly the ear and the hair side, where the face template and the cut by the front face leave
   little; narrowing the cut helped (change 8) but did not remove it. Reverting those frames to the footage's face is what the
   viewer would see there.
2. **The frontal kiss's face behind** is painted at 0.61-0.65 of normal, within the noise of 11.2.210's 0.65-0.67 but on the
   other side of the 0.64 "weak" line (§3.3). The cause is the few pixels' shift of the cut line that the smoothed landmarks
   bring; a version with a compensating dilation did not help and was not shipped.
3. **hide is 0.8 higher in edge shimmer than 11.2.210** (§3.1).
4. **Faces apart for a long time while hidden**: a face passing behind another is carried on its own motion for 20 frames; if
   it stops or turns round behind the other face, its paint lands in the wrong place for that time (mostly under the front
   face's outline, where it is cut). After 20 frames the job's own rules take over, as in 11.2.209.
5. **Real detector scatter is not known.** The synthetic clips move smoothly; a handheld video's landmarks may scatter more,
   which the landmark filter will smooth more, or less where the face really deforms. The strength was chosen on the degraded
   copies (more strength made the seam wobble more, not less).
6. **The measuring tool is not the job.** `mask_steady.py` feeds every detection to the tracker and has no hold or fade; its
   numbers are for comparing versions, not absolute. The job's A face on the crossing clip comes out 7-8 % larger (linear) than the tool's, from the detector's different input.
7. **Three-quarter views (30-45 deg)** fall under the front-view rules; not tested (no synthetic three-quarter contact clip).
8. Synthetic clips only; no footage was looked at. Whether your two reports are fully covered can only be seen on your videos.

## 6. Install

Upload the pack's files to the Space. Changed against 11.2.210: `swap_engine.py`, `core_pipeline.py`, `config.py`. No change to
`packages.txt` or the app. The log at start reads `Phoenix v11.2.211`.

Each change has its switch (§2). If the face behind looks thinner than before in a kiss, `lmk_smooth = 0` brings back
11.2.210's outline on its own; `rival_dilate_hid = 0` its cut.
