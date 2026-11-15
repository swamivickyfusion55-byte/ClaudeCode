# SDOS-210 — TrueTouch: two faces touching, and eating, without the original face coming back

**Pack:** v11.2.210, built on v11.2.209. Engine `aequus-1.2.210-truetouch`.

**Asked:** "You have still not been able to fix face contact occlusion fully, a little revert back and flicker is noted on
contact e.g. smooch or even one face coming in front of other. Secondly, while eating say icecream, lower mouth / lip appears
to come out."

No footage was looked at. Every number below is from synthetic clips made from public sample photos, run through the real
job at **Ultra**, and compared with **11.2.209 at Ultra**.

---

## 1. What was wrong

Four separate causes, each reproduced and measured on 11.2.209.

### 1.1 A face partly behind another one is placed wrongly by the detector

The detector does not report "half a face". It draws a whole, narrower face into the part it can see. Its 5 points for a
320-px face, against the clip's true ones (`det_err.py`):

| Share of the face still visible | 100-85 % | 70 % | 60 % | 50 % | 35-40 % |
|---|---|---|---|---|---|
| Detector's error | 3-6 px | 8-10 px | 10-15 px | 18-25 px | 40-65 px |

The job painted with those points. In the frontal kiss clip the replacement on the face behind shrank and slid toward the
visible side, **only 50-57 % of that face's visible area was painted** (97 % with the faces apart), and it moved from frame to
frame. That is the "little revert and flicker on contact".

### 1.2 A kiss in side view: the occlusion model takes either face for "the face"

Both faces fill each face's crop. The occlusion model (XSeg) then segments one or the other from frame to frame, or a mix. On
those frames the face in front lost most of its paint: **7 reverted and 15 faded frames** of 120 in the side-view clip. The
face behind was cut by the partner's landmark outline, which on a side view is drawn far over this face: within 40 px of the
partner only **12-25 %** of it was painted.

### 1.3 Eating: the footage's own lips were shown on purpose

11.2.92 added a rule for "another lip moves above the lips": while an object is at the mouth, show the footage's own lips
with it. Measured on a new clip (a scoop on a cone coming up to the lower lip three times, `make_lick.py`):

- with the scoop on the lips, **97 % of the visible lips are the original person's**;
- between licks they fade back to the replacement's and out again: the lips' paint changes by up to **0.61 in one frame**.

That is the "lower mouth / lip appears to come out".

### 1.4 Eating: the object was cut only well inside the face

The occlusion cut was limited to the face outline taken 8 % in. The lower part of a scoop on the lower lip was painted over
(16 % of the scoop over the face, in the output video). For a scoop resting on the chin the mask laid 76 % paint on it; in
the output that showed on 2 % of it (12 % on the worst frame), because the swap model mostly redraws the object there.

## 2. What 11.2.210 changes

| | Change | Switch (`swap_engine._P`; 0 = 11.2.209) |
|---|---|---|
| 1 | **A face going behind a front-view face is placed by its own visible part.** When it begins to go under, a reference of it is taken (picture and position, both still good). Its position is then where that reference is found in each frame. The detector is believed less and less as more is covered, and not at all from 15 % on; it takes over again as the face comes out. No jump either way. | `hid_hold` 1 |
| 2 | When nothing of the face can be followed: if it was not moving against the face in front (a kiss, a hug), it stays with that face. If it was (it passes behind), it keeps its own motion for 5 frames, then the job's own rules take it back over 6 frames, as in 11.2.209. | `hid_lost` 5, `hid_rel` 0.02 |
| 3 | For a carried face, the face in front cuts with its forehead too, and with a shorter soft edge. | `rival_forehead` 0.32, `rival_feather_hid` 0.05 |
| 4 | **Side-view pairs** (both turned 45° or more): each face is painted on its own side of the line between the two faces. The partner's outline no longer cuts, the occlusion model no longer cuts, and its face map is used only when it shows this face (else the last one that did). | `contact_side` 1 |
| 5 | **Eating: the replacement's lips stay on.** The 11.2.92 rule is off. | `eat_mouth` 0 (1 = 11.2.209) |
| 6 | The cut sits on the object's own edge (it reached 2-4 px past it, a ring of the footage's lip). | `occ_net_lo` 0.40, `occ_net_hi` 0.80 |
| 7 | An object at the mouth is cut **all of it**: followed to the face's edge and under the swap's soft border, also when it comes in over the chin or rests there. | `occ_net_follow` 1, `occ_net_below` 1 |
| 8 | The face memory forgets slowly, so a slowly arriving object is still known to be in front of face. | `occ_face_forget` 0.04 |

Not changed: identity (11.2.209), the front / behind decision (11.2.176), the 11.2.206 object occlusion away from the mouth,
skin tone.

## 3. Verification

### 3.1 Two faces in contact: the whole job, Ultra, 11.2.209 → 11.2.210

Five clips of two people, 120 frames each, scored against the clips' ground truth (`identity_check.py`, `contact_lag.py`,
`strip_paint.py`, `spill_vis.py`). "A" is the face that goes behind, "B" the face in front.

**Frontal kiss** (heads tilted together, about 60 % of A behind B for 40 frames):

| | 11.2.209 | **11.2.210** |
|---|---|---|
| A's visible face painted, against the same part of the face with nothing in front: mean / lowest frame | 0.67 / 0.54 | **0.90 / 0.85** |
| Frames where that is under 0.70 | 46 | **0** |
| Largest change of it in one frame (flicker of the paint) | 0.10 | **0.05** |
| A's paint position against the moving face: mean / worst | 17.0 / 30.4 px | **10.5 / 14.0 px** |
| A's likeness to its replacement, lowest frame | 0.33 | **0.70** |
| Frames with the other person's face on B | 1 (frame 34) | **0** |
| B's identity frames correct / unclear | 117 / 2 | 119 / 1 |
| B's faded frames | 1 (frame 34) | 1 (frame 34) |
| A's paint left visible on B (of B's area) | 1.7 % | 2.1 % |

**Kiss in side view** (two profiles, lips touching for 80 frames):

| | 11.2.209 | **11.2.210** |
|---|---|---|
| B (in front): reverted / faded frames | 7 / 15 | **0 / 0** |
| A (behind): faded frames | 12 | **0** |
| A painted next to B, mean over the kiss: within 8 px / 8-20 px / 20-40 px | 13 % / 14 % / 20 % | **64 % / 75 % / 89 %** |
| Identity frames clearly the right replacement, A / B | 39 / 98 | **72 / 115** |
| Frames with the other person's face | 0 | 0 |
| Paint left on the other person: A on B / B on A (px, of about 70 000) | 130 / 170 | 549 / 27 |

**One face mostly hides the other** (two thirds of A behind B for 40 frames):

| | 11.2.209 | **11.2.210** |
|---|---|---|
| A's paint position: mean / worst | 39.0 / 66.3 px | **25.6 / 41.2 px** |
| Largest jump of it in one frame | 8.0 px | **4.9 px** |
| A's likeness to its replacement, lowest frame | 0.34 | **0.65** |
| A's visible face painted, against the same part with nothing in front: mean / lowest | 0.91 / 0.72 | **0.97 / 0.90** |
| B's identity frames correct / unclear | 117 / 3 | 119 / 1 |
| A's frames counted "faded" by `identity_check.py` | 3 | 27 (see below) |
| A's paint left visible on B (of B's area) | 2.4 % | 4.9 % (limit 3) |

`identity_check.py` compares the paint on what is visible with the average over a whole face. What is visible of A here is
the outer edge of its face, where the swap is always soft. 11.2.209 painted a whole shrunken face into that strip: more
paint, at the wrong place. Against the same part of the face with nothing in front (the row above it) 11.2.210 paints more.

**Crossing** (A passes fully behind B and comes out on the other side):

| | 11.2.209 | **11.2.210** |
|---|---|---|
| A's paint position: mean / worst | 23.9 / 104.9 px | 17.2 / 104.9 px |
| Largest jump of it in one frame | 5.7 px | 22.9 px |
| A's faded frames / reverted frames | 8 / 0 | 9 / 0 |
| A's likeness to its replacement, lowest frame | 0.38 | **0.64** |
| B's faded frames | 4 | 5 |

Better while A goes under, the same while it comes out (limit 2). The jump of 22.9 px falls where the carry hands A back to
the job's own rules, with A almost fully hidden.

**Harsh step** (two people who never touch): 240 of 240 identity frames correct, placement 1.6 / 1.5 px mean and
4.8 / 2.7 px worst. Identical to 11.2.209; no face was carried.

Mix-ups in all five clips: 1 → 0. Flicker inside the faces (output against input): unchanged, 0.96-1.04.

### 3.2 The carry alone, against the true position

`hidden_hold.py`: the real detector's readings of both people on every frame, handed to the new step as the job does. Person
A's 5 points against the clip's true ones, on the frames it is carried:

| Clip | Frames carried | Detector (what 11.2.209 painted with): mean / worst | 11.2.210: mean / worst | Largest change in one frame: detector / 11.2.210 |
|---|---|---|---|---|
| Frontal kiss (smooch) | 65 | 30.7 / 42.4 px | **12.2 / 12.6 px** | 7.4 / 4.3 px |
| One face mostly hides the other (hide) | 72 | 44.8 / 63.6 px | **14.0 / 17.5 px** | 8.3 / 2.5 px |
| Crossing (the face passes fully behind) | 26, then let go | 43.9 / 100.7 px | 15.4 / 83.6 px | the same (106.9 px, when the face is found again on the other side) |

With the faces apart it carries nothing (0 of 8, 21 and 48 frames). About 5 px of every number is the detector's own
frame-to-frame scatter. In the crossing clip the worst value falls in the 6 frames in which the position is handed back to
the job's own rules.

### 3.3 Eating

The clip: a scoop on a cone comes up to the lower lip, rests there, goes down to the chin and comes back, three times
(`make_lick.py`; the upper lip and the mouth corners stay visible).

**The whole job, Ultra** (`lick_job.py`: the engine's own paint, and the output video against the footage):

| | 11.2.209 strawberry | **11.2.210 strawberry** | **11.2.210 vanilla** |
|---|---|---|---|
| Visible lips shown as the replacement's, scoop on the lips | 0.04 | **0.88** | **0.90** |
| The same between licks: mean / lowest frame | 0.53 / 0.00 | **1.00 / 0.95** | **1.00 / 0.95** |
| Largest change of the lips in one frame | 0.61 | **0.11** | **0.09** |
| Scoop painted over where it covers the face, on the lips: mean / worst frame | 16 % / 20 % | **0 % / 3 %** | **0 % / 0 %** |
| The same with the scoop under the lips | 2 % / 12 % | **0 % / 0 %** | **0 % / 1 %** |

**The mask alone, three colours** (`lick_mask.py`: the real detector and landmarks on every frame, the real mask builder):

| | 11.2.209 strawberry / vanilla / chocolate | 11.2.210 |
|---|---|---|
| Visible lips painted with the replacement, scoop on the lips | 0.03 / 0.02 / 0.04 | **0.90 / 0.92 / 0.91** |
| Face right beside the scoop (8 px) painted | 0.43 / 0.37 / 0.38 | 0.78 / 0.72 / 0.77 |
| Paint on the scoop over the face, on the lips | 0.15 / 0.14 / 0.13 | **0.01 / 0.00 / 0.00** |
| Paint on the scoop resting on the chin | 0.76 / 0.76 / 0.77 | **0.08 / 0.11 / 0.03** |
| Paint on the scoop under the swap's soft border, beyond the face | 0.69 / 0.68 / 0.69 | **0.08 / 0.10 / 0.08** |

The lips are not at 1.00 with the scoop on them because a thin edge beside the scoop still shows the footage (limit 6).

### 3.4 Nothing else moved

All on the final files (checksums compared after the runs).

| Check | Result |
|---|---|
| Harsh-step clip, two people who never touch, full job | 240 of 240 identity frames correct, placement 1.6 / 1.5 px mean, 4.8 / 2.7 px worst: **the same as 11.2.209**. Nothing was carried (no `TOUCH` line). |
| The same with Detect pictures, both right | 120 / 120 correct for both: the same |
| The same with a wrong picture for person B (11.2.208/209 fallback) | B found by elimination at frame 52, 68 of 68 frames correct, 0 mix-ups: the same |
| Object clips of 11.2.206 (`unit206.py`, 11.2.205 against 11.2.210) | 8 of 8 checks pass. Hand over the eye: paint on the object 0.94 → 0.04. Cup at the cheek: 0.67 → 0.004 (11.2.206: 0.07), the whole cup 0.75 → 0.06 (11.2.206: 0.31). Rest of the face: 0.5 % and 0.7 % painted less on average. |
| Faces with nothing in front, 36 runs incl. noise, blur, dark, bright, JPEG 25 (`nofalse206.py`) | 0.00 % more of any face cut than 11.2.205 |
| With the new switches off (`unit210.py`) | the mask equals 11.2.209's on all 150 frames of the eating clip (difference 0) |
| Unit suites | `unit210` 19, `unit209` 13, `unit206` 8, `unit205` 30, `unit204` 7, `unit203` 18, `unit176` 12, `unit176_pipe` 14: all pass |

The 11.2.206 object clips as whole jobs (`obj_metrics.py`, 11.2.206 → 11.2.210): hand over the eye painted over 3.8 % →
0.0 %; cup at the cheek 1.1 % → 0.2 %, with more of the rest of the face kept (36 % → 43 % of the face area changed);
hand sweeping across the face 1.0 % → 0.7 %, rest of the face 34 % → 40 % on average, its lowest frame 26 % (11.2.206: 28 %).

**Cost** (`cost.txt`, 2 threads): the face mask 177.6 → 179.8 ms per face; the carry step 0.07 ms per frame with the faces
apart, 7.8 ms while a face is carried. Against about 3 s per frame of the Ultra swap this does not show.

## 4. What the log says now

`TOUCH: a face partly behind another was placed by its own visible part on N face-frames (K times: found by its reference on
..., by its last motion on ...; handed back to the detector ..., let go when nothing of it was left to follow ..., found again
elsewhere ...). Side-view pairs: M face-frames painted on their own side of the line between the faces (...)`

The `EATING:` line no longer appears (the rule is off).

## 5. Limits

1. **The first frames of a contact.** Which face is in front is still decided as in 11.2.176-209 (identity, sight, box
   size). On the frontal kiss clip the face in front still loses part of its paint on one frame (frame 34) before the
   decision settles. It no longer shows the other person's face there in these runs, but the cause is not removed.
2. **A face that passes fully behind another** is carried only while something of it can be followed. After that it is placed
   as in 11.2.209, and its frames while it comes out on the other side are as faded as before (§3.1, crossing).
3. **A face more than about two thirds hidden for a long time** (the hide clip): the carry holds it with the face in front
   when it cannot be followed. Its position is better than 11.2.209's but not exact (26 px on average). Placed where it
   really is, more of its paint lies under the head in front: the cut uses that face's outline and forehead, not its hair,
   so 4.9 % of the front person's area (hair, temple) carries some of it (2.4 % before). In the output frames I looked at
   this did not show on the hair, but real hair differs.
4. **Side-view pairs:** the line between the faces is straight; real lips and noses interlock. A thin strip beside the line
   can carry the other face's paint (§3.1, "paint left on the other person").
5. **Three-quarter views (30-45°)** fall under the front-view rules. There the outline of the face in front is less exact.
   Not tested: I have no synthetic three-quarter contact clip.
6. **Eating.** The replacement's lips are drawn by the swap model, which cannot see lips hidden by the object. It may draw
   them slightly off where the object covers them. A thin edge (about 2 px in the face crop) of the footage still shows
   next to the object. A wide-open mouth or a tongue was not tested: the synthetic face has a closed smile.
7. **The carry follows a flat motion** (shift, turn, size). A head that turns sideways while mostly hidden is followed only
   approximately.
8. Synthetic clips only; no footage was looked at. Whether your two reports are fully covered can only be seen on your
   videos.

## 6. Install

Upload the pack's files to the Space. Changed against 11.2.209: `swap_engine.py`, `core_pipeline.py`, `config.py`. No change to
`packages.txt` or the app. The log at start reads `Phoenix v11.2.210`.

If the mouth looks worse to you than before while eating, `eat_mouth = 1` in `swap_engine.py` brings back 11.2.209's lips rule
on its own; the tighter cut stays.
