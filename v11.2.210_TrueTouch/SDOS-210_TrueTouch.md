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
(15 % of it), and a scoop resting on the chin was **76 % painted over** with the replacement's chin.

## 2. What 11.2.210 changes

| | Change | Switch (`swap_engine._P`; 0 = 11.2.209) |
|---|---|---|
| 1 | **A face going behind a front-view face is placed by its own visible part.** When it begins to go under, a reference of it is taken (picture and position, both still good). Its position is then where that reference is found in each frame. The detector is believed less and less as more is covered, and not at all from 15 % on; it takes over again as the face comes out. No jump either way. | `hid_hold` 1 |
| 2 | When nothing of the face is left to follow (it passes fully behind), the carry lets go after 5 frames and the job's own rules place it, as in 11.2.209. | `hid_lost` 5 |
| 3 | For a carried face, the face in front cuts with its forehead too, and with a shorter soft edge. | `rival_forehead` 0.32, `rival_feather_hid` 0.05 |
| 4 | **Side-view pairs** (both turned 45° or more): each face is painted on its own side of the line between the two faces. The partner's outline no longer cuts, the occlusion model no longer cuts, and its face map is used only when it shows this face (else the last one that did). | `contact_side` 1 |
| 5 | **Eating: the replacement's lips stay on.** The 11.2.92 rule is off. | `eat_mouth` 0 (1 = 11.2.209) |
| 6 | The cut sits on the object's own edge (it reached 2-4 px past it, a ring of the footage's lip). | `occ_net_lo` 0.40, `occ_net_hi` 0.80 |
| 7 | An object at the mouth is cut **all of it**: followed to the face's edge and under the swap's soft border, also when it comes in over the chin or rests there. | `occ_net_follow` 1, `occ_net_below` 1 |
| 8 | The face memory forgets slowly, so a slowly arriving object is still known to be in front of face. | `occ_face_forget` 0.04 |

Not changed: identity (11.2.209), the front / behind decision (11.2.176), the 11.2.206 object occlusion away from the mouth,
skin tone.

## 3. Verification

CONTACT_TABLE

### 3.2 The carry alone, against the true position

`hidden_hold.py`: the real detector's readings of both people on every frame, handed to the new step as the job does. Person
A's 5 points against the clip's true ones, on the frames it is carried:

CARRY_TABLE

### 3.3 Eating

EAT_TABLE

### 3.4 Nothing else moved

REGRESSION_TABLE

## 4. What the log says now

`TOUCH: a face partly behind another was placed by its own visible part on N face-frames (K times: found by its reference on
..., by its last motion on ...; handed back to the detector ..., let go when nothing of it was left to follow ..., found again
elsewhere ...). Side-view pairs: M face-frames painted on their own side of the line between the faces (...)`

The `EATING:` line no longer appears (the rule is off).

## 5. Limits

LIMITS

## 6. Install

Upload the pack's files to the Space. Changed against 11.2.209: `swap_engine.py`, `core_pipeline.py`, `config.py`. No change to
`packages.txt` or the app. The log at start reads `Phoenix v11.2.210`.

If the mouth looks worse to you than before while eating, `eat_mouth = 1` in `swap_engine.py` brings back 11.2.209's lips rule
on its own; the tighter cut stays.
