# SDOS-213 — TrueTurn: a head turned far away is no longer painted with a face drawn at another angle

**Pack:** v11.2.213, built on v11.2.212. Engine `aequus-1.2.213-trueturn`.

**Asked:** "On extreme side poses.. e.g. where a character is facing away from camera with face very little towards right
left.. where a negligible part of the face is visible say only 10-20%. The pasting is happening with wrong angle of pasted face..
Please fix this."

No footage was looked at. A photo cannot turn, so the test footage is a **3D head** rendered at known angles: the public
"Lee Perry-Smith" head scan (Infinite-Realities, CC BY 3.0, as shipped with the three.js examples), turned from facing the camera
to 110 deg away and back (`make_turn.py`, `head3d.py`; every frame's true turn known), plus the 16 public AFLW2000 sample photos
with their annotated head angles, and the earlier contact clips. Each clip was run through **the whole job at Ultra**, 11.2.212
against 11.2.213.

---

## 1. What was wrong

### 1.1 The swap model cannot draw a face turned that far, and draws a less turned one instead

`inswapper_128` draws the replacement inside a front-facing 128-px face template. Up to a strong profile it follows the head;
beyond about 78 deg of turn it does not: it draws the replacement **less turned than the head** - an eye, the nose's bridge and
the full lips set onto the cheek, the face looking another way than the head it sits on. On the rendered head:

| the head's true turn | what 11.2.212 painted |
|---|---|
| up to about 75 deg | the replacement, following the head (a three-quarter to profile face) |
| 78-85 deg | the replacement drawn at about a three-quarter turn: eye and lips set inward, nose inside the profile |
| 85-100 deg | the same, now plainly a face looking another way, painted across the cheek toward the ear |
| past profile (100-110 deg) | a faint copy of it on the cheek, or an eye fragment |

(`verification/trueturn213/sheet_turn110_212_vs_213.png`, `sheet_turnL_212_vs_213.png`.)

### 1.2 Nothing stopped it

The face detector still finds a head turned 90-110 deg away (score 0.57-0.73), its 5 points bunched at the edge of the profile.
The landmark gate drops most of those detections, but the person stays painted through them (carried), and for a lone face a
weak detection is kept on purpose. The only thing that changed with the turn was the profile mask, which shrinks the painted
area (to 0.06-0.25 of the crop) - inside it the replacement was still drawn at the wrong angle.

### 1.3 The engine's own turn measure cannot tell 70 deg from 90

The tracker's turn measure (the nose's offset from the eyes' midpoint, in eye distances) reads 0.76 on a real 71 deg profile
photo and 3.4 on the rendered head at 72 deg, and beyond 80 deg (eyes 0-15 px apart) it is noise. A fix keyed on it would cut
real profile kisses or miss the turned heads.

---

## 2. What 11.2.213 changes

| | Change | Switch (`swap_engine._P`; 0 = 11.2.212) |
|---|---|---|
| 1 | **The head's turn is read by insightface's 3D-landmark model** (`1k3d68.onnx`, in the same `buffalo_l` face pack the Space already downloads; the job did not load it before). Asked only for a face whose 5 points look turned (about 38 deg and more) or whose track was turned; a front or three-quarter face costs nothing. Read before the landmark gate, so a detection the gate drops still tells the track how far the head has turned. Smoothed on the track (a more turned reading counts 0.7, a less turned 0.5). | `turn_fade` 1, `turn_check` 0.40, `turn_ema_up` 0.70, `turn_ema_down` 0.50 |
| 2 | **Beyond what the swap can draw, the replacement gives way to the person's own face**: full replacement up to a reading of 78 deg, none from 86 (raised cosine between) - on the rendered head, full up to a true 75 deg, about a quarter at 85, none from 90. Applied on top of the paint's own fade, which is left as it was. Coming back, the replacement returns as the head turns back. | `turn_lo` 78, `turn_hi` 86 |
| 3 | **Past profile**: a head turning on, away from the camera, is misread by the model as a 62-75 deg turn - but with its nod and tilt both wild (40-99 deg; at a real 66-80 deg turn they read 16-29). For a head that has already come through the profile such a reading is taken as turned away; a head seen for the first time is read as it shows (a real face turned 66 deg and looking up 40 deg can read like that). | `turn_past_pr` 40 |

Not changed: anything below a 78 deg reading - which includes every contact clip of the earlier versions (profile kisses read
72-76 deg, all others 50 or less) - identity, occlusion, contact, eating, skin tone, quality settings. Ultra and Optimized alike:
this is a fix, not a speed setting.

## 3. Verification

### 3.1 The turning head: how much of the visible face the swap changed, by the head's true turn

The whole job at Ultra on two rendered turns (one to 110 deg, one the other way to 100 deg with the head tilted), each also as
a phone-quality copy (camera noise, blur, H.264 at crf 30). Share of the visible face the output changed (more than 10 grey
levels from the input; a fully swapped front face is about 0.30), mean over the frames in each band (`turn_change.py`):

| true turn | clip 1: 212 | clip 1: 213 | clip 2: 212 | clip 2: 213 | clip 1 phone: 212 | 213 | clip 2 phone: 212 | 213 |
|---|---|---|---|---|---|---|---|---|
| 0-45 | 0.32 | 0.32 | 0.32 | 0.32 | 0.30 | 0.30 | 0.30 | 0.30 |
| 45-60 | 0.26 | 0.26 | 0.29 | 0.29 | 0.24 | 0.24 | 0.27 | 0.26 |
| 60-70 | 0.25 | 0.25 | 0.26 | 0.26 | 0.22 | 0.22 | 0.24 | 0.24 |
| 70-76 | 0.23 | 0.22 | 0.25 | 0.20 | 0.21 | 0.18 | 0.23 | 0.17 |
| 76-80 | 0.22 | 0.12 | 0.24 | 0.12 | 0.19 | 0.05 | 0.23 | 0.09 |
| 80-85 | 0.22 | **0.04** | 0.25 | **0.05** | 0.20 | **0.02** | 0.23 | **0.02** |
| 85-90 | 0.22 | **0.00** | 0.25 | **0.01** | 0.21 | **0.00** | 0.24 | **0.00** |
| 90-100 | 0.24 | **0.00** | 0.24 | **0.00** | 0.23 | **0.00** | 0.25 | **0.00** |
| past 100 | 0.06 | **0.00** | 0.22 | **0.00** | 0.04 | **0.02** | 0.24 | **0.00** |

The replacement is the same as 11.2.212's up to 70 deg, gives way between about 72 and 85, and from 85 deg on the head is its own.
It goes smoothly: the largest change of that share from one frame to the next is 0.08-0.09 in 11.2.213 (0.03-0.09 in 11.2.212),
no frame-to-frame change over 0.15 in any run - no on / off. The bench's own first build of this version left a faint smudge
past profile (the model's misreading, change 3 fixes it; `sheet_past_profile_first_build.png`).

### 3.2 Nothing else moved

The whole job at Ultra, 11.2.212 against 11.2.213, every output frame and every paste mask compared:

| clip | turn as read | result |
|---|---|---|
| three public profile photos (AFLW2000: 71 deg; 66 deg looking up 40; 55 deg) | up to 75 | **the same video, pixel for pixel** (the turn read on every frame, never given back) |
| two people kissing (`smooch`) | 20 | **the same video** |
| eating (an ice cream at the mouth) | 3 | **the same video** |
| profile kiss (two faces at 72-76 deg) | up to 76 | paste masks equal on every frame; the output differs by about 1 grey level over the whole picture - as much as 11.2.212 run twice differs from itself on this clip (up to 37 levels), so not from this change |

(`verification/trueturn213/same_video.txt`.) A first set of profile-photo clips had a mirrored part of the head at the frame's edge
(the clip maker filled the border by reflection) and the detector tracked that instead of the face; they were remade over a blurred
background (`make_profclip.py BG=blur`) and the runs above are on those.

### 3.3 The pose model against the true turn

insightface's 3D-landmark model, on the rendered head at known angles and on the 16 AFLW2000 photos with their annotated angles
(`pose_model_against_true_turn.txt`): on the head within 5 deg of the true turn up to 80 deg, 84-89 beyond 90 (it does not read
past profile, §2 change 3); on the photos within about 8 deg. 26-56 ms per read on this machine's CPU, asked only for faces that
look turned. The "past profile" pattern (a turn of 55+ with nod and tilt both 40+) was not read once on any face of the profile
kiss, the profile photos (remade), the contact or the eating clips (`past_profile_pattern_on_other_clips.txt`).

### 3.4 Unit checks

`unit213.py` (18 checks, among them: the weight by turn; a front face never asks the model; a turned face is followed within 3
detections; past profile kept as turned away, a fresh nodded 3/4 face not; turning back restores the replacement within 2
detections; a carried face takes its track's turn; a missing or failing model changes nothing; the real model on the rendered
head: full replacement up to a 75 deg turn, none from 90) and every earlier suite on the 11.2.213 files: `unit212` 17, `unit211`
23, `unit210` 19, `unit209` 13, `unit206` 4 + 4, `unit205` 30, `unit204` 7, `unit203` 18, `unit176` 12, `unit176_pipe` 14,
`nofalse206` 1 - **180 checks, all pass** (`unit_suites.txt`).


## 4. What the log says now

At the start: `turn model loaded: a head turned beyond what the swap can draw gives way to its own face`. If the face pack has no
`1k3d68.onnx`: a warning, and turned heads are painted as in 11.2.212. At the end: `TURN: heads read for their turn N times (past
profile N); frames where a head turned beyond what the swap can draw was given back to its own face N, partly N`.

## 5. Limits

1. **Beyond about 80 deg the person's own face shows** - which is what this version does on purpose: no swap model of this kind
   can draw the replacement at that angle, and a face drawn at another angle is what you reported. With only 10-20 % of the face
   visible it is the cheek, the nose's tip and an ear.
2. **Between about 78 and 86 deg (as read) the two are mixed** for a few frames while the head turns through.
3. **The thresholds come from one head scan and 16 photos.** The model read the true turn within about 5 deg on the scan and 8 on
   the photos; a face it reads differently (strong make-up, glasses, low light, motion blur) moves the point where the
   replacement gives way by as much.
4. **A head first seen already turned past profile** (someone walking into the picture with their back half to the camera) is
   read as it shows, so its first frames are painted as in 11.2.212 until it turns.
5. Synthetic and public test media only; no footage was looked at. Whether your clips are fully covered can only be seen on them.

## 6. Install

Upload the pack's files to the Space. Changed against 11.2.212: `swap_engine.py`, `core_pipeline.py`, `config.py`. No change to
`packages.txt`, `requirements.txt` or the app. The log at start reads `Phoenix v11.2.213`.

`turn_fade = 0` brings back 11.2.212's painting of turned heads.
