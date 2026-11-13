# SDOS-206 — OcclusionFlow: a hand, cup or phone in front of the face is shown cleanly, from the frame it arrives

**Pack:** v11.2.206, which is v11.2.205 with four changes to the occlusion mask. Engine `aequus-1.2.206-occlusionflow`.
Nothing else changed: the swap, the tracker, the contact rules, colour, restoration, skin tone and the app are 11.2.205's.

**Asked:** "Is there any further opportunity to enhance occlusion smoothness", then "Pls share vs 11.2.206 now". This builds
items 1-4 of the opportunities listed in `Comparison_v11.2.92_vs_v11.2.205.md` (item 5, cut-to-cut between two faces, is not in
this pack).

All tests use synthetic media: insightface's public sample photo, with a hand, cup or phone (drawn shapes in skin and object
colours) moved over a face (`make_objects.py`), and the two-person contact clips made from the same photo. No user footage was
looked at.

---

## 1. What changes on screen

| Situation | 11.2.205 | 11.2.206 |
|---|---|---|
| A hand over an eye or a cheek, a cup at the cheek, a phone at the nose | Only the colour test cuts there; a skin-coloured hand is not "new colour", so the swapped face stays painted over much of it (a see-through hand: the swapped eye shows on the back of the hand) | The occlusion model (XSeg) cuts it too, on the frame it arrives |
| A new object arrives | The colour cut opens 45 % a frame: 2-3 frames see-through | Where the colour test and the model both see it, the cut is complete on that frame |
| An object moves | Last frame's cut is smoothed where it was | Last frame's cut is moved along the object's motion (optical flow) before smoothing; where the model is sure it sees face again, the face comes back on that frame, so the new cut leaves no ghost |
| The edge of the cut | The model's soft 256 px edge: a blurred band a few pixels wide | Fitted to the picture's own edges (guided filter): a cleaner line where a finger or cup crosses the face |
| An object leaving | The model's cut dropped as soon as the piece left over the face got small | The piece that was cut on the last frame stays cut while it leaves |

Hands or objects at the mouth (eating, drinking, a microphone) work as in 11.2.205, plus the instant cut, the motion and the
clean edge.

## 2. How

All in `swap_engine.py`, in the mask step (`build_mask`, `occ_net_keep`) that every tier runs, Ultra included. Four switches in
`_P`; all four at 0 give exactly 11.2.205's mask (checked to 0.00 difference on every frame, §3.1):

| Switch | What it does | Cost |
|---|---|---|
| `occ_flow` | DIS optical flow (OpenCV, ultra-fast preset) between this and the last face crop, at 128 px. Last frame's cuts (colour and model) are moved along it before they are smoothed. A shot cut (the crop changed completely) carries nothing over. | with the edge fit: +7 to +19 ms per face and frame (§3.5) |
| `occ_instant` | Where the cut is opening and the model sees an object, it opens fully on this frame (the slow give-back, 22 % a frame, is kept) | none |
| `occ_snap` | A guided filter on the cut's edge band only (the inside of the cut and the rest of the face are not touched); runs only on frames with a cut | (above) |
| `occ_net_wide` | The model's cut anywhere inside the face outline, not only at the mouth and chin, for big pieces (at least 1 % of the face crop, at least a quarter of it over the inner face). An accepted piece is followed out only where the model saw face on the last frames (a per-face memory), so it never reaches hair, the background or the swap's soft border. A piece that overlaps last frame's cut by 30 % stays cut while the object leaves; where the model is sure it sees face again, that face comes back at once. The lips (with a margin) are left out and stay under 11.2.205's mouth rule, and the eating rule sees only that rule's cut, so a hand on the cheek is not taken for eating. | none (the model already ran) |

The guards against false cuts:
- the wide cut stands aside where 11.2.205's own rules decide: two faces in contact (the face behind is cut by the face in
  front, as before) and a face turned more than about 35 degrees (in a side view the landmark outline reaches past the face,
  where the model rightly sees no face; a first version cut the front face of the side-view kiss there, see §3.3);
- hair along the face's edge is a thin piece at the outline, so it never has a quarter of itself over the inner face;
- nothing is joined in where the model has not seen face;
- a smile's teeth (the model reads them as "not face") stay under the old mouth rule.

§3 measures them. The tuning values (piece sizes, the memory, the yaw limit) are in `_P` next to the switches.

## 3. Verification

### 3.1 The mask, frame after frame (`verification/fastrestore/unit206.py`)

`make_objects.py` clips (80 frames: the object slides in over 4 frames, holds 34 frames with a small jitter, leaves over
4 frames). The hand is skin-coloured, sampled from the face itself — the hardest case for the colour test. The mask of
11.2.205 and of 11.2.206 on the same frames. "Paint over the object" is how much of the swapped face is still painted over
the object, where it covers the face; 0 = the object fully shown.

| Clip | All switches 0 = 11.2.205 | Paint over the object, 11.2.205 → 11.2.206 | The rest of the face painted less than by 11.2.205 (mean / worst frame) |
|---|---|---|---|
| Hand over the eye | identical (0.00 difference, 70 frames) | **0.94 → 0.06** | 0.06 % / 1.1 % |
| Cup at the cheek | identical | **0.67 → 0.07** | 0.57 % / 4.9 % |
| Phone at the nose | identical | 0.050 → 0.004 | 0.22 % / 3.4 % |
| Hand sweeping across the face | identical | 0.36 → 0.20 | 0.10 % / 0.5 % |
| Hand on the cheek | identical | 0.37 → 0.25 | 0.03 % / 0.7 % |
| Hand on the forehead | identical | 0.48 → 0.48 (**no change**, §4) | 0 / 0 |

The last column is the guard against new mistakes: while the object is there and after it moves on, the rest of the face is
painted as by 11.2.205 (the few % on the worst frame are the edge of the cut on the frame the object leaves).

**No false cuts** (`nofalse206.py`): the six faces of insightface's sample photo (hair along the faces, some turned), each
drifting and shaking over 16 frames, nothing in front: 11.2.206 cuts **0.00 %** more of any face than 11.2.205.

Two versions were rejected on the way, both caught by these checks:
- the model's cut reaching the forehead (the painted area above the eyebrows) cut 45-59 % of faces with nothing in front of
  them, because the painted area there includes hair;
- following an accepted piece through everything the model calls "not face" (instead of only where it saw face before)
  stopped painting the swap's soft border around the face while a hand was up: the swap visibly shrank. The job test caught
  it (the rest of the face 45 % → 25 % swapped, 23-39 reverted frames).

### 3.2 The real job (`bench_pipeline.py`, Ultra, one face swapped, 2 threads; `obj_metrics.py`)

The full Space pipeline (detector, swap model, compositor) on the same six clips. (Run before the contact / side-view guard
of §2 was added. The guard does not act on these clips: one frontal face, no second face swapped. The mask test of §3.1 gives
the same numbers with it.) "Painted over" = share of the object
(eroded 3 px) changed by more than 12 grey levels from the footage. A swapped face painted over a skin-coloured hand on a
plain cheek often differs from it by less than that, so the gain shows most where the face has features under the object
(the eye clip).

| Clip | | Painted over: entering / first full frame / holding (worst) | Rest of the face swapped (before → during) | Reverted frames |
|---|---|---|---|---|
| Hand over the eye | 11.2.205 | 21.0 % / 38.3 % / **38.1 % (41.2 %)** | 45 % → 45 % | 0 |
| | 11.2.206 | 5.4 % / 2.1 % / **3.8 % (5.0 %)** | 45 % → 45 % | 0 |
| Cup at the cheek | 11.2.205 | 0.7 % / 2.1 % / 1.5 % (2.2 %) | 45 % → 36 % | 0 |
| | 11.2.206 | 0.9 % / 2.1 % / 1.1 % (2.1 %) | 45 % → 36 % | 0 |
| Phone at the nose | 11.2.205 | 6.7 % / 0.0 % / 0.0 % | 45 % → 32 % | 0 |
| | 11.2.206 | 2.4 % / 0.0 % / 0.0 % | 45 % → 32 % | 0 |
| Hand sweeping across | 11.2.205 | 3.1 % / 0.7 % / 1.9 % (8.4 %) | 45 % → 34 % | 0 |
| | 11.2.206 | 3.2 % / 0.9 % / 1.0 % (3.7 %) | 45 % → 34 % | 0 |
| Hand on the cheek | 11.2.205 | 3.7 % / 2.3 % / 3.8 % (5.3 %) | 45 % → 29 % | 0 |
| | 11.2.206 | 3.7 % / 2.9 % / 4.4 % (5.8 %) | 45 % → 29 % | 0 |
| Hand on the forehead | both | 0.5 % / 0.6 % / 1.3 % (2.6 %) | 45 % → 51 % | 0 |

The model's cut was used on: hand over the eye 0 → 39 of 80 frames; cup 17 → 36; sweep 27 → 33; phone 37 → 39; cheek 35 → 36;
forehead 0 → 0. "Rest of the face during" is the same for both versions on every clip: the face around the object stays
swapped exactly as before, and nothing reverted.

### 3.3 Two people in contact: nothing lost (the same five clips as the 11.2.92 comparison, Ultra)

The full job with two faces swapped, scored against the clips' ground truth (`identity_check.py`, `contact_lag.py`). The
first 11.2.206 build cut the front face of the side-view kiss (the face in profile, where the outline reaches past the face):
2 more weak frames, 7 frames less clearly the right person. The guard (§2: not on contact or turned faces) removed that; the
final numbers:

| Clip | | Identity correct / unclear | Mix-ups | Reverts | Weak frames | Placement mean A / B | Worst A / B | Flicker A / B |
|---|---|---|---|---|---|---|---|---|
| Harsh step | 11.2.205 | 240 / 0 | 0 | 0 | 0 | 1.6 / 1.5 px | 4.8 / 2.7 px | 1.00 / 0.98 |
| | 11.2.206 | 240 / 0 | 0 | 0 | 0 | 1.6 / 1.5 px | 4.8 / 2.7 px | 1.00 / 0.98 |
| Side-view kiss | 11.2.205 | 136 / 104 | 0 | 7 | 27 | 7.9 / 3.7 px | 16.3 / 7.7 px | 1.02 / 0.97 |
| | 11.2.206 | 136 / 104 | 0 | 7 | 27 | 7.9 / 3.7 px | 16.3 / 7.7 px | 1.02 / 0.97 |
| Crossing | 11.2.205 | 191 / 5 | 0 | 0 | 12 | 23.9 / 0.7 px | 104.9 / 1.6 px | 1.01 / 1.00 |
| | 11.2.206 | **192 / 4** | 0 | 0 | 12 | 23.9 / 0.7 px | 104.9 / 1.6 px | 1.01 / 1.00 |
| One face hides behind the other | 11.2.205 | 164 / 3 | 0 | 0 | 3 | 39.0 / 7.2 px | 66.3 / 10.6 px | 1.00 / 1.01 |
| | 11.2.206 | 164 / 3 | 0 | 0 | 3 | 39.0 / 7.2 px | 66.3 / 10.6 px | 1.00 / 1.01 |
| Smooch | 11.2.205 | 168 / 2 | 1 | 0 | 46 | 17.0 / 3.9 px | 30.4 / 6.5 px | 1.01 / 1.01 |
| | 11.2.206 | 168 / 2 | 1 | 0 | **45** | 17.0 / 3.9 px | 30.4 / 6.5 px | 1.01 / 1.01 |

The same or one frame better everywhere. (The smooch's one mix-up frame is 11.2.176's, as in the 11.2.92 comparison.)

### 3.4 Earlier checks, on this pack

`unit176` 12, `unit176_pipe` 14, `unit203` 18, `unit204` 7, `unit205` 30: all pass. `unit206`: all pass on five clips; on the
forehead clip the "less paint" check fails because nothing changes there (§4).

### 3.5 Cost

The mask step timed alone, 11.2.205 and 11.2.206 one after the other on the same two cores (`time206.py`, 70 frames, best of
3): **+6.6 ms** per face and frame at a 256 px face crop (199.7 → 206.3 ms), **+19.2 ms** at 512 px (290.5 → 309.7 ms). The
swap model alone takes 1.7-2.4 s per face and frame on this test machine, so a job takes about 1 % longer at most. (The job
times in the tables above ran beside other work and are not comparable.)

## 4. Limits

1. **A hand on the forehead** (above the eyebrows) is as in 11.2.205: the model's cut zone ends at the face's outline, and
   widening it to the forehead cut hair on faces with nothing in front (§3.1).
2. **The outermost band of the face** (about the outer 3 % of the face crop, the swap's soft border) is as in 11.2.205: a cup
   or hand there keeps some of the swap over it. That is why the cheek clips gain less than the eye clip.
3. **At the mouth** the 11.2.205 rules stand unchanged (eating, drinking, a microphone): the lips are left out of the new cut,
   because the model reads a smile's teeth as "not face".
4. **A skin-coloured hand on a plain cheek** looked almost the same before and after in the job test (3.8 % vs 4.4 % of it
   changed by more than 12 levels): where the face under the hand is plain skin, the swapped face over it is hard to see either
   way.
5. **Synthetic tests only:** drawn objects over a still photo, no real motion blur or changing light; the hand is a flat shape
   with finger strips. The occlusion model (XSeg) is a trained network and can miss an object; then the mask is 11.2.205's.
6. **Side views and faces in contact** (a kiss, cheek to cheek) keep 11.2.205's mask apart from the instant cut, the motion and
   the clean edge: a hand over a face turned more than about 35 degrees is cut as in 11.2.205.
7. **All tiers:** the change is in the mask step that Ultra, Optimized and Balanced share. It does not lower Ultra's quality
   anywhere in these tests (the rest of the face, the contact clips), but it is a change to Ultra. Setting `occ_flow`,
   `occ_instant`, `occ_snap` and `occ_net_wide` to 0 in `swap_engine.py` gives exactly 11.2.205.

## 5. Install

**Space:** upload the pack's files. Changed:
- `swap_engine.py`: the mask step;
- `core_pipeline.py`: the summary log line;
- `config.py`: the version.

No new model, no new package: DIS flow and the guided filter are in `opencv-contrib-python`, which the Space already has (the
skin tone of 11.2.205 uses the same guided filter). If either is missing the step is skipped and the mask is 11.2.205's.

The log at start reads `Phoenix v11.2.206`. At the end of a job:
`OCCLUSION-NET: something in front of the face was cut on N of M face-frames (... away from the mouth: X pieces, Y held while leaving)`.

**App:** no change. Phoenix Mobile 1.2.58 (Azure) works with this pack as with 11.2.205.
