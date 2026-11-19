# SDOS-214 — CleanBite: an object half in the open mouth no longer makes the swap draw lips around it

**Pack:** v11.2.214, built on v11.2.213. Engine `aequus-1.2.214-cleanbite`.

**Asked:** "When the subject is eating something which is half out of the mouth e.g. an ice cream bar, when the mouth opens and
part of the bar is inside it looks like double lips (upper lips) have been created around the bar." And: "Please check for side
profile when eating something."

No footage was looked at. The test media are **synthetic**: a public AFLW2000 sample photo with an open mouth (image03324) and a drawn
bar (a rounded cylinder, shaded, in chocolate, cream and strawberry; horizontal, vertical and tilted) that comes up to the mouth,
goes half in, rests, moves a little and leaves, with a slow camera drift, zoom and sensor noise, 150 frames (`make_bar.py`); the
same at a 38 deg turn (image02616, a wide smile with teeth, mirrored); a phone-quality version (noise, blur, lossy H.264,
`degrade_clip.py`); and two profile photos (65 and 71 deg) with the bar held to the lips (`make_bar_side.py`). Every frame's true
bar, mouth opening and landmarks are known. Each clip was run through **the whole job at Ultra**, 11.2.213 against 11.2.214, and
the bar-free twin of each clip gives the ideal (below).

---

## 1. What was wrong

### 1.1 The swap network draws what it is shown

`inswapper_128` is given the aligned face - with the bar in the mouth - and draws the replacement face from it. It does not know
the bar is an object. What it does with a bar across the open mouth: it **draws the bar on again** (a copy of it, in its own
output) and **draws lip outlines hugging it** - a red line above the bar where the upper lip would be and another below it - while
the replacement's real lower lip comes out thin or is lost. The paste then lays the footage's own bar over that (the object is cut
from the paint, as before), so what is left visible of the network's drawing is its **second set of lips around the bar**, and,
where the cut is not exact (soft edges, a bar the colour of lips, motion), a seam or a dotted edge along the bar.

Measured on the network's own output (`probe3.py`: its 128-px output over the lower face, the bar left out, mean difference in
grey levels from the **ideal** - its output for the same frame *without* the bar). `sheet_swap_input_bar_in_or_out.png` shows the
three pictures side by side: the bar in its input (it draws the bar and the lips around it), the bar taken out (an ordinary open
mouth: the upper lip, the teeth, the lower lip), and the ideal.

| the clip | the bar in its input (11.2.213) | the bar taken out (11.2.214's erase) | the whole true bar taken out (ground truth) |
|---|---|---|---|
| chocolate, from the side | 2.65 | **1.77** | 2.08 |
| cream, from the side | 4.04 | **1.89** | 2.04 |
| strawberry, from the side | 4.00 | **1.89** | 2.03 |
| strawberry, from below | 3.01 | **1.40** | 1.76 |
| cream, from below | 2.92 | **1.47** | 1.83 |
| chocolate, head turned 38 deg | 2.46 | **1.64** | 1.84 |

(frames 45, 75, 105 of each clip; `probe3_final_table.txt`.) The engine's erase does at least as well as erasing the true whole
bar (probably because it also takes the anti-aliased rim the ground-truth mask leaves).

### 1.2 What I could not reproduce

The exact look in your clip - an upper-lip line **doubled** - did not appear on the synthetic clips as a clean second upper lip.
What does appear, on every colour and angle, is the mechanism above (lip outlines hugging the object, the lower lip weak or
missing, a seam or a dotted edge along the bar at a 38 deg turn, `sheet_job_38deg_in_213_214.png`). On the frontal clips, at the
size of the face in the picture, 11.2.213's output is already close to the ideal; the difference is in the numbers of §3.1 and in
the details of the sheets. Whether the doubled lip of your clip is this mechanism can only be seen on your clip.

### 1.3 The paste is not the cause

The object stays solid: on the frontal clips the share of the bar's pixels the swap changed near the lips is 0.01-0.05 in
11.2.213. The strawberry bar's 0.34-0.35 is the same in both versions, and 0.18 of it is also found far from the face, outside the
paste (the enhancer's colour work and the encoder on a saturated pink). The cut was already right where the net sees the object;
the network's drawing around it was the problem.

---

## 2. What 11.2.214 changes

The swap network is given the face **without the object at the mouth**. The object is inpainted out of the frame the network sees
(from its surroundings); the paste, and everything after the network, is as before - it is still the footage's own bar that shows
over the result, and the paste is told to leave the footage alone wherever the object was taken out.

| | Change | Switch (`swap_engine._P`; 0 = 11.2.213) |
|---|---|---|
| 1 | **Where the object is** - the occlusion net (`xseg_2.onnx`, already loaded for the mouth cut) on the face's own 128-px crop: the pixels it is **sure** are not face (probability under 0.05: a bar reads 0.00, the open mouth's cavity 0.04-1.0, so the mouth is not touched), inside the landmark outline, a piece that lies over the lower face and **reaches the edge of the outline** (the cavity is an island inside it). | `obj_erase` 1, `obj_erase_p` 0.05, `obj_erase_min` 0.004 |
| 2 | **All of it** - the net's edge sits a few px inside the object (it is blurred to decide), and the network draws the object on again from any piece left, so the region is grown to the object's real edge (the sharper map, under 0.35, up to 2.5 % of the crop), through the band between the eroded and the real outline (along the object's own axis, over pixels the net calls not face, 16 % of the crop), and widened by 2 % for the anti-aliased rim. Without this the swap network gets 2.5-3.8 instead of 1.4-1.9 (§1.1): a rim or a stub of the bar is enough for it to draw the bar on again. | `obj_erase_edge_blur` 1.2, `obj_erase_edge_p` 0.35, `obj_erase_edge_px` 0.025, `obj_erase_reach` 0.16, `obj_erase_dilate` 0.02 |
| 3 | **Taken out of the network's input** - Telea inpainting of that region of the frame, on a copy (the frame itself is untouched). A region over 480 px is filled at 480 (the network sees the face at 128 px, so a 4K frame costs no more than a 720p one). | `obj_erase_radius` 0.012, `obj_erase_max_px` 480 |
| 4 | **The paste leaves it alone** - the same rule on the crop the paste is built on (the net's map is shared with the mouth cut: no second pass), 1 % wider than the object, laid on after the cut's edge snap. Needed because the network no longer draws the object: the mouth cut stops at the eroded outline, and without this the object's way out through the band to the outline, and its anti-aliased rim, were covered with the face (a smear across the bar, a dark line along its edge - seen on the first builds). | `obj_erase_cut` 1, `obj_erase_cut_dilate` 0.01 |
| 5 | **Where it runs** - the job's key-swap loop (where the network really runs), on every face swapped at **Ultra, Best and Optimized** (`_ERASE_TIERS`); Fast and Balanced as in 11.2.213. Not on two faces in contact (the contact rules decide there), not on a face turned beyond a 0.60 offset (about 48 deg; the profile mask decides there), not where the crop runs out of the frame, never when the net is not loaded. Any failure gives the frame as it was. | `obj_erase_yaw` 0.60, `obj_erase_gate` 0.02 (the legacy single-frame path) |

Not changed: the mouth cut, the contact rules, identity, the turn rules of 11.2.213, skin tone, the quality settings.

`obj_erase` = 0 brings back 11.2.213 exactly.

---

## 3. Measured

### 3.1 The whole job at Ultra, against the ideal

The ideal for a clip = the same job's output on the **bar-free twin** of the clip, with the true bar (the footage's own pixels,
ground-truth alpha) laid over it: the replacement's mouth as it looks with no object, and the object as filmed (`final_err.py`).
Error = mean difference in grey levels over the lips' window (the ground-truth landmarks 52-71, padded 45 px), the bar left out, on
the 115 frames with the bar in the picture.

| the clip (Ultra) | 11.2.213 | 11.2.214 | closer to the ideal by | the bar's pixels the swap changed near the lips, 213 -> 214 |
|---|---|---|---|---|
| chocolate, from the side | 3.93 | **3.33** | 15 % | 0.021 -> 0.012 |
| cream, from the side | 6.28 | **3.61** | 43 % | 0.051 -> 0.054 |
| strawberry, from the side | 6.16 | **3.53** | 43 % | 0.350 -> 0.350 (the enhancer's colour work on a pink) |
| chocolate, GPEN + Cinematic (the app's default) | 4.26 | **3.53** | 17 % | 0.023 -> 0.014 |
| chocolate, head turned 38 deg | 4.53 | **3.40** | 25 % | 0.095 -> 0.049 |
| strawberry, phone-quality | 6.36 | **3.82** | 40 % | 0.337 -> 0.333 |

(`job_numbers.txt`.) The error does not go to 0 because the bar-free job is not the bar job minus the bar: the paste's mask,
the colour match and the temporal states differ with the bar in the picture.

`sheet_job_*.png` (input | 11.2.213 | 11.2.214 | ideal): the bar solid, the lips drawn behind it. At a 38 deg turn 11.2.213 draws a dotted
edge and a seam along the bar; 11.2.214 does not.

### 3.2 Does it fire when it should not, does anything else move

| check | 11.2.213 | 11.2.214 |
|---|---|---|
| a bar-free open-mouth clip, Ultra, 150 frames | - | the erase fires on **0 of 150** face-frames; the output is **identical** to 11.2.213 (0 of 150 frames differ) |
| the same clip offline, every frame (`erase_series.py`) | - | 0 px taken on all 150 frames |
| the 65 deg profile clip, the bar at the lips, Ultra | - | identical to 11.2.213 (0 of 150 frames differ); the turn gate: 0 faces looked at |
| the cream-bar clip at **Fast** | - | identical to 11.2.213 (0 of 150 frames differ); no look, no cut |
| an ice-cream scoop at the mouth (`make_lick.py`, SDOS-210's metrics): the replacement's lips shown while the scoop is on them, mean / lowest | 0.89 / 0.82 | 0.86 / 0.80 |
| ... between licks, mean / lowest | 1.00 / 0.96 | 1.00 / 0.96 |
| ... the scoop painted over, on the lips / under them (mean) | 0 % / 1 % | 0 % / 1 % |
| ... the largest change of the paint from one frame to the next | 0.10 | 0.13 |
| a hand sweeping across the face (`make_objects.py`, SDOS-206's metrics): the hand painted over, holding, mean / worst | 0.7 % / 3.6 % | 0.6 % / 3.3 % |
| ... the rest of the face swapped before / while the hand is on it | 45 % / 40 % | 45 % / 40 % |
| ... reverted frames | 1 | 1 |
| **Optimized**, the cream bar: error against the ideal (Optimized's own bar-free job) | 6.47 | **3.79** |

The erase fired on 70 of the scoop clip's 150 face-frames (the scoop at the mouth and the chin) and on 18 of the hand clip's 80.
The scoop's small drop (0.89 -> 0.86) is the paste leaving a 1 % wider margin around an object it keeps; the sheets
(`sheet_lick_in_213_214.png`, `sheet_handsweep_in_213_214.png`) show no difference one can see.

### 3.3 Side views (asked)

* **A turned head, open mouth (38 deg, a wide smile, the bar in the opening):** covered - the erase's turn gate is a 0.60 offset of
  the nose from the eyes' middle (about 48 deg). Swap-network error 2.46 -> 1.64, whole job 4.53 -> 3.40, the bar's pixels changed
  near the lips 0.095 -> 0.049, and the dotted edge and the seam along the bar that 11.2.213 draws are gone
  (`sheet_job_38deg_in_213_214.png`).
* **A 55 deg head, and two profile photos (65 and 71 deg), the bar held to the closed lips:** **nothing is taken out.** On the 65 deg
  profile clip the job's output is identical to 11.2.213 (0 of 150 frames differ; the turn gate keeps the erase from looking at all).
  In those views the bar lies *outside* the landmark outline, where the face map says "not face" for the bar and for the background
  alike, so there is nothing to tell them apart by; inside the outline the bar covers only 180-950 px of the frame, and the test
  finds none (not even with the turn gate lifted, `erase_probe.py`). What that leaves on the table: with the *true* bar taken out of
  the network's input (ground truth, not something the engine can do) its output is closer to the ideal - 0.72 against 1.31 and
  2.92 against 4.17 on the two profile photos - because in profile too the network draws lip outlines (and a few teeth) around the
  bar's tip. Whether that is wrong depends on the mouth: on my photos the lips are closed, the bar is held against them, and the
  ideal is two closed lips; a real mouth closing on a bar does wrap around it.
* **What could not be tested:** a mouth that is open with a bar in it, in profile. The public profile photos have closed mouths and
  a drawn open mouth would not be a face the network was trained on. The 3/4 view is the closest real test.

### 3.4 Cost

* **A look** (the occlusion net on the face's 128-px crop) is made on every face swapped at Ultra, Best and Optimized: 140 ms on 2
  threads, on a frame with no object as on one with. Taking an object out adds the fill: 17-24 ms at 720p, 51-69 ms at 1080p,
  80-93 ms at 4K (a region over 480 px is filled at 480; before that cap a 4K frame cost 735 ms). Per face-frame looked at, measured
  alone on this machine: **156-165 ms at 720p, 199-210 ms at 1080p, 227-234 ms at 4K** (`erase_cost.txt`).
* **A 42-frame job at Ultra** with the bar in every frame, nothing else running: 11.2.213 45.8 s, 11.2.214 **51.5 s (+12 %)**; the
  swap per frame 661 -> 814 ms. Optimized swaps every 2nd frame when the faces are apart (97 of the 150 frames of the test clip) and
  pays for those only. Fast and Balanced pay nothing.
* The paste's rule takes no network pass of its own (it shares the net's map with the mouth cut).

### 3.5 Unit checks

`unit214.py` - **61 checks**: the object's mask on stand-in face maps (the bar through the outline covered 100 %, the edge and the
way out included, and not covered when those are switched off; the open mouth's cavity untouched; nothing for a face with an open
mouth and no object, for a hole inside the outline, for a speck, for a crop running out of the frame; a failing or missing net gives
nothing); the gates (the switch, contact, partners, a turned face, no landmarks, no net, a failing mask - each gives the frame as it
was; the other tiers' look only on an earlier cut); the paste's rule (`build_mask`: the object's place cut and nothing else changed,
off with the switch, standing aside for contact, partners, a turned face, a tier that does not erase, a face nobody tagged); the
fill over the size limit; the shared map of the net left alone by the erase; the wiring in the job's key-swap loop; **a real
`PredictedFace` (it has `__slots__`) through `_swap_one` and `_reuse_one` at every tier** - the check that would have caught the one
crash of an earlier build, an attribute missing from the slots; and the **real occlusion net** on the synthetic clips (no object:
0 px on every bar-free frame; the bar: at least 97.2 % of it inside the outline taken, mean 99.7 % over 9 frames; the cavity 0.000).
With every earlier suite on the 11.2.214 files: `unit213` 18, `unit212` 17, `unit211` 23, `unit210` 19, `unit209` 13, `unit206`
4 + 4, `unit205` 30, `unit204` 7, `unit203` 18, `unit176` 12, `unit176_pipe` 14, `nofalse206` 1 - **241 checks, all pass**
(`unit_suites.txt`).

How the build was checked: the first full-job runs of earlier builds showed, in the real job and not in any unit check, that the
erase was not reached at all (the network is called in the job's key-swap loop, not through `_swap_one`); that the bar's way out
through the outline was painted over (a smear across the bar); that its anti-aliased rim was too (a dark line along its edge); and,
on the build before this one, a crash (an attribute missing from `PredictedFace.__slots__`). Each is fixed and each is a check now.
The final files were run again at Ultra on the three clips that carry the numbers above: the outputs are identical, frame for
frame, to the ones measured.

---

## 4. What the log says now

At the end of the job: `ERASE: an object at the mouth or chin was taken out of the swap's input (so the lips are drawn behind it, not
around it) on N of M face-frames looked at (mean X % of the face crop); T ms per face-frame looked at; the paste left it alone on K
composited frames (SDOS-214)`. `N = 0` with an object at the mouth means the net did not see it as "not face" (a colour it takes
for skin, or too blurred).

## 5. Limits

1. **Synthetic and public test media only; no footage was looked at.** The exact doubled upper lip of your clip did not appear
   (§1.2). Whether your clips are fixed can only be seen on them.
2. **The object has to read as "not face" to the occlusion net** (`xseg_2`). Chocolate, cream and strawberry bars did (probability
   0.00), at several angles and with phone-quality noise and blur. An object the net takes for face (skin-coloured, very blurred,
   a mirror-bright reflection) is not taken out: the log's `ERASE ... on 0 of N` says so, and the job is then as 11.2.213.
3. **It has to lie over the lower face and reach the face outline.** A bar half out of the mouth does. Something wholly inside the
   open mouth (a sweet on the tongue) is a cavity and is left alone, as before.
4. **The first and last frames of an object's visit are as in 11.2.213**: while the bar is just at the lip (its tip not yet over
   the lower face) nothing is taken - 3 frames at the start and 2 at the end on the test clips - and the network then draws the bar
   and the lips around it as before. The same for any single frame on which the net loses the object (blur, motion). The swaps
   run in parallel, so no earlier frame's finding can hold it.
5. **Profile: not taken out, identical to 11.2.213** (§3.3). Eating with an open mouth in profile could not be tested.
6. **A hand holding the bar is taken out with it** (the net's piece that reaches the outline cannot tell them apart) and pasted
   back from the footage; the face is drawn behind both.
7. **Two faces in contact: the erase and its cut stand aside**; the contact rules decide, as before.
8. **The fill is plain inpainting** from the surroundings. The mouth interior the network then draws where the bar was is its own
   invention (a tongue-like area, a dark cavity); it never shows under the bar, but the part of the opening the bar does not cover
   is the network's, drawn from that fill.
9. **The tiers**: Ultra, Best and Optimized look on every swap; Fast and Balanced are exactly 11.2.213 (no look, no cut). On Optimized the cream clip's erase fired on 45 of its 97 swapped faces (the swaps with the bar at the mouth) and the error against
the ideal fell from 6.47 to 3.79.
10. **The thresholds come from one open-mouth photo, one 38 deg photo and two profile photos** (public AFLW2000 samples) with three
    bar colours; a real bar with another shine or a melting drip is not in them.

## 6. Install

Upload the pack's files to the Space. Changed against 11.2.213: `swap_engine.py`, `core_pipeline.py`, `config.py`. No change to
`packages.txt`, `requirements.txt` or the app. The log at start reads `Phoenix v11.2.214`.
