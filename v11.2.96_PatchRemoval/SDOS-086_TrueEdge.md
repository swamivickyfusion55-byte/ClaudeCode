# SDOS-086 — TrueEdge: the see-through rim where an object touches the face; two people touching (kissing)

**Pack:** v11.2.87 "TrueEdge" (includes v11.2.86 FullCover and everything before). Engine `aequus-1.2.87-trueedge`.
**Reported (after 11.2.86):** "Still the translucence of the object near the point of contact with the body is there." And: "build clear logics of separation on human to human touches, e.g. kissing; it causes disruption in smoothness, face dis-shaping or a revert back to the original (occasionally)."

I have not seen your footage (I only work with synthetic media and logs). Everything below is synthetic objects and synthetic two-person contact clips made from real photographs, through the real detector, swapper and compositor. §6 lists what is not fixed.

---

## 1. The rim at the point of contact: cause

The decision "object or face" is made on the aligned face crop (128–512 px), from a 192-px working copy. In the frame that is 1–3 frame pixels per crop pixel, so around the object the swapped face **faded in over about 10–12 frame pixels**. Inside that rim the picture is a mix of the swapped face and the original footage, which reads as a translucent edge exactly where the object meets the face. Measured in the real pipeline (strawberry scoop at the left cheek, 11.2.86): the face was only 55 % painted 2 px from the object and 86 % at 10 px; inside the scoop 6.2 % of its pixels were still visibly changed.

## 2. What changed (`swap_engine.py`)

| | |
|---|---|
| **Edge found at the pixels of the frame** (`_refine_keep_img`, `occ_img_refine` = 1) | When a crop has an object cut, the cut is re-decided in the frame itself, in a narrow band around the old edge: pixels well inside the cut are the object, pixels well outside are the face, and the band between is flooded from both sides (watershed) over the picture's own edges. The boundary settles on the strongest edge between them, which is the object's outline, at full frame resolution. Shading, a highlight or a darker rim inside the object does not let the face in. |
| **Only where the picture can tell them apart** (`occ_img_sep0/sep1`, `occ_img_radius` 0.10) | Where the object and the skin next to it have nearly the same colour, the old soft decision is kept. |
| **One pixel wider** (`occ_img_bias` 1) | The object's last pixel is half skin; cutting it shows the original skin, which is invisible, while leaving it shows a bright halo. |

A crop with no object cut is returned untouched (0.2 ms).

## 3. Results: the object edge

### 3.1 Real pipeline (real detector, swapper and compositor; face ≈ 320 px wide; Balanced)

| Clip | 11.2.86 | 11.2.87 |
|---|---|---|
| strawberry-pink, left cheek: change inside the scoop (grey levels; ≈ 2.5 = untouched) | 3.80 | **2.57** |
| ... pixels inside the scoop visibly changed (> 12 levels) | 6.2 % | **0.03 %** |
| ... face painted at +2 / +4 / +8 px outside the scoop | 0.55 / 0.48 / 0.72 | **1.08 / 0.87 / 0.92** |
| ... width of the face-to-object transition | ≈ 12 px | **≈ 3 px** |
| mint, right cheek: face painted at +2 / +4 / +8 px | 0.52 / 0.58 / 0.84 | **1.14 / 0.99 / 0.97** |
| mint: inside the scoop | 2.65 (already solid) | 2.64 |
| caramel, right cheek: pixels visibly changed inside | 22.2 % | 21.1 % (**not fixed**, see §6) |
| face away from the object (same as before) | 18.1–19.4 | 18.0–19.4 |

### 3.2 Unit checks (`verification/trueedge/unit87.py`)

On a pale object whose cut was 3–4 px off and 10 px soft: misplaced pixels **1176 → 175**; face opacity 2–7 px inside the object 0.02 → 0.00; the face 2–7 px outside it is not eaten (0.72 → 0.76). Same-colour object and skin: the old decision is kept. A darker 4-px rim inside the object stays cut. Far from the object nothing changes. Degenerate inputs do not raise.

A first version measured the colour on each side in a 0.07-wide window. That window did not reach the sure-face pixels when the old cut was too *wide*, so it fixed a cut that was too narrow but not one that was too wide (misplaced pixels only 1176 → 973). 0.10 fixes both. That is the shipped value.

## 4. Two people touching (kissing, cheek to cheek, a hug)

### 4.1 What I could reproduce

Five synthetic 120-frame two-person clips made from two real faces of the sample photo, moving together until one covers part of the other and then apart again (`cheek`, `near`, `deep`, `hide`, `flip`), real pipeline at Ultra, with the true face geometry recorded. **Only the heaviest case (`hide`: one face about half behind the other) showed your symptoms:** in 11.2.86 the face in front reverted to the original on **18 of 120 frames (15 during the contact)**, and the face behind was pasted **22 % too small** (scale ×0.78) and 0.42 inter-eye distances off.

Causes found:
1. **Who is in front was decided by the size of the detector box.** In a kiss both boxes are about the same size, so the order followed the detector's jitter, and on some frames the face *in front* was treated as being behind the other one: the other face's outline was cut out of it, so patches of the original face came and went. That is the "revert".
2. **A face half hidden is reported smaller by the detector,** and the pasted face shrank with it: the "dis-shaping".

### 4.2 What changed

| | |
|---|---|
| **Who is in front, decided from evidence and kept** (`core_pipeline._DepthOrder`, `depth_order` = 1) | A face whose own key points (eyes, nose, mouth corners) lie inside the other face's outline is the one behind. The detector's confidence and, weakest, the box size also vote. Once decided for a pair, the order changes only after the opposite evidence has been clear for 6 frames in a row (`depth_frames`, `depth_margin`), and it is forgotten when the two have been apart for 15 frames (`depth_forget`). Paint order and the "other face cuts into me" rule both follow it. |
| **Size held during contact** (`contact_scale_lock` = 1, `contact_cover` 0.30) | While more than 30 % of a face's box is covered by another face's box, the pasted face keeps the size it had before the contact. Position and rotation still follow the detector. A real size change with no contact is followed as before. |

### 4.3 Results (real pipeline, Ultra)

| Clip | 11.2.86 | 11.2.87 |
|---|---|---|
| `hide`, face in front: frames reverted (all / during contact) | 18 / 15 | **5 / 4** |
| `hide`, face behind: size during contact | ×0.78 | **×0.86** |
| `hide`, face behind: placement error during contact (inter-eye units) | 0.42 | 0.42 (unchanged, see §6) |
| `hide`, the other person: reverts / placement | 0 / 0.19 | 0 / 0.19 |
| `deep` (one face a third behind the other), face behind: size / placement during contact | ×0.75 / 0.207 | **×0.82 / 0.192** |
| `cheek`, `flip`: reverts, placement, size, slot changes | 0, as 11.2.86 | identical |
| frame-to-frame change of the painted face during contact, against the footage's own (`hide` / `deep`, face behind) | ×0.98 / ×1.03 | ×1.04 / ×1.09 (**slightly more movement**: the held size does not follow the detector's small changes) |
| order changes during contact (`CONTACT:` log line), all five clips | — | 0 |

Unit checks: the key-point rule finds the face behind; ±5 % box jitter over 60 frames never flips the order; a real reversal is followed after exactly 6 frames; the order is forgotten after the pair separates; paint order follows the decision; the size is held at 1.00 under 60 % cover (0.83 without the lock) and a real size change without contact is still followed (0.80).

## 5. Verification

- **Rollback is exact:** with `occ_img_refine`, `depth_order` and `contact_scale_lock` at 0 every composited frame is **bit-identical to 11.2.86** (246 frames, three food clips, real compositor).
- 26 unit checks pass (`unit87.py`).
- The gate's own decision (the keep map) is not touched by this pack, so the false-cut battery, the must-keep sequences and the lip tests of SDOS-085 still apply; I re-ran them on this pack: 0.0 % cut in all must-keep sequences, worst false cut 0.2 % of the crop (4-px alignment noise), lower lip cut 100 %, mean detection of the 7 test colours 76 %.
- **Cost:** the edge refinement costs 30 ms (face region 360 px) to 57 ms (500 px) per face-frame **when an object is cut**, 0.2 ms otherwise. The contact logic is negligible. On a job at 2–3 s per frame: about 1–2 %.

## 6. What this does NOT do — read this

- **The refinement only sharpens an object that was found.** Where the gate does not see the object (caramel, and other colours very close to skin), nothing changes: caramel at the right cheek is still about 21 % see-through. The fix for those needs a face-parsing model, which I cannot download or test here.
- **Kissing: my clips are 2-D cut-outs of still photographs** sliding over each other. A real kiss has 3-D head turns, lips pressed together, noses side by side, profile views and hair. Four of my five contact clips did not show your symptoms at all, so I cannot say these changes cover what you see. They fix the two causes I could measure.
- **When a face is half hidden, its key points are still guessed by the detector:** the placement error of the face behind (0.42 inter-eye distances in `hide`) is unchanged. Holding the size helps the shape; it does not fix the position.
- No real footage.

## 7. Read your log

```
CONTACT: 1 pair(s) of faces touched; the order (who is in front) was changed 0 time(s) during contact
```

A clip with a kiss should show the pair and few or no order changes. Several changes during one kiss = the order is unclear in your footage: send me the line and the time of a frame that looks wrong. The `OCCLUDER:` line is as in SDOS-085.

## 8. Dials (`config.py` → `ENGINE_TUNABLES`)

`occ_img_refine` (1; 0 = off), `occ_img_band` 0.02, `occ_img_radius` 0.10, `occ_img_sep0` 8 / `occ_img_sep1` 12, `occ_img_sharp` 3, `occ_img_bias` 1; `depth_order` (1; 0 = the larger box, as before), `depth_margin` 1.0, `depth_frames` 6, `depth_forget` 15; `contact_scale_lock` (1; 0 = off), `contact_cover` 0.30. All three switches at 0 = 11.2.86 exactly.

## 9. Install

Upload the 8 files (`app.py config.py core_pipeline.py phoenix_api_adapter.py swap_engine.py packages.txt requirements.txt README.md`) to the **root** of the Space, commit, normal restart (or "Push to all Spaces" in the app). The startup log must show `Phoenix v11.2.87` and `swap_engine aequus-1.2.87-trueedge`. Do not restart a Space while a job is running on it.

## 10. Changed files

`swap_engine.py` (`ENGINE_VERSION`, `_refine_keep_img`, `paste_back` / `_composite` take the frame-space refinement, `TrackState.cover` and the size hold, `MultiFaceTracker._mark_crossings` computes the cover, `_P`), `core_pipeline.py` (`_DepthOrder`, `_rival_landmarks`, paint order, `CONTACT:` log line), `config.py` (`VERSION`, `BUILD`, `ENGINE_TUNABLES`). Unchanged: `app.py`, `phoenix_api_adapter.py`, `packages.txt`, `requirements.txt`, `README.md`. Test scripts: `verification/trueedge/` (not part of the Space).
