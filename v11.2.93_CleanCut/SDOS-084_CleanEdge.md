# SDOS-084 — CleanEdge: pale ice cream was see-through at the mouth; sharper object edge; smoother cheeks

**Pack:** v11.2.85 "CleanEdge" (includes v11.2.84 RealFaces and everything before). Engine `aequus-1.2.85-cleanedge`.
**Reported (after 11.2.84):** "two things can be fixed: a sharper demarcation of the edges of the object and the mouth — the part of the object in front of the mouth is a bit translucent and bright; and when the object goes inside the mouth, the shape of the cheeks, the way it swells, can be smoothened."

I have not seen your footage (I only work with synthetic media and logs). Everything below is from synthetic objects on real photographs through the real functions and, for the headline numbers, through the real pipeline (detector, swapper, compositor). §6 lists what is not verified.

---

## 1. Your first point reproduced — and why my earlier tests missed it

My object tests so far used **flat, evenly coloured discs**. A real scoop has shading, texture, a highlight and a cone. I built that (`verification/cleanedge/food.py`) and put it at the mouth of a real face, at a realistic size (face ≈ 320 px wide), through the full pipeline:

| White (vanilla) scoop at the mouth, real pipeline | 11.2.84 | 11.2.85 |
|---|---|---|
| Change inside the scoop against the original footage (grey levels; 0 = scoop untouched) | **15.4** | **2.7** |
| Pixels inside the scoop that are visibly changed (> 12 levels) | **41 %** | **0.02 %** |
| Same for a dark-chocolate scoop (it already worked) | 2.6 | 2.6 |

In the 11.2.84 picture the swapped smile and lips are visible *through* the white scoop: translucent and bright. That is your report.

## 2. Why (measured on the aligned crop, 3 real faces × 6 foods)

1. **The detection flickered.** The scoop's blob hovers around the size limit that lets it count as an object, so the detector said yes, no, yes on alternate frames; each "no" let the face fade back in (closing rate 0.22 per frame). White scoop: detected in 67 % of steady frames; mango 33 %.
2. **The cut covered only part of the object and had a ragged edge.** Highlights, texture and lip-coloured parts of a real scoop look like skin to a colour test. Edge position error 4.3 px per 256-px crop; the cut also ate **4.9 %** of the face right next to the object (mango 21 %).
3. **A moving object's front edge opened at the cautious "first sighting" rate every frame** (arrival: 50 % of the face still shows through).

## 3. What changed (`swap_engine.py`)

| | |
|---|---|
| **Graph-cut edge** (`_gc_refine`, `occ_gc` = 3) | After an object has been found, a graph cut (GrabCut) with the blob's core as sure object and everything away from it as sure background puts the cut on the visible edge of the object. It can only add pixels within 7 % of the crop of what was found, and is ignored if its answer is below 0.6× or above 1.6× the blob. Never runs when nothing has been found. |
| **Continuity** (`occ_cont` = 0.3) | A blob that overlaps what was being cut a moment ago qualifies without the entry tests (size, reaching the edge). Removes the on/off flicker. After the object is gone the face is back in ≤ 4 frames (same closing rate as before). |
| **Fast opening near a confirmed object** (`occ_open_near` = 0.07) | New pixels within 7 % of the crop of a confirmed object open at the fast rate. |
| **Cut a pixel wider** (`occ_bias` = 0.004) | An object's last pixel is half skin; a face pixel left on it reads as a bright halo, a skin pixel from the plate next to it is invisible. |
| **Cheek smoothing** (`cheek_ema` = 0.4) | See §5. |

## 4. Results — the object

### 4.1 Aligned crop, 3 real faces × 6 foods (steady frames; 11.2.84-equivalent = new dials off)

| | 11.2.84 | 11.2.85 |
|---|---|---|
| Frames where the object is detected (all 6 foods) | 61 % | **78 %** (vanilla 67→100, mango 33→100; **strawberry and caramel stay at 33 %**) |
| Face alpha over the arriving object (0 = fully cut) | 0.50 | **0.31** (white/mint/mango 0.2–0.47 → 0.01–0.02) |
| Edge position error, px of a 256 crop | 4.3 | **3.2** (512 crop: 8.6 → 5.4) |
| Face wrongly cut right next to the object | 4.9 % | **0.7 %** |
| Face far from the object wrongly cut | 0.10 % | **0.00 %** |
| Bright halo (face alpha in the object's 3-px inner edge band) | 0.24 | **0.15** (512 crop: 0.23 → 0.10) |
| Interior pixels with face alpha > 0.2 (512 crop) | 7.4 % | **3.7 %** |
| Edge flicker (mean frame-to-frame change of alpha in the edge band) | 0.05 | 0.07 (slightly higher: the graph cut is a little less steady than the old blur) |

Per food and crop size: `verification/cleanedge/` (`harness.py`).

### 4.2 False cuts (clean real faces)

3 new seeds × 12 real faces (6 + mirror images) × 9 harsh perturbation tests (jitter, light ±50 %, light gradient, a wide-open mouth appearing, talking ±10 px, alignment noise 4 px, yaw 14 %, pitch 12 %, rigid 5 px/3°), measured on the **final** keep map (temporal smoothing, edge snap and all): **0.0 % cut in eight of nine tests; 0.2 % of the crop in the worst frame under 4 px alignment noise.** Mean detection of the 7 test colours 76 % (11.2.84: 73 %): vanilla/chocolate/pistachio/white-pink 100 %, mango 99 %, butterscotch 25 %, strawberry 8 %.

### 4.3 Synthetic regression (flat discs, SDOS-074/081/082), re-run

8 must-keep sequences × 3 seeds: **0.0 % cut** (unchanged). Unrelated object near the mouth while talking: worst lip cut 0.01 %. Big scoop at the mouth and small-scoop-then-whole-scoop: lower lip cut 100 %. 7 colours × 3 sizes: face far from the object 0.00 %. Moving scoop, edge band: face opacity left **0.20 → 0.04**. Compositor suites (grain, 512-px crop, 90-frame soak): pass.

### 4.4 Rollback is exact

With `occ_gc`, `occ_cont`, `occ_open_near`, `occ_bias` = 0 and `cheek_ema` = 1 the mask is **bit-identical to 11.2.84** (240 frames compared, largest difference 0.0).

### 4.5 Cost

Gate, per face-frame: 12 ms with no object (unchanged); with an object **25 → 41–47 ms**. Cheek smoothing 1 ms (256-px crop) to 5 ms (512-px crop). On a job at 2–3 s per frame that is under 1 %.

## 5. Your second point: the cheeks

**What I could and could not reproduce.** I made a clip in which the cheeks puff out and chew (a local bulge warp of a real photograph, 2.5 Hz) and ran it through the real pipeline.

- The swap is computed on every Nth frame (Ultra: every frame; Best: every 2nd; Balanced: every 4th) and the frames between reuse the same aligned face. The cheeks, which change slowly, therefore moved **in steps**. Measured as the frame-to-frame "jerk" of what the swap adds in the cheek band: **Ultra 2.10, Balanced 2.67, and Balanced's worst 5 % of frames 5.76 against 2.99 for Ultra.**
- 11.2.85 smooths the **cheek zone only** (rows from the eye line to the chin, from 0.15 of the crop away from the centre line outward; eyes, nose and mouth are returned byte for byte) with a per-frame blend toward the newest swap, which turns each step into a ramp. Balanced: mean jerk **2.67 → 2.4**, worst 5 % **5.76 → 4.4–4.5** (measured at 0.3, 0.5; I ship 0.4). **Ultra is unchanged (2.11 / 2.96 against 2.10 / 2.99).** It is a smoothing, so the cheeks lag the newest swap by about a frame; that is the price.

**Be aware of three things.**
1. The improvement is **modest** (about 10 % mean, 20 % worst-case at Balanced) and only matters when you use Best/Balanced. If you render at Ultra, the cheeks were already as smooth as the swap network makes them.
2. **I found something I did not fix.** On my synthetic swollen cheek the swap network's own output has a small dark smudge on the cheek (it is in the raw network output, not in the compositor). Temporal smoothing cannot remove it. I checked whether a smooth luminance difference between the swap and the footage singles it out so that it could be corrected; it does not (the identity's own cheek shading differs from the footage's by as much), so I left it. If you see a dark spot on a swelling cheek, that is it.
3. I tried an optical-flow follow (move the cached face with the real cheek motion). It gave **no gain** (jerk 2.83 against 2.67) and I removed it. The mask silhouette in the cheek rows was already steady (range 2.8 px of 128 during the swell), so the outline is not the problem.

## 6. What this does NOT do — read this

- **Strawberry-pink, caramel and other skin-like colours are still detected in only about a third of frames** (SDOS-083 §6 stands). A spoon is not cut. A food item that is mostly inside the mouth is cut only where it is visible; the lips closing over it stay face, which is correct.
- **I have no real eating footage.** The ice cream is synthetic; the cheek swell is a warp of a still photo. Real footage may behave differently, and the cheek change in particular is small.
- The face model, thresholds and edge machinery were calibrated on six photographs.

## 7. Read your log

```
OCCLUDER: gate cut something on N of M face-frames; whole-object lock added the rest of the object on G of them (mean +x% of the crop); mouth-zone fill acted on K (mean +y%); edge refit on R; a tracked object kept from dropping out on C
```

`R` > 0 = the graph cut acted; `C` > 0 = the continuity rule saved frames that used to show the face through the object. On my white-scoop test job it read: gate cut something on 75 of 198 face-frames; edge refit on 69; a tracked object kept from dropping out on 61 (a clean unzip of this pack, run end to end: same 2.7 inside the scoop as in §1). Send me the line from an ice-cream job and the time of any frame where something still looks wrong.

## 8. Dials (`config.py` → `ENGINE_TUNABLES`)

`occ_gc` (3; 0 = off), `occ_gc_reach` 0.07, `occ_gc_core` 0.025, `occ_gc_lo` 0.6, `occ_gc_hi` 1.6, `occ_cont` (0.3; 0 = off), `occ_open_near` (0.07; 0 = off), `occ_bias` (0.004; 0 = off), `cheek_ema` (0.4 = weight of the newest swap; **1 = off**), `cheek_x0/x1/y0/y1` (cheek zone). All off = 11.2.84 exactly.

## 9. Install

Upload the 8 files (`app.py config.py core_pipeline.py phoenix_api_adapter.py swap_engine.py packages.txt requirements.txt README.md`) to the **root** of the Space, commit, normal restart (or "Push to all Spaces" in the app). The startup log must show `Phoenix v11.2.85` and `swap_engine aequus-1.2.85-cleanedge`. Do not restart a Space while a job is running on it.

## 10. Changed files

`swap_engine.py` (`ENGINE_VERSION`, `_gc_refine`, `occluder_keep_final` split out of `build_mask`, continuity in `occluder_keep`, `occ_open_near` / `occ_bias`, `_cheek_zone`, `cheek_smooth`, `TrackState.fake_s`, counters, `_P`), `core_pipeline.py` (log line, stats reset), `config.py` (`VERSION`, `BUILD`, `ENGINE_TUNABLES`). Unchanged: `app.py`, `phoenix_api_adapter.py`, `packages.txt`, `requirements.txt`, `README.md`. Test scripts: `verification/cleanedge/` (not part of the Space).
