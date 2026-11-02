# SDOS-083 — RealFaces: the occluder gate could not see objects on real faces

**Pack:** v11.2.84 "RealFaces" (includes v11.2.83 ObjectLock, v11.2.82 LipCover and v11.2.81 Brisk). Engine `aequus-1.2.84-realfaces`.
**Reported (after 11.2.83):** "I don't see any improvement" (ice cream / food at the mouth: the lower lip still comes over the object, the object still looks translucent).

I have not seen your footage (I only work with synthetic media and logs). Everything below was measured on synthetic objects pasted onto **real face photographs** plus synthetic faces. §6 lists what is still unverified.

---

## 1. What went wrong — my part

Versions 11.2.82 and 11.2.83 were tuned and tested on **synthetic faces only**. I did warn in SDOS-081 §5 that on a real photograph the gate had not fired, but I treated that as a side note. It was the whole story: the two previous packs improved a gate that, on a real face, **never switched on**, so there was nothing for them to improve. That is why you saw no change.

## 2. What I measured

I cut six real faces out of the insightface sample photo (aligned crops, plus their mirror images = 12), pasted 7 synthetic objects of different colour onto the mouth area (a held scoop), and asked the gate how much of the object it cuts.

| | 11.2.83 | 11.2.84 |
|---|---|---|
| Object × face combinations (7 colours × 6 real faces) where more than half of the object was cut | **0 of 42** | **32 of 42** (strawberry 0, butterscotch 2 of 6; every other colour 6 of 6) |
| Mean share of the object cut, harsh battery, 3 new seeds × 12 faces | ≈ 0 % | **72 %** (vanilla 96, chocolate 97, pistachio 97, white-pink 94, mango 88, **butterscotch 23, strawberry 8**) |

### Why it was blind

1. **Wrong yardstick.** The gate says "this pixel is not skin" when it differs from the face's skin colour by 2.6 × the skin's own spread. The spread is measured on the forehead and cheeks. On real faces that zone contains hair, shading and texture, so the spread comes out **2–6 times wider than on synthetic faces** (luma spread 30–71 depending on how it is estimated, against 9–12). 2.6 × a wide spread is more than almost any object differs from skin. The gate trusted only extreme contrast.
2. **The face model absorbed the object.** The "this is my own face" model learns every pixel that was not cut, so an object not cut in the first frames was learned as face within ~10 frames.
3. **Pixel-sized settings were off scale.** Small crops (< 192 px) were not rescaled, so the feather, snap and opening radii meant different things at different crop sizes.

## 3. What changed (`swap_engine.py`, `config.py`, `core_pipeline.py`)

All new behaviour lives in `occluder_keep` / `build_mask` and its helpers. A face with nothing in front of it takes the old path.

- **Capped spread** (`occ_cap_y` 10, `occ_cap_c` 3): the skin spread used for the "not skin" test cannot exceed what a clean face has.
- **Tolerant novelty** (`occ_tol` 5 px): a pixel counts as "new" against the face model only if it is outside the range the face showed within 5 px (alignment noise, talking and pitch stop producing false objects).
- **Light plane removed** (`occ_plane`): a smooth brightness change across the crop is fitted and taken out first, so a shadow or a lamp is not an object.
- **Minimum width** (`occ_blob` 6 % of the crop): thin bands (alignment noise, eyebrows, lip line) are ignored.
- **Lower face only** (`occ_rows_min` 0.46): hair, brows and glasses are not what this gate is for.
- **Stricter face model** (`occ_ref_chroma/luma` 1.5, was 2.6): a pixel is vouched for by the model only if it really matches it.
- **Fixed 192-px working size** (`occ_work`): every pixel-sized setting now means the same at every crop size.
- **Whole-object growth** (`_grow_object`, from 11.2.83): the found part of the object is grown through connected pixels of its own colour. Added here: the growth only takes a pixel that is nearer to the object's colour than to the skin's (so a pink object does not eat the cheek), and stops at edges.
- **Mouth-zone closure** (`_close_mouth_zone`, from 11.2.82): convex hull, gaps narrower than a lip, enclosed holes.
- **Edge snap and steepening** (from 11.2.83): the cut follows the object's real edge, with a narrow feather (1.2 % of the crop).
- **First sighting** opens at 0.30 (was 0.45), confirmed sightings at 0.85: a one-frame false detection cannot punch a full hole.
- **Log line** after each job (§5).

## 4. Results

### 4.1 Real faces, synthetic objects (the new measurement)

| Test | Result |
|---|---|
| Detection, harsh battery (3 seeds × 12 real faces × 7 colours) | **72 %** mean (per colour in §2) |
| False cuts on **clean** real faces, worst single frame, 9 perturbation tests (jitter, light ±50 %, light gradient, a wide-open mouth appearing, talking jaw/lips ±10 px, alignment noise 4 px, yaw 14 %, pitch 12 %, rigid 5 px / 3°) | Calibration seeds (7, 8): **0.0 %** in all nine. **New seeds (21–23): 0.0 % in eight; alignment noise 4 px: 3.3 % of the crop in the worst frame of the worst face.** A 4 px alignment noise is far above what the tracker produces; I mention it because it is not zero. Raising the tolerance to 6 left it at 3.2 % and lowered detection (69 %), so I kept 5 |

### 4.2 Real pipeline, end to end

Three 110-frame two-person clips (synthetic scoop moved to the mouth of a real-photo face; real detector, swapper and compositor), Balanced quality:

| | 11.2.83 | 11.2.84 |
|---|---|---|
| `occ_cut` (face-frames where the gate cut something) | **0 / 220** | **70 / 220** (all three clips) |
| `\|output − input\|` over the mouth area (grey levels; lower = the object stays as it was) — pistachio / reddish-brown / dark chocolate | 9.0 / 10.9 / 15.2 | **2.1 / 5.8 / 1.8** |
| Same, over the whole object | 5.7 / 8.5 / 11.7 | **2.1 / 4.2 / 1.8** |
| Upper face (no object), old vs new difference | — | 1.2–1.3 grey levels (encoder noise level) |
| All 110 frames painted, no reverts, no geometry errors | yes | yes |

### 4.3 Synthetic regression (everything from SDOS-074, 081, 082 re-run on the final defaults)

| Test | Result |
|---|---|
| 8 must-keep sequences × 3 seeds (clean, beard, glasses, talking, shadow, dark skin, light drop, light rise, beard + light) | **0.0 % cut** |
| Unrelated object present while the mouth talks (cheek, under chin, corner, below lip) — worst lip cut | 0.01 % |
| Big scoop at the mouth, lower lip cut (vanilla, chocolate, strawberry, butterscotch, mango, closed-mouth history; 3 seeds) | 100 % |
| Small scoop at the lips, then the whole scoop arrives (27 runs) — lower lip cut | 100 % |
| 7 colours × 3 sizes — face far from the object wrongly cut | 0.00 % |
| Moving scoop: face opacity left over the object's inside / edge band | 0.04 / 0.20 mean |
| Must-cut sequences (food, strawberry, spoon, food high at the lips, beard + food, light drop) | strawberry: 1 of 3 seeds late (was 3 of 3); food high at the lips: 1 of 3 (was 3 of 3); spoon: not cut in any seed (unchanged); **beard + food: the cut starts 3–4 frames later in 2 of 3 seeds (was on time) — a side effect of the slower first sighting; it is cut after that** |
| Compositor suites (grain match, 512-px adaptive crop, 90-frame soak) | pass, same numbers |

### 4.4 Cost

Measured on the gate alone, one thread, per face-frame: **11.2.83 ≈ 2 ms (no object) / 5 ms (object); 11.2.84 ≈ 5.6 ms / 14 ms.** On a job at 2–3 s per frame that is well under 1 % extra time.

## 5. Read your log — one line, tell me the numbers

After each job, next to `Done in …`:

```
OCCLUDER: gate cut something on N of M face-frames; whole-object lock added the rest of the object on G of them (mean +x% of the crop); mouth-zone fill acted on K (mean +y%)
```

`occ_cut=N/M` on the `Done` line is the same N and M. In 11.2.83 the equivalent number on my real-face test was 0 of 220.

- **N is clearly above zero during the ice-cream part of your clip:** the gate sees it. If the lip is still over the object at a particular second, send me that time and this line.
- **N is still ~0 while she eats:** the gate is blind on your footage as well. Send me that line and the colour of the ice cream; it decides what I do next.
- A lot of N on a clip with no object (talking only, a beard, a bright lamp): false cuts. Send me the line and the time of a frame where the face looks eaten away.

## 6. What this does NOT do — read this

- **Strawberry-pink, butterscotch and other skin-like objects are weak** (8 % and 23 % on real faces; they differ little from skin or lips). A spoon (metal, thin) is not cut.
- **A small piece of food at the lips** with no larger part of the object qualifying is still not cut. It cannot be told from an open mouth by colour.
- **My calibration is six real faces**, static, with synthetic perturbations and synthetic objects. I have no real eating footage and do not analyse your uploads. Real footage may behave differently; that is what the log line is for.
- Detection can still be wrong in the other direction (§4.1, 3.3 % worst frame under heavy alignment noise). A false cut makes a small piece of the swapped face show the original face; the temporal smoothing keeps it to a frame or two.

## 7. Dials and rollback

`config.py` → `ENGINE_TUNABLES` (all new). To be **less** sensitive: raise `occ_tol` (6), raise `occ_blob` (0.07), raise `occ_rows_min`. To be **more** sensitive: set `occ_cap_y/c` to 0 (= no cap, the 11.2.83 behaviour), lower `occ_ref_chroma/luma` toward 1.0. Exact rollback = re-upload the 11.2.83 pack.

## 8. Install

Upload the 8 files (`app.py config.py core_pipeline.py phoenix_api_adapter.py swap_engine.py packages.txt requirements.txt README.md`) to the **root** of the Space, commit, normal restart (or "Push to all Spaces" in the app). The startup log must show `Phoenix v11.2.84` and `swap_engine aequus-1.2.84-realfaces`. Do not restart a Space while a job is running on it.

## 9. Changed files

`swap_engine.py` (`ENGINE_VERSION`, `_face_stats`, `_occ_work`, `_illum_plane`, `_snap_keep`, `_grow_object`, `_close_mouth_zone`, `occluder_keep`, `build_mask`, `_P`), `core_pipeline.py` (log line, stats), `config.py` (`VERSION`, `BUILD`, `ENGINE_TUNABLES`). Unchanged: `app.py`, `phoenix_api_adapter.py`, `packages.txt`, `requirements.txt`, `README.md`. The test scripts are in `verification/lipcover/` (not part of the Space).
