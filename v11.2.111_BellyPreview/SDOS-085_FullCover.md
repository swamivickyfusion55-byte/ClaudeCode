# SDOS-085 — FullCover: an object in front of the cheeks was see-through; the model also forgot the plain face after the object left

**Pack:** v11.2.86 "FullCover" (includes v11.2.85 CleanEdge and everything before). Engine `aequus-1.2.86-fullcover`.
**Reported (after 11.2.85):** "a simple scenario: the object is in front of the cheeks, it also looks translucent; when it is in front it should be solid and cover the face area fully."

I have not seen your footage (I only work with synthetic media and logs). Everything below is a synthetic object on real photographs, through the real functions and, for the headline numbers, through the real pipeline. §6 lists what is not fixed.

---

## 1. Reproduced

Same realistic scoop as SDOS-084 (shaded, textured, highlight, cone), put at the **left cheek, right cheek and mouth** of six real faces (3 faces × 6 flavours per position, aligned crop):

| Pixels of the object where more than 20 % of the swapped face showed through (11.2.85) | at the mouth | at a cheek |
|---|---|---|
| ordinary colours (vanilla, chocolate, mint) | 0.8 % | 1.8–3.3 % |
| skin-like colours (strawberry, mango, caramel) | 12 % | **41–56 %** |
| object found at all (skin-like colours) | 56 % | **31–68 %** |

So a cheek is clearly worse than the mouth, and worst for pink / orange / caramel ice cream.

## 2. Causes (each measured)

1. **The cut stopped at a fixed row.** The detector looks only below row 0.46 of the face (hair, brows and glasses are above it) and the cut followed that line: a flat edge across the object. A scoop at a cheek reaches above it, so its top slice was left to the face.
2. **The skin reference was sampled from the cheeks.** The "this is not skin" test compares with the skin colour measured on the forehead and the outer cheeks. An object on a cheek is *in* that sampling zone, drags the measured skin colour toward its own colour and widens the spread, so it stops looking unlike skin.
3. **The face model learned the object.** The model of "my own face" learns every pixel that is not cut; an object that is not recognised at once (a skin-like colour, a cone) was learned within about ten frames. Worse, after the object left, the **plain face then read as "new"** against a model that had drifted toward the object, and was cut. In 11.2.85, when the object left a cheek, **4 of 12 right-cheek test cases and 2 of 12 left-cheek cases kept 5–40 % of the crop cut for the rest of the clip** (the cut was never learned, so it never went away, and the continuity rule of 11.2.85 kept it alive). That is a bug of 11.2.85, now fixed.
4. **The "is this new" test was too forgiving at the edge of the face.** It accepted any colour inside the per-channel range of an 11×11 window of the learned face; near the face edge that window holds skin, hair and background, so almost any object colour fell inside it.
5. **Parts attached to the object** (the cone, the hand) are often skin-coloured and were never cut.
6. **The graph cut of 11.2.85 was not reproducible:** OpenCV starts it from k-means with a global random generator, so the same picture gave a slightly different cut on every call.

## 3. What changed (`swap_engine.py`)

| | |
|---|---|
| **Extension above the line** (`occ_ext_min` 0.2, `occ_ext_margin` 0.15) | A found object is followed upward, at most 0.15 of the crop above the part that was found. (First attempt without the margin grew a pale scoop into a pale forehead; the margin fixed that.) |
| **Skin statistics held** (`occ_stat_hold`) | While an object is being cut, the skin statistics of the last clean frame are used. |
| **Face model frozen while an object is cut** (`occ_freeze`, max 450 frames) | Nothing is learned from the object, its cone or its shadow. Resumes when the object has been gone a moment. |
| **New pixels are not learned** (`occ_quar_on`, rate `occ_quar` 0) | For up to 90 frames; also when most of the face is new at once if an object is being cut. (A slow rate was tried and rejected: it lets the model drift toward the object.) |
| **Nearest-colour novelty** (`occ_nn`) | A pixel is new unless one pixel of the learned face within the tolerance matches its luma **and** chroma together. Used on the cheeks and jaw sides only; the mouth zone keeps the old test (everywhere it cut a rim of lip around a held object). |
| **Attached parts** (`occ_attach` 0.10) | New pixels connected to a found object are taken in, each piece only if it has a real edge around most of its free outline, only below the eye line, and at most 10 % of the crop. |
| **Deterministic graph cut** (`occ_gc_seed`) | Fixed seed. |

## 4. Results

### 4.1 Real pipeline, realistic face size (the headline)

A realistic scoop at a cheek of a real face (face ≈ 320 px wide), real detector, swapper and compositor. Change inside the scoop against the original footage, grey levels (≈ 2.5 = the scoop is untouched; the face itself changes by ≈ 18):

| Clip | 11.2.85 | 11.2.86 | Pixels visibly changed (> 12 levels) |
|---|---|---|---|
| mint, right cheek | 5.35 | **2.65** | 10.5 % → **0.01 %** |
| strawberry-pink, left cheek | 11.04 | **3.80** | 31.9 % → **6.2 %** |
| caramel, right cheek | 10.82 | **7.54** | 34.7 % → **22.2 %** (still not solid) |
| vanilla, left cheek / chocolate, left cheek / mango, left cheek | 2.5 | 2.5 | already untouched |

### 4.2 Aligned crop, 3 real faces × 6 flavours (11.2.85 → 11.2.86)

| | left cheek | right cheek | mouth |
|---|---|---|---|
| Ordinary colours: pixels with face alpha > 0.2 | 3.3 → **0.0 %** | 1.8 → **0.0 %** | 0.8 → **0.0 %** |
| Skin-like colours: object found | 31 → **66 %** | 68 → **89 %** | 56 → 56 % |
| Skin-like colours: pixels with face alpha > 0.2 | 55.5 → **14.2 %** | 40.7 → **33.4 %** | 12.3 → 7.6 % |
| Skin-like colours: edge error (px, 256 crop) | 20.3 → 7.3 | 17.2 → 14.0 | 4.9 → 3.9 |
| Over-cut right next to the object (ring) | 2.5 → 3.1 % (ordinary) | 9.3 → 6.0 % | 0.5 → 0.8 % |
| Cone / hand cut | 45 → 40 % (ordinary), 19 → 37 % (skin-like) | 55 → 55 %, 48 → 54 % | — |

### 4.3 The face comes back (new test)

After an object vanishes, frames until less than 0.5 % of the crop is still cut, 12 cases per position: **11.2.85: 4 of 12 never (right cheek), 2 of 12 never (left cheek), 0 of 12 at the mouth. 11.2.86: all 36 cases ≤ 4 frames.**

### 4.4 False cuts on clean real faces

12 faces (6 + mirror images) × 9 harsh perturbation tests (jitter, light ±50 %, light gradient, a wide-open mouth appearing, talking ±10 px, alignment noise 4 px, yaw 14 %, pitch 12 %, rigid 5 px/3°), final keep map: seeds 21–23: 0.0 % cut in eight tests, **0.2 %** of the crop in the worst frame under 4 px alignment noise; seeds 31–33 (not used to tune): **0.0 % in all nine.** Mean detection of the seven test colours 76 % / 75 %.

### 4.5 Synthetic regression (SDOS-074…084), re-run

8 must-keep sequences × 3 seeds: 0.0 % cut. Unrelated object near the mouth while talking: worst lip cut 0.01 %. Big scoop at the mouth and small-scoop-then-whole-scoop: lower lip cut 100 %. 7 colours × 3 sizes: face far from the object 0.00 %. "Food, then a light drop": unchanged.

### 4.6 Rollback and tests

With `occ_ext_min`, `occ_stat_hold`, `occ_nn`, `occ_quar_on`, `occ_attach`, `occ_freeze` at 0 the mask is **bit-identical to 11.2.85** (480 frames, three positions, three colours; the graph cut seeded in both). 23 unit checks pass (extension limit, nearest-colour test, held statistics, learning hold, frozen model, attached pieces with and without an outline, the cap).

### 4.7 Cost

Gate per face-frame: **no object 13 → 25 ms; object at a cheek 46–54 → 72–75 ms** (the nearest-colour test costs ≈ 11 ms on every frame). On a job at 2–3 s per frame: about 1 %.

## 5. Read your log

```
OCCLUDER: … edge refit on R; a tracked object kept from dropping out on C; attached part (cone, hand) taken in on A
```

`A` > 0 = a cone / hand / stick was taken in. Send me the line from a job with an object at the cheek, and the time of any frame that is still wrong.

## 6. What this does NOT do — read this

- **Skin-like colours are improved, not solved.** Caramel at the right cheek is still 22 % see-through in the real pipeline; where the object's colour equals the skin or hair next to it, colour cannot separate them and I did not find a safe structural cue. A face parser would, but I cannot download or test a new model here, so I did not ship one.
- **The cone / hand is cut in about 40–55 % of its pixels,** not all.
- **Mouth:** skin-like colours at the mouth are unchanged (56 % found); the new novelty test is deliberately off there.
- **No real footage;** the ice cream is synthetic, the faces are six photographs.
- Edge flicker was not re-measured after the graph cut was made deterministic.

## 7. Dials (`config.py` → `ENGINE_TUNABLES`)

`occ_ext_min` (0.2; 0 = off), `occ_ext_margin` 0.15, `occ_stat_hold` (1; 0 = off), `occ_freeze` (1; 0 = off), `occ_freeze_max` 450, `occ_quar_on` (1; 0 = off), `occ_quar` 0, `occ_quar_age` 90, `occ_quar_global` 0.30, `occ_nn` (1; 0 = off), `occ_nn_mouth_y0` 0.60 / `occ_nn_mouth_dx` 0.22, `occ_attach` (0.10; 0 = off), `occ_attach_edge` 10 / `occ_attach_edge_frac` 0.5, `occ_gc_seed`. All off = 11.2.85.

## 8. Install

Upload the 8 files (`app.py config.py core_pipeline.py phoenix_api_adapter.py swap_engine.py packages.txt requirements.txt README.md`) to the **root** of the Space, commit, normal restart (or "Push to all Spaces" in the app). The startup log must show `Phoenix v11.2.86` and `swap_engine aequus-1.2.86-fullcover`. Do not restart a Space while a job is running on it.

## 9. Changed files

`swap_engine.py` (`ENGINE_VERSION`, `_ext_floor`, `_novel_nn`, `_attach_novel`, `_face_stats` held statistics, `occluder_keep` novelty split, `TrackState` slots, `_quarantine`, frozen model in `occluder_keep_final`, graph-cut seed, `_P`), `core_pipeline.py` (log line), `config.py` (`VERSION`, `BUILD`, `ENGINE_TUNABLES`). Unchanged: `app.py`, `phoenix_api_adapter.py`, `packages.txt`, `requirements.txt`, `README.md`. Test scripts: `verification/fullcover/` (not part of the Space).
