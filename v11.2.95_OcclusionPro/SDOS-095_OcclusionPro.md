# SDOS-095 — OcclusionPro: two faces in contact, objects in front, no reverting

**Pack:** v11.2.95 "OcclusionPro" (v11.2.94 Natural + occlusion work). Engine `aequus-1.2.95-occlusionpro`. No app change.
**Asked:** "Do a deep research and find the best way to make the occlusion management perfect, cut to cut. When one face comes in contact with another, say kissing, both should be distinctly outlined. It should never revert to the original and flicker when there is any obstruction. When any object comes in front of a face, the area covered by the object should go straight away."

Research report: `reports/Occlusion handling in face swap video.md` (notes in `research_notes/Occlusion handling in face swap video/`). All tests use synthetic clips made from public photos; no user footage.

---

## 1. What the research said, and what was built

The research's main finding was that no new model is needed. FaceFusion 3.9 (the closest open reference) uses the same XSeg occluder this engine already runs. What was missing was the logic around it. These items from the report's ranked list were built:

| # (report) | Recommendation | Built as |
|---|---|---|
| 1 | Exclusive per-pixel ownership for touching faces | **One face per pixel**: the face in front is cut out of the face behind along its own outline (forehead included), at full strength (was 85 %); next to the front face, the face behind keeps only its own outline; the front face's soft edge is cut where it lies over the face behind |
| 5 | Front/behind decision with hysteresis | **Outline-edge evidence**: the face whose outline edge is visible across the contact line is in front; the order only flips after the evidence favours the other face by 10 % for 4 frames in a row; with no evidence (faces apart) the order is held |
| 2 | Objects cut on the frame they arrive | **Edge novelty**: a ring around each face's outline remembers what it normally looks like; something new arriving there (a hand, a cup, a phone) is cut on that same frame, and stays cut while it stays |
| 3 | Occluder over the whole face, not just the mouth | The XSeg occluder now acts **anywhere on the face** (it was limited to the mouth and chin), for regions of at least 0.4 % of the crop |
| 4, 8 | Never revert; reset at cuts | Unchanged and re-verified: tracking carries the face through contact and objects (0 reverted frames in every test below); the per-face state already restarts at shot changes (v11.2.93) |

Speed: unchanged. The two-face contact clips took 497–498 s on v11.2.95 against 500–512 s on v94.

## 2. Two faces in contact (kiss, cheek to cheek)

Four synthetic contact clips were used: cheek to cheek, a deep overlap, one face hiding the other, and the order flipping. Two public faces, 120 frames each, with B in front of A where they overlap.

Earlier metrics counted the whole elliptical cut-out of each person, which also holds wall, hair and an arm. The final measure is taken on each person's **face**: the 106-point outline found on the source photo and moved into every frame by the exact transform the clip used (`contact_face.py`).

| Clip | A's paint seen on B's face, worst frame (v94 → v95) | B's paint on A's face, worst frame (v94 → v95) | front face B painted, weakest frame (v94 → v95) |
|---|---|---|---|
| cheek | 16.2 % → **0.3 %** | 7.0 % → 7.0 % | 72 % → **89 %** |
| deep | 11.1 % → **0.3 %** | 13.1 % → **6.6 %** | 76 % → **87 %** |
| hide | 5.9 % → **0.3 %** | 21.2 % → **6.4 %** | 82 % → **87 %** |
| flip | 0.1 % → **0.0 %** | 13.2 % → **9.3 %** | 92 % → **91 %** |

On average across contact frames, B's paint on A's face dropped from 3.6 / 7.3 / 12.1 / 6.9 % to 3.4 / 1.4 / 1.6 / 2.0 %.

- **Reverted frames: 0** for both people on all four clips.
- **Flicker** (frame-to-frame change against the footage's own): ×1.03–1.11 in contact, the same range as v94.
- The front/behind cue was correct on 89–100 % of contact frames. Face size alone, the old rule, was correct on 0 % of the cheek-to-cheek frames, because there the face behind was the larger one.

## 3. Objects in front of a face

Hand, cup and phone clips: an object slides in over 4 frames, holds for 34 frames, then leaves. The figure is the share of the object that the swap still painted over (it should be 0). These runs used the final pack, straight from the zip.

| Object | v94 entering → v95 | v94 holding (worst) → v95 |
|---|---|---|
| hand over the eye | 14.7 % → **0.2 %** | 7.1 % (10.1) → **0.4 % (1.2)** |
| cup over the forehead | 4.9 % → **0.1 %** | 15.7 % (17.8) → **0.6 % (1.5)** |
| phone over the nose | 7.5 % → **2.0 %** | 0.0 % → 0.0 % |
| hand on the cheek (skin colour, the hardest case) | 1.8 % → **0.5 %** | 2.9 % (4.7) → **2.6 % (3.7)** |

The rest of the face stayed swapped (the same share as before the object arrived), with **0 reverted frames**.

## 4. Unchanged elsewhere (regression checks)

- **Side views:** on the fast side-view clip, paint off the face is 0.0 %, the face never lost its paint, and the wobble is 1.89 %, identical to v94.
- **Edge novelty on side views:** it is switched off from about 45° of turn (side-view weight ≥ 0.5). On a side-view clip it cut 36 of 120 frames before this gate and 0 after.
- **Unit checks:**
  - `unit95.py`: 17/17. Covers the order hysteresis, order held without evidence, one face per pixel in both directions, the behind face clamped beside the front face, the edge novelty (cut on the arrival frame, kept while the object stays, gone after) and the outline cue.
  - `unit94.py`: 29/29.
  - `unit90.py`: all pass.

## 5. Tunables (`swap_engine.py` `_P`; 0 = the v94 behaviour)

| Tunable | Value | What it does |
|---|---|---|
| `own_pixels` | 1 | one face per pixel |
| `own_grow` | 0.02 | grows the front face's outline by this share of the crop before it is cut |
| `own_feather` | 0.03 | softness of that cut |
| `own_near` | 0.12 | the face behind keeps only its own outline within this share of the crop of the front face |
| `rival_cut` | 1.0 | strength of the cut |
| `occ_novel` | 1 | edge novelty on |
| `occ_novel_grow` | 0.20 | width of the edge ring |
| `occ_net_zone` | 1 | the XSeg occluder acts on the whole face (0 = mouth and chin only) |

## 6. Limits

- **Synthetic tests only.** The contact clips are two cut-out photos sliding together; a real kiss involves 3-D head turns and partly hidden landmarks.
- **Not landmark-perfect at the touching line.** The front face's own swap can still paint up to about 7–9 % of the face behind on its worst frame, at the touching line, where the two outlines are within a few pixels of each other.
- **Small hand on the cheek.** A hand of exactly the skin's colour that arrives inside the face (not at its edge) is cut only by XSeg (2.6 % still painted).

## 7. Files and install

- **Pack:**
  - Upload the 8 files (`app.py config.py core_pipeline.py phoenix_api_adapter.py swap_engine.py packages.txt requirements.txt README.md`), or use "Push to all Spaces" in app 1.2.24.
  - Log lines: `Phoenix v11.2.95`, `swap_engine aequus-1.2.95-occlusionpro`.
  - New log line: `OCCLUSION-PRO: objects arriving at a face's edge cut on n of m face-frames; two faces kept apart (one face per pixel) on k face-frames`.
- **Test scripts** in `verification/fastrestore/`:
  - `make_objects.py`, `obj_metrics.py`
  - `contact_metrics.py`, `contact_spill.py`, `contact_face.py`
  - `order_cue.py`, `occ_zone_test.py`, `unit95.py`
