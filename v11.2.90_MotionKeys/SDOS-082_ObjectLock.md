# SDOS-082 — ObjectLock: the object is found whole and cut cleanly, not see-through

**Pack:** v11.2.83 "ObjectLock" (includes v11.2.82 LipCover and v11.2.81 Brisk). Engine `aequus-1.2.83-objectlock`.
**Reported (after 11.2.82):** "the separation between the object and the body is still not clear, there is a bit of overlap, the lower lip still comes over the object, all relatively translucent."

I have not seen your footage (I only work with synthetic media and logs), so every number below is from synthetic faces and the real functions. §5 says what this cannot fix and §6 how to read your log.

---

## 1. Turning your description into numbers

I built a synthetic object that moves up to the mouth, holds, and leaves, and measured the **face's remaining opacity over the object** (0 = the object is fully in front, 1 = the face is painted over it), 3 places: the object's inside, its 4-px edge band, and the face right next to it.

| Version 11.2.82 | inside | edge band | face next to it wrongly cut |
|---|---|---|---|
| mean over a 39-frame pass | **0.23 – 0.32** | **0.56 – 0.60** | 0.25 |
| frames where more than 20 % of the face shows through the inside | **23 – 26 of 39** | | |

That is "translucent" (inside), "overlap" (edge band) and "lower lip over the object".

## 2. Three causes, all measured

1. **Half the object was never cut.** A picture of the alpha shows a clean diagonal across the object: one half cut, the other half painted over. Two things can leave part of an object uncut: the face model learns every pixel that was not cut (a moving object passes over the same pixels for several frames), and the mouth area's learned "appearance" is a smear of lips, teeth and cavity, so an object over the mouth can read as "my own face". I measured the result (half the object uncut), not which of the two did it; either way the part is vouched for. The 11.2.82 fix only covered a lip-sized gap, not half an object.
2. **Slow opening made it see-through.** A new object pixel opened at 45 % per frame: first frame 55 % face, second 30 %, third 17 %.
3. **A soft, lagging edge.** The decision is blurred by a twentieth of the crop and smoothed over time, so its edge is several pixels off the object's real edge and the swapped face shows in that band.

## 3. What changed (`swap_engine.py`)

1. **Whole-object lock** (`_grow_object`). Once an object has qualified under the existing tests, take the colour of the pixels that qualified and grow through **connected pixels of the same colour, stopping at any edge**. Lips differ from the object and are separated from it by an edge; the uncut half of the object is neither. It uses colour of the object itself, not novelty against the face model, so a learned or lip-vouched part is found. Capped at 16 % of the crop; skipped if the seed is not one colour; no-op when nothing qualified.
2. **Fast, confirmed opening.** A hole opens at 85 % per frame once **two consecutive frames** agree; a first sighting still opens at 45 %. The confirmation is what keeps a one-frame false detection from punching a full hole (§4).
3. **A crisp edge.** The cut's edge is snapped onto the object's real edge in the picture with a guided filter (the frame itself is the guide), the blur is a quarter of what it was (0.012 of the crop instead of 0.05) and the edge is steepened so there are few half-transparent pixels.
4. The v11.2.82 mouth-zone fill stays.

## 4. Results (synthetic)

| Test | 11.2.82 | 11.2.83 |
|---|---|---|
| Moving object: opacity left **inside** | 0.27 | **0.02** |
| ...frames with more than 20 % showing through (2 seeds × 39) | 23 – 26 | **1** |
| ...edge band (4 px) | 0.58 | **0.21** |
| Static object, 7 colours × 3 sizes: part of the object cut | 95 – 98 % | **100 %** |
| ...the 12-pixel ring of face around it wrongly cut | 11 – 17 % | **0.1 – 1.4 %** |
| ...face far from the object wrongly cut | 0.03 % | **0.00 %** |
| Big dark scoop at the mouth: lower lip cut | 88 – 100 % | **100 %** |
| Small scoop at the lips, then the whole scoop (27 runs): mean lower-lip cut | 97.4 % | **100 %** |
| 8 must-keep sequences × 3 seeds (clean, beard, glasses, talking, shadow, dark skin, light ×2) | 0.0 % | **0.0 %** |
| Unrelated object present while the mouth talks: worst cut over the lips | 0.04 % | **0.00 %** |
| A **one-frame** false detection: hole-frames it costs | 1.38 | **0.33** |
| Object seen for 2 frames: hole | 2.47 hole-frames | 2.90 |
| Compositor suites (grain match, 512-px adaptive crop, 90-frame soak) | pass | pass, same numbers |
| Cost per face-frame **while an object is in front** (512 crop) | 9.1 ms | **17.7 ms** (+8.6; zero extra with nothing in front) |
| Full job through the real pipeline | completes | completes |

Objects the gate never detects (pale pink, skin-coloured) stay uncut: nothing qualified, so there is nothing to lock onto.

## 5. What this does NOT fix

- **Everything above starts from the existing gate firing.** On a *real photo* face in my end-to-end test the gate did not fire at all (SDOS-081 §5: the face's colour spread is 2–6× wider on real faces than on the synthetic ones the thresholds were tuned on). On your footage it does fire (`occ_cut` 3.5–9 % of face-frames in your logs), so this helps wherever it does; where it does not, nothing here can.
- **Small food at the lips that never qualifies**, **pale pink / skin-coloured food** and a **metal spoon** are unchanged.
- **Fast motion:** a pixel the object newly covers shows face at 0.2–0.3 for that one frame (first sighting opens at 45 %). Raising `occ_open_first` removes it at the cost of a larger flash on a false detection (0.65 → a one-frame false detection costs 1.4 hole-frames, as before).
- The end-to-end clip never exercised the new code (gate silent), so the evidence is the synthetic measurements.

## 6. Reading your log

After each job, next to `Done in …`:

```
OCCLUDER: gate cut something on N of M face-frames; whole-object lock added the rest of the object on G of them (mean +x.x% of the crop); mouth-zone fill acted on K (mean +y.y%)
```

- **G > 0 in an ice-cream job:** the new code found parts the old one missed. If the lip is still over the object, send me the time in the clip and this line.
- **N small while she is eating:** the gate is not firing on your footage and none of this can help; send me the line.

## 7. Dials and rollback (`config.py` → `ENGINE_TUNABLES`)

```python
"occ_grow": 1.0,  "occ_grow_max": 0.16,         # whole-object lock; 0 = off
"occ_open": 0.85, "occ_open_first": 0.45,       # hole opening speed (0.45/0.45 = v11.2.82)
"occ_feather": 0.012, "occ_snap": 0.02, "occ_snap_eps": 0.0005, "occ_steep": 3.0,   # edge; 0.05 / 0 / – / 1.0 = v11.2.82
"occ_bridge": 0.09,                             # v11.2.82 mouth-zone fill; 0 = off
```
To go back to v11.2.82 exactly: `occ_grow 0`, `occ_open 0.45`, `occ_feather 0.05`, `occ_snap 0`, `occ_steep 1.0`.

## 8. Install

Upload the 8 files (`app.py config.py core_pipeline.py phoenix_api_adapter.py swap_engine.py packages.txt requirements.txt README.md`) to the **root** of the Space (or "Push to all Spaces"), commit, normal restart. Startup log must show `Phoenix v11.2.83` and `swap_engine aequus-1.2.83-objectlock`. Do not restart a Space that has a job running. Quality and ordering of everything else is unchanged.

## 9. Changed files
`swap_engine.py` (`ENGINE_VERSION`; `_grow_object`, `_snap_keep`; `build_mask` opening rule, snapping, steepening; `TrackState.occ_raw`; defaults), `core_pipeline.py` (log line and stats), `config.py` (`VERSION`, `BUILD`, tunables). Unchanged: `app.py`, `phoenix_api_adapter.py`, `packages.txt`, `requirements.txt`, `README.md`. Test scripts: `verification/lipcover/` in the repository.
