# SDOS-081 — LipCover: the lower lip painted over the ice cream

**Pack:** v11.2.82 "LipCover" (includes everything in v11.2.81 Brisk). Engine `aequus-1.2.82-lipcover`.
**Reported:** "when the character is eating, the lower lip comes over the ice cream instead of being covered by it, while the shine is correctly covered by the object."

I have not seen your footage (I only work with synthetic media and logs). This pack fixes **one cause I could reproduce and measure**. §5 lists what it does not cover, and §6 how to tell from your own log whether it helped.

---

## 1. What I reproduced

The occluder gate (SDOS-074) cuts a hole in the swapped face where an object is in front of it. Every sentence below was measured on synthetic faces with the real `occluder_keep`, over three random seeds.

**A big dark scoop held at the mouth:** the scoop is cut almost everywhere except over the **lower lip**, which is painted on top of it:

| Scoop at the mouth | Scoop cut away from the lips | Lower lip under it cut |
|---|---|---|
| chocolate, mouth was talking beforehand | 71–82 % | **10–18 %** |
| chocolate, mouth was closed beforehand | 75–89 % | 15–68 % |

A picture of that run shows a ring of cut around an uncut lip-shaped hole exactly where the lips are.

## 2. Why

Three things combine:

1. **Lip colour.** The lips are "off skin" by nature, so an object close to lip colour (chocolate, caramel, strawberry) is *less* different from the lips than from the skin around it. The colour tests find the skin-side of the object and miss the lip-side.
2. **The model learns an uncut object.** The face model that vouches for "this is my own face" learns every pixel that was not cut. An object that sits at the lips for a few frames before it is large enough to qualify (a spoonful, then the whole scoop) is absorbed into that model within about ten frames, and from then on the lip-side of it is "my own face". The part that arrives later is new, so it is cut. That is the pattern you describe: most of the object covers correctly, the lower lip does not.
3. **Interior blobs are not cut on purpose.** A blob has to reach the face edge or be large before it counts, otherwise an open mouth (teeth, tongue, cavity) would be cut. A small piece of food at the lip is such a blob.

## 3. What changed

`swap_engine.py` → new `_close_mouth_zone`, called only after an occluder has **already qualified** under every existing test. Inside the mouth zone (rows 0.58–0.92, columns 0.24–0.76 of the aligned crop) it adds, without using colour:

- the **convex hull** of each qualified component (a lip-coloured part of an object ends up inside the hull of the rest);
- a gap between two parts of the object narrower than a lip;
- any hole the object **completely surrounds** (a patch of face cannot be enclosed by the thing held in front of it).

It never adds more than 9 % of the crop in one go (if it would, it adds nothing). With no qualified occluder it does nothing at all, so a face with nothing in front of it is unchanged by construction.

## 4. Results (synthetic)

| Test | v11.2.81 | v11.2.82 |
|---|---|---|
| Big chocolate scoop at the mouth, lower lip cut | 10–18 % (talking history), 15–68 % (closed) | **88–100 %**, **91–100 %** |
| Same, butterscotch | 93–95 % | 99 % |
| Small scoop sits at the lips, then the whole scoop arrives (27 runs: 3 colours × 3 delays × 3 seeds) — mean lower-lip cut | **43.8 %** | **97.4 %** |
| 8 must-keep sequences × 3 seeds (clean, beard, glasses, talking, shadow, dark skin, light drop ×2) | 0.0 % cut | **0.0 % cut, output identical** |
| Unrelated object present (cheek, under the chin, beside the corner of the mouth, scoop below the lip) while the mouth talks: worst cut over the lips | 0.03 % | 0.04 % |
| Compositor suites (grain match, 512-px adaptive crop path, 90-frame soak) | pass | pass, same numbers |
| Full job through the real pipeline (detector, swapper, compositor), 2 clips | completes | completes, outputs differ from v11.2.81 by ≈1 grey level (encoder noise) |

## 5. What this does NOT fix — read this

- **A small piece of food at the lips with nothing else of the object qualifying** (under about 4.5 % of the face, not reaching the face edge) is still not cut. In a synthetic test a small scoop at the lower lip is cut 0 % at every colour. Telling it from an open mouth by colour is not possible (white ice cream looks like teeth, chocolate like the mouth cavity). It needs a mouth-opening signal I cannot validate without footage.
- **Pale or lip-coloured food and a metal spoon** remain the honest misses of SDOS-074.
- **New finding, not changed.** On a *real photograph* face the gate did not fire at all in my end-to-end test, even for a bright scoop. The face's own colour spread, measured over the forehead and cheeks, comes out 2–6× wider on real faces (brightness spread 30–56 on the six faces of the sample photo, because hair and shading fall in that zone) than on the synthetic faces the thresholds were tuned on (9–12). The gate therefore only trusts high-contrast objects. I did **not** change it: making it more sensitive without your footage could cut real face (hair, shadows) and a false cut is worse than a miss. If the lip fix does not show in your footage, this is the next thing to look at, and I need a log from an ice-cream job to do it (§6).
- I could **not confirm any of this on your footage**, and my end-to-end test never exercised the new code (the gate did not fire on the real-photo clips), so the evidence is the unit measurements above.

## 6. How to tell if it helped — one log line

After each job the log now has, next to the `Done in …` line:

```
OCCLUDER: gate cut something on N of M face-frames; the mouth-zone fill (object held at the lips) acted on K of them (mean x.x% of the crop when it did)
```

- **K > 0 during an ice-cream job:** the new code acted. If the lip is still over the ice cream, send me the exact moment (time in the clip) and this line.
- **N is small and K = 0 while she is eating:** the gate is not firing on your footage (the §5 finding), and the fix cannot help. Send me that line; it decides what I do next.

Your earlier jobs for reference (`occ_cut`, from the `Done` line): 413/5305, 114/3253, 311/3490, i.e. the gate cut something on 3.5–9 % of face-frames.

## 7. Dial and rollback

`config.py` → `ENGINE_TUNABLES`:

```python
"occ_bridge":     0.09,   # gap closed, as a fraction of the crop; 0 = the whole fix off (v11.2.81 behaviour exactly)
"occ_bridge_max": 0.09,   # never add more than this fraction of the crop at once
"occ_hole_max":   0.08,   # largest enclosed hole that is filled
```

## 8. Install

Upload the 8 files (`app.py config.py core_pipeline.py phoenix_api_adapter.py swap_engine.py packages.txt requirements.txt README.md`) to the **root** of the Space, commit, normal restart (or "Push to all Spaces" in the app). Startup log must show `Phoenix v11.2.82` and `swap_engine aequus-1.2.82-lipcover`. Do not restart a Space while a job is running on it.

## 9. Changed files
`swap_engine.py` (`ENGINE_VERSION`, `_close_mouth_zone`, three `_P` entries, `OCC_STATS` +2 counters), `core_pipeline.py` (one log line, stats reset), `config.py` (`VERSION`, `BUILD`). Unchanged: `app.py`, `phoenix_api_adapter.py`, `packages.txt`, `requirements.txt`, `README.md`. The test scripts are in `verification/lipcover/` of the repository (not part of the Space).
