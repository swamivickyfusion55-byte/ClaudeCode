# SDOS-074 — See Food
### The occluder gate could never fire. Now it can.

**Report:** *"taking care of occlusions on face. E.g. when someone is eating something say ice cream… there is a mix up, lower lip and chin not clearly differentiated."*
**Baseline:** `v11.2.74` / engine `aequus-1.2.74-cutwall`
**Delivered:** `v11.2.75` / engine `aequus-1.2.75-seefood`

---

## 1. The occluder gate was arithmetically dead

`occluder_keep()` — whose own docstring names your exact case (*"The oval used to paint the new face over ice cream or food"*) — searched for a foreign object inside this window:

```python
mouth = (row >= 0.58) & (row <= 0.66) & (col >= 0.36) & (col <= 0.64)
```

**0.08 × h by 0.28 × w = 2.24% of the crop.** It then required the connected component to be at least:

```python
min_area = max(80, int(0.04 * h * w))      # 4% of the crop
```

**The minimum area is larger than the entire search window.** At every crop size:

| Crop | Search window | Minimum area required | |
|---|---|---|---|
| 128 | 367 px | 655 px | impossible |
| 256 | 1,468 px | 2,621 px | impossible |
| 384 | 3,303 px | 5,898 px | impossible |
| 512 | 5,872 px | 10,485 px | impossible |

The component could never reach the threshold, so the function returned all-ones on **every frame of every job**. Verified against the real function:

```
clean face (no occluder)       removes 0.00%
vanilla ice cream at mouth     removes 0.00%
chocolate ice cream            removes 0.00%
strawberry ice cream           removes 0.00%
metal spoon at mouth           removes 0.00%
hand over cheek/jaw            removes 0.00%
```

**The engine has had zero occluder rejection for its entire history.** The new face was painted straight over whatever was in front of it — which is exactly the "mix up" you see, and why the lower lip and chin read as undifferentiated: the swap paints a complete mouth and chin where the ice cream should be.

### Why it had been crippled

The code tells you itself: *"Cutting every pixel that was slightly off skin colour also punched the lower lip and the chin, and that edge moved every frame."* Its only evidence was *this pixel is off the face's median chroma* — and **lips are off it**, by construction. So the window was clamped down and the floor raised until the false positives stopped. They stopped because the detector stopped.

---

## 2. The replacement: four pieces of evidence, all required

The lip-versus-object problem is only unsolvable with one signal. With four it is tractable, and the search can cover the whole face instead of 2% of it.

**1. Not this face's skin.** Off its own chroma or luma, measured in units of the face's own spread — so one threshold works on any skin tone under any light.

**2. Not this face's own learned appearance.** A per-pixel model of what this identity looks like in aligned space (`TrackState.occ_ref`). The aligned crop is **pose-normalised**, so a beard, glasses, a birthmark or a scar sits at the same crop coordinates on every frame and matches its own history — ice cream in front of them does not. This is also what catches **food held over a beard**, which a location-only exemption cannot.

> Stored as **normalised residuals**, not absolute YCrCb, so a lighting change cancels. With absolute values a 45% light drop made the whole beard read as novel and cut 14.5% of the face on every frame afterwards. I measured that, then fixed it. The variance floors had to become *proportional* to the face's brightness for the same reason — absolute floors broke scale invariance.

**3. Reaching the edge of the face, or large.** A held object is attached to a hand, an arm, a cone, so its foreign region continues to the face's edge. The lips are an interior island; the mouth cavity likewise. Originally I required the blob to extend fully *outside* the face oval — too brittle, because the part that bridges to the hand is often itself near skin colour and goes undetected, leaving the blob wholly inside with nothing to measure. Reaching the **rim band** is the robust form of the same fact (the lip blob stays ~0.16 of the crop clear of it). Food held right at the lips reaches neither, so a large novel blob also qualifies on size alone — safe *only* because the appearance model already vouches for the lips, gated on that model being well warmed.

**4. Big enough** — as a fraction of the face area, not of the crop.

Nothing is cut during the first `occ_ref_warm` (12) frames, while the model forms; cutting then would punch a hole in a beard before the model can vouch for it.

### The hole also has to open in time

```python
track.occ_keep = track.occ_keep * 0.90 + keep * 0.10
```

That is **22 frames to reach 90% of the hole** — most of a second of the new face being painted over the spoon before the hole was even open, and for food moving to and from the mouth every second or two it never opened at all. Opening is now the fast direction (`occ_open` 0.45, ~4 frames); closing stays slower as insurance against a detector blink. That is only safe because the detector no longer fires on lips, an open mouth, a beard, glasses, a shadow or a lighting change.

---

## 3. Results

Across two independently-seeded synthetic faces with realistic skin statistics:

### Must NOT be cut — **zero false positives**

| Case | Peak removed |
|---|---|
| clean face | 0.0% |
| lips, mouth opening and closing | 0.0% |
| **closed mouth for 24 frames, then wide open** | 0.0% |
| beard | 0.0% |
| glasses | 0.0% |
| hard side shadow | 0.0% |
| light drops 45% mid-shot | 0.0% |
| light rises 70% mid-shot | 0.0% |
| beard + light drop | 0.0% |

The "closed mouth then wide open" case was added specifically to attack the large-blob rule — an expression the model has never seen is the obvious way that rule could misfire.

### Must be cut

| Case | Removed | |
|---|---|---|
| vanilla ice cream at the mouth | 8.2–8.4% | **cut** |
| chocolate ice cream | 4.3–8.2% | **cut** |
| **ice cream held right at the lips** | 8.2–8.4% | **cut** |
| **beard with ice cream over it** | 8.2–8.3% | **cut** |
| ice cream, dark skin | 11.2–11.3% | **cut** |
| food, then a lighting change | 8.4–8.5% | **cut** |
| strawberry (pale pink) ice cream | 0.0% | **missed** |
| low-contrast metal spoon | 0.0% | **missed** |

### The two honest misses

Both are objects whose colour sits within ~2σ of the face's own skin and lip chroma — a pale pink lolly against pink lips, and a grey spoon measured at 2.0σ against light skin where the threshold is 2.6σ. A real metal spoon has specular highlights that make it far more distinct than my fixture; a pale pink ice lolly genuinely is close to lip colour.

Lowering `occ_chroma`/`occ_luma` to 2.2 still produced **zero false positives** in testing and would catch more. I left the default at 2.6 because precision matters more here: a false cut puts the subject's real face back through a hole in the swap, which is worse than failing to cut. **If you hit this, drop both to 2.2 in `config.py` and tell me how it looks.**

A **hand matching the subject's own skin tone** remains out of reach for a chroma method — that needs segmentation. The existing `rival_cut` already covers another person's hand or cheek via their landmarks.

---

## 4. Cost

The decision is Gaussian-blurred to a twentieth of the crop before use, so computing it at full resolution bought nothing:

| | Cost/frame | Model memory per track |
|---|---|---|
| at full crop resolution (512) | 36.7 ms | 3.15 MB |
| **fixed working resolution 192** | **7.9 ms** | **0.44 MB** |

Both are now independent of crop size — which also means the appearance model stays valid when the adaptive crop size changes as the subject moves toward or away from the camera.

At 512 the composite goes from ~34.8 ms to ~42.7 ms per output frame: about **+3.6 s** on a 451-frame clip.

---

## 5. Dials (`config.py` → `ENGINE_TUNABLES`)

```python
"occ_chroma":     2.6,   # lower to catch more (pale food, dull utensils)
"occ_luma":       2.6,
"occ_ref_chroma": 2.6,   # lower = less willing to accept a region as "my own face"
"occ_ref_luma":   2.6,
"occ_min_frac":   0.010, # smallest occluder, as a fraction of the face
"occ_big_frac":   0.045, # blob this large counts without reaching the face edge
"occ_ref_warm":   12,    # frames before the model may veto anything
"occ_work":       192,   # working resolution; cost and memory follow this
"occ_open":       0.45,  # how fast a hole opens
"occ_close":      0.22,
```

**Rollback:** `occ_min_frac = 1.0` disables all cutting (nothing can be that large), restoring v11.2.74 behaviour exactly.

---

## 6. Verification

| Test | Result |
|---|---|
| Old gate's window vs its own minimum area, all crop sizes | **unsatisfiable at every size** |
| Old gate against ice cream / spoon / hand | **0.00% removed, confirmed empirically** |
| 8 must-keep scenarios × 2 seeds | **0 false positives** |
| 6 must-cut scenarios × 2 seeds | all cut |
| Beard learned, then food **over** the beard | both correct |
| Absolute vs proportional variance floors | absolute caused a 14.5% false cut after a light change; fixed |
| `keep` returned at crop size after downscaled working pass | verified at 128/256/384/512 |
| Appearance model never learns under a cut region | by construction, learning is weighted by `keep` |
| **Regression:** grain matching | 95% → 8%, unchanged |
| **Regression:** 90-frame soak, all passes | no crash, EMA jump 0.011 |
| **Regression:** affine maths incl. the 448 trap | 16/16 |
| **Regression:** cut barrier | wrong-shot frames 2 → 0, 0 starved |
| **Regression:** cut detector | 12/12 |
| `py_compile` all modules | clean |

**Honest limit:** calibrated against synthetic faces built to realistic skin statistics (Y std 9–12, Cr/Cb std 3–4), not your footage — this environment has no source video. My first synthetic had flat skin (std ≈ 3) and I caught myself tuning thresholds to that artifact; the chroma variation had to be generated at low resolution and upscaled, because blurring standard normals at σ=28 collapsed the amplitude ~100×. The thresholds are the part most worth checking against real material.

---

## 7. Changed files

| File | Change |
|---|---|
| `swap_engine.py` | `ENGINE_VERSION` → `aequus-1.2.75-seefood`; `occluder_keep` rewritten (four-signal, whole-face, working-resolution); `+ _face_stats`, `_occ_work`, `update_face_ref`; `TrackState` + `occ_ref`/`occ_nref`; `build_mask` passes the model and learns from the decision; hole open/close rates split; 14 new `_P` entries |
| `config.py` | `VERSION` → `v11.2.75`; `BUILD` → `SeeFood · the occluder gate actually fires now`; 14 new `ENGINE_TUNABLES` |
| `core_pipeline.py`, `app.py`, `phoenix_api_adapter.py`, `requirements.txt`, `packages.txt`, `README.md` | unchanged — shipped for set integrity |

**Startup log must read:** `aequus-1.2.75-seefood` and `v11.2.75`.

Still open: the five v11.2.52 tracker edits (SDOS-071 §6), the lazy-recognition lever (SDOS-072 §8), the motion-lag item (SDOS-073 §6), and the v11.2.7 edge-revert item #21.
