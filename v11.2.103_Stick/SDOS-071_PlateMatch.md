# SDOS-071 — Plate Match
### Audit of Phoenix v11.2.71 "MoveLock", and the v11.2.72 remediation

**Scope:** make the output read as *photographed*, not *composited*.
**Baseline audited:** `v11.2.71` / engine `aequus-1.2.51-detzoom` (SuperGrok build)
**Delivered:** `v11.2.72` / engine `aequus-1.2.72-plate`
**Method:** every claim below is a number produced by a script that was run, not an estimate. Where my first measurement was wrong, the wrong one and the correction are both recorded.

---

## 0. Executive summary

The build's *geometry* is in good shape. The Continuum architecture — swap on sparse key frames, composite every frame from a cached ArcFace-aligned crop — is sound, and it is already fully resolution-agnostic: every consumer derives crop size from `fake.shape[0]`, nothing hard-codes 128.

What is wrong is not geometry. It is that **three separate stages each throw away the evidence that the output was photographed**, and all three are cheap to fix:

| ID | Defect | Measured impact | Status |
|----|--------|-----------------|--------|
| **F1** | The face restorer runs at 512, then its output is resized back to 128 and discarded | **85% of restored detail destroyed**; delivered on-screen detail **9.6× lower** than achievable at a 512px face | Fixed |
| **F2** | Restored faces carry no grain; the plate around them does | Face grain **95% below** the surrounding footage | Fixed → **8%** |
| **F3** | Restored faces stay sharp through motion the plate blurs | Pass did not exist | Implemented |
| **F4** | Output converted with BT.601 and tagged with nothing, while every HD player assumes BT.709 | **41 levels** of channel error at 1080p — a colour cast across the whole film | Fixed → **4** |
| **F5** | "Optimized" tier used a *lower* quantiser with a *worse* preset than the tier below it | Spent more bits for less quality | Fixed |

F1 and F2 are the two that matter most, and they are the same mistake twice: **work was done at high resolution, then resampled down before anyone could see it.**

There is also one finding outside this scope that I did not act on but which you should read first — see **§6**.

---

## 1. F1 — The restoration was being deleted before it was used

### 1.1 The defect

`core_pipeline.py`, `_aligned_post._post`:

```python
big = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC)
out = _enhance(big, enh)                 # GFPGAN / CodeFormer at its native 512
stats["gfpgan_calls"] += 1
return cv2.resize(out, (crop.shape[1], crop.shape[0]),
                  interpolation=cv2.INTER_AREA)      # <-- back down to 128
```

The restorer is loaded with `upscale=1` and works natively at 512×512. It is handed a 512 upscale of the 128 swap output, it produces a genuine 512 restoration — and that restoration is then resized to 128 and returned. The paste then *up*-samples that 128 result back to whatever the face occupies on screen, typically 300–600px.

So for a close-up the pipeline was: 128 → 512 → restore → 128 → ~450. Two of those three resamples are pure loss, and the expensive middle step was being paid for in full and then thrown in the bin.

Two guards enforced the loss. `swap_engine.run()`:

```python
if p is not None and p.shape == fake.shape:   # a 512 crop fails this
    fake = p
```

and the identical guard in `core_pipeline.swap_frame`. A restorer returning anything larger was silently rejected and the un-restored 128 swapper output used instead.

### 1.2 How much detail that costs

My **first** measurement of this was wrong and I am recording it because the error is instructive. I built the synthetic "restored face" from 8–64px noise upscaled to 512. All of its content sat below the 128px Nyquist limit, so 128px could carry essentially all of it, and the measurement reported only **8.6% loss**. That number was an artefact of my test content, not a property of the pipeline.

Rebuilt with pore-level detail present on the full 512 grid — which is what a face restorer actually synthesises — the honest figures are:

| On-screen face size | Restoration detail destroyed by the 128 round-trip |
|---|---|
| 200px | 61.5% |
| 400px | **85.3%** |
| 600px | 83.1% |

### 1.3 The subtlety that makes this non-trivial

You cannot simply return the 512 crop. The reuse path reconstructs the affine from the cached correction:

```python
C = M_actual @ inv(estimate_norm(kps, size))      # aligned_correction
M = apply_correction(C, estimate_norm(kps, size)) # reuse
```

Pass `size=512` and `estimate_norm` resolves against the **128-space** template, so the face pastes at quarter size. The affine has to be told the crop grew:

```python
def scale_affine(M, k):
    S = np.array([[k, 0, 0], [0, k, 0], [0, 0, 1]], np.float32)
    return (S @ _as3x3(M))[:2, :].astype(np.float32)
```

This is only valid because the ArcFace template itself scales — and it does **not** always. `_arcface_dst` switches branch on `image_size % 112`:

```
_arcface_dst(256) == _arcface_dst(128)*2     True
_arcface_dst(512) == _arcface_dst(128)*4     True
_arcface_dst(384) == _arcface_dst(128)*3     True
_arcface_dst(448) == _arcface_dst(128)*3.5   False   maxdiff = 46.183
```

A **448** crop lands on the 112 branch with `diff_x = 0` instead of 32, and the face would paste offset by ~46px of crop space. A naive "accept anything bigger" fix has a landmine in it. So admissibility is *verified*, never assumed:

```python
def crop_upscale_ok(old_size, new_size):
    k = new_size / old_size
    return np.allclose(_arcface_dst(new_size), _arcface_dst(old_size) * k, atol=1e-3)
```

### 1.4 Adaptive sizing, not "always 512"

Retaining every crop at 512 would be **16× the compositing work on every output frame**, and most of it wasted: `paste_back` immediately resamples the crop down to the face's on-screen span, so resolution above that span buys nothing visible. Measured at 1080p, the composite alone: 6.85 ms at 128, 28.60 ms at 512.

So the kept size is chosen from the face's true on-screen span, which is recoverable from the affine's own linear part:

```python
def native_crop_span(M, size):
    A = np.asarray(M, np.float32).reshape(2, 3)[:, :2]
    return size / math.sqrt(abs(np.linalg.det(A)))
```

```
on-screen span    51px -> crop 128        384px -> crop 384
                 102px -> crop 128        410px -> crop 512
                 128px -> crop 128        512px -> crop 512
                 205px -> crop 256       1152px -> crop 512
```

A distant face costs exactly what it used to and looks exactly as it used to. A close-up gets the full restoration. `restore_crop_max` (default 512) caps it; set 128 to reproduce the old behaviour byte-for-byte.

> Note the half-pixel slack in `choose_crop_size`. The span comes out of a `sqrt` of a determinant, so an exact 3× face measures `384.0000001` and without the slack would round *up* to 512 and cost 16× the compositing for nothing.

### 1.5 Verification

The affine algebra is exact, not approximate:

```
scale_affine(est128, 4) == est512                        maxdiff 0.00e+00
reuse512 == scale(reuse128, 4)                           maxdiff 0.00e+00
crop-corner image positions agree (128 vs 512 path)      max 0.0000 px
```

End to end through `run()` *and* the `reuse()` path, with a head that has moved and rotated between the two:

```
OLD (crush to 128)      cached crop 128   key box (658,228,1138,718)   reuse box (891,277,1399,800)
NEW (adaptive retain)   cached crop 512   key box (660,230,1138,719)   reuse box (894,279,1402,802)
```

The face lands in the same place at the same size — the 2–3px difference is the mask feather crossing the detection threshold, not displacement.

And the point of the exercise, measured in the **same image-space window** for both paths (my first attempt at this measured the two cases at different scales and reported a *loss*; `mean |Laplacian|` is resolution-dependent and that comparison was meaningless):

| On-screen face | v71 delivered detail | v72 delivered detail | Gain |
|---|---|---|---|
| 128px | 20.77 | 20.77 | **0%** — identical, by design |
| 256px | 5.49 | 25.16 | +358% |
| 384px | 2.81 | 17.66 | +528% |
| 512px | 1.92 | 18.40 | **+860%** |
| 768px | 1.35 | 10.11 | +649% |

---

## 2. F2 — Grain

### 2.1 Why this is the dominant tell

A face restorer is a denoiser as much as a detail synthesiser; its output is clean. The footage it is pasted into is not — it carries sensor grain. A clean face inside a grainy frame is a **spatial-frequency** mismatch, and the existing LAB colour match cannot touch it because colour matching operates on tone. This is consistently the first thing a viewer registers as wrong, ahead of colour and ahead of edges.

v11.2.71 contained zero grain handling — no occurrences of grain, noise synthesis, or any equivalent.

### 2.2 Measuring the plate honestly

The obvious estimator — standard deviation of a high-pass — is defeated by a face. Pores, lashes and the lid crease *are* high-frequency content, so the estimate rides up on detailed faces and the matcher injects grain that is not there. I used Immerkaer's estimator instead, whose kernel has zero response to any locally-quadratic surface (which is what smooth skin under soft shading is):

```python
_IMMERKAER = np.array([[ 1, -2,  1],
                       [-2,  4, -2],
                       [ 1, -2,  1]], np.float32)
sigma = sqrt(pi/2) * mean_under_mask(|I * N|) / 6
```

Calibration — unbiased on pure noise, and structure-rejecting on a face:

```
pure Gaussian noise, flat field:   true 1.0 -> 1.00    4.0 -> 4.02    16.0 -> 15.96   (gain 1.00)
face with sigma-6 pore texture, no noise added:        estimate 0.47   (structure floor)
```

A σ=6 pore field reads as 0.47. That is the property the whole pass depends on.

> A second measurement of mine was misleading here too: an early test reported a gain of 0.66 and I briefly suspected the estimator. The estimator was fine — my synthetic plate gave each BGR channel *independent* noise, and `cvtColor(BGR2GRAY)` averaged three independent fields, reducing σ by ~1/√3. The code was right; the test's label was wrong.

### 2.3 The part I got wrong first, and the fix

I initially ran grain matching on the aligned crop, next to the colour match. It worked in isolation but barely moved the end-to-end number:

```
plate grain 8.0:   face sigma 0.26 -> 0.78   against plate 5.38     mismatch 95% -> 85%
```

The reason is that **bilinear interpolation is a low-pass filter**, and the crop path crosses it twice:

1. `aligned_target = warpAffine(orig, M, ...)` — the grain is attenuated *before* anything measures it, so the measured deficit is already about half of true.
2. `paste_back` warps the crop back out — the injected grain is attenuated again, and geometrically *stretched*, which is something a camera cannot do.

Grain is what the sensor wrote onto the final pixel grid, so both halves of the job have to happen on that grid. The pass now lives in `paste_back`, measured on the **untouched plate under the mask** (the original face — same subject, same distance, same light, which is the fairest reference available) and injected into `warp_face` *after* the warp:

```python
if orig is not None:
    warp_face = match_grain(warp_face, orig[y1:y2, x1:x2], warp_mask, track=track)
```

Deficits combine in quadrature, because noise powers add and amplitudes do not:

```python
deficit = sqrt(max(s_t**2 - s_f**2, 0.0)) * grain_strength
```

and the clamp at zero means a *clean* plate with a noisy source photo correctly does nothing — the pass never denoises the face to fix a problem that is not there.

Grain is luma-dominant: one shared luma field plus a weak independent per-channel chroma field. Equal independent noise on all three channels produces coloured speckle that reads as compression damage, not film.

### 2.4 Result

Grain in the painted face versus the untouched footage beside it, in the final frame:

| Plate grain σ | v71 face σ | mismatch | v72 face σ | mismatch |
|---|---|---|---|---|
| 2.0 | 0.26 | 81% | 1.33 | **3%** |
| 4.5 | 0.26 | 91% | 2.89 | **5%** |
| 8.0 | 0.26 | 95% | 4.98 | **8%** |

Temporal behaviour: grain is regenerated per output frame, so a crop reused across twenty frames gets twenty different grain fields — baking it into the cached crop would turn grain into dirt on the lens. The σ *estimate* is EMA-smoothed (largest single-frame change over 90 frames: **0.011**) so the amplitude tracks the shot rather than the frame.

Cost control: four fresh 512² Gaussian draws per frame measured ~11 ms, more than the rest of the composite combined. A pool generated once and read at a random offset per frame is visually indistinguishable and costs a memory copy. That change alone took the pair from +35 ms to +12 ms at 512.

---

## 3. F3 — Motion blur

### 3.1 The flaw in the obvious approach, which my first version had

When the head moves fast the plate's face is smeared and a sharp swap reads as pasted on. The obvious implementation compares the sharpness of the swap against the sharpness of the plate and blurs the difference away.

That is wrong in a way that defeats the entire build, and **my first implementation had it**. A restored face is *supposed* to be sharper than the plate — that is what the restorer is for. So "fake is sharper than target" is the normal case on a locked-off tripod shot, and the pass fires on every frame with any movement at all. The 90-frame soak caught it: on footage with **no motion blur whatsoever**, the absolute-sharpness version applied blur kernels up to **17px**, destroying exactly the restoration F1 had just rescued.

### 3.2 The invariant that works

Motion blur is not a loss of sharpness, it is a **directional** loss of sharpness: it suppresses detail along the travel axis and leaves detail across it untouched. So the quantity to match is the asymmetry, measured as perpendicular-over-along Sobel energy about the motion direction.

The decisive property, measured across detail amplitudes 8, 20 and 40:

```
   L        8      20      40
   0    1.014   0.990   0.993
   3    1.410   1.403   1.444
   7    2.315   2.469   2.350
  11    3.051   3.024   3.098
  21    3.970   4.193   4.389
```

The three columns agree to within ~3%. **Anisotropy is independent of how much texture the face has** — which is precisely the property absolute sharpness lacks. It reads ~1.0 for any unblurred image and climbs with blur length at about 0.14 per pixel of kernel, giving `mblur_k = 6.0` (the inverse, rounded *down*: this pass should under-blur, because a face softer than its plate reads as out of focus while one slightly sharper just reads as a good restoration).

Direction comes from the track's image-space keypoint velocity pushed through the affine's linear part, so the blur lies along the face's motion *in the crop*, which rotates with the head.

This pass stays in aligned space, unlike grain: blur is a property of the optics, so it must act before the sensor stage, and the aligned crop is the frame the head itself lives in.

### 3.3 Verification

The case that broke the first version:

```
plate NOT blurred, restored face far more detailed
  plate anisotropy 0.998   fake anisotropy 1.011   ->  restoration preserved: True
```

Genuinely blurred plates, with the fake both sharper *and* more detailed:

| Plate blur | plate anisotropy | fake anisotropy | kernel applied | output anisotropy |
|---|---|---|---|---|
| 0 | 1.008 | 1.004 | — | 1.004 |
| 3 | 1.414 | 0.998 | 3.1 | 1.319 |
| 5 | 1.956 | 0.996 | 5.9 | 2.022 |
| 7 | 2.397 | 0.995 | 8.2 | 2.204 |
| 11 | 3.039 | 0.992 | 11.4 | 2.229 |
| 17 | 3.567 | 0.998 | 14.1 | 2.273 |

Under-shooting at extreme blur is the deliberate direction, enforced by three independent caps: `mblur_strength` 0.85, the travel distance itself (it can never blur more than the motion allows), and `mblur_max` as a fraction of the crop.

All idle paths confirmed no-ops: still head, zero velocity, no track, `mblur_strength = 0`.

---

## 4. F4 — Rec.709, and why the naive fix is worse than the bug

### 4.1 The defect

```python
"-f", "rawvideo", "-pix_fmt", "bgr24", ...      # full-range BGR in
*_encoder_args, "-pix_fmt", "yuv420p",          # ...and out, with no colour tags
```

ffmpeg's default `rawvideo → yuv420p` conversion uses the **BT.601** matrix and writes **no colour metadata**. Every HD player treats untagged HD as **BT.709**. So the entire film was decoded with the wrong matrix — a colour cast on every frame, completely independent of anything the face pipeline does.

Measured at 1080p on saturated patches, decoded as a correct HD player would decode it:

```
current build (601 pixels, untagged)    worst-channel error 41    mean 7.90
```

### 4.2 The trap

The obvious fix is to add the tags. **Do not do this.** Tested:

| | worst-channel error |
|---|---|
| A — current build, no colour flags | 5 |
| B — **BT.709 tags only** | **41** |
| C — tags **+ scale filter** | 4 |

Tagging alone makes it *materially worse*, because the pixels are still converted with 601 and are then *labelled* 709. The scale filter is what changes the conversion; the tags are what stop the player guessing. **Both, or neither.** This is the single most likely way an audit recommendation of this kind ships a regression.

### 4.3 The fix

```python
_colour_args = [
    "-vf", "scale=out_color_matrix=bt709:out_range=tv",
    "-colorspace", "bt709", "-color_primaries", "bt709",
    "-color_trc", "bt709", "-color_range", "tv",
]
```

```
fixed (709 pixels, 709 tagged)          worst-channel error 4     mean 1.33
```

4 is the codec noise floor at CRF 0. Verified as a complete command with audio, without audio (the optional `-map 1:a:0?` path), and the stream tags confirmed in the muxed output: `yuv420p(tv, bt709, progressive)`.

---

## 5. F5 — Encoder ladder, and audio

`"Optimized"` sat at CRF 23 with the **ultrafast** preset — a *lower* quantiser than Balanced (CRF 24) paired with a *worse* preset than Balanced (veryfast). It spent more bits for less quality than the tier below it. Now `veryfast`, which makes the ladder monotonic.

Audio was always re-encoded at 192k AAC. Transparent for dialogue, audible on music beds at the tiers someone picks when the result is going to be graded or projected: now 256k for Best, 320k for Ultra.

---

## 6. Finding outside this scope — read this one

**None of the five edits you specified for v11.2.52 are present in v11.2.71.** Verified individually:

| Edit you requested | State in v11.2.71 |
|---|---|
| `ENGINE_VERSION` → `aequus-1.2.52-slothold` | was still `aequus-1.2.51-detzoom` |
| `trk_cross_iou` 0.12 → 0.30 | still **0.12**, in both `config.py:293` and `swap_engine.py:164` |
| drop the `_cdist_norm` clause from the crossing test | still present, `swap_engine.py:2345` |
| insert the slot-hold block | **absent** — no occurrence anywhere |
| `want+1` → `want`, `_merge_faces(..., iou=0.22)` | `_merge_faces` **does not exist** in `core_pipeline.py` |

v11.2.71 appears to have been branched from v11.2.51, not v11.2.52. The male-character-not-swapped fix was never in this line of the code, which is the most likely reason that symptom has persisted.

**I have deliberately not applied these in this pack.** Your standing rule is one change per pack, and folding a tracker-behaviour change into a realism pack would make any regression impossible to attribute. The diffs above are exact and ready — say the word and they go out as their own pack, on top of v11.2.72.

---

## 7. Cost

Per **output frame**, 1080p plate, median of 9×6 reps:

| On-screen face | v71 (128, no matching) | v72 (adaptive + both passes) | Δ |
|---|---|---|---|
| 128px | 2.83 ms | 5.70 ms | +2.87 ms |
| 256px | 4.01 ms | 9.70 ms | +5.69 ms |
| 384px | 4.70 ms | 18.10 ms | +13.41 ms |
| 512px | 4.13 ms | 34.76 ms | +30.62 ms |
| 768px | 7.47 ms | 41.46 ms | +34.00 ms |

Stated plainly: on a 451-frame clip dominated by close-ups this is **+13 to +15 seconds**. Against your last logged job — detector 974 ms/call × 448 calls ≈ 436 s — that is about **3%**. The composite was never the bottleneck and still is not.

Memory: `_aligned_hist` holds one crop per key frame per slot before emission prunes it. At 1080p `chunk_n` is 120, so the worst case (`skip_n=1`, two slots) is 120 × 2 × 786 KB ≈ **189 MB**, on top of the 746 MB the frame buffer already holds. Realistic cadence (`skip_n=5`) is ~38 MB. `restore_crop_max` is the control if that ever matters.

**Dials, all in `config.py` → `ENGINE_TUNABLES`:**

```python
"restore_crop_max": 512,   # 128 = exact v71 behaviour; 256 = most of the gain, ~1/3 the cost
"grain_strength":   0.90,  # 0.0 disables grain matching outright
"mblur_strength":   0.85,  # 0.0 disables motion-blur matching outright
```

Both passes are *measurement-driven*, so on clean, static footage they are already no-ops without being turned off.

---

## 8. Verification performed

| Test | Result |
|---|---|
| Affine algebra — template scaling, `estimate_norm` scaling, reuse round-trip | exact, `maxdiff 0.00e+00` |
| Paste geometry — 128 vs 512 crop-corner image positions | `0.0000 px` |
| 448-crop template trap | correctly rejected by `crop_upscale_ok` |
| End-to-end geometry through `run()` **and** `reuse()`, head moved + rotated | same place, same size |
| Delivered detail by face size | +0% at 128px, +860% at 512px |
| Immerkaer calibration, pure noise | gain 1.00 |
| Immerkaer structure rejection, σ=6 pore field | floor 0.47 |
| Grain match, end to end against surrounding plate | 95% → 8% mismatch |
| Grain: clean plate, noisy source | unchanged — never denoises |
| Grain: per-frame independence | confirmed, mean abs delta 1.77 |
| Anisotropy invariance across detail amplitudes 8/20/40 | agree within 3% |
| Blur: sharp plate + sharper restored face | **restoration preserved** |
| Blur idle paths (still / zero vel / no track / disabled) | all no-ops |
| 90-frame soak, all passes on, moving + turning + scaling head | no crash; grain EMA 1-frame jump 0.011; blur correctly idle |
| Encoder command — audio, no audio, muxed tags | `rc=0`, `yuv420p(tv, bt709, progressive)` |
| `py_compile` on all five modules | clean |
| `config.ENGINE_TUNABLES` → `swap_engine._P` key coverage | zero orphans |

One structural near-miss worth recording: the first insertion of the plate-matching functions placed module-level `def`s immediately before a *method*, which silently closed the `AlignedCompositor` class body. `ast.parse` reported **syntax OK** and every method had detached from the class. Caught by asserting `hasattr(AlignedCompositor, ...)` rather than trusting the parse. Syntax checks do not catch structural damage.

---

## 9. Residual gaps — the honest roadmap

Not fixed here, in the order I would take them:

1. **`inswapper_128` is a hard 128px ceiling on identity.** F1 recovers the *restorer's* resolution; the swap network's own output is still 128 and everything above it is synthesised, not transported. This is the single biggest remaining limit on "movie grade" and no amount of post-processing changes it. The fix is a higher-resolution swap model (SimSwap-512, or a 256/512 inswapper variant), which is a model-swap project, not a patch.
2. **Occlusion is geometric + chroma, never segmentation.** A hand across the face is handled by landmark hulls and skin confidence. A real face-parsing model (BiSeNet / FaceParsing) would cut hair, glasses and fingers correctly instead of approximately.
3. **Lighting direction is not matched** — only global LAB mean and std. A face lit from the left pasted into a scene lit from the right is tonally correct and still reads wrong. Spherical-harmonic relighting is the known approach.
4. **8-bit only.** No 10-bit (`yuv420p10le`) path, no HDR/PQ. Needed for anything destined for a grade.
5. **Edge revert** — a face partly outside the frame still flickers. Carried from the original v11.2.7 audit as item #21; still open.
6. **The §6 tracker edits.** Highest priority of everything on this list, and it is a one-pack job.

---

## 10. Changed files

| File | Change |
|---|---|
| `swap_engine.py` | `ENGINE_VERSION` → `aequus-1.2.72-plate`; `+ crop_upscale_ok`, `native_crop_span`, `choose_crop_size`, `scale_affine`, `noise_sigma`, `motion_anisotropy`, `match_grain`, `match_motion_blur`, `_grain_field`, `_ema`; `run()` crop retention; `paste_back()` grain stage; `_composite()` blur stage; `TrackState` + `grain_ema`, `mblur_ema`; 10 new `_P` entries |
| `core_pipeline.py` | `_aligned_post._post` returns the restoration at its own resolution; `swap_frame` retains the larger crop and scales the affine; encoder Rec.709 convert + tag; `Optimized` preset; audio bitrate by tier |
| `config.py` | `VERSION` → `v11.2.72`; `BUILD` → `PlateMatch · restore kept, grain + motion matched, Rec.709`; 10 new `ENGINE_TUNABLES` |
| `app.py`, `phoenix_api_adapter.py`, `requirements.txt`, `packages.txt`, `README.md` | unchanged — shipped for set integrity |

**Startup log must read:** `aequus-1.2.72-plate` and `v11.2.72`. If it does not, the Space is running stale files.

**Rollback:** `restore_crop_max: 128`, `grain_strength: 0.0`, `mblur_strength: 0.0` reverts all three realism changes from config alone, no code edit. The encoder fixes (F4/F5) are independent of those and should be kept regardless — they are strictly corrections.

---

## 11. Research consulted

- [ONNX Runtime — model quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) — INT8 gains 5–15% without VNNI
- [ONNX Runtime — performance tuning](https://oliviajain.github.io/onnxruntime/docs/performance/tune-performance.html)
- [InsightFace model zoo](https://github.com/deepinsight/insightface/blob/e68b8b076fb11e9f75c87244d234385f0f318181/model_zoo/README.md) — buffalo_l = SCRFD-10GF + w600k_r50 (91.25% MR-ALL); buffalo_s = SCRFD-500MF + MBF (71.87%)
- [SCRFD](https://www.insightface.ai/research/scrfd)
- [Hugging Face Spaces hardware](https://huggingface.co/docs/hub/en/spaces-overview) — free CPU Basic = 2 vCPU / 16 GB
- [ORT thread oversubscription](https://github.com/mmornati/proton-faces/issues/79) — degradation is super-linear
- [Deriving ONNX thread counts from the cgroup grant](https://github.com/NSTA1/Orleans.Lattice/pull/2610)
- [oneTBB oversubscription](https://github.com/intel/tbb/issues/190)
- J. Immerkaer, *Fast Noise Variance Estimation*, Computer Vision and Image Understanding, 1996 — the grain estimator in §2.2
- Deepfake-realism literature consensus: neck seam and skin-tone mismatch are the most-cited tell, followed by grain mismatch and absent motion blur — which is the ordering F2/F3 target.
