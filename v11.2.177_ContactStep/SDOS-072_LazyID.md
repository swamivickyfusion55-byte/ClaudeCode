# SDOS-072 — Lazy ID
### Why the detector costs 974 ms, and the fix

**Scope:** detector throughput, with no change to what the tracker sees.
**Baseline:** `v11.2.72` / engine `aequus-1.2.72-plate`
**Delivered:** `v11.2.73` / engine `aequus-1.2.73-lazyid`
**Method:** profiled against the real `buffalo_l` models (SCRFD-10GF + w600k_r50) under `onnxruntime` pinned to 2 threads, to approximate HF free CPU Basic. Every number below came out of a script that ran.

---

## 1. The headline: detection is 7% of "detector time"

Your log reports one number, `detector_avg=974.3ms`, and the natural reading is "the face detector is slow". It isn't. Profiled on a 720p frame holding 6 detectable faces:

| Stage | Model | Cost | Runs |
|---|---|---|---|
| Detection | SCRFD-10GF @ 640 | **77.4 ms** | once per pass |
| Landmarks | landmark_2d_106 | **5.4 ms** | **per face** |
| Gender/age | genderage | **0.6 ms** | **per face** |
| **Recognition** | **w600k_r50 (ArcFace R50)** | **84.6 ms** | **per face** |

Full `FaceAnalysis.get()` on that frame: **1030 ms**, which reproduces your 974 ms almost exactly. Of it, **508 ms is recognition** — 93% of the per-face cost, and 53% of the whole call.

`FaceAnalysis.get()` runs all three auxiliary models on **every** detection, unconditionally:

```python
for i in range(bboxes.shape[0]):
    face = Face(bbox=bbox, kps=kps, det_score=det_score)
    for taskname, model in self.models.items():
        if taskname == "detection": continue
        model.get(img, face)        # landmark + genderage + RECOGNITION, every face
```

Measured against face count, same frame:

```
max_num=2  ->  440 ms   (2 faces)
max_num=4  ->  735 ms   (4 faces)
max_num=8  -> 1023 ms   (6 faces)
max_num=20 -> 1030 ms   (6 faces)
detection ONLY, all 6  ->   74.5 ms
```

## 2. And the pipeline throws most of those faces away

This is the part that makes it expensive rather than merely inefficient. A single "detector call" in this build is up to four separate `_fa_get` invocations:

1. `_detect_want` pass 1.
2. `_detect_want` pass 2 — blacks out the faces already found and detects again, **whenever fewer than `want` were found**. So it runs on exactly the frames that were already the slowest.
3. The hi-res probe, when a slot is missing or a face looks marginal.
4. `_zoom_detect`, once per still-missing slot.

Every face from every one of those passes gets an 84.6 ms embedding. They are then merged, deduped, keypoint-reliability-gated and overlap-filtered down to the **one or two people actually being swapped**. Every discarded face paid full price.

Back-solving your logged 974.3 ms against the measured components:

```
if 1 pass  per detect  ->  9.9 faces per pass
if 2 passes per detect ->  4.5 faces per pass
if 3 passes per detect ->  2.7 faces per pass
```

All three of those are the wasteful regime. A genuinely efficient frame — two people, one pass — costs **259 ms**, not 974 ms. So roughly **three quarters of your detector time is embeddings on faces that were discarded, or on passes that duplicated work.**

---

## 3. The fix: defer recognition to the survivors

Recognition is removed from the detection path and paid for once, at the single point where faces are handed to the tracker.

The catch is coordinates. The ArcFace crop is cut from the **detection-space** keypoints on the **detection-space image**, but by the time faces reach the tracker, `_scale_faces` has rescaled them to full-frame coordinates and `_shift_faces` has translated the zoom-crop ones. Handing the recogniser those keypoints would crop the wrong region of the wrong picture. So the detection-space keypoints and the source image are stashed on the face and restored for the inference:

```python
if defer_embedding and _DEFER_EMB_ON:
    f._emb_img = img
    f._emb_kps = np.array(f.kps, np.float32, copy=True)
```

```python
saved = getattr(f, "kps", None)
try:
    f.kps = kps          # detection-space keypoints
    rec.get(img, f)      # against the detection-space image
finally:
    f.kps = saved
    f._emb_img = None    # do not pin a megabyte of pixels per tracked face
    f._emb_kps = None
```

Filled at the choke point, immediately before `tracker.assign`:

```python
faces = _as_face_list(faces)
_embed_faces(faces)
faces = _dedupe_faces(faces)     # second pass, now that embeddings exist
assigned = tracker.assign(faces, ref_map, dt_frames=dt_frames)
```

**The tracker receives exactly the faces it always did, carrying exactly the embeddings it always did.** Verified: embeddings are **bit-identical**, max absolute difference `0.000e+00`, including after the coordinate transforms.

### Dedupe keeps its strength

`_dedupe_faces` uses embeddings as its *second* test and already has a complete fallback — IoU first, then embedding cosine, then `elif not dup:` a centre-distance test. It is written to degrade that way. So inside `_fa_get` it now runs on IoU plus centre distance, and then runs **again** on the embedded survivors, so the one case the fallback cannot see — two barely-overlapping boxes that are the same person — is still caught. Verified: deferred and eager dedupe keep the same 6 of the same raw detections.

---

## 4. A latent bug fixed on the way

`_fa_get` passed `max_num=20` to SCRFD. When more than 20 faces are present, SCRFD picks the top 20 by:

```python
values = area - offset_dist_squared * 2.0   # "some extra weight on the centering"
```

That does not merely *prefer* central faces, it overwhelms area. A face 300 px off centre loses 180,000 units against a typical face area of ~10,000 — so an off-centre face is ranked below essentially any central one. **That is precisely the "second character entering from the side is dropped" failure mode.** It needed >20 faces to bite, so it was latent, but the ordering was working against you.

`max_num` now defaults to `0` (unlimited). This was previously a sensible trade — each extra face cost ~91 ms. It now costs ~6 ms, so the cap has no reason to exist.

I considered capping candidates myself to buy more speed and **rejected it** for the same reason: any ranked cap can drop the person you want, and that bug has cost you weeks.

---

## 5. What this will actually do for your job

| Scenario | v11.2.72 | v11.2.73 | Speedup |
|---|---|---|---|
| Both people visible, 1 pass | 259 ms | 259 ms | **1.0×** — no change |
| One visible → blackout 2nd pass | 336 ms | 251 ms | 1.3× |
| 2 people + 2 bystanders, 1 pass | 440 ms | 271 ms | 1.6× |
| 2 people + 4 bystanders, 1 pass | 621 ms | 283 ms | 2.2× |
| 1 visible + 4 bystanders, 2 passes | 1061 ms | 299 ms | **3.5×** |
| 2 people + 4 bystanders, 2 passes | 1242 ms | 396 ms | 3.1× |
| + hi-res probe, 3 passes | 1863 ms | 509 ms | **3.7×** |

**Stated plainly: where the old code was already efficient, this changes nothing.** The gain is exactly proportional to the wasted work, which is the honest shape of the fix. Your 974 ms average says you are deep in the wasteful rows.

Projected on your logged job (448 detector calls, 436.5 s of detector time):

| If your frames look like... | Detector time becomes | Saved |
|---|---|---|
| 2 people + 2 bystanders, 1 pass | 268.6 s | 167.9 s |
| 2 people + 4 bystanders, 1 pass | 198.6 s | 237.9 s |
| 2 people + 4 bystanders, 2 passes | 139.2 s | **297.3 s** |
| 3 passes, 6 faces | 119.4 s | **317.1 s** |

Your 7.6-minute job becomes roughly **2 to 4.5 minutes**.

---

## 6. Instrumentation — so the next round is measured, not guessed

`detector_avg=974ms` cannot distinguish "one pass over ten faces" from "four passes over two", and those want opposite fixes. The summary line now carries the three quantities that multiply:

```
det_passes=<n> raw_faces=<n> embeds=<n> (<x> embeds per raw face)
```

- `det_passes` well above `detector_calls` → the extra probes are firing; the lever is their trigger conditions.
- `raw_faces / det_passes` high → bystanders; the lever is a size/score floor.
- `embeds per raw face` near 1.0 → the deferral is not helping and the next lever is §8.

The one-time fallback is now a **warning**, not a debug line:

```
staged detect unavailable (<Error>: <msg>) - falling back to FaceAnalysis.get
for the whole job; lazy recognition is OFF
```

A silent fall back to the slow path is a job that takes four times as long for no stated reason, which is indistinguishable from "the optimisation doesn't work". I hit exactly this while testing — a missing module-level constant sent every call down the fallback, and the only symptom was that the speedup vanished.

---

## 7. What I measured and deliberately did NOT change

**`det_size` 640 → 512** saves 34 ms of a 605 ms call (5.6%). I measured the cost in findability, which is the thing that matters:

```
face crop width   640    512    480    416    320
          110px  0.90   0.88   0.86   0.89   0.86
           60px  0.88   0.84   0.76   0.73   0.34
           45px  0.76   0.67   0.57   0.33   0.43
           34px  0.56   0.32   0.53     --     --
           20px  0.54     --     --     --     --
                           (det_score; -- = NOT FOUND)
```

512 holds down to ~45 px and fails below ~34 px. The faces at risk are too small to swap usefully — but the gain is 5.6% and the risk touches your worst bug, so **the default stays 640**. `PHOENIX_DET_SIZE` already exposes it.

**ORT threading** is already correct: `intra_op_num_threads = native`, `inter_op = 1`, `ORT_SEQUENTIAL`, `allow_spinning = 0`, EXTENDED graph optimisation. Nothing to win there.

**Capping the candidate set** — rejected, see §4.

---

## 8. Residual gaps — the next lever, and its risk

The row this pack does **not** improve is "both people visible, one pass": 2 faces × 84.6 ms = 169 ms of a 259 ms call, and both embeddings are genuinely handed to the tracker.

The next lever is to notice that once a track is **identity-locked and established**, assignment is largely spatial (`trk_gate_hold` is deliberately low) and a fresh embedding adds nothing. Embeddings are genuinely needed only when binding, crossing, or re-acquiring after a miss. Skipping recognition on steady frames would cut that 169 ms to near zero — a further ~1.6× on top of this pack, and it compounds with it.

**I did not ship it, because its failure mode is the worst one you have.** It requires pre-matching a detection to a track *before* the tracker runs, on spatial evidence alone, and handing the tracker a cached embedding. If that pre-match is wrong — two people crossing is exactly when it would be — the tracker confirms the **wrong identity** with high confidence. That is the same defect family as every bug in this project's history: *a check reading state that does not mean what it assumes.* It needs the conservative gate (locked, established, high IoU, unambiguous in both directions, not crossing) and real two-person footage to validate against, which I do not have.

Run this pack first. The `embeds per raw face` figure will say whether §8 is worth the risk for your material.

Also still open: the five v11.2.52 tracker edits (SDOS-071 §6) and the v11.2.7 edge-revert item #21.

---

## 9. Verification

| Test | Result |
|---|---|
| Staged `_fa_get` vs stock `FaceAnalysis.get()` — bbox, kps, landmark_2d_106, embedding, det_score, gender, age | **bit-equivalent** |
| Deferred embeddings after `_scale_faces` + `_shift_faces` coordinate mutation | **bit-identical, max abs diff 0.000e+00** |
| Landmarks + gender present on deferred faces (gates stay informed) | yes |
| Dedupe without embeddings vs with | same 6 of same raw detections |
| Stash cleared after embedding (no frame pinned) | yes |
| Counters prove the staged path ran, not the fallback | `passes=27 raw_faces=162 embeds=76` → **0.47 embeds per raw face** |
| `None` image | `[]` |
| `DET_DEFER_EMBEDDING=False` | embeddings computed eagerly, as before |
| Staged-path exception | falls back to `FaceAnalysis.get`, warns once |
| `py_compile` all modules | clean |

Tests ran against the real downloaded `buffalo_l` models, with the functions **extracted from the shipped `core_pipeline.py`** rather than reimplemented.

Honest limit on this: I verified the detection subsystem against live models, not a full video render — this environment has no gradio and no source footage. The integration argument is that the tracker's inputs are provably unchanged.

---

## 10. Changed files

| File | Change |
|---|---|
| `core_pipeline.py` | `_fa_get` staged with `defer_embedding`; `max_num` default 20 → 0; `+ _embed_faces`; `+ _DET_COUNTERS`; `_detect_want` defers both passes; hi-res probe and `_zoom_detect` defer; `_pair_faces` embeds + re-dedupes before `tracker.assign`; one-time fallback warning; summary line reports passes/raw_faces/embeds |
| `config.py` | `VERSION` → `v11.2.73`; `BUILD` → `LazyID · recognition deferred to the faces that survive`; `+ DET_DEFER_EMBEDDING` (env `PHOENIX_DEFER_EMB`) |
| `swap_engine.py` | `ENGINE_VERSION` → `aequus-1.2.73-lazyid` |
| `app.py`, `phoenix_api_adapter.py`, `requirements.txt`, `packages.txt`, `README.md` | unchanged — shipped for set integrity |

**Startup log must read:** `aequus-1.2.73-lazyid` and `v11.2.73`.

**Rollback:** `DET_DEFER_EMBEDDING = False` in `config.py`, or `PHOENIX_DEFER_EMB=0` in the Space environment. No code edit. The `max_num` fix (§4) and the instrumentation stay either way, and both are strict improvements.
