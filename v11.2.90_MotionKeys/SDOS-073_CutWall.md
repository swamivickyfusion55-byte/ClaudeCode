# SDOS-073 — Cut Wall
### Why the pasted face holds on frame changes

**Report:** *"output is good, however there is some lag or some visible pasting confusion, maybe the pasted face holds — ON FRAME CHANGES"*
**Baseline:** `v11.2.73` / engine `aequus-1.2.73-lazyid`
**Delivered:** `v11.2.74` / engine `aequus-1.2.74-cutwall`

---

## 1. Diagnosis: nothing in the build knew what a scene change was

Grepped the whole codebase for any notion of a shot boundary:

```
scene_?cut | shot_?cut | scene_?change | shot_?change | hard_?cut | cut_detect
   -> no matches
```

There was no shot-cut detection of any kind. Three separate mechanisms were therefore free to carry the previous shot into the next one, and **all three are the same root cause** — no barrier at the cut.

### 1.1 The geometry timeline interpolates straight through a cut

`_geom_for_frame` brackets a real detection before a frame with one after it and lerps between them, bounded only by `max_bracket_frames = max(taper+1, round(out_fps * 0.55))` — about **17 frames at 30 fps**. A cut almost always has detections within 17 frames on both sides, so the face was interpolated from where the old shot left it to where the new shot found it.

Simulated with detections at frame 26 (x=120) and the cut at 30 (x=500):

```
frame    v11.2.73 x    v11.2.74 x
   26         120.0         120.0
   27         215.0         120.0
   28         310.0         120.0
   29         405.0         120.0
   30         500.0         500.0
```

**The face walks 380 px across the screen over the four frames before the cut.** That is the "lag" — and note it happens *before* the cut, because the detection at the new shot acts as a future anchor pulling the geometry forward.

### 1.2 `_nearest_aligned` had no bound of any kind

```python
def _nearest_aligned(slot, g):
    best, bd = None, None
    for gi, fake, corr in hist:
        d = abs(gi - g)                 # absolute - searches BOTH directions
        if bd is None or d < bd:
            best, bd = (fake, corr), d
    return best
```

No temporal cap and no direction limit. Two consequences:

- **It reaches across a cut in either direction.** Modelled on the real key-frame grid (`skip_n=5` plus the forced key at the cut), the two frames *before* the cut were painted with the **next** shot's face:

```
frame  shot   v11.2.73        v11.2.74
   28     A       B@30            A@25      <-- WRONG SHOT
   29     A       B@30            A@25      <-- WRONG SHOT
   30     B       B@30            B@30
```

- **One crop could ride for a very long time.** Combined with the chunk-boundary trim that keeps a single entry (`_aligned_hist[slot] = (...)[-1:]`), a single aligned crop could be stamped onto a second or more of output with nothing to stop it.

### 1.3 The tracker carried the old shot's head for up to 24 detections

`trk_max_missed = 24` is counted in *detections*, not frames. At `skip_n = 5` that is up to **120 frames — four seconds** of predicted geometry extrapolated from a shot that no longer exists. And `track.fake` — the cached aligned crop — was never invalidated, so `reuse()` kept pasting the old shot's face, with the old shot's lighting, colour match and grain.

---

## 2. The fix: a cut is a barrier

Geometry, cached pixels and tracker state all stop at it.

### 2.1 Detecting the cut — precision over recall, deliberately

**A missed cut leaves exactly the behaviour this build already had. A false cut resets live tracks and can put the original face back for a few frames** — the defect family this engine has been fixed for repeatedly. So the test requires **two independent signals to agree**:

- **`mad`** — mean absolute difference of a 64 px greyscale, each frame first normalised to zero mean and unit variance. The normalisation makes a flash or an exposure ramp cancel instead of reading as a cut, and makes the threshold scene-independent so it can be absolute rather than relative to a running median.
- **`corr`** — correlation of a coarse 8×8×8 BGR histogram between consecutive frames. A pan keeps nearly all its content, so `corr` stays ~0.99; a cut to different material collapses it.

Calibrated on synthetic sequences, **12/12 correct**:

| Sequence | mad | corr | Verdict |
|---|---|---|---|
| static | 0.02 | 1.000 | no cut |
| fast pan | 0.93 | 0.997 | no cut |
| **whip pan (px=40)** | **1.16** | **0.999** | **no cut** — rejected by corr |
| **whip pan (px=70)** | **1.15** | **0.999** | **no cut** — rejected by corr |
| zoom | 0.56 | 0.998 | no cut |
| **flash / explosion** | **0.22** | **−0.008** | **no cut** — rejected by mad |
| **slow dissolve** | **0.18** | **0.747** | **no cut** — rejected by mad |
| hard cut | 1.13 | −0.007 | **CUT** |
| cut between two pans | 1.16 | −0.007 | **CUT** |
| cut pan → static | 1.12 | 0.333 | **CUT** |
| cut whip → whip | 1.12 | −0.009 | **CUT** |
| two cuts | 1.13 | −0.009 | **both CUT** |

Thresholds `mad > 1.00` **and** `corr < 0.70`.

The whip pans are the point of the table: they sit **above** the mad threshold and are rejected purely by correlation, while the flash sits **below** the corr threshold and is rejected purely by mad. **Neither signal alone is safe — the `AND` is load-bearing.**

I first tried a *relative* mad test (spike against a running median). It false-positived on fast pans and, worse, **missed cuts that happen during a pan** — the pan's own baseline hid the cut. The normalisation is what made absolute thresholds possible.

**Known limit:** a cut between two shots with near-identical colour palettes keeps `corr` high (measured 0.999) and is **missed**. That is the status quo, not a regression.

### 2.2 The barrier is applied where history is READ, not where it is written

My first implementation cleared `_geom_hist` and `_aligned_hist` at the cut, inside the detection pass. **That was a bug, and I caught it before shipping.** The detection pass runs a whole chunk ahead of emission, so destroying pre-cut history at that moment strips the geometry and crops that the frames *before* the cut still have to be rendered with — they would have emitted the original face. The same mistake made `_last_cut_g` wrong: it holds the chunk's *latest* cut, which is in the future for most frames being rendered.

So the barrier is two lookups over a job-level sorted list of cuts:

```python
def _shot_lo(g):   # first frame of the shot containing g
def _shot_hi(g):   # first frame of the NEXT shot after g
```

and history is **filtered at read time**, which is correct for every frame regardless of which side of the cut it sits on:

```python
_lo, _hi = _shot_lo(g), _shot_hi(g)
_tl = [(gg, rr) for (gg, rr) in _geom_hist[slot] if _lo <= gg < _hi]
```

Verified: frames before a cut keep their pre-cut history; frames after it see only post-cut history; **no frame is left with no crop at all.**

### 2.3 The cut is forced to be a detector key frame

Detection otherwise runs on the `skip_n` grid, so the first frames of a new shot would be painted from the old shot's geometry before anything looked at them. Forcing a key frame at the cut means the new shot's timeline starts exactly at the cut — which is why the barrier starves no frames (verified above).

### 2.4 Tracker state: geometry forgotten, identity kept

`TrackState.reset_for_cut()` clears position, velocity, alpha-beta filter state, the mask/hull/colour/grain/blur EMAs, and the cached aligned crop. It deliberately **keeps `id_lock`, `emb` and `slot_gender`**:

> The person we were told to swap is still the person we were told to swap; we have only lost track of where they are. Clearing identity here would hand a virgin slot to whoever the detector happens to see first in the new shot — which is the "wrong person bound to the slot" defect this engine has been fixed for twice.

### 2.5 The last leak: `track.fake` fallback

`_reuse_one` falls back to whatever `track.fake` still holds when no cached crop is found, and emission walks forward — so on the first frame of a new shot that fallback is the **last frame of the old one**. Nulled explicitly in the emission loop at each cut, along with any in-progress fade, which belongs to the old shot too.

---

## 3. Result

| | v11.2.73 | v11.2.74 |
|---|---|---|
| Frames painted with the wrong shot's face (around one cut) | **2** | **0** |
| Frames left with no crop by the barrier | — | **0** |
| Geometry travel across a cut | **380 px over 4 frames** | none — holds, then snaps |
| Cached crop maximum age | **unbounded** | 48 frames (~1.6 s) |
| Tracker carry-through past a cut | up to 24 detections (~120 frames) | ends at the cut |

## 4. Cost

| | per frame | over a 451-frame clip |
|---|---|---|
| 720p | 1.24 ms | 0.56 s |
| 1080p | 2.37 ms | **1.07 s** |

About **one second** on your whole clip, and it runs once per frame regardless of face count. Against the ~120–270 s the v11.2.73 detector work now takes, it is under 1%.

## 5. Dials

```python
CUT_DETECT            = True    # False restores the old behaviour exactly
CUT_MAD_MIN           = 1.00    # raise to make it more reluctant
CUT_HIST_CORR_MAX     = 0.70    # lower to make it more reluctant
ALIGNED_MAX_AGE_FRAMES = 48     # cached-crop age cap, independent of cuts
```

Env overrides: `PHOENIX_CUT_DETECT`, `PHOENIX_CUT_MAD`, `PHOENIX_CUT_CORR`, `PHOENIX_ALIGNED_MAX_AGE`.

The summary log now reports `shot_cuts=<n>`. **If that reads 0 on footage you know has cuts**, the palette-similarity limit in §2.1 is the likely reason — tell me and I will add a structural-similarity third signal.

## 6. If the symptom persists

The report mentioned "lag" as well as holding. If, after this, the face still *trails the head during continuous motion* (no cut involved), that is a different defect: the geometry interpolation lags because `_geom_for_frame` is bounded by `max_bracket_frames` and falls into a frozen branch during fast movement. A prior note in the code records that 61% of frames on a rapid-motion clip land in that branch, and that tightening the detection cadence did **not** help — the anchors are sparse because detections are being *rejected* downstream while the subject moves, not because they are scheduled too rarely. That is a worthwhile next pack, and it is separate from this one.

Still open: the five v11.2.52 tracker edits (SDOS-071 §6), the §8 lazy-recognition lever (SDOS-072), and the v11.2.7 edge-revert item #21.

## 7. Verification

| Test | Result |
|---|---|
| Cut detector: static / pan / whip pan ×2 / zoom / flash / dissolve | **0 false positives** |
| Cut detector: hard cut / cut in pan / pan→static / whip→whip / two cuts | **all detected** |
| Same-palette cut | missed, as documented (status quo) |
| Shot bounds `_shot_lo` / `_shot_hi` across two cuts | correct at every boundary incl. the cut frame itself |
| Pre-cut frames keep pre-cut history | yes |
| Post-cut frames see only post-cut history | yes |
| Barrier starves no frame of a crop | 0 frames |
| Crop never crosses a cut in either direction | verified |
| Age cap rejects a 60-frame-old crop | verified |
| `TrackState.reset_for_cut` clears geometry + crop + EMAs | verified |
| `reset_for_cut` **keeps** `id_lock` and `emb` | verified |
| `MultiFaceTracker.mark_cut` resets all slots | verified |
| Chunk-boundary cut seen (signature carried) | by construction, `_cut_prev_sig` |
| **Regression:** lazy-ID embeddings still bit-identical | `0.000e+00` |
| **Regression:** grain matching | 95% → 8% mismatch, unchanged |
| **Regression:** 90-frame soak, all passes | no crash, EMA jump 0.011 |
| **Regression:** affine maths, 448 trap | all pass |
| `py_compile` all modules | clean |

Honest limit: calibrated and verified on synthetic sequences and extracted pipeline logic, not on your footage — this environment has no source video. The cut thresholds are the part most worth checking against real material, and `shot_cuts=<n>` in the log is how you check them.

## 8. Changed files

| File | Change |
|---|---|
| `core_pipeline.py` | `+ _cut_signature`, `_shot_cuts`, `_CUT_STATS`; `_cut_prev_sig` / `_last_cut_g` / `_cut_list` / `_shot_lo` / `_shot_hi`; cuts forced to key frames; cut barrier in the detection pass; `_records_at` clips the timeline to the shot; `_nearest_aligned` shot-scoped + age-capped; `track.fake` and fade nulled at cuts in emission; `shot_cuts` in the summary |
| `swap_engine.py` | `ENGINE_VERSION` → `aequus-1.2.74-cutwall`; `+ TrackState.reset_for_cut`; `+ MultiFaceTracker.mark_cut` |
| `config.py` | `VERSION` → `v11.2.74`; `BUILD` → `CutWall · a shot change is a barrier; nothing crosses it`; `+ CUT_DETECT`, `CUT_MAD_MIN`, `CUT_HIST_CORR_MAX`, `ALIGNED_MAX_AGE_FRAMES` |
| `app.py`, `phoenix_api_adapter.py`, `requirements.txt`, `packages.txt`, `README.md` | unchanged — shipped for set integrity |

**Startup log must read:** `aequus-1.2.74-cutwall` and `v11.2.74`.

**Rollback:** `CUT_DETECT = False`. The `ALIGNED_MAX_AGE_FRAMES` cap is independent and should be kept regardless — an unbounded crop age is a defect with or without cuts.
