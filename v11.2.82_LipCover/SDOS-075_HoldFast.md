# SDOS-075 — Hold Fast
### Why it reverts to the original, and why it ghosts

**Report:** *"still reverting to original intermittently kept happening and ghosting."*
**Baseline:** `v11.2.75` / engine `aequus-1.2.75-seefood`
**Delivered:** `v11.2.76` / engine `aequus-1.2.76-holdfast`

---

## 1. First: it is not my last two packs

Both symptoms could plausibly have come from mechanisms I added in v11.2.74 and v11.2.75, so I tested those before anything else.

**The occluder gate (v11.2.75)** punches holes in the mask, and a false hole shows the original face through it. Tested against seven hazards my original battery never had — all of them the subject's own face:

| Hazard | Cut |
|---|---|
| closed mouth 24 frames → **wide smile showing teeth** | 0.0% |
| teeth + tongue | 0.0% |
| strong specular highlight appearing | 0.0% |
| lock of hair falling across the cheek | 0.0% |
| dark lipstick applied mid-shot | 0.0% |
| sudden motion blur | 0.0% |
| talking — teeth on/off every 2 frames | 0.0% |

Clean. It is not the cause. (It is now instrumented anyway — see §5 — so it can be ruled out from your log rather than from my testing.)

---

## 2. The actual cause: the evidence was weighed backwards

There is exactly **one** path in the whole pipeline that emits a frame with no face on it:

```python
records = _records_at(g)
if not records and not _active_fade:
    writer_ok = _safe_result_put(frm)      # the original frame, untouched
```

So a revert means `_records_at` returned nothing for every slot, which happens when no **real** detection lies within `taper` frames. And `taper` is not constant:

```python
eff_taper = _cadence_taper if _blind else taper
```

- normal: `REACQUIRE_GRACE_SEC` × fps = **60 frames (2.0 s)**
- inside a "blind span": `_cadence_taper` = **3–12 frames (0.1–0.4 s)**

A blind span forms from a run of marks saying "nothing paintable here". **Two different readings set those marks, and they were weighted in exactly the wrong order:**

| Reading | What it actually means | Frames needed to collapse the grace window |
|---|---|---|
| detector returned **nothing** | weak evidence the face may have gone | `BLIND_AFTER_SEC` → **22.5** |
| detector returned a face, every candidate **rejected** downstream | **positive evidence the face is there**, only that this read wasn't trusted | `_ACTIVE_REJECT_STREAK = 2` **detector calls** |

Two calls. At the default cadence that is **10 output frames**; at `det_n=1` it is **2 frames**.

```
skip_n=1:  2 rejected calls =  2 output frames -> grace window drops 60 -> 3
skip_n=3:  2 rejected calls =  6 output frames -> grace window drops 60 -> 6
skip_n=5:  2 rejected calls = 10 output frames -> grace window drops 60 -> 10
```

**The reading that proves the subject is present collapsed the grace window up to ten times faster than the reading that suggests they are absent.**

Once demoted, the held face fades within a few frames and the original shows through — until the next accepted detection brings it back. That is the intermittent revert, exactly. And the fading, extrapolated face on the way down is the ghost. **Both symptoms are one mechanism.**

### The fix

The threshold is now a **duration in output frames**, cadence-independent, and no quicker than the absence path:

```python
REJECT_BLIND_SEC = 0.75      # same as BLIND_AFTER_SEC
_reject_blind = max(2.0, out_fps * REJECT_BLIND_SEC)

def _run_is_blind(run):
    if len(run) < 2:
        return False
    return (float(run[-1]) - float(run[0])) >= _reject_blind
```

A rejected run still demotes — a face that can never be confirmed must not be painted forever — but only once it has persisted as long as an empty detector would have to.

| Mark pattern (per detector call, skip_n=5) | v11.2.75 blind frames | v11.2.76 |
|---|---|---|
| steady, 2 hard calls mid-shot | 10 | **0** |
| steady, 3 hard calls | 15 | **0** |
| alternating hard pairs (flicker) | 30 | **0** |
| sustained rejection, 6 calls | 30 | 30 |
| subject genuinely leaves (empty) | 38 | 38 |
| subject leaves while being rejected | 60 | 60 |
| brief empty blip | 0 | 0 |

The three cases that must still demote all do.

### A bug I introduced doing this, and caught

My first version applied the duration test to every run. That silently broke **"subject genuinely leaves"** — it stopped forming a span at all, which is the "ghost carried 411 frames past her exit" defect this code was written to prevent.

The reason: the absence path pushes a single **constant** synthetic marker (`quiet[0] + BLIND_AFTER`) each time it fires, so every entry in that run has the same value and a duration test on it measures zero forever. Absence has already applied its own threshold, so it now qualifies on its own:

```python
if run and (run_absent or _run_is_blind(run)):
```

Caught by the table above, not by reading.

---

## 3. The ghost, and the tension the fix creates

Holding the face longer is only an improvement if it is held in the **right place**. During a gap, placement came from the most recent timeline entry — which is a naive constant-velocity extrapolation:

```
Last real detection: frame 100 at x=400. Tracker extrapolates 12 px/frame.

 frame   v11.2.75 x   v11.2.76 x   alpha
   105          460          460    0.99
   110          520          520    0.97
   115          580          400    0.94
   130          760          400    0.75
   145          940          400    0.44
```

In v11.2.75 the face slides **540 px away from the head it belongs to** before finally fading out. That is the ghost — and my §2 fix, on its own, would have made it last *longer*.

So the extrapolation is now bounded. It is a good estimate for a few frames and an increasingly wrong one after that, so past `hold_after` (the cadence window) placement falls back to the **last confirmed position** and stays there while the alpha continues to fade:

```python
if hold_after is not None and real_dist > float(hold_after):
    side = obs_lo          # the last REAL sighting, not the latest guess
```

A face that holds still in the right place reads as a brief freeze. One that slides off the head reads as a ghost.

---

## 4. Fail-safe on my own cut barrier

A false shot-cut resets the tracker and clips the geometry timeline — which produces the same revert. The barrier now checks its own plausibility:

```
 3 cuts in 451 frames (1 per 150.3) -> allowed
12 cuts in 451 frames (1 per  37.6) -> allowed
38 cuts in 451 frames (1 per  11.9) -> DISABLES barrier
```

Past 1 cut per 12 frames sustained, the detector has lost its mind on this material; the barrier switches itself off for the rest of the job and says so in the log, rather than quietly shredding the timeline.

---

## 5. Instrumentation — so the next round is measured

Every previous round of this defect was tuned blind, including some of mine. The summary line now attributes it:

```
REVERTS=<n> blind=<n> faint=<n> vis_rejected_calls=<n> occ_cut=<n>/<n>
```

| Field | Meaning | How to read it |
|---|---|---|
| `REVERTS` | frames emitted with **no face at all** | the symptom, counted directly |
| `blind` | frames rendered inside a blind span (short taper) | if this is ≫ 0, §2 is still firing — raise `REJECT_BLIND_SEC` |
| `vis_rejected_calls` | detector calls where **every** candidate was rejected | if high, the content/identity gate is the root cause, not the taper |
| `faint` | frames painted below 0.5 opacity | ghost-prone frames |
| `occ_cut` | frames where the occluder gate cut anything | if ≈ total, the v11.2.75 gate is misfiring — raise `occ_chroma`/`occ_luma` |

If `REVERTS` is near zero but you still see the original, the cause is opacity (`faint`), not geometry — a different fix, and the number tells us which.

---

## 6. Verification

| Test | Result |
|---|---|
| Occluder gate vs 7 real-face hazards (teeth, tongue, specular, hair, lipstick, blur, talking) | **0 false cuts** |
| Blind-span formation, 7 mark patterns | transient runs hold; absence and sustained rejection still demote |
| Absence path after the duration change | regression found and fixed |
| Extrapolation drift bound | 540 px → 0 px past the cadence window |
| Cut-density fail-safe | trips at 1-per-12, allows 1-per-37 |
| **Regression:** grain matching | 95% → 7% mismatch |
| **Regression:** 90-frame soak, all passes | clean, EMA jump 0.011 |
| **Regression:** affine incl. 448 trap | 16/16 |
| **Regression:** cut detector | 12/12 |
| **Regression:** cut barrier | wrong-shot frames 2 → 0, 0 starved |
| **Regression:** occluder battery | ice cream cut, beard/mouth/clean kept |
| `py_compile` all modules | clean |

**Honest limit, and it matters here.** I verified this on extracted pipeline logic and synthetic sequences; I have no footage, so I cannot prove the fix removes *your* reverts. What I can say precisely: the mechanism in §2 demotes the grace window after as little as 2 output frames of rejected-but-present detections, that is a logic inversion rather than a tuning choice, and it produces exactly the symptom you describe. If reverts persist, the new log line distinguishes the remaining possibilities in one run instead of another round of guessing.

---

## 7. Dials

```python
REJECT_BLIND_SEC = 0.75   # raise to ride through longer rejection runs
REACQUIRE_GRACE_SEC = 2.0 # the long grace window itself
BLIND_AFTER_SEC = 0.75    # how long an EMPTY detector may run before demoting
```

**Rollback:** `REJECT_BLIND_SEC = 0.0` restores the old hair-trigger. The drift bound reverts by passing `hold_after=None` at the `_geom_for_frame` call site.

## 8. Changed files

| File | Change |
|---|---|
| `core_pipeline.py` | `_ACTIVE_REJECT_STREAK` (call count) → `_run_is_blind` (duration); `run_absent` so absence keeps its own threshold; `_geom_for_frame` gains `hold_after` and falls back to the last confirmed position; cut-density fail-safe; 5 attribution counters and their reporting |
| `swap_engine.py` | `ENGINE_VERSION` → `aequus-1.2.76-holdfast`; `+ OCC_STATS` |
| `config.py` | `VERSION` → `v11.2.76`; `BUILD` → `HoldFast · a rejected read is not an absent face`; `+ REJECT_BLIND_SEC` |

**Startup log must read:** `aequus-1.2.76-holdfast` and `v11.2.76`.

Still open: the five v11.2.52 tracker edits (SDOS-071 §6), the lazy-recognition lever (SDOS-072 §8), the motion-lag item (SDOS-073 §6), and the v11.2.7 edge-revert item #21.
