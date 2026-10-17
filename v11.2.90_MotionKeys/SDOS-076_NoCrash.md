# SDOS-076 — No Crash
### My bug. The cut reset killed the job.

**Report:** 4 of 9 jobs failed with `float() argument must be a string or a real number, not 'NoneType'`
**Baseline:** `v11.2.76` / engine `aequus-1.2.76-holdfast`
**Delivered:** `v11.2.77` / engine `aequus-1.2.77-nocrash`

---

## 1. Do this first — you do not need to wait for the pack

In your Space, set the environment variable:

```
PHOENIX_CUT_DETECT = 0
```

and restart. That disables the shot-cut barrier, `reset_for_cut()` is never called, and the crash cannot happen. **Re-run the 4 failed jobs immediately.** You lose only the cut-barrier fix from v11.2.74; everything else is unaffected.

Then install v11.2.77 and remove the variable.

---

## 2. What happened

This is my defect, introduced in v11.2.74 (CutWall). Your log pins it exactly:

```
03:06:28  shot cuts in this chunk at frames [294]
03:08:03  JOB CRASH F7581D27
          core_pipeline.py:5697  _record_geometry(pairs or [], g)
          core_pipeline.py:4199  _render_alpha(f, tr)
          core_pipeline.py:3205  return float(track.smooth_alpha(target))
          swap_engine.py:1525    cur = float(self.alpha_ema)
          TypeError: float() argument must be ... not 'NoneType'
```

A shot cut was detected, and the job died ~95 seconds later on the first frames after it.

`reset_for_cut()` — which I wrote — nulls the track's state so the previous shot cannot bleed into the next one. It set:

```python
self.alpha_ema = None
```

But `alpha_ema` is **never None anywhere else in the class**:

```
__init__          self.alpha_ema = 1.0
line 1246         self.alpha_ema = 1.0
line 1312         self.alpha_ema = 1.0
line 1469         self.alpha_ema = 1.0
reset_for_cut     self.alpha_ema = None     <-- mine, the only one
```

So `smooth_alpha` was written as `cur = float(self.alpha_ema)` with no guard, because it never needed one. **I broke an invariant the rest of the class depended on.** The opacity EMA is not evidence about the old shot — it is smoothing state, and its correct post-cut value is "fully opaque", exactly as at construction.

It only hit 4 of 9 jobs because only those clips contained a detected shot cut.

### Why the review that should have caught it didn't

I nulled fourteen fields in one go and reasoned about them as a group — "forget the old shot" — instead of checking each one against its consumers. So I audited all of them properly this time, comparing every field `reset_for_cut` writes against its `__init__` default:

| Field | `__init__` | `reset_for_cut` | |
|---|---|---|---|
| `alpha_ema` | `1.0` | `None` | **BUG — fixed to 1.0** |
| `_reacquired` | `False` | `True` | intentional — it *is* a re-acquisition |
| `missed` | `0` | `max(missed, 1)` | intentional — treat as not currently seen |
| all other 11 fields | `None` / `0` / `False` | identical | correct |

`alpha_ema` was the only field where I invented a value the rest of the code could not handle.

---

## 3. The three fixes

**1. The bug itself.** `self.alpha_ema = 1.0`, matching `__init__`.

**2. A guard, so this class of mistake cannot cost a render again.** A smoothing state is never worth crashing a twenty-minute job over:

```python
cur = self.alpha_ema
if cur is None:
    cur = 1.0
    log.debug("alpha_ema was None; recovered to 1.0")
```

**3. The geometry bookkeeping no longer takes the job down with it.** One frame missing from the timeline is a brief hold — the engine is built for exactly that — while an exception loses everything rendered so far. It is caught, logged **loudly once** with a full traceback (so a real defect stays visible rather than being silently swallowed), counted, and reported as `geom_errors=<n>` in the summary.

---

## 4. Verification

Rather than test only the line that crashed, I exercised **every** `TrackState` method after a cut reset:

```
cache_fake ok   predict ok   predicted_face ok   reset_appearance ok
reset_for_cut ok   smooth_alpha ok   smooth_colour ok   smooth_hull ok
smooth_mask ok   update ok   flow_correct ok   ramp_occlusion ok
_commit_identity ok   predict x30 ok
full cycle: reset -> predict -> update -> smooth_alpha  ok

every TrackState method survives reset_for_cut()
```

And the reported sequence end to end — two identities tracked to frame 294, a cut, then the detections that killed the job:

```
before cut: alpha_ema per slot = [1.0, 1.0]
cut at 294 -> mark_cut() reset 2 tracks
after cut : alpha_ema per slot = [1.0, 1.0]
post-cut frames 294..339: all painted, no exception
identities preserved across the cut: True
```

| Regression | Result |
|---|---|
| grain matching | 95% → 7% mismatch |
| 90-frame soak, all passes | clean |
| affine incl. 448 trap | 16/16 |
| cut detector | 12/12 |
| ghost drift bound | 540 px → 0 px |
| `py_compile` all modules | clean |

---

## 5. Also fixed: the UI was lying about its version

Your screenshot header reads **v11.2.71 (Detect)** while the log reads **v11.2.76**. Three hardcoded strings in `app.py` had drifted several releases behind, so a bug report could not say which code was running. They now read `config.VERSION` directly:

```
UI will now show: v11.2.76 (HoldFast)   ->  v11.2.77 (NoCrash)
```

---

## 6. Changed files

| File | Change |
|---|---|
| `swap_engine.py` | `ENGINE_VERSION` → `aequus-1.2.77-nocrash`; `reset_for_cut` sets `alpha_ema = 1.0`; `smooth_alpha` guards against a non-float |
| `core_pipeline.py` | `_record_geometry` call wrapped so one frame cannot abort the job; `record_geom_errors` counter; `geom_errors` in the summary |
| `app.py` | header and footer read `config.VERSION` instead of hardcoded strings |
| `config.py` | `VERSION` → `v11.2.77`; `BUILD` → `NoCrash · the cut reset no longer kills the job` |

**Startup log must read:** `aequus-1.2.77-nocrash` and `v11.2.77`, and the page header should now match.

The `REVERTS=… blind=… faint=… vis_rejected_calls=… occ_cut=…` line from SDOS-075 is still what I need to see once a job completes — the crash prevented any job from reaching it.
