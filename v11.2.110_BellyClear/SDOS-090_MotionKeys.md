# SDOS-090 — MotionKeys: evaluation of SDOS-089 and what was built

**Pack:** v11.2.90 "MotionKeys" (v11.2.89 Steady + one change to the swap scheduler + diagnostics). Engine `aequus-1.2.90-motionkeys`.
**Request:** review the attached SDOS-089 (Professional-Grade Face Quality, Pose Robustness and Temporal Consistency), keep only what actually helps, evaluate before implementing.

I have not seen your footage (I only work with synthetic media and your logs). Every number below is a synthetic clip made from real photographs, through the real detector, swapper and compositor.

---

## 1. Verdict on each SDOS-089 proposal

| ID | Proposal | Verdict | Why (measured or read in the code) |
|---|---|---|---|
| F-02 | Adaptive swap cadence from face-local motion | **Built** (§2) | The one proposal that matches a measured defect: Best/Balanced step on fast motion (×1.35–1.38 flicker against ×1.06 at Ultra, v11.2.88/89). The SDOS's reading of the code is right in outline, but the real mechanism is more specific (§2.1) |
| F-04 | Low-light path: analysis-only enhanced frame for detection | **Rejected, measured** (§3) | The detector's recall stays 100 % down to a mean luma of 8; the enhanced copy did not improve landmarks and made them 10–20 % worse in dim and dark frames. Added a log counter instead, so real footage can decide |
| F-03, F-05 | `PoseQuality` record, calibrated pose estimator, bounded profile fallback, separate identity and render confidence | **Rejected as a rewrite; counters only** (§4) | The engine already has a graded pose confidence, hysteresis, a fail-closed rule and bounded holds, calibrated on 5940 poses. A 3-D pose model is a new dependency that cannot be validated without real profile footage, which I may not analyse |
| F-01 | Fixture manifest, `tests/` tree, benchmark runner | **Partly** | The scripts I use for every pack are in `verification/` and are extended here (`verification/motionkeys/`). A consent-tracked real-footage corpus is not something I can build |
| F-06 | Compositing checks (reused crops use current geometry, state reset on cuts, …) | **Already true** | `swap_frame` cuts the crop at the same interpolated geometry the renderer uses, cuts reset state, grain/blur/colour are capped (SDOS-072 onward). No change |
| F-07 | Structured errors, replace silent `except` | **Counters only** | A blanket rewrite of the hundreds of handlers in a 7,500-line file, without tests, is a bigger risk than the silent fallbacks. Added the two log lines of §5 |
| F-08 | UI: explain the presets | **Done, small** | The hint under "Swap every N" claimed multi-face jobs "always process every frame regardless": true for tracking, **not for the swap**. It now says what really happens (§2.1) |
| F-09 | Extract modules | **Rejected** | No tests to hold the behaviour, no gain in output |
| F-10 | API contract tests, privacy cleanup, CI | **Checked, nothing to change** | The uploaded face photos and the source video are deleted in the job's `finally` block (`core_pipeline.py`, end of `_run_job_body`); a sweeper deletes orphans after 24 h; outputs are kept 12 h by design |

## 2. F-02 built: swap on extra frames while a face moves or changes fast

### 2.1 What is really going on (and a correction of my SDOS-088)

- In a **2-face job** (your usual: `Face mode: 2`) the engine tracks and detects on every frame, and **how often the swap network runs is set by the Quality preset alone**: Ultra every frame, Best every 2nd, Balanced every 4th, Fast every 6th. The "Swap every N" setting is ignored there. The frames in between reuse the last swapped face.
- `_adaptive_swap_gap()` can shorten that interval when the whole frame changes (a 160×90 grey difference). A head turning in a locked-off shot changes little of the frame, so it rarely fires; and in **single-face jobs** the swap can only run on a key frame, which sit on the same grid, so shortening the gap changes nothing.
- On fast motion the reused face goes stale and **steps** to the new swap: measured on a fast clip (sweep ±180 px with blur, roll ±22°, squeeze to 35 % yaw), through the real pipeline, painted face compared in its own frame (head motion removed with the true key points):

| 2-face job, fast clip | change per frame, output / footage | worst 5 % of frames | largest step / median |
|---|---|---|---|
| Ultra | ×1.06 | 8.7 (footage 8.5) | 1.3 |
| Best (v11.2.89) | ×1.35 | 16.3 | 2.2 |
| Balanced (v11.2.89) | ×1.38 | **20.8** | 3.0 |

### 2.2 The change (`core_pipeline._FaceMotionKeys`, `adaptive_key` in `config.py`)

Every frame of a chunk is looked at, for each tracked face separately, using the geometry the renderer already has. A face is swapped on that frame as well when, since it was last swapped,
- its centre has moved more than **0.08 face widths** (`adaptive_key_move`), or
- its look has changed by more than **0.20** summed per frame (`adaptive_key_change`): its patch (cut at sub-pixel accuracy) is compared with the previous frame's patch at the previous frame's own position, so plain travel cancels and what is left is turning, rolling, blur and light.

Only measured faces count (not held or predicted geometry). Extra swaps are capped at 60 % of the frames (`adaptive_key_max_frac`). `adaptive_key = 0` restores v11.2.89 exactly. Ultra is untouched (it swaps every frame anyway).
Calibrated on the clips: sensor noise reads 0.002 per frame, a slow drift or a talking face 0.03–0.07, a fast turn 0.3–0.4.

### 2.3 Results (real pipeline, 2-face job)

| Clip, preset | | change per frame (out / footage) | worst 5 % of frames | largest step | swap calls | run time |
|---|---|---|---|---|---|---|
| fast mixed motion, Balanced | v11.2.89 | ×1.38 | 20.8 | 3.0 | 106 | 91 s |
| | **v11.2.90** | **×1.07** | **8.75** | **1.2** | 138 | 116 s |
| fast mixed motion, Best | v11.2.89 | ×1.35 | 16.3 | 2.2 | 114 | 101 s |
| | **v11.2.90** | **×1.06** | **8.72** | **1.3** | 138 | 115 s |
| fast mixed motion, Ultra (reference) | | ×1.06 | 8.73 | 1.3 | 140 | |
| fast sideways sweep with blur, Balanced | v11.2.89 | ×2.51 | 8.4 | 4.0 | 70 | |
| | **v11.2.90** | **×1.47** | **3.3** | 4.0 | 119 | 118 s |
| fast sideways sweep, Ultra (reference) | | ×1.65 | 3.6 | 2.5 | 120 | |
| calm clip (slow drift and roll), Balanced | both | same | | | 63 = 63 | 69 s = 69 s |
| two faces coming together to a kiss, Balanced | both | 0 extra swaps | | | 126 = 126 | 106 s vs 108 s |
| a scoop moving to the mouth, Balanced | both | 0 extra swaps | | | 116 = 116 | 97 s vs 100 s |

On these clips the whole clip is fast, so the extra cost is the worst case: **Balanced/Best end up at about Ultra's cost and Ultra's smoothness, and only for as long as the face moves fast**. Calm, talking and contact footage is not touched: the calm, kiss and food clips make no extra swap and render the same (kiss clip: identical to the pixel; the food clip differs by encoder-level noise that also differs between two runs of 11.2.89 itself, up to 66 levels).
Placement of the painted face against the true geometry is unchanged (median 0.173, worst 0.238 inter-eye distances on the mixed clip; 0.153 / 0.165 on the sweep) and so is sharpness.

### 2.4 Unit checks (`verification/motionkeys/unit90.py`, 13)

Still face asks for nothing; a slow drift nothing in the first 12 frames (the longest preset gap is 6); a fast sweep asks on every frame; turning in place asks by change of look only; steady 12 px/frame travel asks by travel only; minimum gap and the cap hold; held or predicted geometry never triggers; no records, a box outside the frame and a missing first swap do not raise; the old swap-gap function is unchanged.

## 3. F-04 measured: low light

Real-photo clip darkened (gain 0.30 / 0.12 / 0.06, signal-dependent noise) and a backlit variant (face ×0.12, bright background), the pipeline's own detector (SCRFD, 640 input, threshold 0.28), 40 frames each, against the true key points:

| | mean luma | recall as is | landmark error as is | with analysis copy (gamma + CLAHE on luma) |
|---|---|---|---|---|
| normal | 100+ | 100 % | 0.038 | |
| dim | 39 | 100 % | 0.036 | 100 %, 0.041 (worse) |
| dark | 16 | 100 % | 0.038 | 100 %, 0.043 (worse) |
| very dark | 8 | 100 % | 0.061 | 100 %, 0.071 (worse) |
| backlit face | 102 | 100 % | 0.046 | 100 %, 0.041 (slightly better) |

Nothing to recover, so nothing was built. Caveat: synthetic noise; real low light adds motion blur and compression. The job log now has a `LIGHT:` line (§5): if your dark clips show many dark faces and still look wrong, that tells me where to look.

## 4. F-03/F-05: why no pose-quality rewrite

`_pose_confidence` / `_kps_reliable` (`core_pipeline.py`) are already a graded score from landmark fit and pose proxies with two thresholds: 0.5 to start painting, 0.10 to keep painting a track that was painting a good pose a moment ago. That is the "separate whether this is a face from whether to keep painting it" the SDOS asks for; it fails closed on error; the thresholds were swept over profile turns to yaw 0.9 and every roll. Holds are bounded by the miss budget, `paste_frozen`, `_hold_fade` and the blind spans. Wrapping that in a dataclass renames it; replacing it needs a 3-D pose estimator and real profile footage to calibrate, neither of which I have. Without that data, any threshold change is a guess, and a wrong guess shows a wrong-angle paste.

## 5. Read your log (two new lines)

```
LIGHT: 12 of 4120 detected faces were dark (mean luma under 45 of 255); mean face luma 131
MOTION-KEYS: 214 extra swaps for fast motion (140 for travel, 74 for change of look) out of 3594 frames checked; peak change per frame 0.41
```
`MOTION-KEYS` with a small number = calm footage, nothing changed. A large number = a lot of fast motion; expect the job to take that share longer. If it climbs on footage you think is calm, send me the line and I will raise the limits.

## 6. What this does NOT do

- **Real side poses** are still untested (no real footage); this pack does nothing for a wrong angle that comes from the detector's landmarks or the swap network.
- **The extra swaps cost time.** On a clip that is fast from start to end Balanced costs about +30 % swaps (+27 % run time here); Best +21 % swaps (+14 %). To see the old speed set `adaptive_key` to 0.
- **Single-face jobs**: the same code runs, but I could not give it a separate end-to-end test (the bench builds 2-face jobs); it uses the same scheduler path and the unit checks.
- Ultra gains nothing (it already swaps every frame). The 12 h retention of outputs, API limits and error handling were reviewed, not changed.

## 7. Dials (`config.py` → `ENGINE_TUNABLES`)

`adaptive_key` (1; 0 = v11.2.89), `adaptive_key_move` 0.08, `adaptive_key_change` 0.20, `adaptive_key_min_gap` 1, `adaptive_key_max_frac` 0.60. Fewer extra swaps: raise the two limits, or lower `adaptive_key_max_frac` (0.2 allows extra swaps on at most a fifth of the frames).

## 8. Install

Upload the 8 files (`app.py config.py core_pipeline.py phoenix_api_adapter.py swap_engine.py packages.txt requirements.txt README.md`) to the **root** of the Space, commit, normal restart (or "Push to all Spaces" in the app). The startup log must show `Phoenix v11.2.90` and `swap_engine aequus-1.2.90-motionkeys`. Do not restart a Space while a job is running on it.

## 9. Changed files

`core_pipeline.py` (`_FaceMotionKeys`, the swap-selection loop of Phase 2, per-job reset, progress estimate, `LIGHT:` and `MOTION-KEYS:` log lines, dark-face counters), `swap_engine.py` (`ENGINE_VERSION`, five tunables), `config.py` (`VERSION`, `BUILD`, tunables), `app.py` (one hint text). Unchanged: `phoenix_api_adapter.py`, `packages.txt`, `requirements.txt`, `README.md`. Test scripts: `verification/motionkeys/` (not part of the Space).
