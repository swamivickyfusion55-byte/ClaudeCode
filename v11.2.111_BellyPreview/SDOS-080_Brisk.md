# SDOS-080 — Brisk: the AI models were running on every core of the host, not the 2 vCPUs the Space owns

**Pack:** v11.2.81 "Brisk" (includes everything in v11.2.80 Resume).
**Reported:** "Is the app optimised for speed? I am not happy with the time it takes. Check for non-value-added tasks or inefficiencies."

---

## 1. Short answer

No. There was one real inefficiency and it is fixed. The rest of the time is the swap network doing the work you asked for, and the only big lever there is a choice that belongs to you (§4).

**Where the time goes** (2-person Ultra job, 80 frames, measured on the real `inswapper_128` / `det_10g` under a 2-vCPU quota): about **three quarters is the swap network** (2 swap calls per frame, one per person), most of the rest is detection. Encoding, compositing, reading and writing are noise (0.3 s of 283 s).

**What was wasted — fixed:**

| Finding | Evidence | Fix |
|---|---|---|
| The tuning code that was meant to limit ONNX Runtime's threads **never ran**. `_ort_cpu_load_context()` swaps out `onnxruntime.InferenceSession`, but insightface builds its sessions from its own subclass of the *original* class, so the swap is never reached. Every model ran on ORT's stock default: **all cores the machine shows, with idle threads spin-waiting**. | Read back from the real sessions: `intra_op_num_threads = 0` before, `2` after | Options are now passed through insightface's own `sess_options` argument (`FaceAnalysis(...)` and `model_zoo.get_model(...)`) |
| A cpu-basic Space owns **2 vCPUs of CPU time** (a cgroup quota) but `os.cpu_count()` shows the host's cores. Every thread pool was sized from that. A CFS quota punishes that hard: the threads burn the quota in the first milliseconds of each 100 ms period and the whole container is frozen for the rest. | One swap call, 2-vCPU quota: **2 threads 1.4 s · 4 threads 2.1 s · 16 threads 3.3 s · 32 threads 5.3 s** (4-vCPU quota: 4 threads 0.9 s) | New `_cpu_allowance()` reads the affinity and the cgroup v1/v2 quota (own cgroup and every ancestor), rounds up, never raises. Model threads = that, capped at 4 as before |
| The startup line said `graph=EXTENDED · intra=N` whatever was really configured. | Hard-coded string | Replaced by a `CPU budget:` line read back from ORT itself (§5) |

**Measured gain** (same clip, same Space-like 2-vCPU quota, two runs each): **211 s → 154 s and 200 s → 158 s, i.e. 21–27 % less wall time.** The two versions' videos differ by the same amount two runs of *one* version differ by (mean 1.37 vs 1.28 grey levels per pixel; the grain is random), so this is not a visible change. My sandbox shows 4 cores; a host that shows more cores than that should gain *more*, because the 16- and 32-thread rows above are worse than the 4-thread row. I could not measure on a real Hugging Face host, so treat 21–27 % as the floor I observed, not a promise.

Quality is untouched: the same models and the same maths. Thread count changes only how the same arithmetic is scheduled.

---

## 2. Your own logs, for reference

| Job | Frames | Wall (s) | s / frame | Detector calls / frame |
|---|---:|---:|---:|---:|
| 11 Sep | 451 | 394 | 0.87 | 0.99 |
| 8 Oct | 1808 | 8182 | 4.53 | 1.55 |
| 8 Oct | 2022 | 5332 | 2.64 | 2.43 |
| 8 Oct | 2205 | 5882 | 2.67 | 2.46 |
| 9 Oct | 2250 | 11932 | 5.30 | 1.55 |
| 9 Oct | 2205 | 6233 | 2.83 | 2.46 |
| 10 Oct | 3225 | 9426 | 2.92 | 2.05 |

The same two-person clip costs 2.6–2.9 s/frame on 2 vCPUs in my sandbox, which is where most of your jobs sit; I cannot tell from a log why the 4.5–5.3 s/frame jobs were slower. Detection runs 1.5–2.5 times per frame because a missing person triggers extra search passes; those passes were doing real work in my tests (§3).

---

## 3. Checked and deliberately NOT changed

| Idea | Verdict |
|---|---|
| **Search back-off** (stop hunting for a missing person after N empty searches) | Built and tested: frame-identical on three scenes (80/80 frames, same painted-person timeline), but it saved only **0–6 % of detector calls and no measurable wall time** (283 vs 288 s, 286 vs 283 s, 276 vs 278 s), because the extra passes were finding the person (7 of 51 searches in one scene). A risk with no measured payoff. **Not shipped.** |
| **Skip identity recognition** on more frames | It is what keeps person 1 and person 2 from being swapped over. Not worth the risk. |
| **Detector on a fitted canvas** instead of 640×640 | Tried. A portrait test showed boxes that were *not* equivalent at frame edges. Reverted. |
| **ORT graph level / OpenVINO** | My first test suggested 20 % from the graph level. That baseline was wrong: stock ORT is already at its highest level. The real gain was the threads. |
| Lower detection resolution | Costs small and distant faces. Not a free win. |

---

## 4. The one big lever is yours: swap cadence, and splitting across Spaces

The swap network is about three quarters of the time and it runs once per person per swapped frame.

**Swap cadence** (`quality`), same 80-frame two-person clip at 10 fps, 2-vCPU quota, forced initial window of 1.5 s included:

| Quality | Swap calls | Wall time | vs Ultra |
|---|---:|---:|---:|
| Ultra (swap every frame) | 160 | 273 s | — |
| Best (swap every 2nd frame) | 94 | 173 s | **−37 %** |
| Balanced (swap every 4th) | 62 | 126 s | **−54 %** |

The saving grows with clip length: the first 1.5 s is always swapped on every frame, so a long clip approaches the full ratio (Best ≈ −50 %). Every output frame is still composited from the cached face crop with its own interpolated landmarks, so what you give up is expression *freshness* between swaps, not paste accuracy. **I have not looked at any real footage**, so I cannot tell you where that trade is visible on your clips; try Best on one clip you know well before changing your habit. The default stays as it was.

**Split across Spaces** is linear and needs no code: N parts on N Spaces finish in about 1/N of the time. The app already has the part-split and the SG1–SG20 list for exactly this.

**`cpu-upgrade` hardware** helps only if the pipeline is told to use the extra CPUs: set `PHOENIX_NATIVE_THREADS` (1–8) as a Space variable. The allowance code sizes the default automatically (capped at 4); I measured 4 vCPUs at 0.9 s per swap call but nothing above 4. Costs money; try one Space first.

---

## 5. How to check on the Space

After the restart the log must show:

```
Phoenix v11.2.81
CPU budget: allowed=2 of <N> visible cores · model threads=2 · ORT tuning on · swap intra=2 inter=1 · detect intra=2 inter=1 · load <s>
```

`allowed` is the Space's real CPU quota, `<N>` is what the host shows. If `allowed` equals `<N>` on a cpu-basic Space, tell me — the quota file was not readable and the old behaviour is in effect (no harm, no gain).

## 6. Kill switches

| Variable (Space → Settings → Variables) | Effect |
|---|---|
| `PHOENIX_ORT_TUNE=0` | Back to stock ORT session options (the v11.2.80 behaviour) |
| `PHOENIX_NATIVE_THREADS=N` | Force the model thread count (1–8) |

## 7. Verification

| Check | Result |
|---|---|
| `_cpu_allowance()` against emulated cgroup v1/v2, parent-cgroup quotas, unreadable/garbage files, affinity | 14/14 pass |
| Real inswapper_128 + det_10g sessions read back (`get_session_options()`) | intra 2 / inter 1 on every model |
| Same clip, v11.2.80 vs v11.2.81, 2-vCPU quota, 2 runs each | 211.5 → 153.8 s, 200.2 → 158.3 s; same-size, same frames, difference within run-to-run noise |
| Full jobs on three synthetic two-person scenes (second person absent, blinking, tiny) | 80/80 frames painted, identical painted-person timeline |
| Clean extract of the zip: every `.py` compiles, versions match, a full job runs and logs `CPU budget:` | pass |

**Not tested — please read:**
- **A real Hugging Face host.** My sandbox is a 4-core machine with a 2-vCPU cgroup quota, which is the same mechanism but not the same machine, ORT build or Python (3.11 here, 3.10 on the Space).
- **Real footage.** Everything above is synthetic media; I do not analyse your uploads.
- **cpu-upgrade beyond 4 vCPUs.**

## 8. Install / rollback

Upload the 8 files (`app.py config.py core_pipeline.py phoenix_api_adapter.py swap_engine.py packages.txt requirements.txt README.md`) to the **root** of the Space, commit, normal restart. (Or use "Push to all Spaces" in the app, 1.2.10.) `app.py`, `phoenix_api_adapter.py`, `swap_engine.py`, `packages.txt`, `requirements.txt`, `README.md` are byte-identical to v11.2.80. Rollback: `PHOENIX_ORT_TUNE=0`, or re-upload v11.2.80.

## 9. Changed files
`core_pipeline.py` (`_cpu_allowance`, `_ort_session_options`, options passed to `FaceAnalysis` and both `get_model` calls, truthful `CPU budget:` log), `config.py` (version, build). New: this document.
