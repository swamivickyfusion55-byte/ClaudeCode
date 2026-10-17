# SDOS-078 — Re-ID
### A locked identity that could never be re-recognised

**Report:** *"The FACE was not swapped in this job, please investigate"* — the last of six jobs in a log from `v11.2.78`.
**Baseline:** `v11.2.78` / engine `aequus-1.2.78-twoup`
**Delivered:** `v11.2.79` / engine `aequus-1.2.79-reid`

---

## 0. What the log proves, and what it does not

I do not have your footage and I do not analyse real face-swap footage, so everything here comes from your log lines and from tests against the real tracker code. Where a conclusion is a mechanism I reproduced rather than something I saw happen on your video, it is labelled that way.

**Good news first.** The v11.2.78 fix for the two-person grace window worked. On the two jobs where both people were in frame, `REVERTS` and `blind` fell from 750 / 829 and 1674 / 1789 to **0 / 0** and **48 / 96**, and the per-person strip finally shows both people.

**Not established:** which mechanism caused the last job's result. §3 shows a proven defect that produces exactly "recognised only intermittently", but I cannot prove it fired on your video. The new log fields in §6 will say.

---

## 1. The six jobs

| job | slot order | Detect refs | painted | REVERTS | blind | per person (painted / recorded) |
|---|---|---|---|---|---|---|
| 58B6A787 | F, M | both | 100% | **0** | **0** | #1 1333/1333 · #2 1452/1452 |
| 98817E87 | M, F | both | 63% | 750 | 829 | **#2 only** 1267/1207 |
| 4DC51EAA | M, F | both | 66% | 750 | 829 | **#2 only** 1449/1390 |
| 528BD900 | F, M | both | 98% | 48 | 96 | #1 2081/2051 · #2 1913/1890 |
| 56B1C320 | M, F | both | 66% | 750 | 829 | **#2 only** 1449/1390 |
| **3AF99A1C** | M, F | **`[True, False]`** | 48% | 1674 | 1789 | #1 1338/1326 · #2 664/589 |

Three observations that do not depend on any theory:

1. **Jobs 4DC51EAA and 56B1C320 are the same clip** — every detector number is identical to the digit (`swap_calls=1390`, `kps_rejected=187`, `weak_kept=13`, `REVERTS=750`) — run with two *different* female photos. The female photo changed; the result did not. So the miss is upstream of the swap: detection and binding, not the replacement image.
2. **In all three `M, F` jobs the male slot was never recorded at all** (`recorded` has no key 0). Either he is absent from that part of the clip, or he was never bound. The log cannot say which; the new strip (§6) can.
3. **The last job is the only one where the Detect frame produced a reference for just one person.** The female had no reference to match against and was identified purely by elimination.

---

## 2. A bug in my own counters (certain)

`occ_cut` across the six jobs reads `326/2785 → 365/4052 → 404/5501 → 571/9495 → 610/10944 → 901/12946`. That is a running total, and so are `det_passes`, `raw_faces` and `embeds`. Five module-level counters I added in v11.2.73 and v11.2.75 were **never reset**, so after the first job every per-job ratio in the summary line was meaningless.

Worse, `_CUT_STATS["disabled"]` is a **process-wide latch**. The cut-density fail-safe sets it when the detector claims implausibly many cuts, and nothing cleared it, so one bad job would have silently switched the shot-cut barrier off for every later job until restart.

**Fix:** `_reset_job_counters()` runs at the start of every job and clears all five, including the latch. Verified: after a reset the latch is `False` and every counter is zero.

---

## 3. The identity wall (proven mechanism)

### 3.1 How a slot recognises a person

`MultiFaceTracker.assign()` compares each detection to the slot's **lock** — the embedding of the *first frame it ever bound*, at whatever pose and lighting that frame had. The Detect-frame reference is consulted only before a lock exists; once locked, `_identity()` ignores it. Later detections must then clear:

- **0.38** against the lock (**0.40** after a gap), or
- a spatial rescue: `sim >= 0.22` *and* sitting where the track expects the person.

After pairing, a **second** absolute floor discards any face under **0.32** as "not this slot's person".

### 3.2 It cannot recover

`_commit_identity` refuses to update the lock from anything below 0.38. So the lock can never move toward the appearance that is failing. And after `trk_max_missed` (24 *detections* — about 0.8 s at your every-frame cadence) the track is rebuilt with its geometry cleared but its lock kept, which removes the spatial rescue and leaves only the wall.

Measured on the real `assign()`, a person with **no reference** who leaves and re-enters at a different place, with her embedding's similarity to the stored lock set directly:

| measured similarity to lock | v11.2.78 re-bound |
|---|---|
| 0.60, 0.46 | 100% |
| 0.41 | 98–100% |
| 0.37 | 89–94% |
| **0.33** | **0%** |
| **0.29 and below** | **0%** |

and it stays 0% for any gap length (20, 60, 150 frames). **It is a cliff, not a slope, and nothing brings her back.**

### 3.3 A second finding inside it: the spatial rescue was dead code

Tracing one failing case frame by frame showed a tracked person at similarity 0.29 bound **once**, then rejected every frame after. The gate's spatial rescue admits faces down to 0.22 when they are where the track expects; the very next check (the 0.32 floor above) discarded them. So that rescue never worked for 0.22–0.32 — a continuously tracked person whose similarity dipped under 0.32 for a moment started accumulating misses and slid into the stricter gap rules.

### 3.4 The fix: relative admission

With two or more identified people, "is this above an absolute number?" is the wrong question. The right one is "**which person is this, relative to the others?**" — and that is easy, because different people sit near 0 against each other's locks.

A face the absolute test rejects is now admitted only if **all** hold:

| condition | default | why |
|---|---|---|
| similarity ≥ floor | **0.25** | never admit below this, however clear the margin |
| ahead of **every** other identified slot by ≥ margin | **0.15** | it is clearly *this* person, not another |
| not sitting on another track's position | — | `_belongs_to_other` |
| seen on consecutive detections at ~the same place | **2** | one frame of look-alike never binds |
| at least one *other* identified slot exists | — | with nothing to be relative to, keep the absolute rule |

The same test now replaces the blind 0.32 floor after pairing, and the Detect-frame **reference is kept as a second prototype** (`max(lock, ref)`) instead of being ignored once locked.

### 3.5 Result, same test, same measurements

| measured similarity to lock | v11.2.78 | **v11.2.79** |
|---|---|---|
| 0.37 | 89–94% | **98–100%** |
| 0.33 | **0%** | **98–100%** |
| 0.29 | **0%** | **97–99%** |
| 0.26 | 0% | **~70%** |
| 0.23 and below | 0% | 0% (the floor, by design) |

### 3.6 The cost, measured — not hidden

A wrong-person swap is worse than a missed frame, so I measured bystander false binds in the **worst** case I could construct: the female absent for 300 frames so her slot is wide open, a bystander standing where she was, and the bystander's similarity to each lock drawn from a distribution at several levels of recognition noise.

| bystander similarity to each lock | v11.2.78 | **v11.2.79** |
|---|---|---|
| mean 0.05, sd 0.12 | 0.14% of frames | 0.11% |
| mean 0.10, sd 0.12 | 0.37% | 0.43% |
| mean 0.12, sd 0.15 (heavy noise) | 1.83% | 2.43% |
| **sustained look-alike at 0.30** | **0%** | **43%** |

For ordinary noise the false-bind rate is at baseline. **The honest weakness is a bystander who genuinely resembles a target** (similarity ≥ 0.25–0.30 *and* far from the other person, for a sustained stretch): the old wall rejected them, this does not. That is real but rare in practice.

If it ever happens, tighten with no code change:

```python
"trk_rel_floor":   0.30,   # was 0.25
"trk_rel_margin":  0.18,   # was 0.15
"trk_rel_confirm": 3,      # was 2
```

which I measured as: look-alike case **43% → 4%**, ordinary-noise false binds unchanged, at the price of giving up recovery below ~0.32.

### 3.6.1 A mistake worth recording

My first version also made the post-pairing floor honour the *original* spatial rescue, to fix §3.3. The sweep showed false binds **identical across four different rescue settings** — which meant the extra binds were not coming from the new rule at all. They came from that change: the 0.32 floor had been the only thing stopping a bystander standing in a tracked person's spot at 0.22–0.32 from taking the slot. Removing it raised false binds from 0.4% to 2.0%. The final structure routes that band through the margin-and-confirmation test instead of either bypassing it or killing it blindly, and false binds returned to baseline.

Two further bugs in my own first draft, found by tracing rather than guessing: a pending-confirmation record was being cleared by *unrelated* faces (the male, evaluated first, reset the female's count on every call so it never reached 2), and a face hovering around the 0.32 line kept breaking its own "consecutive" chain until I carried the chain through successful binds.

---

## 4. The reference warning

`target refs from Detect frames: [True, False, False, False]` was the only trace that person 2 had no reference. It is now a warning at job start, and the finished-job message says:

> ℹ person 2 had no Detect-frame reference (matched by elimination) — capture everyone in the Detect frame for a firmer match

For the best result, capture **both** people in the Detect frame. A reference is the strongest evidence the tracker has.

---

## 5. What is *not* wrong

- **`ClientDisconnect` tracebacks** in your log are `upload_file` requests where the browser dropped mid-upload (phone sleeping, tab closed, network change). They are noisy but are not pipeline failures and no job was affected.
- **The v11.2.78 grace-window fix, picker and dedupe** all held: the picker kept 2 of 2 chosen people on real faces, close-contact pairs are no longer merged, and replaying your logged job through the real span function still gives `blind=2321`.
- **The occluder gate** (`occ_cut` once the cumulative counter is read as a delta: roughly 6–15% of composites) and the **cut barrier** (`shot_cuts=0` on every job here) are not implicated.

---

## 6. Diagnostics added

```
... ID-GATE {0: {'id_reject': 412, 'rel_rescued': 31, 'post_reject': 9, 'rel_waiting': 6,
                 'rel_no_margin': 188, 'rel_below_floor': 1894}, 1: {...}}
TIMELINE (2 s per char; . none, a=person 1, b=person 2, #=both): ##aaaaa##aaaaabb....aa#aaaaaaa
```

| counter | meaning |
|---|---|
| `id_reject` | faces the identity gate refused for this slot |
| `rel_below_floor` | refused because similarity < 0.25 — **if this dominates, the person genuinely does not resemble the stored lock** |
| `rel_no_margin` | refused because another person scored nearly as well |
| `rel_rescued` / `rel_rescued_post` | faces the new rule admitted |
| `post_reject` | refused after pairing |

The **TIMELINE** strip is the most useful single line: it shows *where in the video* each person was swapped, which aggregate coverage cannot. If you tell me "she is on screen from 0:40 to 1:10 and the strip shows `.` there", the `ID-GATE` numbers say which gate refused her.

---

## 7. What I need next

After one run on `v11.2.79`, please send:

1. the summary line **and** the `TIMELINE` line, and
2. one sentence: *at roughly what timestamps is the person visible but not swapped?*

If `rel_below_floor` dominates for her slot, her embedding really is far from the stored lock (a wrong lock, or a very different look), and the next step is rebuilding the lock rather than loosening the gate. If `rel_no_margin` dominates, the two people are being confused with each other.

I have still **not** applied the five v11.2.52 tracker edits (SDOS-071 §6). I now have a measured defect in the admission logic those edits did not touch, and stacking more tracker changes would make the next log unreadable.

---

## 8. Verification

| Test | Result |
|---|---|
| Real `assign()`: re-entry at similarity 0.33 / 0.29 | **0% → 98–100% / 97–99%** |
| Same, similarity 0.23 (below floor) | 0% (by design) |
| Bystander false binds, 3 noise regimes | at baseline (0.11 / 0.43 / 2.43% vs 0.14 / 0.37 / 1.83%) |
| Sustained look-alike at 0.30 | 43% — **documented limit**, tightening gives 4% |
| 4 rescue-strictness settings swept, recall vs false binds | defaults chosen from the table in §3 |
| Counters reset at job start; latch cleared | verified |
| Log line `Done in` | 35 placeholders = 35 arguments |
| Timeline strip on a synthetic job | matches the constructed pattern |
| Re-bind after a cut, 12 cases | 60/60 both people |
| Stable realistic embeddings down to reference similarity 0.28 | 90/90 both people |
| **Regression:** picker on 7 real faces | 2/2 chosen people kept |
| **Regression:** dedupe close contact / real duplicates | 5/5 kept; 3/3 removed |
| **Regression:** logged-job blind-span replay | `blind=2321` / 0 |
| **Regression:** lazy-ID embeddings | bit-identical, 0.000e+00 |
| **Regression:** grain / affine (448 trap) / cut detector / cut barrier | 95%→7% / 16/16 / 12/12 / 2→0 |
| **Regression:** TrackState after cut reset; ghost drift bound; occluder battery | all pass |
| `py_compile` all modules | clean |

**Honest limit.** The embeddings in these tests are synthetic, built to realistic geometry (a person at a set similarity from a stored lock, impostors drawn from stated distributions). Real distributions are messier, which is why the dials are exposed and why §7 asks for the diagnostic fields before any further tuning.

## 9. Changed files

| File | Change |
|---|---|
| `swap_engine.py` | `ENGINE_VERSION` → `aequus-1.2.79-reid`; `_rel_admit`; relative admission at the gate and in place of the post-pairing 0.32 floor; `_identity` keeps the reference as a prototype; pre-computed per-slot similarity table; per-reason `diag`; 3 new `_P` entries |
| `core_pipeline.py` | `_reset_job_counters` (incl. the cut latch) called at job start; `_timeline_strip`; per-frame mask buffer; missing-reference warning and message; `ID-GATE` and `TIMELINE` in the log |
| `config.py` | `VERSION` → `v11.2.79`; `BUILD` → `ReID · a locked identity can be re-recognised, relative to the others`; `trk_rel_floor` / `trk_rel_margin` / `trk_rel_confirm` |

**Startup log must read:** `aequus-1.2.79-reid` and `v11.2.79`.

**Rollback:** `trk_rel_floor: 1.0` makes the relative rule unreachable and restores the old admission behaviour (the counters fix and diagnostics are independent and should be kept).
