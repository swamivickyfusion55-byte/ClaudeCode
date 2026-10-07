# SDOS-077 — Two Up
### Why one of two people stays original for most of a video

**Report:** *"One face (male) was swapped, female face remained original for most of the video duration, and occasionally the replacement appeared."*
**Log analysed:** job `03377653-2` on `v11.2.77` — the one with the end-of-job summary.
**Delivered:** `v11.2.78` / engine `aequus-1.2.78-twoup`

---

## 0. What is proven and what is not

I do not have your footage. Everything below is tied to a number from your log or to a test against the real functions and real models. Where my reproduction is weaker than your real result, I say so.

**Proven:** one defect reproduces your logged `blind=2321` to the digit (§2). Three more defects are proven on real code, one of them on real face models (§3–5).

**Not proven:** that these four account for *all* of your reverts. My simulation of the main fix moves the weaker person from ~90% to 100% painted, while your log shows 39% of frames with no face at all. The gap means something else also contributes, and §7 lists exactly what I need to see next to find it.

---

## 1. The log

```
FACE PAINTED ON 1484/2465 FRAMES (60%); detector_calls=3098 (empty=23); swap_calls=1303;
det_faces_avg=1.17 max=3; kps_rejected=1572; shot_cuts=1;
REVERTS=963 blind=2321 faint=122 vis_rejected_calls=0 occ_cut=20/1522; geom_errors=0
```

| Field | Reading |
|---|---|
| `REVERTS=963` | 39% of frames emitted with **no face painted at all** |
| `blind=2321` | **94%** of frames rendered with the *short* grace window |
| `det_faces_avg=1.17, max=3` | almost never more than two faces in frame — so the "extra people" theory is **not** the main driver here |
| `occ_cut=20/1522`, `geom_errors=0`, `shot_cuts=1` | my v11.2.74–77 additions are clean on this job |
| `vis_rejected_calls=0` | exactly zero — which, in a two-person job, turned out to be a symptom (§2) |

`blind=2321` was the number that did not add up: 23 empty detector returns cannot normally blind 94% of a video.

---

## 2. The main defect: the blind span never closes in two-person mode

### Mechanism

The grace window for holding a face through a gap is not constant. `_records_at` uses

```python
eff_taper = _cadence_taper if _in_blind_span(g) else taper     # 3–12 frames vs 60
```

A "blind span" is built by `_blind_spans()` from a list of verdicts, one per detector call:

| mark | meaning |
|---|---|
| `True` | a face was found and painted |
| `False` | faces were found but none was usable |
| `None` | the detector returned nothing |

A run of `None` is **only ever closed by a `True` or `False` mark.** Those two marks are written at exactly one place — inside the *single-face* branch's content gate:

```
line 5092   else:                      <- the else of `if multi_face_safe:`
line 5525       if frm is not None and pairs:
line 5598           _vis_marks.append((g, bool(_kept)))
```

**In two-person mode that branch never runs.** The only mark it ever writes is `None`, for an empty return. So the `None` run is never closed, `quiet` accumulates every empty return in the job, and once two of them fall more than `BLIND_AFTER_SEC` (0.75 s) apart, one span opens and stays open to the end of the video.

### Reproduction against the real function

I lifted the span logic into a module-level function (`_compute_blind_spans`) so it can be tested directly instead of through a replica, then fed it your job's numbers: 23 empty returns, the first near frame 122.

```
A) marks as two-person mode writes them today (None only):
   spans: [(144, END)]
   blind frames: 2321 of 2465        <- your log said blind=2321

B) same job, a verdict written on every detector call:
   spans: []
   blind frames: 0 of 2465
```

122 + 22 = 144, and 2465 − 144 = 2321. **It matches to the digit.**

### Why it produces reverts

With the detector running on every frame (`detector_calls=3098` over 2465 frames), the short window at `skip_n=1` is **4 frames**. Any stretch longer than ~8 frames in which a person has no usable detection falls out of the timeline entirely, and `_records_at` returns nothing — the one and only path that emits an unpainted frame. With the long window (60 frames) the same stretch is simply held.

This defect is **not** from my recent packs. It is in the v11.2.71 lineage you uploaded. My v11.2.75 attribution counters are what made it visible.

### The fix

After each detection call in two-person mode, write the verdict the single-face path already writes, derived from the per-person `_seen_ok` flags the code already computes:

```python
if multi_face_safe and not (_vis_marks and _vis_marks[-1][0] == g):
    _any_seen = any(_seen_ok.values())
    _vis_marks.append((g, True if _any_seen else False))
```

`True` when any person is bound to a real detection, `False` when faces were found but none was usable — the same meaning as in single-face mode. A genuinely absent subject still demotes (the `None` path is unchanged), and a long rejection run still demotes at `REJECT_BLIND_SEC`.

**Effect, with the caveat in §0.** Replaying a two-person job through the real `_geom_for_frame` and `_compute_blind_spans` (detector every frame, one person rejected in runs):

| | blind frames | male painted | weaker person painted |
|---|---|---|---|
| v11.2.77 | 2321 / 2465 | 98.9% | **89.9%** |
| v11.2.78 | 0 / 2465 | 100% | **100%** |

That is milder than your 39% reverts. My synthetic rejections are shorter and less clustered than the real ones; the fix removes the mis-shortened window but cannot invent detections for stretches where the person is genuinely not found for more than 2 s.

---

## 3. The picker lost its identity input (my v11.2.73 change, exposed by v11.2.74)

`_keep_n_faces` is the identity-aware picker that keeps the people you chose in the Detect frame and ignores bystanders. It scores each face by embedding similarity to your reference photos (`picked = 1 if idsim >= 0.32`). It runs at lines 4751 and 4851 — **before** the tracker. Since v11.2.73 defers recognition, `_embed_faces` did not run until `_persistent_track_pairs`, which comes later. So the picker saw faces with no embeddings: every `idsim` was −1, nobody counted as "picked", and the ranking fell through to overlap-with-tracker, then face **area**.

That was masked while the tracker held anchors. My v11.2.74 cut reset clears `last_hit_bbox`, so after any cut every overlap is zero and the largest two faces win.

```
3 faces (male, female, large bystander), keep 2

v11.2.72 eager embeddings, right after a cut      FEMALE + MALE
v11.2.74 deferred,         right after a cut      BYSTANDER + MALE     <-- loses her
```

And on **real face models**, with a photo of seven real faces and the two *smallest* designated as "the people you chose":

```
v11.2.77 (before): picker keeps 2: 0/2 are the people the user chose   <-- WRONG PEOPLE
v11.2.78 (after) : picker keeps 2: 2/2 are the people the user chose   OK
```

It also sustains itself: a slot that never binds never acquires an anchor, so the area-only ranking keeps excluding her.

**Honest scope:** your log shows `det_faces_avg=1.17, max=3`, so this picker only acts on the minority of calls with three faces. It is a real regression and it matters for any clip with bystanders, but I do **not** claim it is the main cause for *this* job.

**Fix:** `_keep_n_faces` now fills embeddings (and runs the cosine dedupe) itself, but only when it has a real choice to make — more faces than slots — so ordinary two-face frames stay cheap. It also counts how often it drops a face that matches a reference (`picker_dropped_ref`).

---

## 4. Pre-embedding dedupe merged people in close contact (my v11.2.73 change)

`_dedupe_faces` uses IoU plus cosine similarity when embeddings exist. For a pair missing an embedding it falls back to a centre-distance test (`dist < 0.35 × bbox diagonal`). My deferred path ran its first dedupe *before* embeddings existed, i.e. in the fallback — which merges two different people whose faces overlap:

```
situation                        centre dist   IoU   eager   deferred
kiss, faces overlapping                55      0.29     2        1   <-- LOSES A PERSON
one partly in front of the other       40      0.43     2        1   <-- LOSES A PERSON
```

**Fix:** the pre-embedding pass now uses `centre_frac=0.20`, which only catches boxes sitting on top of each other. Real duplicates (same person, box shifted 8 / 20 / 30 px) are still removed; all five close-contact cases now keep both people. The cosine dedupe runs once embeddings exist.

## 5. Single-face pairing read embeddings that were never filled

`_pairs_for_frame` ranks candidates by `np.dot(f.normed_embedding, ref0)` inside a `try` that returns −1.0 on failure. On a deferred face the embedding is `None`, so it silently degraded to "no identity information". `_embed_faces(faces)` is now called before it.

---

## 6. Things I tested and cleared — including a mistake of mine

- **The tracker re-binds both people after a cut**, with and without a bystander, at realistic embedding noise: 60/60 and 60/60.
- **Identity admission at low reference similarity.** My first test suggested binding collapses around cosine 0.6. That was an artefact of modelling every frame's embedding as independent noise. With realistic, stable per-person embeddings (≈0.9 frame-to-frame), both people bind 90/90 down to a reference similarity of **0.28**. So I did *not* touch admission — loosening it on the strength of an unrealistic test would have been a regression.
- **v11.2.75 occluder gate** and **v11.2.74 cut barrier** are clean on this job (`occ_cut=20/1522`, `shot_cuts=1`).

---

## 7. Making this measurable — and what I need next

`FACE PAINTED ON 60%` is an **OR across people**: one swapped person plus one untouched person still reads as a great job. That is why a one-sided result could hide.

The summary line and the in-app "Done" message now report each person separately:

```
Done — 2465 frames · face on 2434/2465 (99%) · people: #1 98% · #2 8% ⚠ person 2 swapped on far fewer frames · ...

PER-PERSON painted={0: 2410, 1: 190} recorded={0: 2440, 1: 1800} picker_calls=N picker_dropped_ref=N; weak_kept=N weak_dropped=N
```

| Field | What it separates |
|---|---|
| `painted` low, `recorded` high | the tracker knew she was there but she was not composited (opacity / gates) |
| `painted` ≈ `recorded`, both low | she is not being detected or bound at all |
| `picker_dropped_ref` > 0 | the picker discarded someone you chose |
| `weak_kept` / `weak_dropped` | how many keypoint-gate rejections (`kps_rejected=1572`) were actually lost |

**Please send me the summary line from the next run.** If `people:` still shows a lopsided result after this build, those four numbers tell me which stage is responsible.

## 8. Not applied

The five v11.2.52 tracker edits flagged in SDOS-071 §6 are still absent from this lineage and I have still not applied them. I have a proven cause for the symptom without them, and stacking a tracker rewrite on top would make the next log unreadable. If the per-person numbers show she is *never bound*, that is the next candidate.

I also considered making the grace window per-person rather than global. I did not, because a person who is detected only intermittently needs the *long* window to be held continuously, and a per-person blind span would take it away from exactly the person this is about.

---

## 9. Verification

| Test | Result |
|---|---|
| Real `_compute_blind_spans` replays the logged job | **blind = 2321**, exact |
| Same job with a verdict every call | blind = 0 |
| Picker on 7 **real** faces, 2 smallest chosen | before 0/2 → after **2/2** |
| Picker, synthetic, after a cut | BYSTANDER+MALE → FEMALE+MALE |
| Dedupe: 5 close-contact cases | all keep 2 (before: 2 of 5 lost a person) |
| Dedupe: real duplicates shifted 8/20/30 px | all still removed |
| Tracker re-bind after a cut | 60/60 both people |
| Admission at realistic low reference similarity | 90/90 down to 0.28 |
| Stash survives `_scale_faces`/`_sort_faces_left`/`_merge_detected`/`_smooth_faces` | none copy the face object |
| `Done in` log line | 34 placeholders = 34 arguments |
| **Regression:** lazy-ID embeddings | still bit-identical (0.000e+00) |
| **Regression:** grain | 95% → 7% |
| **Regression:** 90-frame soak | clean |
| **Regression:** affine incl. 448 trap | 16/16 |
| **Regression:** cut detector / barrier | 12/12; wrong-shot frames 2 → 0 |
| **Regression:** TrackState after cut reset | every method survives |
| **Regression:** ghost drift bound | 540 px → 0 px |
| `py_compile` all modules | clean |

## 10. Changed files

| File | Change |
|---|---|
| `core_pipeline.py` | `+ _compute_blind_spans` (lifted out of a closure, same logic); two-person verdict marks; `_keep_n_faces` embeds + dedupes when it must choose; `_embed_faces` before single-face pairing; `_dedupe_faces(centre_frac)` with a conservative pre-embedding value; per-person counters, message and log; `_PICK_STATS`; `weak_kept`/`weak_dropped` |
| `swap_engine.py` | `ENGINE_VERSION` → `aequus-1.2.78-twoup` |
| `config.py` | `VERSION` → `v11.2.78`; `BUILD` → `TwoUp · two-person grace window fixed, picker sees identity again` |
| `app.py`, `phoenix_api_adapter.py`, `requirements.txt`, `packages.txt`, `README.md` | unchanged — shipped for set integrity |

**Startup log must read:** `aequus-1.2.78-twoup` and `v11.2.78`.

**Rollback:** there is no config switch for the verdict-mark fix; it is a restore of `core_pipeline.py` from v11.2.77. The dedupe and picker changes are independent of it.
