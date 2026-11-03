# SDOS-093 — CleanCut: no deformed face for a moment after a fast move or a shot change

**Pack:** v11.2.93 "CleanCut" (v11.2.92 FastRestore + two fixes). Engine `aequus-1.2.93-cleancut`. No app change (1.2.24 stays).
**Reported:** "On fast movement or shot change, for a fraction of a second the face becomes deformed." (job 40A6249F, v11.2.92)

I have not seen your footage (I only work with synthetic media, public test photographs and your logs).

---

## 1. What your log says

- `shot_cuts=0`: the shot change in your clip was **not detected**. The cut detector is known to miss a cut between two shots with very similar colours (same scene, another angle), so for the engine the new shot was the same face moving very fast.
- `RESTORE: GPEN ... on 510 swapped face crops ... mean strength 0.20`: GPEN + Cinematic was on.

## 2. The cause: my v11.2.92 GPEN "steady detail"

To stop GPEN's restored detail from shimmering, v11.2.92 mixed half of the previous frame's restored detail into the current one. It did that **without checking that the previous frame still showed the same face in the same pose**, and it was not cleared at cuts. After a shot change (detected or not) or a very fast turn, the previous face's edges, hairline and features were blended into the new face, fading over 2-3 frames: the "deformed for a fraction of a second". The mouth rule of v11.2.92 had a related weakness: after a missed cut, the previous shot's mouth state could carry over into the new face.

Seen on a synthetic shot change between two public photos, cut not detected, one replacement face (frames 30 and 31, just after the cut): v11.2.92 shows a ghost of the previous shot's face outline across the cheek and forehead on both frames; v11.2.91 and v11.2.93 do not.

## 3. The fixes

1. **Steady detail only while the face looks the same.** The coarse look of the face (32x32 grey of its blurred centre) is compared with the previous frame's: full mix up to a change of 1 grey level, none from 4. Measured on the clips: a still or slowly moving face 0.2-1, the fastest turn and roll 4-6, a shot change 67. The previous detail is also never used across a detected cut or more than 2 frames back.
2. **Per-face state restarts when the look jumps** (change above 20; only a cut gets there): the mouth (eating) rule, the side-view weight and the occlusion map start fresh, so nothing from the previous shot is carried into the new one. The eating rule also no longer acts on a held / predicted face (one not measured on this frame). New log line: `LOOK-JUMP: n times a face's look changed at once ...`, so a missed shot change shows up in the log.

## 4. Results (real pipeline, 2-face job setting, Ultra, Fast blend, GPEN + Cinematic; v11.2.91 with Cinematic)

Shot change not detected, compared with the second shot rendered alone (mean grey-level difference of the aligned face):

| | frame 30 (first after the cut) | 31 | 33 | settled |
|---|---|---|---|---|
| v11.2.91 | 4.6 | 4.1 | 3.7 | 1.9 |
| v11.2.92 | **5.4** (ghost outline) | 4.6 | 3.9 | 2.0 |
| **v11.2.93** | **4.6** | 4.3 | 3.8 | 2.0 |

(The remaining 4-5 grey levels after a cut, in all versions, are the colour match settling on the new shot over about half a second; no shape change.)

Fast movement:
- Fast sweep / roll / turn clip: no one-frame spike above 3x the footage's own change in any version; GPEN adds a little frame-to-frame change (x1.15 of the footage against x1.04 without GPEN), the same in v11.2.92 and v11.2.93.
- Fast side views with motion blur (±160 px sweep), against v11.2.91: paint off the face 7.9 → 0.2 % and 10.1 → 0.7 %, frame-to-frame wobble 5.95 → 4.42 % and 6.51 → 5.04 %, no frame where the face lost its paint (0 in both versions). The side-view outline of v11.2.92 holds in fast motion.
- The occlusion model on fast motion: on my fast clip it cuts 28 % of a crop on half the face-frames, but checked per face, all of that is on a background "face" the detector finds in the clip; the real face keeps 97-98 % paint at the mouth and chin with or without the model.

A test-setup note: my first comparison ran two jobs at once with the same job id, and the pipeline names its temporary video `/tmp/result_<job id>.mp4`, so the two overwrote each other. Those runs were discarded and redone with a job id per run. On your Spaces this cannot happen (every job has its own id and one runs at a time).

## 5. What this does NOT do

- **Missed shot changes are still missed** by the cut detector (similar colours). v11.2.93 only makes sure nothing of the previous shot is blended into the new face; the colour match still takes about half a second to settle on the new shot, as in every earlier version.
- Untested on real footage: the cuts and fast moves are synthetic, from public photos.

## 6. Verification

`verification/fastrestore/unit93.py` (27 checks: the 21 of v11.2.92 plus: steady detail mixed for the same face, not without the previous look, not for another face, fading with the change; mouth / side-view state kept for the same face and restarted for another face in the same track; eating rule not on a held face). `make_cutclip.py` + `transient_metrics.py` (shot change, fast clip), `make_profclip.py` with `FAST=1` + `profvideo_metrics.py` (fast side views), `bench_pipeline.py` (one job id per run with `BENCH_UNIQUE_JID=1`). The 13 checks of v11.2.90 pass.

## 7. Install

Upload the 8 files (`app.py config.py core_pipeline.py phoenix_api_adapter.py swap_engine.py packages.txt requirements.txt README.md`), or use "Push to all Spaces" in app 1.2.24 (Spaces that are rendering are updated when their job ends). Log: `Phoenix v11.2.93`, `swap_engine aequus-1.2.93-cleancut`.
