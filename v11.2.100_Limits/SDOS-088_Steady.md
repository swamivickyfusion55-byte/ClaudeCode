# SDOS-088 — Steady: back to the v11.2.81 swap; what I found about fast turns and side poses

**Pack:** v11.2.89 "Steady". Engine `aequus-1.2.89-steady`.
**Reported (after 11.2.88):** "On fast turns the pasting flickers and is a bit hazy; in side poses the face angle is incorrectly pasted. I think somewhere we have introduced issues in the code."

I have not seen your footage (I only work with synthetic media and your logs). Side poses here are a synthetic squeeze of real face photographs, not a real 3-D head turn. §6 says what that cannot show.

---

## 1. What I did

I rendered the same fast-moving synthetic clips (sweep ±180 px with motion blur, roll ±22°, squeeze to 35 % yaw; real detector, swapper and compositor) through **v11.2.71** (before my first change of this series), **76, 79 and 88** (81 has the same swap engine as 79 and gave the same result on a longer clip), and compared them against the true face geometry of the clip.

| Measured on the fast clip (Balanced) | 71 | 76 | 79 | 88 |
|---|---|---|---|---|
| Placement error of the painted face, median / worst (inter-eye distances) | 0.173 / 0.238 | 0.173 / 0.238 | 0.173 / 0.238 | 0.173 / 0.238 |
| Frames where the swap is missing | 0 | 0 | 0 | 0 |
| Sharpness of the face vs the footage (Laplacian) | 1.20 | 1.28 | 1.28 | 1.26 |

**Conclusion 1: nothing in the tracking, placement or compositing path changed the result between 71 and 88 on these clips.** The base I restored in 11.2.88 was not broken by the eating work in that path.

## 2. What I did find

### 2.1 Side poses and turns: the eating work of v11.2.82/83 makes the occluder gate cut the face more

The occluder gate (the "is something in front of the face" test) has run since v11.2.75. A turning head uncovers what is behind the cheek (a wall, a window), and the gate can read that as an object and cut a piece of the swapped face away; the original face then shows through, at the wrong angle. I made the head "turn" in the test (aligned face squeezed 35 % / 50 %, the uncovered side filled with a plain wall), 12 real faces, final cut after temporal smoothing:

| Share of the face area cut | v11.2.81 | v11.2.83 (= 88) |
|---|---|---|
| bright wall, yaw 35 %: mean of the worst frame / worst face | 0.12 % / 1.5 % | **1.44 % / 17.3 %** |
| dark wall, yaw 50 %: mean of the worst frame / worst face | 1.64 % / 6.3 % | **3.93 % / 17.4 %** |
| dark wall, yaw 50 %: while the head is turning | 0.90 % | **2.24 %** |
| dark wall, yaw 50 %: held side pose | 1.62 % | **3.75 %** |
| faces with more than 3 % cut in some frame (dark wall, 50 %) | 4 of 12 | 5 of 12 |

v11.2.82/83 (faster opening of a cut, whole-object growth, mouth-zone fill) were written to catch ice cream. On turning faces they open a false cut faster and larger. That is the only change between 81 and 88 that touches the output, and it is the likely reason side poses look worse. **11.2.89 removes it: the swap engine, tracker and compositor are v11.2.81 exactly** (v11.2.79 and 80 have the same swap engine; you described that state as excellent on 14 Oct). The eating gate of 82/83 was, per SDOS-083, blind to ice cream on real faces anyway ("0 of 42" combinations), so nothing measurable is lost.

v11.2.81's gate still cut up to 6 % of the face in my harshest turn (dark wall, 50 % yaw). I did not change that: it is the behaviour you rated excellent, and I will not ship a new gate on synthetic tests alone.

Correction to SDOS-087: I wrote that the `OCCLUDER:` line should be about 0 on clips without food. It was not: your 11.2.88 jobs show 112 of 1854, 306 of 2439 and 526 of 4944 face-frames cut (6–12 %). The same rate appeared on your 11.2.78/79 jobs (3.5–12 %), so the rate is not new, but the 82/83 tuning makes each false cut larger.

### 2.2 Flicker on fast turns: it depends on the quality setting, not on a regression

Same fast clip, painted face compared in its own frame (head motion removed with the true key points), mean change per frame, output against the footage's own; "worst 5 %" and the largest step against the median:

| Quality | change per frame (output / footage) | worst 5 % of frames (output / footage) | largest step / median |
|---|---|---|---|
| Ultra | 6.9 / 6.5 (**×1.06**) | 8.7 / 8.5 | 1.3 |
| Best | 8.8 / 6.5 (×1.35) | 16.3 / 8.5 | 2.2 |
| Balanced | 9.0 / 6.5 (×1.38) | **20.7** / 8.5 | 3.0 |
| 71 vs 88 at Balanced | 8.9 vs 9.0 | 20.7 vs 20.7 | same |

At Best and Balanced the swap network runs every 2nd / 4th frame and the frames between reuse the last swapped face; while the head moves fast, that face is a little stale and **steps** to the new one at each swap. That is the flicker. **Ultra (a swap on every frame) shows no extra flicker (×1.06).** It was the same in 71, so it is how those modes are built, not something introduced since.

From your logs: the first and second 11.2.88 jobs made about one swap per painted face-frame (Ultra); the third made one per two (Best). If the clips that flicker were rendered at Best or Balanced, render the fast-turn clips at Ultra.

## 3. What this pack is

v11.2.81 with only `VERSION`, `BUILD` and `ENGINE_VERSION` changed (11.2.81 still announced engine "1.2.79"). Checked: `core_pipeline.py`, `app.py`, `phoenix_api_adapter.py`, `packages.txt`, `requirements.txt`, `README.md` are byte-identical to 11.2.81; `swap_engine.py` and `config.py` differ only in those three strings.
Nothing from 11.2.84–87 is in it: the cheek smoothing, edge refinement, depth order and size-hold of those versions stay out.

## 4. What this does NOT do

- **Real side poses (3-D head turns) are not tested.** My side pose is a 2-D squeeze of a frontal photograph. If the pasted angle is wrong because the detector's key points are poor on a real profile, or because the swap network renders a profile badly, none of this addresses it, and I cannot reproduce it without footage I may not analyse. If it persists on 11.2.89, send me the `Done` and `OCCLUDER:` lines of that job and the time of a bad frame.
- **Best and Balanced still step on fast turns.** Use Ultra for those clips, or tell me and I will make a fast-turn mode that swaps on every frame only while the head moves quickly.
- The eating scenes (food at the mouth or the cheeks) get no special handling beyond what 11.2.81 has.

## 5. Verification

- Fast clips, real pipeline, versions 71/76/79/88 compared as in §1 (identical placement in all four), 81 against 88 on a second, longer clip (identical).
- Turn / side-pose false-cut battery (12 real faces × yaw 20/35/50 % × 6 uncovered-background colours), v11.2.81 against v11.2.83, as in §2.1.
- Flicker at Ultra, Best, Balanced (§2.2).
- `Fast (blend)` against `Off` for smooth motion: identical on these clips.
- The zip unpacks to exactly these files; all `.py` compile; end-to-end run of the extracted pack on the fast clip (Balanced): `swap_engine aequus-1.2.89-steady` in the log, face painted on 70 of 70 frames, `REVERTS=0`, placement error median 0.173 / worst 0.238, sharpness 1.28 (the same numbers as 11.2.79).

## 6. Install

Upload the 8 files (`app.py config.py core_pipeline.py phoenix_api_adapter.py swap_engine.py packages.txt requirements.txt README.md`) to the **root** of the Space, commit, normal restart (or "Push to all Spaces" in the app). The startup log must show `Phoenix v11.2.89` and `swap_engine aequus-1.2.89-steady`. Do not restart a Space while a job is running on it.
