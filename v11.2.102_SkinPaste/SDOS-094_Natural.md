# SDOS-094 — Natural: GPEN + Cinematic toned down; eye placement in side views (investigated)

**Pack:** v11.2.94 "Natural" (v11.2.93 CleanCut + a milder look). Engine `aequus-1.2.94-natural`. No app change.
**Reported:** "Cinematic and GPEN make it a bit too clean and bright, which looks a bit unnatural. Please marginally normalise. Also see the placement of the eyes in the side pose of the man."

The two screenshots were not analysed: they are frames of real people, and the standing rule for this project is to work only from synthetic media, public test photographs, logs and descriptions.

---

## 1. Marginally more natural

Measured on the sharp test clip (a public photo, 494-px face, nothing upscaled), painted face against the footage:

| | v11.2.93 GPEN + Cinematic | **v11.2.94** | footage |
|---|---|---|---|
| brightness (L, 0-255) | +2.6 | **+1.4** | 0 |
| bright highlights on the face (L > 200) | 2.8 % | **2.4 %** | 1.3 % |
| skin texture (fine detail) | x0.87 | x0.84 | x1 |
| sharpness (Cinematic alone: 0.24) | 0.39 | 0.34 | 1 |
| shimmer | 1.25 | 1.23 | |
| identity match to the replacement photo | 0.768 | 0.767 | |

Changes (`config.py`):
- Cinematic: glow 0.18 → **0.10** (less sheen), HDR 0.12 → **0.10**, vibrance 0.12 → **0.08**, and new `keep_brightness` 1: the touch may move light around the face but not brighten it as a whole (measured +0.05 grey levels on a face crop).
- GPEN: strength 0.6 → **0.5**.

The remaining +1.4 is the colour match's existing half-strength correction towards the replacement photo's skin tone (unchanged since v11.2.81, `cm_strength_l` 0.42). In the frames, the nose highlight and the lip gloss of v11.2.93 are toned down; eyes and lips stay crisper than with plain Cinematic. For a still more natural look: `ENHANCE_GPEN["blend"]` 0.4, or plain "Cinematic" in the app.

## 2. Eye placement in side views: what I measured

The swap is placed with one similarity fit (move, rotate, scale) of the detector's 5 points (eyes, nose tip, mouth corners) onto a frontal face template. On AFLW2000's side views (50-71 deg, true 3-D landmarks):
- The detector's point for the **visible** eye is 1.0-3.4 % of the face size off; the 106-point landmarks place it at 0.2-1.3 % (better on 5 of 6 faces). For the **hidden** eye neither is reliably better (both 3-14 %).
- But where the template's eye lands through the fit is **5-8 % off whichever eye point is used**, and with the better visible-eye point it got slightly worse on 3 of 6 faces (e.g. 5.2 → 6.5 %): a side view cannot be fitted to a frontal template by moving, rotating and scaling, so better points alone do not move the drawn eye.
- On the swapped side-view photos themselves, the swap model draws the visible eye where the real eye is (it follows the target's own pose), on all three.

So the change was **built, measured and switched off** (`side_eye` 0 in `swap_engine.py`; code kept). I could not reproduce a misplaced eye on public side-view photos. To go further I need a description of what looks wrong (see §4).

## 3. Verification

`verification/fastrestore/unit94.py` (29 checks: the 27 of v11.2.93, Cinematic keeps the face's brightness, side-eye refinement off), `look_metrics.py` (brightness, highlights, texture, colour against the footage), `kps_accuracy.py` (detector vs 106-point vs the pack's points, and the drawn eye, against AFLW2000's true landmarks). The 13 checks of v11.2.90 pass.

## 4. What I need for the side-view eyes

A description of the frame (no picture needed): is the visible eye too high / too low / too far towards the nose or the ear, too big or too small, or is there a second, hidden-side eye drawn near the nose? Roughly how far is the head turned (a three-quarter view, or a full profile)? And the job's `SIDE-VIEW:` log line.

## 5. Install

Upload the 8 files (`app.py config.py core_pipeline.py phoenix_api_adapter.py swap_engine.py packages.txt requirements.txt README.md`), or "Push to all Spaces" in app 1.2.24. Log: `Phoenix v11.2.94`, `swap_engine aequus-1.2.94-natural`.
