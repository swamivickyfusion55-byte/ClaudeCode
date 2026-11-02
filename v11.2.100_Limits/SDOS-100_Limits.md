# SDOS-100 — Limits: Patch Removal takes up to 10 painted positions per mark

**Pack:** v11.2.100 "Limits" = v11.2.99 PatchBlend with only the limits changed. Engine `aequus-1.2.99-patchblend` (unchanged).
**App:** 1.2.28 (APK Builder 1.2.28).

**Asked:** "applying patch removal in 10 frames with subject in different positions" - not 10 separate marks: one mark painted at
up to 10 frames. "This is a simple change, deliver immediately without testing."

## What changed

| | before | now |
|---|---|---|
| positions (frames) of ONE mark | 6 (app) | **10** (app) |
| painted positions in all, per video | 12 | **20** (app and Space) |
| marks per video | 4 | **10** (app and Space) |

- App 1.2.28: `PatchFrames.MAX_VIEWS_PER_MARK` 6 -> 10, `MAX_VIEWS` 12 -> 20, `MAX_PATCHES` 4 -> 10. At the limit the add button
  greys out and a line says why.
- Space v11.2.100: `PATCH_MAX` 4 -> 10, `PATCH_MAX_VIEWS` 12 -> 20 in `config.py`, and the same defaults in
  `phoenix_api_adapter.py` and `core_pipeline.py`. Nothing else differs from v11.2.99.
- To remove a mark across 10 frames: open Patch Removal, paint the mark, tap **+ Position**, go to the next frame where the
  person has moved, paint the same mark, tap **+ Position** again ... and **Freeze** on the last one.
- The app and the Space must match: a Space on v11.2.96-v11.2.99 refuses more than 4 marks or 12 positions
  ("at most 12 painted positions per job") before anything is processed.

## Not tested

Delivered on request without the usual test run. It is constants only (3 numbers in the app's `PatchRemoval.kt`, 2 in the Space's
`config.py` plus the same defaults in two other files) on top of the tested v11.2.99. The Space files pass `py_compile`; the
app's compile check passed on 1.2.28 before the last constant (6 -> 10 per mark) was changed, and was not repeated for it.
NOT run: a job with 10 positions, the app on a phone.

## Cost

Each position of a mark is a separate anchor: the Space fills it (LaMa, ~1-3 s) and follows it through the video. Ten positions of
one mark cost about ten times one position in planning time, and the per-frame work uses the nearest position in time
(unchanged), so the render itself is not ten times slower.

## Install

- Space: upload the 8 files (or "Push to all Spaces"). Log: `Phoenix v11.2.100`.
- App: upload the 6 files of `Phoenix_APK_Builder_1.2.28.zip` to the APK Builder Space and build; install over 1.2.27.
