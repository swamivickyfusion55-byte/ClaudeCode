# SDOS-178 — the Patch Removal Preview button works again

**Pack:** v11.2.178, which is v11.2.177 plus the `phoenix_patch_preview` endpoint. Engine `aequus-1.2.178-contactstep`. The face swap
is the same code as 177 (only version strings changed). SDOS-176 and SDOS-177 hold for it.

**Reported:** "in manual patch clean, preview button is not working, not showing cleaned patch.. Is manual patch working at
all.."

All tests use synthetic frames and clips. No user footage was looked at.

## 1. Cause

The app's **Preview** button (Patch Removal) calls the Space endpoint `phoenix_patch_preview`, with the frame and the
finger-painted mask. The uploaded v11.2.175, which is based on 11.2.92, had the manual patch code (`patch_manual.py`) only in
the video job. It never registered that endpoint, so Preview had nothing to call. The app then shows "This Space has no Patch
Removal yet ..." or "Preview failed". 176 and 177 inherited this unchanged. I did not test the Preview button before
delivering them.

The **video job** was not affected: the patches the app sends with a job reach `patch_manual.prepare()` and are removed in
the render (section 3).

## 2. The change

- `phoenix_api_adapter.api_patch_preview(image, mask)`: reads the frame and the painted mask (an alpha or grey mask, as the
  app sends it). It removes the mark with `patch_manual.remove_patch_image`, the same still fill the job uses on the painted
  frame. It returns the cleaned frame (JPEG) and a JSON string: `ok`, `marks`, `area_px`, `inpainter` ("still fill"), `ms`,
  `width`, `height`, `version`.
- `app.py`: the hidden controls and `api_name="phoenix_patch_preview"` (concurrency 1). These are the same component types as
  our earlier line, which the app was written against: [0] a file, [1] a JSON string.

## 3. Verification

1. **The function** (`verification/patchremoval/preview178.py`, a synthetic skin frame with a dark painted mark, 1280x720): 4 of 4 checks
   pass.
   - It returns a cleaned frame and JSON `ok`.
   - The mark is gone: its pixels' mean went from 48 to 170, against 170 for the skin around it.
   - The frame away from the mark is unchanged: 0.8 levels, from the JPEG.
   - With no mask painted, it returns a clear error and does not crash.
2. **The real Gradio app:** `app.py` built with Gradio 4.44.1, the Space's version, with only the model packages stubbed (`app_check178.py`).
   - Its API lists `phoenix_patch_preview` with the 7 others.
   - Launched locally and called as the app calls it (`gradio_client`, frame + mask), it returned the cleaned frame file
     and `{"ok": true, "marks": 1, "area_px": 10776, "inpainter": "still fill", "ms": 1853, ...}`.
3. **The video job with a painted patch** (`verification/patchremoval/make_patchclip.py proc_tattoo`: procedural skin, a
   tattoo that moves, turns, scales and leaves the picture; `bench_pipeline.py patches=`; `bench_eval.py` against the same
   clip without the tattoo):
   - v11.2.178: the patch is anchored on the painted frame (match 1.00) and followed on 141 of 150 frames. The tattoo left
     is 11.2 % on average, missed on 7 of the 140 frames it is in the picture.
   - v11.2.175 on the same clip: identical, 11.2 % left and missed on 7 of 140 frames. The video patch removal is the same
     code, unchanged by 176-178.

## 4. Limits

- Synthetic frames and clips only. A real body has folds, hair and changing light.
- The Preview shows the still fill on the painted frame. In the video, each frame is filled after the mark is tracked, so
  frames where the mark turns or is partly covered can differ from the Preview.
- Not tested on a phone: the endpoint was called through `gradio_client`, the same HTTP route the app uses
  (`/call/phoenix_patch_preview`).

## 5. Install

Upload the pack's files. `app.py`, `phoenix_api_adapter.py`, `config.py` and `swap_engine.py` (version string) differ from
177. The log at start reads `Phoenix v11.2.178` and `swap_engine aequus-1.2.178-contactstep`. The app (1.2.49) needs no change.
