# SDOS-179 — the app's painted mask is read as the paint, not as the whole frame

**Pack:** v11.2.179, which is v11.2.178 plus one function fixed in `phoenix_api_adapter.py` (`_mask_from_image`). Engine
`aequus-1.2.179-contactstep`. The face swap is the same code as 177 and 178.

**Reported on 178:** "The preview itself is not working properly. The entire screen became a bit dark and patch was not
removed."

All tests use synthetic frames and clips. No user footage was looked at.

## 1. Cause

The app saves the painted mask (`PatchFrames.maskBitmap`) as an **ARGB_8888 bitmap: a black background with white paint**,
written as a PNG. That PNG is **opaque everywhere** (alpha 255). The uploaded v11.2.175 adapter, kept in 176-178, read any
4-channel mask by its alpha channel only. For the app's mask that is 255 on every pixel, so **the whole frame counted as
painted**:

- **Preview:** the whole picture was filled, which darkened it (measured: 82 levels mean change), and the mark stayed.
- **Video job:** the same function reads the mask the app sends with a job (`mask_b64`). There too the painted mark
  covered the whole frame.

Our earlier line (v11.2.96-148) read the alpha channel only when it is used (some pixels transparent, some not), and the
brightest colour channel otherwise. 179 restores that.

**Why 178's tests missed it:** they drew the mask as a transparent PNG with the paint in the alpha channel, and the video test
wrote a grey mask file directly. Neither is the app's format. The tests below use the app's exact format.

## 2. Verification (`verification/patchremoval/preview179.py`, an app-format mask: opaque ARGB PNG, black, white paint)

| Check | v11.2.178 | v11.2.179 |
|---|---|---|
| Mask read as | 100 % of the frame | the strokes (overlap 1.000 with the paint, 1.1 % of the frame) |
| Preview: the mark's pixels | 48 -> 21 (not removed, darker) | 48 -> 170 (skin around: 170) |
| Preview: the rest of the frame | changed by 82 levels (darkened) | 0.8 levels (JPEG only) |
| A job's mark from the app (`mask_b64`), as saved for the render | 100 % of the frame | the strokes (overlap 1.000) |

- **Through the real Gradio 4.44.1 app** (`app_check178.py`, called as the phone app calls it, with the app-format mask):
  `ok`, 10776 px filled, a mean change of 1.2 levels on the whole frame.
- **A video job with the mark sent as the app sends it** (`bench_pipeline.py patches=... app_mask=1`: frame JPEG + opaque
  ARGB mask, base64, through the adapter's own `_load_patches`; the `proc_tattoo` clip):
  - v11.2.178: the mark is read as 921600 px, the whole 1280x720 frame. The tattoo was not removed on any of the 140 frames
    it is in the picture ("left 129 %": the frame around it was changed too).
  - v11.2.179: the mark is read as 19472 px, the strokes. It is anchored on the painted frame and followed on 141 of 150
    frames. 11.2 % of the tattoo is left on average, missed on 7 of 140 frames. That is the same as with a mask written
    directly as a grey file.

## 3. Install

Upload the pack's files. Only `phoenix_api_adapter.py`, `config.py` and `swap_engine.py` (version string) differ from 178.
The log at start reads `Phoenix v11.2.179`. The app (1.2.49) needs no change. Paint over the mark and press **Preview**.
