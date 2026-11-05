# Phoenix v11.2.118 — LockFinder

- BodyLock no longer switches off the mark finder for painted dark marks (moles, tattoos). Fast motion: a mole 36 % -> 16 % still
  visible, a tattoo 17 % -> 1 %. Hair and the one-go belly clean are unchanged (the one-go positions keep the v11.2.117 behaviour).
- Known: a fast-moving mole still loses its hold on ~24 of 150 frames; the v11.2.112 base holds it (SDOS-118).

# Phoenix v11.2.117 — OneGoBelly

- The one-go belly clean (job setting `clean_belly`) works on the Space: MediaPipe Pose Tasks (0.10.35), the band from just below the
  navel to the hip line, the navel and clothing left alone (`belly_clear.py`). Before, it failed on `mp.solutions` and painted nothing.
- New route `phoenix_clean_belly_preview`: one frame -> the frame with the marks removed, the band outlined, and the info.
- `requirements.txt`: `mediapipe==0.10.35`; `packages.txt`: `libgles2`, `libegl1`.
- Tested on synthetic skin with MediaPipe's test photo (SDOS-117). Fast motion under BodyLock is not yet tested on this pack.

# Phoenix v11.2.112 — BodyLock

The paint is placed from the surface **around** the stroke, then smoothed. It is no longer driven by the dark texture inside the stroke, which is what made a removal sit still or flicker while the body moved.

- Dark-on-skin strokes (body hair and similar patches) are detected on the anchor frame.
- Their motion is an ECC fit of the surrounding ring, with the stroke excluded.
- That pose is smoothed on a wider window than the global track.
- The per-frame "find the dark mark and snap to it" step is off for these strokes, so strands cannot yank the paint.
- The filled area is grown by 5 px so the soft fringe is covered.
- Other marks (moles, tattoos, ink) still use the existing local fit and mark finder.

Synthetic check: a dark patch translating 3 px / 2 px per frame on a textured plate stayed within 2.9 px of the true position (mean 0.8 px), with frame-to-frame residual jitter of 0.5 px mean.
