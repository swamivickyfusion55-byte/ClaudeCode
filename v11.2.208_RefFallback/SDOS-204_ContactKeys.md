# SDOS-204 — ContactKeys: Optimized is conservative in contact and in side poses; Ultra untouched

**Pack:** v11.2.204, which is v11.2.203 with the changes below. Engine `aequus-1.2.204-contactkeys`. SDOS-203 holds for
everything else (the 176-179 contact fixes, encrypted saved uploads).

**Asked:**
- "Will it be much quality compromise, if I make the swap interval 1 when bodies are connected or else say 2 or 3 (in ultra
  mode)? And update such setting in any of the existing presets."
- "Please update this in optimised."
- "I want Ultra to be the highest in quality, hence not to be touched.. Make these smart changes only in Optimised, also
  would it also be a good idea to be more conservative in extreme side poses.. say more than a certain degree?"
- "To summarise optimised will follow conservative when in contact and extreme sides."

All tests use synthetic clips made from public sample photos. No user footage was looked at.

---

## 1. The rule

| Tier | Two faces apart | Two faces touching | A face turned past ~40 deg | One face |
|---|---|---|---|---|
| **Ultra** | every frame | every frame | every frame | every frame (unchanged) |
| **Optimized** | **every 2nd frame** (was every 5th) | **every frame** | **every frame** | **found on every frame, swapped every 2nd** (was both every 5th); **every frame in a side pose** |
| Best, Balanced, Fast | as v11.2.203 | as v11.2.203 | as v11.2.203 | as v11.2.203 |

In every case a face that moves or changes fast still gets extra swaps (MotionKeys, SDOS-090).

- **Touching** means the two face boxes overlap at all, or their centres are under 1.25 face sizes apart. That is the
  tracker's own contact test.
- **Side pose** uses the engine's own profile measure: the nose's sideways offset from the eye midpoint, roll-corrected,
  in eye distances. Calibrated on AFLW2000's true head yaw: frontal 0.00-0.25, 36-45 deg 0.34-0.51, 55-71 deg 0.76-1.48.
  0.40 is about 40 deg.
- **One face:** after "Yes make it like 2 face logic, every 2nd". Optimized now finds the face on every frame, as Ultra
  does, so placement is exact on every frame. It swaps every 2nd frame while the face is frontal, and every frame while it
  is turned past ~40 deg. In v11.2.203 it found and swapped the face on every 5th frame only.

**Ultra is untouched.** It swaps every frame, as in v11.2.203. The one other change: the ArcFace skip of v11.2.203 is now off
in Ultra (Optimized only). The Space's Quality / HQ presets are as in v11.2.203. The phone app needs no change: its
Optimized preset sends "Auto" and lets the Space decide.

Switches (in `swap_engine._P`):
- `smart_keys`: 0 makes Optimized behave as v11.2.203 except the two-face gap.
- `side_key_h`: the side-pose threshold (0.40); higher = only stronger side poses.
- `arc_skip`: 0 turns the ArcFace skip off.

## 2. Verification

**Unit checks**, run from the final files: **51 of 51 pass**.
- `unit204.py` 7: the contact test, and the side-pose test at frontal, 36, 45 and 60 deg, with a tilted head, and one face
  of two in profile.
- `unit176.py` 12 and `unit176_pipe.py` 14: the tracker and contact rules.
- `unit203.py` 18: the ArcFace skip and the encryption.

**The mechanism, measured.** The first 204 build tried the same contact rule on Ultra with "swap every 2". It was then
removed from Ultra at your request, but it is the same code Optimized now uses. Same bench as SDOS-203 (synthetic clips, the
real pipeline on CPU, 2 threads):

| Clip | Ultra, every frame (v11.2.203) | Every 2nd apart, every frame in contact |
|---|---|---|
| Harsh step (two people apart throughout) | 498 s, 240 swaps; worst 4.8 px; flicker A x1.00 | **399 s (20 % faster)**, 168 swaps; worst 4.8 px; flicker A x1.02 |
| Step clip (apart) | 479 s, 240 swaps; worst 18.7 px (frame 84) | **380 s (21 % faster)**, 172 swaps; the same scores |
| Side-view kiss (in contact most of the clip) | 585 s; A revert 0 / mix-up 0, B revert 7 / mix-up 0; other's paint 4 % | 581 s, 101 frames swapped for contact; **identical scores** |
| Cross | 486 s; weak 8 + 4, no revert or mix-up | 479 s, 84 frames swapped for contact; **identical scores** |

- In contact the result is the same as Ultra, and so is the time: every frame is swapped there.
- Apart, the time drops by about a fifth. The only measured difference is the face's frame-to-frame change: x1.02 against
  x1.00 of the footage's own.
- **What these clips cannot show:** their faces are still photos (no talking, no blinking), so the expression refresh that
  every-frame swapping buys does not appear in any score. Judge that on real footage.

**Optimized against Ultra** (the final v11.2.204 files; Ultra is the same code as v11.2.203; same bench, 2 threads):

| Clip | Ultra | Optimized v11.2.204 |
|---|---|---|
| Harsh step (two people apart throughout) | 497.5 s, 240 swaps; identity correct 120 + 120, no revert or weak frame; worst 4.8 px | **385.5 s (23 % faster)**, 168 swaps; identity correct 120 + 120, no revert or weak frame; worst 4.8 px (the same placement) |
| Side-view kiss (in contact most of the clip) | 585.1 s; B: revert 7 / weak 15, A: weak 12; no mix-up | **492.5 s (16 % faster)**, every frame in contact swapped (101 contact + 17 side-pose swaps); B: revert 7 / weak 15, no mix-up - the same as Ultra |
| One face, a strong side pose on every frame | 62.3 s and 81.9 s (two runs) | 78.2 s, 87.3 s and 77.1 s (three runs): every frame is a side pose, so it swaps every frame like Ultra |

- Apart and frontal, Optimized is about a fifth faster than Ultra with the same scores.
- In contact it swaps every frame, like Ultra. It was still 16 % faster on the kiss clip, from its lighter encoding and the
  ArcFace skip (Optimized only).
- On the all-side-pose clip it does the same swaps as Ultra. The runs differed by up to 25 s from one run to the next on this
  shared CPU: that spread, not a slowdown, is the difference between the columns (an earlier reading of "Optimized slower than
  Ultra" was this noise; re-runs confirmed it).

## 3. Limits

1. **Synthetic tests only, and still faces.** The expression refresh (mouth, eyes) is not measured by any score here.
2. **Against Ultra, Optimized saves time mainly where the faces are apart and frontal** (23 % on the step clip). In a kiss or
   a hug it swaps every frame like Ultra; the 16 % it still saved on the kiss clip comes from its lighter encoding and the
   ArcFace skip, not from fewer swaps.
3. **Optimized keeps its other settings:**
   - a lighter encoding (CRF 20, where Ultra uses 18);
   - audio at 192k, where Ultra uses 320k;
   - a one-face detector input of 560 px, where Ultra uses 720.

   For the same picture quality as Ultra outside the swap cadence, these would also have to match. Not changed here.
4. **The side-pose threshold (~40 deg) is a calibrated proxy** from the 5 key points, not a measured 3-D angle. A tilted or
   foreshortened face reads a few degrees off. `side_key_h` moves it.
5. From SDOS-203: the ArcFace skip (now Optimized only) can show the other person's face for 1 frame at a shot cut where the
   two land exactly on each other's previous spots.

## 4. Install

Upload the pack's files. Only `core_pipeline.py`, `swap_engine.py`, `config.py`, `app.py` (hint text) and this document
differ from 203. The log at start reads `Phoenix v11.2.204`. An Optimized job logs `OPTIMIZED, two faces: ...` and
`SMART-KEYS (Optimized): N extra swaps because two faces touched, N because a face was turned more than about 40 degrees`.
