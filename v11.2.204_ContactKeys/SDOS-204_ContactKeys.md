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
| **Optimized** | **every 2nd frame** (was every 5th) | **every frame** | **every frame** | every 5th, **every frame in a side pose** |
| Best, Balanced, Fast | as v11.2.203 | as v11.2.203 | as v11.2.203 | as v11.2.203 |

In every case a face that moves or changes fast still gets extra swaps (MotionKeys, SDOS-090).

- **Touching** means the two face boxes overlap at all, or their centres are under 1.25 face sizes apart. That is the
  tracker's own contact test.
- **Side pose** uses the engine's own profile measure: the nose's sideways offset from the eye midpoint, roll-corrected,
  in eye distances. Calibrated on AFLW2000's true head yaw: frontal 0.00-0.25, 36-45 deg 0.34-0.51, 55-71 deg 0.76-1.48.
  0.40 is about 40 deg.
- **One face:** Optimized detects only every 5th frame, and a swap on points interpolated between detections is weaker
  than a reused swap cut on a real detection (the pipeline's own rule). So while the face was last seen turned past
  ~40 deg, the next chunk is **detected and swapped** on every frame.

**Ultra is untouched.** It swaps every frame, as in v11.2.203. The one other change: the ArcFace skip of v11.2.203 is now off
in Ultra (Optimized only). The Space's Quality / HQ presets are as in v11.2.203. The phone app needs no change: its
Optimized preset sends "Auto" and lets the Space decide.

Switches (in `swap_engine._P`):
- `smart_keys`: 0 makes Optimized behave as v11.2.203 except the two-face gap.
- `side_key_h`: the side-pose threshold (0.40); higher = only stronger side poses.
- `arc_skip`: 0 turns the ArcFace skip off.

## 2. Verification

UNITS

TABLE

## 3. Limits

LIMITS

## 4. Install

Upload the pack's files. Only `core_pipeline.py`, `swap_engine.py`, `config.py`, `app.py` (hint text) and this document
differ from 203. The log at start reads `Phoenix v11.2.204`. An Optimized job logs `OPTIMIZED, two faces: ...` and
`SMART-KEYS (Optimized): N extra swaps because two faces touched, N because a face was turned more than about 40 degrees`.
