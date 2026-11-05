# SDOS-118 — LockFinder: painted dark marks keep their mark finder

**Pack:** v11.2.118 "LockFinder" = v11.2.117 (the one-go belly clean) with one engine change. Engine `aequus-1.2.118-lockfinder`.
**App:** no change (1.2.32).

**Asked:** "Do whatever option is best" — after the v11.2.113 report ("paint moves along with the body, but in a few frames it opens
up at the centre like a split").

All tests use synthetic clips (`make_patchclip.py`, `make_hairclip.py`, the belly clip from `make_bellyreal.py`); no user footage.

---

## 1. What we found

The upload's BodyLock (dark-on-skin strokes: the paint is placed from the ring of skin around the stroke) is switched on for **any dark
mark**, not only for body hair. Its code also switches off the **mark finder** for those marks. The release notes say moles and tattoos
keep the finder; the code does not do that.

On fast motion (the mole and tattoo clips at four times the speed) that costs a lot:

| Clip (video_test, share of the mark still visible / frames missed) | v11.2.112 (base) | v11.2.117 (before) | v11.2.118 (this) |
|---|---|---|---|
| Fast mole | 0.4 % / 0 of 150 | 35.9 % / 50 of 150 | **15.6 % / 24 of 150** |
| Fast tattoo | 3.5 % / 0 of 150 | 16.8 % / 12 of 150 | **1.1 % / 3 of 150** |
| Hair strands (synthetic, moving) | not run | 0.2 % / 0 of 150 | **0.1 % / 0 of 150** |
| Belly one-go (moving, 90 frames) | not run | 1.5 % / 0 of 90 | **1.5 % / 0 of 90** |

The failing frames on the fast mole are frames where the mark is not removed at all: the engine refuses to paint them, and the mole
shows at full strength. We did not see a "split" through the middle in these clips. If the split you saw is a different pattern, the
frames from your job will show it (see section 5).

### The variants that were compared (all on the same clips)

| Variant | Fast mole | Fast tattoo | Hair | Belly one-go |
|---|---|---|---|---|
| A: as shipped in 117 | 35.9 / 50 | 16.8 / 12 | 0.2 / 0 | 1.5 / 0 |
| B: the finder kept on for every locked mark | 15.6 / 24 | 1.1 / 3 | 0.1 / 0 | **72.4 / 68 (broken)** |
| C: BodyLock off | 30.0 / 44 | 2.4 / 0 | 1.0 / 0 | **66.8 / 56 (broken)** |
| D1: local placement off (BodyLock on) | 35.9 / 50 | 16.8 / 12 | — | — (same as A on the fast clips) |
| **E = 118: finder kept for painted marks, not for the one-go belly positions** | **15.6 / 24** | **1.1 / 3** | **0.1 / 0** | **1.5 / 0** |

- The finder is what holds fast tattoos and moles, but it breaks the one-go belly positions (B: 72 %).
- BodyLock is what makes the one-go belly work (C: 67 % with it off), and it is small for the hair.
- Local placement (D1) made no difference on these clips.

## 2. The change

- A painted mark's track is marked `keep_finder` (every group except the one-go `belly` group). In `apply()` the mark finder is used on
  a locked track only when `keep_finder` is set, and `lock_finder` (default on) allows it.
- Nothing else changes: BodyLock, the ring pose, the seal of the hole and the drop logic still run for dark marks, as in 117.
- The one-go belly positions keep the v11.2.117 behaviour exactly (group `belly`).

## 3. What is not fixed

- **Fast moles still lose the hold** on 24 of 150 frames (15.6 % of the mark visible), where the v11.2.112 base holds them (0.4 %). The
  difference is not isolated: turning BodyLock off (C) and the local placement off (D1) did not close it. If fast moles matter most,
  the base engine is better on that clip. That needs a decision: a v11.2.112-based pack, or a further engine investigation.
- The centre opening itself was not reproduced as a split in these clips. The synthetic clips show it as frames where the mark is not
  removed, not as a seam through the middle.

## 4. Verification

- `video_test.py` (the pack's own engine, the Space's LaMa model): the four clip sets above, the final files of this pack.
- `unit117_onego.py` on the 118 pack: all 20 checks pass (the one-go job positions, the navel, the shorts, the preview route).
- Not run: the older unit suites `unit96`-`unit111` (written for our v11.2.1xx engine, not for the upload's BodyLock path); a full job on
  the Space (no insightface models here); the variant D2 (local placement and BodyLock both off), which was stopped to save time.

## 5. Install and what to send

- **Space:** upload the pack files as for 117 (the same files as 117; `swap_engine.py` and `config.py` changed). The Space rebuilds only if the
  requirements changed (they did not).
- Log at start: `Phoenix v11.2.118`, `swap_engine aequus-1.2.118-lockfinder`.
- If a fast-moving mole or tattoo still opens up, send the frame numbers from the job log (`PATCH-EVENTS`) and a short clip; that is what
  decides the next step.
