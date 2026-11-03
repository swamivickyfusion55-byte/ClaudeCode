# SDOS-101 — PatchSimple: simpler Patch Removal with a stronger cover; no more silent stall at 11 %; a second job finds an idle Space

**Pack:** v11.2.101 "PatchSimple". Engine `aequus-1.2.101-patchsimple`. **App 1.2.29** (APK Builder 1.2.29).
Replaces v11.2.100 "Limits" (whose engine still kept only 4 marks / 12 positions - see section 5).

**Reported:**
1. "Change the patch removal logic to the earlier one, more simplified and more time efficient; just make appropriate fixes so that the intensity of the patch is higher to cover the left over of the mole or tattoo."
2. "Patch removal in 10 frames with subject in different positions" (not 10 marks).
3. "The video processing is stuck at 11 % for a long time."
4. "I have 20 Spaces, one job is running split into 10, however I am not able to start another job." (with the Space log of the stuck job)

All tests use synthetic clips (procedural skin, public photos); no user footage.

---

## 1. The simpler engine

"The earlier one" is taken as **v11.2.98's** per-frame logic (the version just before the heavy v11.2.99 blend), because it keeps the several-positions marks that the app 1.2.26+ sends.

**Taken out (v11.2.99):** the per-frame local light field (a membrane solved across the painted area), the texture re-adding, and the lift judged under the light field. These were the slow part.

**Kept from v11.2.99, on purpose:**
- the **lighting-free tracker** (compares the mark with the lighting divided out, with 2 px of tolerance for bending skin). It costs no time (planning time is the same) and it is what stops the mark showing when a shadow moves over it: on the moving-shadow clip v11.2.98 lost the mark on 50 of 150 frames, now 0.
- the **kiss hold** (pipeline): a face lost while touching another face is held, not dropped.
- one switch undoes the tracker: `PATCH_CFG["detail_track"] = False` in `swap_engine.py`.

**Per-frame cost** of the painted area (one mark, 2 cores, synthetic clips; machine idle):

| | v11.2.98 | v11.2.99 | **v11.2.101** |
|---|---|---|---|
| mole (about 16 px) | 15 ms | 26 ms | **about 20 ms** |
| big tattoo (150 x 90 px) | 141 ms | 161 ms | **about 135 ms** |

The simple blend is back; the small extra over v11.2.98 is the cover step below (about 0-7 ms).

## 2. The stronger cover ("intensity")

The painted stroke covers plain skin around the mark, and in the current frame that skin is there to be seen. New per-frame step `_pr_cover`:
- over those margin pixels (inside the painted area, 2 px from its edge, away from the mark's core, not in front of an object) the fill is compared with the frame;
- where the fill is darker than the frame's own skin, its level is brought **up** to the frame's, smoothly and also over the mark itself - and a little over: `cover = 1.5` (half of the deficit extra), at most 10 levels;
- brightening only; it fades to nothing at the painted area's edge; nothing outside the painted area changes;
- it comes before the v11.2.98 lift, which stays.

**Measured** (synthetic; lightness 0-255, + = darker than the clean clip):

| Clip | inside the mark: cover off -> **on** | on the stroke around it: off -> **on** |
|---|---|---|
| procedural mole | +1.09 -> **+0.93** | +0.51 -> **+0.20** |
| mole shifting 1-3 px against the skin | +0.73 -> **+0.69** | +0.42 -> **+0.27** |
| procedural birthmark | +1.07 -> +1.06 | +1.01 -> +0.95 |
| photo mole (fill already lighter than the truth) | unchanged (never darkened) | unchanged |

**What this does and does not show.** On these clips the leftover is small and flat: the darkness is the same +0.6-0.7 L 18-24 px away, outside the painted area, where nothing is painted (a measuring offset between the two encodings). There is no ghost spot of the mark to remove; the cover cuts the darkness on the stroke around the mark by about 40-60 % and inside the mole by 5-15 %. **A faint residue under the paint on real footage may have a cause these clips do not model** (relief of a real mole, skin sliding, auto-exposure, compression). The cover is the stronger safety net for it; it cannot be proven here. If a faint leftover is still seen, send the job log (text only).

## 3. Positions and marks

- One mark can be painted at **up to 10 positions** (app 1.2.28+), **20 positions** and **10 marks** in all.
- v11.2.100 raised these only in the app and in the Space's settings check; the engine kept its own caps of 4 marks / 12 positions and **silently dropped** the rest (found by running 10 marks through it: 4 were followed). Fixed: engine caps 10 / 20, and a `WARNING: ... left out` line in the job log whenever anything is dropped. One mark at up to 10 positions worked in v11.2.100.
- Test (engine, a clip with 10 marks, from the zip): all 10 followed and removed (2.3-10.4 % of each mark left, 2-16 of 150 frames missed). Full job through the pipeline: 10 anchored, removed on 1463 frames (10 marks x about 146).

## 4. The stall at 11 %

**From the pasted Space log** (job D4734B93, 429 MB video, `patches=10`, Space v11.2.100):
- 17:55:28 model load; **LaMa only loads at 18:00:44**: five minutes went on reading the whole video into the patch cache.
- After that every one of the 10 painted positions was **followed through the entire video** (forward and back), one after the other, with **no progress message and no way to cancel** (the Stop button is not looked at while following). For a long video that is hours. The job shows 11 % because 11 % is where "finding and following marks" is announced.
- Two jobs (17:55:26 and 17:56:57, both with 10 positions) were running on this Space at once and shared its 2 CPU cores.

**Fix (Space v11.2.101):**
- **Each position is followed only through its own stretch**: from the previous painted position (and 45 frames before it) to the next (and 45 frames past it); the first goes back to the start, the last to the end. Ten positions no longer cost ten whole-video passes. (Test, 10 positions on a 150-frame clip: planning 104 s -> 71 s with identical result: 2.8 % of the mark left, 136 of 150 frames covered. The saving grows with the length of the video: each position used to cost the whole video; now about two stretches.)
- **Progress**: the job message updates about once a second ("Patch removal: following mark 2 of 3 - 45 % - about 3 min left"). The percentage stays at 11 % during this step, but it is no longer silent.
- **Cancel** is checked on every frame of the following (a cancel took 4 s in the test, not the whole plan).
- The frames of a position found away from the painted time are fetched in one pass over the video, not by decoding from the start for each.
- A warning in the Space log when a job starts on a Space that is already running one.

**Not changed:** reading the whole video into the cache first still takes minutes on a very long video (5 min for 429 MB in the log). A split job reads only its own part.
**For the job that is stuck now:** a Space on v11.2.100 ignores Stop while it follows marks; only restarting that Space ends it (the job is lost). Update Spaces only when idle.

## 5. App 1.2.29: a second job when one is split

- **Cause (from the app code and the log):** a new job went to the **selected** Space, which the split had already used for part 1 (the log shows two jobs on one Space), and a split counted its free Spaces from the phone's own History, compared by raw address text (a trailing "/" made a busy Space look free) and counting entries in states such as "Cancelled" or "Error" as busy for ever. A split that was still cutting and uploading its later parts also left their Spaces free-looking.
- **Now:**
  - before a job starts, every Space is asked whether it is really idle (`/phoenix/busy`, v11.2.92+; the phone's knowledge when a Space does not answer); addresses are compared normalised;
  - one job: the selected Space if idle, else the first idle one ("Space 1 is busy - using Space 11");
  - a split uses as many idle Spaces as it needs; with fewer it splits into that many parts and says so; "Every Space is busy" only when none is idle;
  - the Spaces a split has picked are held until each part has started;
  - terminal History states now include Cancelled, Error, Stopped, Expired, Interrupted, Downloading.
- Compile check: 0 errors (kotlinc 2.0.21 + Compose plugin). **Not run on a phone.**

## 6. Limits

- Synthetic tests only; the faint-leftover cause on real footage is not reproduced (section 2).
- Under a moving shadow the fill follows the shadow less closely than v11.2.99's relight (shadow test: error about 3 levels against 2; the shadowed tattoo clip 3.8 L against 2.0-2.6 L): the mark is still covered on every frame (0 missed), the tone of the patch matches a little less. The fill's fine texture is the inpainter's own (about 50-80 % of the surrounding skin on photo skin) - v11.2.99's texture step is not in this build.
- The Space still accepts a second job while one runs (it only warns); the app now avoids it.
- The adapter's new warning line is inside a try/except and was not exercised by a test.

## 7. Verification (all from the zip's own files)

- `unit101.py` 21/21 (new: lighting-free view, stroke-margin cover, tracking under a shadow, the planner's bounded follow / progress / cancel, engine caps = config limits, the kiss hold); `unit98` 11/11, `unit97` 17/17, `unit96` 18/18 (their limit checks now follow the configured limits).
- Clips: moving shadow + bending 0 missed frames; mole shifting 1-3 px 0.1 % left; two positions across a cut 2.0 % / 1.4 % left.
- 10 marks: engine test and full pipeline job (above). 10 positions: `views_test.py`.
- Kiss with the detector blind for 40 frames of contact: `REVERTS=0`, as in v11.2.99.
- New scripts: `multi_test.py`, `views_test.py`; `make_patchclip.py` gains `jit` (mark shifting against the skin) and `multi` (10 marks).

## 8. Install

- **Space:** upload the 8 files (or "Push to all Spaces", which skips busy Spaces). Log: `Phoenix v11.2.101`, `swap_engine aequus-1.2.101-patchsimple`.
- **App:** upload the 6 files of `Phoenix_APK_Builder_1.2.29.zip` to the APK Builder Space and build; install over 1.2.28.
