# SDOS-087 — Restore: the face swap of v11.2.83 is back; the eating changes of v11.2.84–87 are taken out

**Pack:** v11.2.88 "Restore". Engine `aequus-1.2.88-restore`.
**Reported (after 11.2.87):** "The swap quality has worsened, in an attempt to solve the scenario of eating. Check whether you introduced any bug and revert the base swap logic to the latest one of yesterday; you may keep the confident corrections of today."

I have not seen your footage. I used your job log (no video) and synthetic clips.

---

## 1. What went wrong (from your log)

```
job 1 (2 people, 246 frames):    OCCLUDER: gate cut something on 402 of 492 face-frames ... attached part taken in on 195
job 2 (1 person, 3122 frames):   OCCLUDER: gate cut something on 1744 of 1915 face-frames; whole-object lock added ... mean +19.5% of the crop
```

The "object in front of the face" detector added for eating (v11.2.84 onward) **fired on 82 % and 91 % of all face-frames** of your two jobs. Each time it handed part of the face back to the original footage: on average about a fifth of the face crop in job 2. Unless someone is eating in nearly every frame, those are false cuts: patches of the original face showing through the swap. That is the drop in quality you see. My tests had found almost no false cuts. They used six still photographs with synthetic disturbances, which is clearly not representative of your footage.

The same detector also fires when two faces touch. On my synthetic kissing clip (no food at all) v11.2.84–87 changed up to 21–28 % of the face area in the worst frame, against v11.2.83.

## 2. Bugs found

| | Since | Effect |
|---|---|---|
| Eating detector fires on normal frames (above) | v11.2.84 | patches of the original face; the main cause |
| Cheek smoothing (`cheek_ema`) is not reset at a shot cut or when a face is picked up again | v11.2.85 | for ~4 frames after each cut the previous shot's cheeks are blended in; during movement the cheeks lag the rest of the face |
| The frame-space edge data was passed through the shared compositor object | v11.2.87 | wrong mask on the other frame **only if** `PHOENIX_VIDEO_WORKERS=2` (default is 1, so not on your Space) |

## 3. What this pack is

**The face swap, the tracker and the compositor are exactly v11.2.83** (14 Oct, the last pack before the eating detector was switched on for real faces). On faces it gives the same output as v11.2.81, the version you called excellent. Only the version strings differ. Checked:

- `core_pipeline.py` is byte-identical to v11.2.83; `swap_engine.py` and `config.py` differ only in `ENGINE_VERSION`, `VERSION` and `BUILD`.
- Composited output, real compositor, two synthetic two-person clips (240 frames): **identical to v11.2.83 on every pixel**, and v11.2.83 identical to v11.2.81 on the same clips.
- Real pipeline run end to end on a two-person clip: see §5.

**Kept from today: nothing that changes the output.** Every change of 15 Oct (v11.2.85–87) and the detector switch-on of v11.2.84 either works through the eating detector or touches every face on every frame (cheek smoothing). Your log shows the detector is not reliable on your footage, so none of them qualifies as a confident correction. The kissing logic of v11.2.87 (who is in front, size held during contact) was only ever tested on synthetic clips; I left it out too, rather than ship it on an untested base.

## 4. What you lose

The eating improvements of v11.2.84–87 (object cut out over the lips and cheeks). If you need them for a particular clip, keep v11.2.87 on a separate Space and render only eating scenes there. A version of the eating detector that only acts when it is sure needs testing on real eating footage; I will not ship one again on synthetic tests alone.

## 5. Verification

- Clean faces, real compositor, against v11.2.83: identical (0 of 240 frames differ).
- End-to-end smoke test (real detector, swapper, compositor; Balanced; two people touching, 120 frames): face painted on 120/120 frames, `REVERTS=0`, and **`gate cut something on 0 of 240 face-frames`**. v11.2.86/87 on the same clip: 96 of 240.
- The zip unpacks to exactly these files; all `.py` files compile.

## 6. Note on your job 2

`REVERTS=1202` there are frames in which **no face was detected at all** (detector empty on 1218 calls, 0.15 faces per frame on average): someone turned away, out of frame or too small. That is detection, not today's changes. The `LOPSIDED` warning says the second person was never found in that clip. If a face is clearly visible in some of those frames, send me the times.

## 7. Install

Upload the 8 files (`app.py config.py core_pipeline.py phoenix_api_adapter.py swap_engine.py packages.txt requirements.txt README.md`) to the **root** of the Space, commit, normal restart (or "Push to all Spaces" in the app). The startup log must show `Phoenix v11.2.88` and `swap_engine aequus-1.2.88-restore`. The `OCCLUDER:` line should now read close to 0 on clips without food (v11.2.83's detector does not fire on real faces). Do not restart a Space while a job is running on it.
