# SDOS-098 — PatchStrong: no faint darkness under the mark, nothing outside the painted area, every button on screen

**Pack:** v11.2.98 "PatchStrong" (v11.2.97 PatchViews + this). Engine `aequus-1.2.98-patchstrong`. **App 1.2.27** (APK Builder 1.2.27).

**Reported:**
- "The Freeze button is still going out of the screen; keep all buttons well within the screen."
- "I intended to keep both portrait and landscape for the patch removal screen."
- "Enhance the intensity of the paint marginally, as a bit of leftover (very light) darkness remains."
- "Under the mark but within the marked area. In any case the area outside the mark need not be changed."

All tests use synthetic clips made from procedural skin and public photos; no user footage.

---

## 1. The app (1.2.27): the editor

**Why Freeze went off the screen.** The editor was a separate dialog window. A dialog window does not get reliable system-bar insets on every phone. In landscape the navigation bar sits on the right, so the right-hand button ended up under it.

**Now:**
- The editor is a full screen inside the app's own window, where the insets are reliable. The whole screen sits inside the safe drawing area with a 12 dp margin.
- The Preview / + Position / Freeze row is never inside a scroll area.
- Rows of small buttons scroll sideways on a narrow screen instead of being cut off.
- The back button closes the editor.

**Portrait and landscape both work:**
- **Portrait:** picture on top, controls below, the three buttons in one row at the bottom.
- **Landscape:** picture on the left, controls on the right.

Turning the phone keeps everything: the frame, the paint, the positions, the preview and the zoom. The landscape lock of 1.2.26 is gone.

## 2. The faint darkness under the mark

**Where it was.** On the frames where the mark was removed, the result was still darker than the clean skin under the mark (synthetic clips, lightness 0–255).

| Clip | Inside the mark (darker than clean) | Mark's rim | Around the mark (inside the painted area) |
|---|---|---|---|
| procedural mole | +1.3 | +1.4 | +0.6 |
| mole on the cheek (photo) | **+2.8** | −2.4 | −9.8 |

So on the cheek there was a darker spot where the mark was, inside a fill that is otherwise slightly lighter. In the preview (the fill itself) there was none: within ±0.5 on the skin clips. The darkness builds up on the other frames, as the fill is carried with the skin and relit.

**The fix: a per-frame lift, inside the painted area only.**
- On every frame, the level over the mark's core (the pixels that were dark) is compared with the fill around it in the painted area and the untouched skin just outside.
- Where the core is darker, it is brought up to that level.
- It follows the spot's shape, brightens only, by at most 6 levels, so a crease or a lip line inside stays.

**Results** (the mark's spot; synthetic test spot of −4.0 levels → −0.6):

| Clip | v11.2.97 | **v11.2.98** |
|---|---|---|
| procedural mole | +1.31 | **+1.08** |
| tattoo | +1.15 | **+0.89** |
| birthmark | +1.21 | **+1.08** |
| mole on the cheek | +2.78 | **−1.34** (the spot is gone; slightly lighter) |
| mole on the lip edge | +1.41 | +2.64 (see §4) |

- On the procedural clips about +1 level remains across the whole painted area, not as a spot. The encoding offset between the clean and the marked clip is exactly 0, so this is real but at the limit of what can be seen.
- The mark left on average and the frames where it stayed are the same as v11.2.97.
- Speed is the same: the lift is solved at reduced size for big marks (the tattoo is 144 ms per frame against 134 ms).

## 3. Nothing outside the painted area changes

Turned off, because they changed pixels outside the area you painted:
- the **edge repair** of v11.2.97, which covered dark pixels just outside the painted area;
- the **extension over a missed rim**, which added dark pixels touching the painted area;
- the **outward feather**, which faded the fill into the skin around.

The fill's soft edge now lies inside the painted area. What remains is the 2-px halo around your paint, which keeps the edge seamless.

I tried a 1-px halo. It made the cheek fill lighter (the mark measured −5.4 %, overshot) and let a wrong position through on two frames, so it stays at 2 px.

Pixels changed outside the painted area (beyond 1 px), on the painted frame:

| | v11.2.97 | **v11.2.98** |
|---|---|---|
| preview | 21–227 px per clip | 9–98 px, only the halo ring |
| job | 3–84 px | 2–38 px |

**What this costs:**
- On the lip-edge clip the mark's soft rim reaches past the painted area. The rim extension used to cover it (6.9 % of the mark left); now it stays (8.7 %).
- **Paint a little past the mark's edge.** The editor's hint now says so.

## 4. Limits

- **Synthetic tests only.**
- **The lift cannot tell a ghost from a real darker shade under the mark.** On the cheek it made the spot 1.3 levels lighter than the true skin; the 6-level cap limits it.
- **Lip-edge clip:** the residual inside the mark rose (+1.41 → +2.64), because without the rim extension the part of the mark beyond the paint stays.
- On the out-and-back swing of the test clips, a mirrored copy of the face created by the clip's own border reflection is sometimes treated as the mark (up to 130 px on 1–3 frames). Real footage has no such copy.
- App 1.2.27: compile check 0 errors, audit 44/44; not yet run on a phone.

## 5. Verification

- `verification/patchremoval/unit98.py`: 12 checks, all pass. They cover:
  - the lift: a faint spot is lifted (−4.0 → −0.6), no spot gives no lift, it is capped, a lighter spot is not darkened, a big mark is lifted at reduced size;
  - nothing changes outside the painted area: the lift, the fill alpha, and the preview on a clip (0 px beyond the halo);
  - the edge repair and rim extension are off.
- Earlier suites on this pack: unit97 16/16, unit96 18/18, unit95 17/17, unit94 29/29, unit90 all pass.
- New scripts:
  - `residual.py`: the leftover inside the mark, on its rim and around it, on removed frames;
  - `preview_residual.py`: the preview's leftover and its changes outside the painted area;
  - `video_test.py` also reports the pixels changed outside the painted area on the painted frame.
- Full jobs from the zip (the real render path):
  - procedural mole: 8.0 % of the mark left on average, missed on 6 of 136 frames (the earlier full job missed 13), with 4 short gaps bridged;
  - cheek mole with a face swap in the same job: faces on 150 of 150 frames, the mark removed on 133.

## 6. Install

**Space:**
- Upload the 8 files, or use "Push to all Spaces".
- Log lines: `Phoenix v11.2.98`, `swap_engine aequus-1.2.98-patchstrong`.

**App:**
- Upload the 6 files of `Phoenix_APK_Builder_1.2.27.zip` to the APK Builder Space and build.
- Install over 1.2.26.
