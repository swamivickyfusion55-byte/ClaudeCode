# SDOS-097 — PatchViews: a mark painted at several positions; no flicker in full contact

**Pack:** v11.2.97 "PatchViews" (v11.2.96 PatchRemoval + this). Engine `aequus-1.2.97-patchviews`. **App 1.2.26** (APK Builder 1.2.26).

**Reported:**
1. "In the same position it comes out okay, just a minor opening of the patch in very few frames. As soon as the position of the person changes it doesn't work. Add an option to mark on multiple positions. Also add zoom-in for correct marking. The buttons (Freeze etc.) are almost out of the screen. Make the patch editor landscape only (currently it goes to the main screen of the app)."
2. "Occlusion management is much better, but there is minor flicker between original and replacement during full contact, like a smooch."

All tests use synthetic clips made from procedural skin and public photos; your footage was not used.

---

## 1. Patch Removal: several positions of the same mark

**Why it failed when the person moved.** Each mark was followed from one painted frame. When the person turns, or the shot changes, the area no longer looks like that frame. It is then either lost, or found and covered by a fill made for a different angle and scale.

**Now.** In the editor, after painting the mark, tap **+ Position**. Move to a frame where the person has moved or turned and paint the same mark there. That gives up to 6 positions per mark and 12 in total.

On the Space, every position is anchored on its own frame and followed through the whole video. On each frame, the position **nearest in time** that is followed there is used. If that one doesn't fit the frame, the next one is tried. A position with clearly lower tracking confidence gives way to a better one.

**The "opening" on a few frames:**
- **One-frame losses:** gaps of 1–3 frames between followed frames are now bridged (only between poses that agree; the existing safeguards still refuse a bridged pose that doesn't fit).
- **The mark's edge:** dark mark pixels just outside the filled area are covered with the painted frame's skin, relit. This is only done where nothing is in front of the mark.

**Test.** A shot change at frame 75 shows the same skin turned 35°, 1.7× larger and darker.

| Clip | 1 painted position | 2 positions (frames 20 and 110) |
|---|---|---|
| tattoo | 2.1 % of the mark left, missed on 2 frames | **1.4 %**, missed on 1 frame |
| mole on the cheek (public photo) | 7.3 % left, 209 px changed away from the mark | **4.1 %**, **3 px** |

The second position mostly improves the fill: after the cut, the fill matches the size and angle seen there.

On real footage the gain should be larger: there a 3-D turn changes how the area looks, not just its size and angle. Here even one position was found again after the cut.

**Full jobs from the zip** (the real render path; tattoo on procedural skin, no face):

| | Mark left | Frames where the mark stayed |
|---|---|---|
| 1 position | 4.7 % | 2 of 150 |
| 2 positions | 4.2 % | 2 of 150 |

With a face swap in the same job (the mole on the cheek), both positions anchored and together followed the mark on 150 of 150 frames.

**Earlier clips: identical to v11.2.96** on the final engine, run from the zip:

| Clip | Mark left | Frames where the mark stayed |
|---|---|---|
| procedural mole | 10.6 % | 13 of 136 |
| mole on the cheek | 4.0 % | 6 of 136 |
| tattoo | 5.0 % | 6 of 140 |
| birthmark | 17.3 % | 21 of 140 |
| mole on the lip edge | 7.2 % | 6 of 136 |

Nothing changed away from the mark, and objects stay untouched (0.0–1.7 %). The gap bridging and the edge repair seldom trigger on these synthetic clips (the edge repair acted on 3–27 frames per job). They target the openings seen on real footage and cost nothing where they don't trigger.

**A wider halo was measured and not taken.** A 3-px halo cut the frames where the mark stayed on the procedural mole (13 → 10) and the birthmark (21 → 13). But it left more of the mark on the photo clips (cheek 4.0 → 7.1 %, lip edge 7.2 → 8.4 %, tattoo 5.0 → 5.7 %). It stays at 2 px; `halo_px` in `swap_engine.py` is the tunable.

## 2. App 1.2.26: the editor

- **Landscape only, and it no longer closes when the phone turns.**
  - Before, the screen was rebuilt on rotation and the editor's state was lost, so it dropped back to the main screen.
  - Now the app handles the rotation itself (`configChanges`), and everything in the editor is kept in the ViewModel: frame, strokes, positions, preview and zoom.
- **Layout:** the picture on the left, the controls in a side panel on the right. **Preview**, **+ Position** and **Freeze** are fixed at the bottom of the panel, inside the safe area, so they can no longer fall under the navigation bar.
- **Zoom:**
  - **+ / − / Fit** buttons, as well as two-finger pinch, with a zoom badge.
  - While painting, a **3× magnifier** in the corner shows what is under the finger, with the brush outline.
- **Frame stepping:** one frame / one second back and forward, for picking the exact frame.
- Positions already added are listed; "Remove the last added position" undoes one.
- **Before / After** toggle on the preview.

## 3. Full contact (a smooch): the flicker

A new synthetic clip, `kiss_smooch`: heads tilted towards each other, mouths and noses pressed together (about 60 % of A behind B), a rhythmic push, and a little detector-size jitter.

**The face behind (A) was the one flickering.** Its painted share on its own visible face jumped between 58 and 76 % from frame to frame as contact began (largest jump 11.9 points), then hovered around 60 %. The face in front stayed steady at 89 %.

**The cause, found by switching mechanisms off one at a time:**

| Variant | A's frame-to-frame change |
|---|---|
| v11.2.96 | 1.31 (largest 11.9) |
| edge novelty off | 1.17 |
| XSeg limited to the mouth | 1.28 |
| the clamp beside the front face off | 1.31 |
| colour occluder off | 1.31 |
| **XSeg off** | **0.46 (largest 2.3)**; A painted 80 % instead of 75 %, lowest frame 70 % instead of 58 % |

In A's crop, most of what the occluder net (XSeg) sees is the **other** face. Trained on one face per crop, it takes A's own visible strip for "not face", a little differently on every frame. Its cuts are applied after the temporal smoothing, so they flicker straight through.

**The fix:**
- Where another face overlaps a face, the net's cuts are ignored within that face's outline (forehead included) plus 8 % of the crop.
- In **full contact**, when that zone covers 15 % or more of the crop, the net is off for the whole face. The face-against-face rule (one face per pixel, held front/behind order) decides alone.
- Faces not touching another face are unchanged: objects are still cut on the frame they arrive.

A first version only calmed the net within that zone next to the other face. It changed nothing (the per-frame series was identical), because the net's cuts fell on the face behind's own visible strip, not on the contact line. Hence the whole-face rule in full contact.

Results:

| Clip | Face behind: frame-to-frame change, mean / largest jump (v11.2.95 → v11.2.97) | Face behind painted, average (lowest) | Front face painted |
|---|---|---|---|
| **smooch** | 1.31 / 11.9 → **0.48 / 2.3** | 75 % (58) → **80 % (70)** | 89 % → 89 % |
| deep overlap | 0.89 / 14.8 → **0.54 / 2.7** | 77 % (63) → 78 % (67) | 89 % → 89 % |
| face mostly hidden | 1.05 / 5.7 → 0.99 / 5.6 | 68 % (46) → 69 % (46) | 89 % → 89 % |
| cheek to cheek | 0.16 / 2.6 → 0.16 / 2.6 | 88 % → 88 % | 90 % → 90 % |

On all four clips:
- **0 reverted frames.**
- **The face behind does not show on the front face** (0.3 % at most), as in v11.2.95.
- **Overall flicker against the footage** ×1.03–1.11, as before.

Objects, run from the zip: hand over the eye 0.4 % still painted while it holds, the same as v11.2.95.

## 4. Limits

- **Synthetic tests only.** The contact clips are two cut-out photos; the shot-change clip is a flat rotation and scale, not a 3-D turn.
- **During full contact the occluder net is off** for the touching faces. A hand or a cup between two kissing faces is then cut only by the colour occluder and the face-against-face rule.
- **The face behind is still only partly painted in a full kiss.** Its visible strip is mostly forehead and outer cheek, where the swap mask is soft. In the clip about 70 % of that strip is painted, steadily.
- **Painted positions have to be added by hand** (+ Position). A turn far from every painted position can still lose the mark for those frames.

## 5. Verification

- `verification/patchremoval/unit97.py`: 16 checks, all pass. They cover:
  - the nearest painted position is used, the next one if it refuses, and a low-confidence one gives way;
  - gap bridging: short gaps are filled, long gaps and jumps are not;
  - the job settings: groups, limits, and app 1.2.25 entries without a group;
  - contact calm: light contact (zone only, far cuts still applied), full contact (net off), and no other face (unchanged).
- Earlier suites on this pack: unit96 18/18, unit95 17/17, unit94 29/29, unit90 all pass.
- Clips:
  - `make_patchclip.py OUT cut`: the shot-change clips. `video_test.py --views=2` runs them with two positions.
  - `make_kiss.py smooch`: the full-contact clip. `contact_face.py` with `SERIES=1` prints per-frame paint and dips.
  - `bench_pipeline.py patches=… views=2`: a full job with two positions.
- App 1.2.26: compile check 0 errors; the builder's static audit passes 44/44; not yet run on a phone.

## 6. Install

**Space:**
- Upload the 8 files, or use "Push to all Spaces".
- Log lines: `Phoenix v11.2.97`, `swap_engine aequus-1.2.97-patchviews`.
- New log text:
  - `PATCH-REMOVAL patch 1.2: anchored on frame …`, then `patch 1: 2 positions, together followed on …`.
  - `occluder net calmed next to the other face on n (off for the whole face in full contact on m)`.

**App:** upload the 6 files of `Phoenix_APK_Builder_1.2.26.zip` to the APK Builder Space, build, and install over 1.2.25.

App 1.2.25 keeps working with this Space: each of its marks is one position.
