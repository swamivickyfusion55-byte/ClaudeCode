# SDOS-096 — Patch Removal: paint a mark on one frame, it is gone from the whole video

**Pack:** v11.2.96 "PatchRemoval" (v11.2.95 OcclusionPro + Patch Removal). Engine `aequus-1.2.96-patchremoval`. **App 1.2.25** (APK Builder 1.2.25).
**Asked:** "Add an optional Patch Removal button to remove any patch on any part of the body by finger touch (erase-like). It should identify exactly where the finger moved and make its colour and texture exactly like the skin around it: a tattoo, a mole or any dark mark, of any shape. Once identified and erased on a frame, show an erased preview. Once frozen, the app should remove that patch across the video, even if that part moves to another position."

All tests use synthetic clips made from procedural skin and public photos; no user footage.

---

## 1. How it works

**In the app (1.2.25).** Under the target video there is a new **Patch Removal (optional)** button. It opens a full-screen editor:
1. **Choose the frame.** Slide to a frame where the mark is clearly visible. The app grabs that exact frame on the phone (not the nearest key frame).
2. **Paint.** Paint over the mark with one finger. Brush sizes are S, M and L, and Undo and Clear are available. For a small mole, pinch with two fingers to zoom and pan; **Fit** resets the view.
3. **Preview.** The frame and the painted mask go to the Space, which returns the frame with the mark removed. Use **Show before / Show after** to compare. If it's not right, paint more or Undo, then preview again.
4. **Freeze.** The mark is kept. Up to 4 marks per video, each removable again from the list. Choosing another video clears them.

Frozen marks travel with every job on that video: in a split job, every part gets them.

**On the Space (v11.2.96)**, before the face swap sees each frame:

| Step | What happens |
|---|---|
| **Find the painted frame** | The app's frame is matched against the job's frames (near the time sent, then everywhere), and the painted mask is aligned to the video's pixels. If that frame is not in this video at all (another part of a split job), the painted **area** itself is searched for in the frames. |
| **Fill once** | The mark is filled **once**, on that frame, with **LaMa**, the standard open model for object removal (Suvorov et al., WACV 2022, Apache-2.0). LaMa continues skin texture and also lines that cross the mark (a lip line, a crease, a shadow edge). The finger's rim is widened by the dark edge of the mark if the finger missed it. Without LaMa, an exemplar fill (Criminisi et al. 2004, guided by a smooth fill) takes over. |
| **Follow the skin** | The area around the mark is tracked forwards and backwards through the video. Optical-flow points (forward-backward checked, RANSAC) give the motion. An ECC alignment of the painted frame's patch then fixes the position exactly, always against the painted frame, so it never drifts. When the area leaves the picture or is hidden, it is found again by ORB features and template matching. |
| **Remove on every frame** | The fill moves with the skin (move, turn, scale, shear). Only the change of **lighting**, measured on the skin around the mark, is added, and it is softened when the frame is motion-blurred. Because the texture is fixed to the skin, it cannot shimmer. |
| **Never paint the wrong thing** | Anything in front of the mark (a hand, a cup) does not look like the painted frame there and is left untouched. If the mark's own core is not where expected (covered, out of the picture, or a doubtful position), nothing is painted on that frame. |

New API route: `phoenix_patch_preview(frame, mask)` returns the erased frame and a JSON status.

Job settings: `"patches": [{"image_b64", "mask_b64", "t"}]`, where `t` is in seconds in the uploaded file (the app corrects for a local trim). At most 4 marks; bad input is refused with a reason.

## 2. Results

### Still frame (the preview)

| Clip | Mark's darkness against the skin, before → after | Texture in the fill vs the skin around it |
|---|---|---|
| mole on procedural skin | 47.4 → 1.0 | 1.49 vs 1.84 |
| tattoo, about 190×120 px | 59.3 → 0.8 | 1.83 vs 1.94 |
| birthmark | 21.0 → 0.6 | 1.85 vs 1.87 |

On the public photo (2× enlarged), LaMa against my first approach, a smooth fill plus copied texture:
- **Mole on the cheek:** LaMa's fill is seamless; the first approach left a faint dot pattern.
- **Mole on the lower-lip edge:** LaMa rebuilds the lip line and the teeth; the first approach smeared lip colour over the teeth.

The preview took about 5–10 s here on 2 CPUs. The first preview also downloads LaMa (92 MB) once.

### Video

Five 150-frame clips (5 s). Over the clip the mark:
- moves ±160 px, turns ±9°, scales ±10 % and changes brightness ±12 %;
- has an object pass over it (frames 40–60);
- swings right out of the picture and back at up to about 140 px per frame (frames 92–112).

| Clip | Mark left (average) | Frames where the mark stayed | Object painted over | Changed away from the mark |
|---|---|---|---|---|
| mole, procedural skin | 10.6 % | 13 of 136 | 0.0 % | 0 px |
| mole on the cheek (photo) | 4.0 % | 6 of 136 | 0.6 % | 0 px |
| tattoo | 5.0 % | 6 of 140 | 1.7 % | 0 px |
| birthmark (soft, light) | 17.4 % | 21 of 140 | 0.0 % | 0 px |
| mole on the lip edge (photo) | 7.2 % | 6 of 136 | 0.6 % | 0 px |
| tattoo, **second part of a split job** (painted frame not in it) | 7.4 % | 6 of 80 | 0.0 % | 0 px |
| mole on the cheek, second part | 5.9 % | 6 of 79 | 0.0 % | 1 frame (see §3) |

- **Where the misses come from:**
  - On every clip, the 6 frames of the out-and-back swing (about 140 px per frame, beyond what optical flow follows).
  - On the smooth procedural clips, the frames where the object covers most of the mark. There, the visible part of the mark stays for those frames.
- **Stability:** the frame-to-frame change on the removed mark beyond the clean clip's own is 3–6 grey levels. On the lip-edge clip it is 18, because LaMa's rebuilt lip line differs slightly from the true one; the frames look steady.
- **Full jobs** (the real render path, encoder included; the final runs are from the zip):
  - Tattoo: missed on 6 of 140 frames (the swing frames). Birthmark: missed on 20 of 140. Both match the engine test. Procedural mole: missed on 13 of 136 before the fix below, 10 after (planning re-run).
  - Mole on the cheek with **a face swap in the same job**: faces swapped on 150 of 150 frames, the mark removed on 133.
  - A fix found by these runs: the job first kept the painted frame as a JPEG. On the soft birthmark, the JPEG artefacts made the object check hold the fill back on 43 frames (54 missed). The frames around the painted time are now kept lossless: 1 frame held back, 20 missed.
- **Cost:**
  - Mole job: planning 19 s, then 14 s of fill and 6 s of tracking over the job, about 4 % of the 891 s job.
  - A large tattoo: about 56 s of planning and 32 s of fill for 150 frames.
- **Earlier behaviour unchanged:**
  - Without patches, the swap path is unchanged: v11.2.95's unit checks pass on this pack (unit95 17/17, unit94 29/29, unit90).
  - The hand-over-the-eye job was rerun from the zip: 0.4 % of the hand still painted while it holds, the same as v11.2.95.

## 3. Limits (read before using it on a client job)

- **Synthetic tests only:** procedural skin, and a public photo moved as a whole. On a real arm or neck the skin also bends and twists. The fill follows a straight (affine) motion of the area around the mark, so a mark on skin that folds may show a thin edge on some frames.
- **The fill is made from the painted frame.** If the body part later turns far (well beyond about 30°), the lighting is still adjusted but the texture is the painted frame's. Pick a frame where the mark faces the camera.
- **Very fast motion** (more than about 100 px per frame) is not followed for those frames; the mark shows there. Motion blur at such speed usually hides it anyway.
- **Heavy cover on smooth skin:** if an object covers most of a mark, its visible remainder can stay visible for those frames. The engine prefers that to painting skin where it shouldn't.
- **Large marks:** marks wider than about 200 px (at 720p) are filled at LaMa's 512-px working size, so fine grain is added back to match the skin around them.
- On the split-part test, one frame of the test clip showed a mirrored copy of the face, made by the clip's border reflection, and the engine treated that copy as the mark. Real footage has no such copy.
- The mark must be on a frame **of the uploaded video**. A Space older than v11.2.96 ignores the marks; the app's preview says so.

## 4. Verification

In `verification/patchremoval/`:
- `make_patchclip.py`: the clips and their clean twins.
- `video_test.py`: the engine on whole clips; `--start=N` simulates a later part of a split job.
- `perframe.py`: per-frame residue and stray changes.
- `bench_eval.py`: the full job output against the clean twin.
- `unit96.py`: 18 checks, all pass. They cover:
  - the preview removing the mark and changing nothing elsewhere;
  - the LaMa path and the fallback without it, including a crashing model;
  - the app's opaque mask PNG, alpha masks, and refused bad input;
  - the preview route and the job config.

App 1.2.25:
- compiles with 0 errors (kotlinc 2.0.21 + Compose plugin against Android 15 classes; a missing `ViewModelProvider` compile stub was added to the verification stubs);
- the builder's static audit passes 44/44;
- not yet run on a phone.

## 5. Install

**Space:**
- Upload the 8 files (`app.py config.py core_pipeline.py phoenix_api_adapter.py swap_engine.py packages.txt requirements.txt README.md`), or use "Push to all Spaces" in the app.
- Log lines: `Phoenix v11.2.96`, `swap_engine aequus-1.2.96-patchremoval`.
- For a job with marks, the log also shows `PATCH-REMOVAL patch n: anchored on frame …` and `PATCH-REMOVAL: n mark(s), … removed on F frames …`.
- LaMa is downloaded at the first preview or job that uses it: 92 MB, checked by SHA-256, URL in `config.py` `LAMA_URL`.

**App:**
- Upload the 6 files of `Phoenix_APK_Builder_1.2.25.zip` to the APK Builder Space and press Build APK.
- Install over 1.2.24; do not uninstall.
