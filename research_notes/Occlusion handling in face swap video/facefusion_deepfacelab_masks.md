# Occlusion masks in FaceFusion 3.x and DeepFaceLab / DeepFaceLive (XSeg)

Method note: the FaceFusion, DeepFaceLab and DeepFaceLive repositories were cloned and read directly. FaceFusion: all tags 3.0.0 (2024-10-03) through 3.9.2 (2026-10-26, current master `decc6c9`). The FaceFusion occluder and parser ONNX files were downloaded from facefusion-assets and inspected and timed locally with onnxruntime 1.30 CPU on a 4-core Intel Xeon @ 2.10 GHz. The network blocked docs.facefusion.io, deepfakevfx.com and deepwiki.com, and the GitHub API for iperov repos, so community-guide coverage is thin. File links below point at the tag the fact was read from.

---

## Q1. FaceFusion mask types, CLI options, defaults and exact combination formulas

### Takeaway
FaceFusion 3.x has exactly four mask types: `box`, `occlusion`, `area` (since 3.3.0) and `region`. The default is `box` only, so occlusion handling is off unless the user enables it. Every enabled mask is a float map at the swap-crop resolution. They are combined with a pixel-wise `numpy.minimum.reduce` (intersection, not a product), clipped to 0..1, inverse-affine-warped and alpha-blended. The occlusion, area and region masks share one post-process, `(GaussianBlur(m, sigma=5).clip(0.5, 1) - 0.5) * 2`. That is a blur followed by a soft threshold at 0.5 which re-stretches 0.5..1 to 0..1. Occlusion is computed on the target crop before swapping. Region and area are computed on the swapped crop after swapping.

### Cited Findings
**Mask types and history**
- 3.0.0: `FaceMaskType = Literal['box', 'occlusion', 'region']`. There is no occluder or parser model choice: a single `dfl_xseg.onnx` and `bisenet_resnet_34.onnx` come from release `models-3.0.0`. 3.0.0 also has an internal `create_mouth_mask` (landmark hull, eroded with a (21,3) kernel, blurred sigmaX=1, sigmaY=15), used by the lip syncer. — [face_masker.py @3.0.0](https://github.com/facefusion/facefusion/blob/3.0.0/facefusion/face_masker.py), [typing.py @3.0.0](https://github.com/facefusion/facefusion/blob/3.0.0/facefusion/typing.py)
- 3.1.0 adds `FaceOccluderModel = Literal['xseg_1', 'xseg_2']` and `FaceParserModel = Literal['bisenet_resnet_18', 'bisenet_resnet_34']`. — [typing.py @3.1.0](https://github.com/facefusion/facefusion/blob/3.1.0/facefusion/typing.py)
- 3.2.0 adds `xseg_3` (commit 7f90ca7, 2025-01-10, "Add another xseg model"). — [types.py @3.2.0](https://github.com/facefusion/facefusion/blob/3.2.0/facefusion/types.py)
- 3.3.0 adds the `area` type and `FaceMaskArea = Literal['upper-face', 'lower-face', 'mouth']` (commits 2cad038 "Add landmark powered face areas" and 4d34a92 "Implement area mask, refactor box mask", June 2025). — [types.py @3.3.0](https://github.com/facefusion/facefusion/blob/3.3.0/facefusion/types.py)
- 3.4.0 adds `'many'` to `FaceOccluderModel`, from PR #910 "Introduce many to the occlusion mask". Its commit note reads "Then we use minimum". — [types.py @3.4.0](https://github.com/facefusion/facefusion/blob/3.4.0/facefusion/types.py), [PR #910](https://github.com/facefusion/facefusion/pull/910)
- 3.4.0 through 3.9.2 have no new mask types. The current set is `FaceOccluderModel = Literal['many','xseg_1','xseg_2','xseg_3']`, `FaceParserModel = Literal['bisenet_resnet_18','bisenet_resnet_34']`, `FaceMaskType = Literal['box','occlusion','area','region']`. — [types.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/types.py)

**CLI options and defaults (3.9.2; the same values since they were introduced)** — [program.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/program.py), [choices.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/choices.py)
- `--face-occluder-model`: default `xseg_1`, choices `many, xseg_1, xseg_2, xseg_3`. The ini key `[face_masker] face_occluder_model` exists since 3.1.0. In 3.1.0 the code read it from the `face_detector.` section, a bug fixed by 3.3.0.
- `--face-parser-model`: default `bisenet_resnet_34`.
- `--face-mask-types`: default `box` (nargs `+`). This is unchanged from 3.0.0 to 3.9.2.
- `--face-mask-areas`: default is all of `upper-face lower-face mouth`.
- `--face-mask-regions`: default is all 10 regions: `skin left-eyebrow right-eyebrow left-eye right-eye glasses nose mouth upper-lip lower-lip`.
- `--face-mask-blur`: default `0.3`, range 0.0–1.0 in steps of 0.05. It affects only the box mask.
- `--face-mask-padding`: default `0 0 0 0` (top, right, bottom, left, in % of crop), range 0–100. Since 3.5.0 it is sanitised with `sanitize_int_range`.
- Area landmark sets (68-point indices): `upper-face: [0,1,2,31,32,33,34,35,14,15,16,26,25,24,23,22,21,20,19,18,17]`, `lower-face: [3..13,35,34,33,32,31]`, `mouth: [48..67]`. — [choices.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/choices.py)
- Region to parser label index: `skin 1, left-eyebrow 2, right-eyebrow 3, left-eye 4, right-eye 5, glasses 6, nose 10, mouth 11, upper-lip 12, lower-lip 13`. Indices 0, 7–9 and 14–18 (background, ears, earring, neck, cloth, hair, hat in the standard CelebAMask-HQ order) are never selectable, so a region mask always excludes hair, ears and neck. — [choices.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/choices.py)

**Exact formulas (face_masker.py @3.9.2)** — [face_masker.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/face_masker.py)
- Box mask:
  ```
  blur_amount = int(crop_w * 0.5 * face_mask_blur)
  blur_area = max(blur_amount // 2, 1)
  mask = ones
  zero top rows    [: max(blur_area, int(crop_h*pad_top/100))]
  zero bottom rows [-max(blur_area, int(crop_h*pad_bottom/100)):]
  zero left cols   [: max(blur_area, int(crop_w*pad_left/100))]
  zero right cols  [-max(blur_area, int(crop_w*pad_right/100)):]
  if blur_amount > 0: GaussianBlur(sigma = blur_amount*0.25)
  ```
  For example, a 256 crop at blur 0.3 gives blur_amount 38, 19 px zeroed borders and sigma 9.5. 3.0.0 cached this as `create_static_box_mask` via `lru_cache`.
- Occlusion mask:
  - Input: the BGR crop, `cv2.resize` to (256,256) with default bilinear interpolation, `/255.0`, NHWC (the `transpose(0,1,2,3)` is a no-op). There is no RGB swap and no mean or std normalisation.
  - Each model's output `[0][0]` (256×256×1) is clipped to 0..1 and resized back to the crop size.
  - With `'many'`, the three models (xseg_1, xseg_2, xseg_3) are combined with `numpy.minimum.reduce`, i.e. the most conservative face estimate.
  - Post-process: `(cv2.GaussianBlur(m.clip(0,1), (0,0), 5).clip(0.5, 1) - 0.5) * 2`. Sigma is a fixed 5 px at crop resolution, whatever the crop size.
- Region mask:
  - Input: crop resized to 512×512, BGR→RGB, `/255.0`, ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225), NCHW.
  - Then `argmax(0)` over the 19 logits of output `[0][0]` and `numpy.isin(selected label ids)`, which gives a hard 0/1 mask.
  - That mask is resized to crop size, then gets the same blur(5) and soft threshold as above.
- Area mask: `cv2.convexHull` of the selected 68-landmark points (transformed into crop space with the affine matrix), `fillConvexPoly` with 1.0, then the same blur(5) and soft threshold.

**Order and combination inside the swapper (face_swapper/core.py `swap_face`, identical structure in 3.0.0 and 3.9.2)** — [face_swapper/core.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/processors/modules/face_swapper/core.py), [face_swapper.py @3.0.0](https://github.com/facefusion/facefusion/blob/3.0.0/facefusion/processors/modules/face_swapper.py)
1. `warp_face_by_face_landmark_5(temp_vision_frame, landmark '5/68', model_template, pixel_boost_size)` produces the crop and affine matrix. Crop size equals `--face-swapper-pixel-boost`, whose default is the first choice: 256x256 for 256 models, 128x128 for inswapper_128.
2. Box mask, if enabled, on the original target crop.
3. Occlusion mask, if enabled, on the original target crop, i.e. **before** the swap.
4. Swap: tiles via `implode_pixel_boost`, model inference, then `explode_pixel_boost`.
5. Area mask, if enabled, on the swapped crop, using target 68 landmarks transformed by the affine matrix.
6. Region mask, if enabled, on the **swapped** crop: the parser segments the generated face, not the target.
7. `crop_mask = numpy.minimum.reduce(crop_masks).clip(0, 1)`.
8. `paste_back`: `cv2.warpAffine(crop_mask, paste_matrix, ...)` (bilinear) then `.clip(0,1)`; the crop is warped with `BORDER_REPLICATE`, then `out = frame*(1-m) + swapped*m` inside the paste bounding box. — [face_helper.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/face_helper.py)
- The default swapper model is `inswapper_128_fp16` in 3.2.0–3.3.x and `hyperswap_1a_256` from 3.4.0 to 3.9.2 (template `arcface_128`, size 256). — [face_swapper/core.py @3.4.0](https://github.com/facefusion/facefusion/blob/3.4.0/facefusion/processors/modules/face_swapper/core.py)

### Inferences
- **The post-process is a hard-ish threshold.** Blur(sigma 5) followed by `clip(0.5,1)` remaps to 0..1, so a pixel survives only where the blurred XSeg probability is above 0.5 and is fully opaque only where it reaches 1.0. This erodes the mask edge slightly (about a few pixels at the 0.5 contour) and gives a short ramp, not a wide feather. It is a spatial operation only and does nothing for temporal stability.
- **Sigma does not scale with resolution.** Because sigma is fixed at 5 px of crop resolution, the effective feather relative to the face is about 4× larger at a 128 crop (inswapper default) than at a 512 boost.
- **Computing occlusion on the pre-swap target crop is the key design choice.** Hands, cups and food are segmented in the real frame. The swapped face is never pasted over them, even if the swap model hallucinates a face there. Region and area are computed post-swap, so they describe the generated face's semantic layout.
- **Cost of an occlusion-only paste mask.** For a ~0.1–0.3 s per-face CPU budget, the cheapest faithful replica is: box mask (essentially free), plus one XSeg at 256 (about 0.3 s on the test CPU, see Q2), combined with `min`. This alone likely exceeds the budget at the low end; see Q2 timings.

### Gaps
- docs.facefusion.io could not be fetched (the domain is blocked in this session), so official prose descriptions of each option were not checked. Values above come from code, which is authoritative.

---

## Q2. Occluder models (xseg_1/2/3, many) and face parser models: provenance, input/output, cost

### Takeaway
All three FaceFusion occluders are DeepFaceLab XSeg networks exported with tf2onnx: about 70 MB each, NHWC 256×256×3 float BGR input in 0..1, sigmoid 256×256×1 output. FaceFusion's metadata only says "vendor DeepFaceLab, GPL-3.0, 2021"; training data for each variant is undocumented. The parsers are yakhyo's BiSeNet (ResNet-18 or ResNet-34), trained on CelebAMask-HQ at 512×512 with 19 classes. On a 4-thread 2.1 GHz Xeon, one XSeg pass costs about 0.29–0.35 s and BiSeNet about 0.23 s (R18) or about 0.39 s (R34). 'many' triples the XSeg cost.

### Cited Findings
- **Model set metadata (`create_static_model_set`, @3.9.2):**
  - `xseg_1` and `xseg_2`: release `models-3.1.0`, `'vendor': 'DeepFaceLab', 'license': 'GPL-3.0', 'year': 2021`, `size (256,256)`.
  - `xseg_3`: release `models-3.2.0`, same metadata.
  - `bisenet_resnet_18`: `models-3.1.0`; `bisenet_resnet_34`: `models-3.0.0`. Both `'vendor': 'yakhyo', 'license': 'MIT', 'year': 2024`, `size (512,512)`.
  - [face_masker.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/face_masker.py)
- **Measured with onnx and onnxruntime 1.30 on files downloaded from facefusion-assets** ([models-3.1.0](https://github.com/facefusion/facefusion-assets/releases/tag/models-3.1.0), [models-3.2.0](https://github.com/facefusion/facefusion-assets/releases/tag/models-3.2.0), [models-3.0.0](https://github.com/facefusion/facefusion-assets/releases/tag/models-3.0.0)); 4 intra-op threads, Intel Xeon @ 2.10 GHz, mean of 10 runs after warm-up:

  | Model | File size | Input | Output | Exporter / opset | CPU time |
  |---|---|---|---|---|---|
  | xseg_1 | 70,324,286 B | `input` [N,256,256,3] | `output` [N,256,256,1], last ops `Conv → Sigmoid → Reshape`, 394 nodes | tf2onnx, opset 15 | ≈352 ms (≈367 ms single-thread) |
  | xseg_2 | 70,324,286 B (same size as xseg_1, different weights) | same | same | tf2onnx, opset 15 | ≈304 ms |
  | xseg_3 | 70,327,709 B | same | same | tf2onnx, opset 17 | ≈294 ms |
  | bisenet_resnet_18 | 53,205,356 B | `input` [N,3,512,512] | 3 outputs, each [N,19,512,512] (main plus 2 auxiliary heads; FaceFusion uses output 0) | pytorch | ≈231 ms |
  | bisenet_resnet_34 | 93,632,546 B | same | same | pytorch | ≈390 ms |

  Raw outputs on random input were in [0,1] for XSeg and in logit range (about −2.7 to 7.3) for BiSeNet.
- **XSeg architecture (DFL):** `nn.XSeg(3, 32, 1)` with base_ch=32, a U-Net-style encoder of ConvBlocks doubling channels (32→64→128→256…), resolution fixed at 256. — [facelib/XSegNet.py](https://github.com/iperov/DeepFaceLab/blob/master/facelib/XSegNet.py), [core/leras/models/XSeg.py](https://github.com/iperov/DeepFaceLab/blob/master/core/leras/models/XSeg.py)
- **yakhyo face-parsing:** trained on CelebAMask-HQ ("High-quality face parsing dataset with 30,000 images"), default `--num-classes 19` and `--image-size 512 512`. ResNet18 is about 11.2M parameters (~43 MB); ResNet34 is about 21.3M parameters (~82 MB). The README does not list label names. — [yakhyo/face-parsing](https://github.com/yakhyo/face-parsing)
- **'many' semantics:** the outputs of xseg_1, xseg_2 and xseg_3 are combined with `numpy.minimum.reduce` before the blur and threshold. Anything any one model calls "not face" (occluder) is excluded. — [face_masker.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/face_masker.py), [PR #910](https://github.com/facefusion/facefusion/pull/910)

### Inferences
- **Channel order.** The occluders take BGR because FaceFusion passes OpenCV frames without a channel swap and DFL trains on `load_bgr()` images. A reimplementation must feed BGR in 0..1 in NHWC layout.
- **Likely provenance of xseg_1.** 3.0.0 shipped `dfl_xseg.onnx`. xseg_1 (`models-3.1.0`) is plausibly a renamed or re-exported copy, possibly DFL's generic WF XSeg. This is not confirmed: the release pages show no notes and the code metadata gives no training description.
- **Budget fit.** For a 0.1–0.3 s per-face budget on a similar CPU, a single XSeg pass alone (~0.3 s) uses the whole budget, and 'many' (~0.9 s) or BiSeNet-34 is out of reach. The XSeg graph is heavy (70 MB of float32 weights). Options worth evaluating separately: int8 quantisation, running XSeg at a lower rate, or a lighter occlusion segmenter.

### Gaps
- No primary documentation of what data xseg_1, xseg_2 and xseg_3 were trained on, or which face type each uses (WF or FF). The facefusion-assets release pages contain only the tag name and the asset list failed to render. The weights' training provenance remains unknown.
- No published FaceFusion benchmarks of mask cost were found; the timings above are local measurements on one CPU.

---

## Q3. Temporal handling and multiple or overlapping faces in FaceFusion

### Takeaway
FaceFusion does **no temporal smoothing of masks**: occlusion, region and area masks are recomputed independently per frame. Since 3.7.0 there is an optional face tracker. It IoU-links detections across a small frame window and linearly interpolates face geometry (bbox and landmarks) only for frames where a detection is missing. It does not smooth existing detections and does not touch masks. Multiple faces are pasted sequentially onto the same frame with no inter-face exclusion. Each face's crop is taken from the frame that already contains earlier swaps, and there is no "other face" mask.

### Cited Findings
- **Face tracker (3.7.0+).** `--face-tracker-score` defaults to `0.0`, which is off. Its range is 0.0–0.5 in steps of 0.05; help text: "specify the overlap score used to match the tracked faces". How it works:
  - When the score is above 0, `select_faces` calls `track_faces(target_vision_frames, score)`.
  - `create_face_tracks` detects faces in every frame of the window. Each face is assigned to the existing track whose last face has the highest bounding-box IoU above `score`; otherwise it starts a new track.
  - For each track spanning the middle (target) frame, `refill_faces` fills gaps by linear interpolation (`average_face_geometry`: bbox and all landmark sets interpolated by `average_factor`; embedding, age and gender come from the nearer anchor) and returns the face for the middle frame.
  - [face_tracker.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/face_tracker.py), [face_creator.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/face_creator.py), [face_selector.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/face_selector.py), [locales.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/locales.py)
- **Window size.** `--target-frame-amount` (first appears in 3.8.0) defaults to `2`, range 0–10; help text: "specify the amount of target frames forwarded to the processor". `select_video_frames(path, n, offset)` returns frames n−offset … n+offset, so the default is a 5-frame window centred on the current frame. — [program.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/program.py), [vision.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/vision.py), [workflows/core.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/workflows/core.py)
- **Rejected Kalman tracker.** An earlier community PR #1156 "Implement face tracker" (Kalman prediction plus IoU and embedding association, stable track IDs, intended to keep ordering consistent in many-face mode and survive brief occlusions or missed detections) was closed on 2026-06-14. The maintainer asked for "a new pr based on multi frame approach", which became the 3.7.0 tracker. — [PR #1156](https://github.com/facefusion/facefusion/pull/1156)
- **Face selection.**
  - `--face-selector-mode` defaults to `reference` (choices `many`, `one`, `reference`); `--face-selector-order` defaults to `large-small`; `--reference-face-distance` defaults to `0.3`; `--reference-face-position` defaults to `0`.
  - In reference mode, every target face whose embedding distance is below the threshold (after `numpy.interp(d, [0,2], [0,1])`) is swapped.
  - In many mode, all faces are swapped, sorted by order.
  - [program.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/program.py), [face_selector.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/face_selector.py)
- **Sequential pasting.** `process_frame` loops `for target_face in target_faces: temp_vision_frame = swap_face(source_face, target_face, source_vision_frame, temp_vision_frame)`. Each subsequent crop is warped from the already-modified frame, and its mask is computed only from its own crop. — [face_swapper/core.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/processors/modules/face_swapper/core.py)

### Inferences
- **Two faces in contact (kiss, cheek to cheek).** FaceFusion's only defence is the XSeg occluder judging that the other person's cheek, nose or lips inside the crop are "not this face". XSeg is trained to separate face from non-face, not one face from another, so another person's skin in the crop is likely labelled face and gets swapped over. The swapped face's box or region extent can then cover part of the other person.
- **Reference mode with one source.** If only one identity is replaced, the other face is never processed, but its pixels can still be overwritten by the target face's paste wherever the target's mask spans them.
- **Overlapping pastes.** With many mode, the later paste wins in the overlap; the order follows `face_selector_order` (default large-small).
- **Temporal stability.** Because masks are per-frame and thresholded at 0.5 after the blur, boundary pixels near the 0.5 contour can toggle between frames, which is the mechanism behind visible occlusion-mask flicker. A downstream implementation would need its own temporal filter (for example, EMA of the warped mask in a stabilised face frame). FaceFusion provides none.

### Gaps
- No FaceFusion code or documentation was found that explicitly addresses two-face contact, and no "other face" exclusion option exists in any 3.x tag.
- FaceFusion's GitHub issue search for "occlusion", "xseg", "flicker" and "kiss" returned only PRs (#1156, #910, #230). User-reported occlusion problems are therefore not documented on GitHub; FaceFusion directs support to Discord, which was not searchable.

---

## Q4. DeepFaceLab XSeg training and merging, and DeepFaceLive's handling

### Takeaway
In DFL, XSeg is a per-project segmentation model. The user hand-labels about 50–100 varied faces with include and exclude polygons (obstructions are exclude polygons), trains a 256-px XSeg, applies it to the facesets, and selects a merge mask mode. `XSeg-prd*XSeg-dst` is the product, i.e. the intersection of the mask of the generated face and the mask of the real destination face. Obstructions in dst are removed via XSeg-dst. DeepFaceLive runs **no** XSeg at runtime. It uses the DFM model's learned masks, which inherit XSeg only if the faceset was XSeg-labelled before training. iperov states generic XSeg is too slow for real time.

### Cited Findings
**Workflow (iperov's XSegEditor commit message)** — [DFL commit 01d8167 "added new XSegEditor !"](https://github.com/iperov/DeepFaceLab/commit/01d81674fdf0acb1b896db3b984828c9812d9111)
- Mask about 50–100 dst faces with include or exclude polygons, choosing only frames where the face changes noticeably (closed eyes, new head direction, different lighting).
- Keep the polygon logic consistent across frames; use exclude polygons for obstructions.
- Train, then check the "XSeg dst faces" preview. If masks glitch, label more faces and continue training.
- Merger modes are described as: XSeg-prd (requires labelled src), XSeg-dst (requires labelled dst), and XSeg-prd*XSeg-dst, "the smallest area of both".
- Without a trained XSeg model in the workspace, the XSeg modes give a transparent mask.

**XSeg training code** — [models/Model_XSeg/Model.py](https://github.com/iperov/DeepFaceLab/blob/master/models/Model_XSeg/Model.py)
- `face_type` defaults to `wf`; choices `h, mf, f, wf, head`, with the help text "Choose the same as your deepfake model."
- Batch size defaults to 4, range 2–16. Resolution is fixed at 256.
- Optimiser: `RMSprop(lr=0.0001, lr_dropout=0.3)`.
- Loss: `sigmoid_cross_entropy_with_logits`. The optional pretrain mode uses DSSIM at two filter sizes plus 10×MSE.

**Augmentation (`SampleGeneratorFaceXSeg.batch_func`)** — [samplelib/SampleGeneratorFaceXSeg.py](https://github.com/iperov/DeepFaceLab/blob/master/samplelib/SampleGeneratorFaceXSeg.py)
- Labels: polygon labels, or else applied XSeg masks binarised at 0.5.
- 50% of samples get the non-face area of another labelled sample composited in. The other sample's background is randomly rotated (±180°), scaled (±10%) and HSV- or levels-shifted, then mixed with `c_mask = 1-(1-bg_mask)*(1-mask)`, `rnd = 0.15+U*0.85`.
- Random flip, rotation ±10°, scale, translation and warp.
- 50% "face flare" and 50% "bg flare" (large Gaussian-blurred additive glow).
- Local HSV or levels shifts, sharpen or motion blur and Gaussian blur, nearest or bilinear down-resize, JPEG — each at 25% chance inside random faded circles.
- Masks are re-binarised at 0.5 after warping.

**Inference (`XSegNet.extract`)** — [facelib/XSegNet.py](https://github.com/iperov/DeepFaceLab/blob/master/facelib/XSegNet.py), [mainscripts/XSegUtil.py](https://github.com/iperov/DeepFaceLab/blob/master/mainscripts/XSegUtil.py)
- `np.clip(net_run(x), 0, 1)`, then `result[result < 0.1] = 0  # get rid of noise`.
- When applying to a faceset, faces are resized to 256 with LANCZOS4. A "same" or different face type can be chosen, for example to apply a WF XSeg to a HEAD faceset.

**Merger (`MergeMaskedFace`)** — [merger/MergeMasked.py](https://github.com/iperov/DeepFaceLab/blob/master/merger/MergeMasked.py), [merger/MergerConfig.py](https://github.com/iperov/DeepFaceLab/blob/master/merger/MergerConfig.py)
- `mask_mode_dict = {0:'full', 1:'dst', 2:'learned-prd', 3:'learned-dst', 4:'learned-prd*learned-dst', 5:'learned-prd+learned-dst', 6:'XSeg-prd', 7:'XSeg-dst', 8:'XSeg-prd*XSeg-dst', 9:'learned-prd*learned-dst*XSeg-prd*XSeg-dst'}`.
- Default `mask_mode = 4` in the constructor; the interactive prompt default is `1`.
- XSeg-prd: the predicted face is resized to 256 (INTER_CUBIC) and run through XSeg.
- XSeg-dst: the **original frame** is re-warped with `get_transform_mat(landmarks, 256, face_type)` and run through XSeg.
- Combinations are elementwise products (mode 5 is a clipped sum).
- Then `wrk_mask[wrk_mask < 1/255] = 0`, and the mask is resized to `mask_subres_size = input_size*4`.
- Erode (positive `erode_mask_modifier`) or dilate (negative) uses an elliptical kernel of that size; the range is −400..400 and the default is 0.
- The border is zeroed to `input_size + blur//2`; then `GaussianBlur((blur,blur))` with an odd kernel (`blur_mask_modifier`, default 0).
- The mask is inverse-warped with `WARP_INVERSE_MAP|INTER_CUBIC` and values below 1/255 are zeroed again.

**SAEHD masked training** — [models/Model_SAEHD/Model.py](https://github.com/iperov/DeepFaceLab/blob/master/models/Model_SAEHD/Model.py)
- `masked_training` defaults to True: "Masked training clips training area to full_face mask or XSeg mask". It is available for whole_face and head.
- `blur_out_mask` defaults to False and "requires exact xseg mask in src and dst faceset".

**DeepFaceLive** — [apps/DeepFaceLive/backend/FaceMerger.py](https://github.com/iperov/DeepFaceLive/blob/master/apps/DeepFaceLive/backend/FaceMerger.py), [FaceSwapDFM.py](https://github.com/iperov/DeepFaceLive/blob/master/apps/DeepFaceLive/backend/FaceSwapDFM.py), [FaceSwapInsight.py](https://github.com/iperov/DeepFaceLive/blob/master/apps/DeepFaceLive/backend/FaceSwapInsight.py), [xlib ImageProcessor.erode_blur](https://github.com/iperov/DeepFaceLive/blob/master/xlib/image/ImageProcessor.py)
- Mask sources:
  - `face_mask_source` (the DFM's predicted dst mask): default on.
  - `face_mask_celeb` (the DFM's predicted celeb mask): default on.
  - `face_mask_lmrks` (landmark convex hull from FaceAligner): default off.
- `face_mask_erode` defaults to 5 (range −400..400); `face_mask_blur` defaults to 25 (range 0..400).
- Combination:
  - CPU path: product of the enabled masks.
  - GPU path: product, then binarised `O = (X <= 0.5 ? 0 : 1)`, then `binary_morph(erode, blur, fade_to_border=True)`.
  - Erode uses a 3×3 ellipse with `iterations = erode//2`.
  - The warped mask is zeroed below 1/255.
- The InsightFace swapper path sets both masks to all-white (`np.full_like(..., 255)`), so there is no occlusion handling there beyond the optional landmark hull.
- `DFMModel.convert` returns `out_celeb, out_celeb_mask, out_face_mask` from the ONNX DFM. — [modelhub/DFLive/DFMModel.py](https://github.com/iperov/DeepFaceLive/blob/master/modelhub/DFLive/DFMModel.py)
- **iperov on XSeg-dst in DeepFaceLive** (issue #63, 2022-06): a user's XSeg-excluded open mouth worked in DFL merge (mask_mode XSeg-dst) but not in DeepFaceLive. iperov replied that it was not related to DeepFaceLive and that "generic xseg model has low performance to work in real time" ("no, code it for yourself"). — [DeepFaceLive #63](https://github.com/iperov/DeepFaceLive/issues/63)
- DeepFaceLive FAQ: "Src faceset should be xseg'ed and applied. You can apply Generic XSeg to src faceset." — [doc/user_faq/user_faq.md](https://github.com/iperov/DeepFaceLive/blob/master/doc/user_faq/user_faq.md)
- Both DeepFaceLab and DeepFaceLive repositories were archived (read-only) on 2024-11-13. — [DeepFaceLive #63 page](https://github.com/iperov/DeepFaceLive/issues/63)
- Third-party tutorials (secondary sources, partly AI-generated summaries) list hands, hair, glasses, piercings and tattoos as things to exclude, and say never to label a face with only an exclusion polygon. — [DeepfakeVFX XSeg tutorial (search snippet; site blocked for fetch)](https://www.deepfakevfx.com/tutorials/deepfacelab-2-0-xseg-tutorial/)

### Inferences
- **Two faces in contact in DFL.** The mechanism available is to draw exclude polygons over the other person's face, lips or nose wherever it enters the target face crop, on enough varied frames, and to merge with XSeg-dst or XSeg-prd*XSeg-dst. XSeg-dst then removes the other person from the paste area. This follows directly from the include/exclude design, but no primary source explicitly describing a kissing workflow was found.
- **The 50% background-compositing augmentation** in XSeg training teaches "outside the labelled polygon is not face", including foreign textures. It does not specifically teach "another face is not this face" unless such cases are labelled. This is why generic XSeg models (as reused by FaceFusion) tend to include a second face's skin.
- **DFL's product (`*`) combination vs FaceFusion's `min`.** For soft masks, product is more aggressive (0.7×0.7 = 0.49 < min 0.7). DFLive then binarises at 0.5 on the GPU path, which makes the two roughly equivalent at the threshold.

### Gaps
- Community guides (deepfakevfx.com, mrdeepfakes, deepwiki) could not be fetched; those domains are blocked in this session. The recommended label counts per obstruction type, the iteration counts for XSeg training, and kissing-specific guidance remain unverified from primary community guides.
- The search snippet for DFL issue #686 (an obstruction-masked XSeg auto-selecting the WF face type) was not verified by fetching.

---

## Q5. Known weaknesses reported by users (flicker, beards, hair, glasses, kissing) and fixes

### Takeaway
Little primary user-report material was reachable. The FaceFusion GitHub tracker exposes essentially no occlusion issues, and the community guides were network-blocked. The well-documented limitations come from code and maintainer statements: per-frame masks with no temporal filtering, generic XSeg being too slow for real time (iperov), and DeepFaceLive not supporting XSeg-dst.

### Cited Findings
- **DeepFaceLive:** an XSeg-excluded open mouth was not respected because DFLive uses only the DFM's learned masks. The maintainer gave no fix ("code it for yourself") and cited low real-time performance of generic XSeg. — [DeepFaceLive #63](https://github.com/iperov/DeepFaceLive/issues/63)
- **FaceFusion's response to temporal instability** is in face geometry, not masks:
  - The 3.7.0 multi-frame tracker interpolates missed detections across a ±`target_frame_amount` window.
  - The rejected Kalman PR aimed at "brief occlusions / missed detections" and ordering consistency.
  - [face_tracker.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/face_tracker.py), [PR #1156](https://github.com/facefusion/facefusion/pull/1156)
- **FaceFusion's 'many' occluder (3.4.0)** takes the pixel-wise minimum of three XSeg models. It is a precision-over-recall ensemble: any single model's occluder detection removes the swap there. — [PR #910](https://github.com/facefusion/facefusion/pull/910), [face_masker.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/face_masker.py)
- **FaceFusion region masks** cannot include hair, ears or neck (labels 7–9 and 14–18 are not exposed), and `glasses` (label 6) is selectable as a region to *keep* swapped; it is selected by default. — [choices.py @3.9.2](https://github.com/facefusion/facefusion/blob/3.9.2/facefusion/choices.py)

### Inferences
- **Expected failure modes, from mechanism rather than user reports:**
  - **Flicker.** Per-frame XSeg plus a 0.5 soft threshold after a 5 px blur, with no temporal filter, produces boundary chatter.
  - **Beards, stubble and glasses** sit at the semantic edge of "face" for XSeg. The minimum in 'many' mode amplifies false negatives, i.e. the swap is removed from beards and glasses frames.
  - **Kissing or cheek contact.** XSeg generally treats any facial skin as face. The defence is DFL-style manual exclude labelling of the partner's face, or an instance-aware rule: intersect with the target's own landmark hull or face-parse skin, or subtract the other detected face's hull.
- **Fixes used in practice** (from tool design): `--face-occluder-model many` for more conservative occlusion; combining `occlusion` with `region` (post-swap parse) or `area` masks; increasing box padding or blur; in DFL, labelling more frames and retraining XSeg, then using XSeg-prd*XSeg-dst and tuning erode and blur.

### Gaps
- No primary user reports (GitHub issues, forum threads) on FaceFusion occlusion flicker, beard or glasses false positives, or kissing failures could be retrieved. FaceFusion issue search returned only PRs, and the forums and guides were blocked. Treat the failure modes above as mechanism-based inferences, not reported evidence.
