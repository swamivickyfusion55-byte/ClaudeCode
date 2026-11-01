# Temporal stability of occlusion masks and face tracking through occlusion (for video face swap)

Research date: 2026-11-01. Network note: arxiv.org, ar5iv, huggingface.co, docs.opencv.org and cristal.univ-lille.fr could not be reached from this session (these hosts would have to be allowed in the session's network settings). Paper claims below therefore come from search-result snippets and from the authors' official GitHub code, which was cloned and read directly. Statements marked **[unverified]** come from background knowledge and were not confirmed against a source in this session. The CPU timings marked **[measured here]** come from a benchmark I ran in this session: OpenCV 5.0.0, Python, Intel Xeon @ 2.10 GHz, `cv2.setNumThreads(1|2)`, synthetic textured images with a known (2.5, -1.5) px shift, mean of 50 runs (20 for Farneback). The script is at `/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/flow.py`.

---

## Q1. Temporal mask stabilisation: hysteresis, attack/release, median, EMA, optical-flow warping, recurrent matting / VOS, and what face-swap tools actually do

### Takeaway
None of the open-source face-swap tools I inspected (FaceFusion 3.9.2, Deep-Live-Cam as of Oct 2026) applies any temporal filter to the occluder mask. Each frame's XSeg mask is used as-is, which is a direct cause of occluder flicker. The cheapest robust fix fits well under the budget. Compute the occluder probability in aligned-crop space, warp the previous frame's state forward with DIS ULTRAFAST flow (about 0.6 ms at 256 px on one CPU thread, measured here), and combine three steps: a centred 3-frame median using the 1-frame look-ahead, which gives instant attack and rejects 1-frame blips; a peak-hold envelope with slow exponential decay, which gives a smooth release; and hysteresis thresholds. Heavy learned temporal models (RVM, XMem, Cutie) have published speeds only for GPUs and are not needed.

### Cited Findings

**What face-swap tools do (source code read directly)**
- FaceFusion 3.9.2 (commit decc6c9, 2026-10-26) builds the occlusion mask per frame. It resizes the aligned crop to the XSeg input size of (256, 256), runs one or more models (`xseg_1/2/3`; mode `many` takes the pixel-wise `numpy.minimum` of all of them), resizes the result back, and then applies `(cv2.GaussianBlur(mask.clip(0,1), (0,0), 5).clip(0.5, 1) - 0.5) * 2`. That is a spatial blur followed by a hard re-threshold at 0.5 and a ramp. No previous-frame state is used anywhere in `face_masker.py`. The default occluder model is `xseg_1`, the default mask type is `box` (occlusion is opt-in), and the default `face_mask_blur` is 0.3 — [FaceFusion face_masker.py / program.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_masker.py)
- FaceFusion's box mask feathers by `blur_amount = int(crop_size * 0.5 * face_mask_blur)` with sigma `blur_amount * 0.25` — [FaceFusion face_masker.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_masker.py)
- Deep-Live-Cam (commit 2026-10-27) has an optional whole-frame blend, `final = prev_result*(1-w) + current*w`, with default `interpolation_weight = 0.2` (an EMA over the full output frame, which ghosts during motion). Its face enhancer reuses the cached GFPGAN result on alternate frames (`_ENH_INTERVAL = 2`). Code comments say the blend mask was deliberately derived from the swap's own affine transform rather than from per-frame landmarks, so it follows the face "with no independent jitter source, no EMA, no lag", and the mask is cached when the face is nearly still. There is no temporal occluder logic — [Deep-Live-Cam face_swapper.py / face_enhancer.py](https://github.com/hacksider/Deep-Live-Cam/tree/main/modules/processors/frame)
- Users report the problem in the tracker: Deep-Live-Cam issue #1610 "Flickering issue with face mask" (2025-12-23) and #325 "Handle Face Jitter..." (2024-08-14). Only the titles were seen; the bodies were not accessible — [Deep-Live-Cam issues](https://github.com/hacksider/Deep-Live-Cam/issues/1610)
- A search of FaceFusion's GitHub issues for "flicker" and "occluder" returned no hits through the API (the search may be incomplete) — [FaceFusion issues](https://github.com/facefusion/facefusion/issues)

**Optical-flow mask propagation and its cost**
- DIS (Kroeger et al., ECCV 2016) reports 300–600 Hz on a single CPU core at 1024×436 for its fastest operating points, and 10–600 Hz across operating points. That drops to about 42/46 Hz when preprocessing such as disk access and rescaling is included — [arXiv 1603.03590](https://arxiv.org/abs/1603.03590) (via search snippet)
- OpenCV exposes DIS presets `PRESET_ULTRAFAST`, `PRESET_FAST` and `PRESET_MEDIUM` — [OpenCV DISOpticalFlow docs](https://docs.opencv.org/4.12.0/javadoc/org/opencv/video/DISOpticalFlow.html). OpenCV issue 14473 reports that the OpenCL (UMat) path of DIS can be slower than the CPU path at common resolutions — [opencv#14473 mirror](https://hub.harborframework.com/tasks/abundant/opencv__opencv-14473)
- **[measured here]** Per-call CPU times, 1 thread / 2 threads:

  | size | DIS ULTRAFAST | DIS FAST | DIS MEDIUM | Farneback (0.5,3,15,3,5,1.2) | LK, 68 pts, win 21, 3 levels | `remap` float mask | open 7×7 + connectedComponents |
  |---|---|---|---|---|---|---|---|
  | 128 px | 0.21 / 0.11 ms | 0.51 / 0.39 | 1.65 / 1.35 | 3.5 / 3.1 | 0.65 / 0.45 | 0.04 | 0.12 |
  | 256 px | **0.64 / 0.59 ms** | 1.26 / 1.72 | 5.6 / 5.4 | 13.9 / 13.6 | 0.95 / 0.62 | 0.13 / 0.07 | 0.39 / 0.26 |
  | 512 px | 2.7 / 1.8 ms | 5.1 / 4.6 | 27.8 / 15.2 | 59 / 60 | 1.2 / 1.1 | 0.52 | 1.7 |

  All three DIS presets recovered the global shift to within about 0.1 px (median). Synthetic texture with pure translation is an easy case; real faces with non-rigid motion and occluder edges will be noisier.
- A patent on reducing "chatter" in video object segmentation computes a weighted average of masks flow-warped from neighbouring frames together with the current frame's mask, then blurs and thresholds the result spatio-temporally ("gaussholding") — [US 8358691](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8358691)
- VToonify (2022) smooths face-parsing maps over a temporal window using optical flow plus occlusion masks, and adds a flicker-suppression loss — [arXiv 2209.11224](https://arxiv.org/pdf/2209.11224)
- A 2026 preprint proposes inference-time temporal smoothing of per-frame SAM2 probability maps. It reports that SAM2 predictions stay sensitive to motion, occlusion and appearance change — [alphaxiv 2604.17115](https://www.alphaxiv.org/abs/2604.17115.md)

**Learned temporally consistent matting and VOS (cost)**
- Robust Video Matting (RVM, WACV 2022) gets its temporal consistency from ConvGRU recurrent states carried from frame to frame (`fgr, pha, *rec = model(src, *rec, downsample_ratio)`). Reported speed is 4K at 76 FPS and HD at 104 FPS on a GTX 1080 Ti; the README table says 74 FPS at 4K in FP32, a small inconsistency with the paper. The downsample ratio is 0.25 for HD and 0.125 for 4K. The source gives no CPU timings — [RVM GitHub](https://github.com/PeterL1n/RobustVideoMatting); [WACV paper](https://openaccess.thecvf.com/content/WACV2022/papers/Lin_Robust_High-Resolution_Video_Matting_With_Temporal_Guidance_WACV_2022_paper.pdf)
- Cutie (2023/CVPR 2024): Cutie-small runs at 45.5 FPS, Cutie-base at 36.4 FPS and XMem at 22.6 FPS (GPU, YouTubeVOS). Cutie keeps a bounded memory of 5 frames by default — [arXiv 2310.12982](https://arxiv.org/abs/2310.12982v1). An independent benchmark on an A6000 measured XMem at 7.3 FPS and 12.5 GB — [LiVOS arXiv 2411.02818](https://arxiv.org/pdf/2411.02818). I found no CPU numbers for XMem or Cutie.

### Inferences
- Because there is no temporal state, any XSeg probability near 0.5 at a hand, hair or second-face boundary flips the swap on and off between frames. FaceFusion's `clip(0.5,1)` re-threshold after the blur makes this worse, since it is a hard threshold applied after smoothing.
- **Recipe A: occluder-mask temporal filter, all in 256×256 aligned-crop space, about 2–3 ms per face:**
  1. Compute `o_t = 1 − xseg_face_prob_t` as a float occluder probability. Do not binarise it yet.
  2. Motion-compensate the state. Run DIS ULTRAFAST on grey 256 px crops from frame t−1 to frame t (backward flow, used with `remap` to pull state from t−1). Aligned crops already remove most head motion, so the flow mainly tracks the occluder itself. If the face alignment changes between frames, re-express the previous state through the relative affine before running flow.
  3. Use the 1–2 frame look-ahead for **fast attack with spike rejection**. Take `o'_t = median(o_{t−1}^warped, o_t, o_{t+1}^warped)` per pixel. A real occluder that lasts at least 2 frames gives `median(0,1,1) = 1` on the frame it first appears, so there is zero delay, while a single-frame blip gives `median(0,1,0) = 0`. An occluder that disappears drops immediately (`median(1,0,0) = 0`), so the release has to come from step 4.
  4. Apply a **slow release envelope** (peak-hold with decay): `S_t = max(o'_t, d · S_{t−1}^warped)`, with a decay `d` of about 0.6–0.75 per frame at 30 fps, which takes about 3–6 frames to fall from 1 to below 0.2. The output mask follows a rise in a single frame and fades on removal, so the face returns smoothly. These constants are my suggestion, not from a source.
  5. Apply **hysteresis** where a hard decision is needed, for example a per-component "occluder present" flag: switch on at `S > 0.6`, switch off only when `S < 0.3` for at least 2 consecutive frames. Paste with the soft mask `S` remapped through a smoothstep between 0.3 and 0.6, not a 0.5 cut. Suggested values, not sourced.
  6. Optional and cheap (about 0.3 ms): run a morphological open of 3–7 px on the binarised `S > 0.5`. Then drop connected components smaller than about 0.5–1% of the crop area unless they touch the crop border, because real occluders such as a hand or another face enter from outside the face. Suggested values, not sourced.
  7. Check flow validity. If the forward-backward flow error is large, or the mean photometric error after warping is high (fast motion or a cut), skip step 2 and use the unwarped `o_t` so a stale mask is not dragged along.
- Learned recurrent models (RVM ConvGRU, XMem/Cutie memory) are the right idea, but they are GPU-oriented and segment whole people or objects. On 2 vCPU the flow-warp plus envelope approach costs about 100× less and needs no new model.

### Gaps
- No published CPU timing for RVM, XMem or Cutie at 256 px; whether a 256 px RVM fits in about 0.1 s on 2 vCPU is unknown.
- No source found that gives specific attack/release or hysteresis constants for segmentation masks; the constants above are engineering suggestions to tune on test clips.
- The DIS paper's per-preset Hz figures could not be read (arxiv blocked); the OpenCV preset-to-paper mapping is unverified.
- Rope, roop forks, DeepFaceLab/DeepFaceLive and commercial tools were not inspected (no reachable source in this session). **[unverified]** DeepFaceLab's XSeg is a per-frame segmentation model with no temporal component.

---

## Q2. Avoiding flicker of a per-frame binary occluder decision without delaying a real occluder

### Takeaway
Keep the decision soft for as long as possible. Use the look-ahead frame as a confirmation vote instead of a persistence counter, so a real occluder is accepted on the frame it appears. Make only the release slow (envelope decay plus hysteresis), and filter tiny or interior components spatially.

### Cited Findings
- FaceFusion's only anti-noise step is spatial: a Gaussian blur with sigma 5 then `clip(0.5,1)` ramp. There is no persistence or temporal vote — [FaceFusion face_masker.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_masker.py)
- FaceFusion's `many` occluder mode combines several XSeg models with a pixel-wise minimum of face probability, which favours declaring an occlusion. This is an ensemble along the model axis, not over time — [FaceFusion face_masker.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_masker.py)
- The VOS chatter-reduction patent averages warped neighbour-frame masks and then thresholds, which is a temporal vote on soft values — [US 8358691](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8358691)
- A 2026 video-segmentation paper suppresses flicker by penalising prediction changes only in stable regions, keeping real changes sharp. It is a training-time loss, but the principle is the same as change-gated smoothing — [arXiv 2602.09648](https://arxiv.org/pdf/2602.09648)
- **[measured here]** Morphological open (7×7 ellipse) plus `connectedComponentsWithStats` costs about 0.26–0.39 ms at 256 px.

### Inferences
- A **persistence counter** of the form "on after N frames" always delays a real occluder by N−1 frames. With a 1-frame look-ahead the equivalent rule has no delay: declare the occluder at frame t if it is present at t and at t+1 (or the centred median of 3). That rejects one-frame false positives while accepting at t. With a 2-frame look-ahead, "2 of {t, t+1, t+2}" also rejects 2-frame blips but accepts at t.
- Asymmetric gains: attack gain 1.0 (take the new value immediately when it is larger), and release by multiplying by about 0.6–0.75 per frame, or equivalently an EMA with α ≈ 0.25–0.4 used only when the value falls. A symmetric EMA such as Deep-Live-Cam's α = 0.2 frame blend delays the attack by about 4–5 frames and ghosts, which is exactly the failure to avoid.
- To avoid "erosion flicker" along the occluder edge, apply the envelope to the soft probability rather than to a binary mask, then feather the paste mask. That gives a temporally stable soft edge. A small dilation of 2–4 px of the occluder region before feathering keeps swapped pixels from bleeding onto fingers. Suggested, not sourced.
- Special case, a second face occluding during a kiss: XSeg can classify the other person's face as "face". Combine the occluder mask with a geometric test: pixels inside the other tracked face's projected hull that lie in front, judged by depth order (the larger or nearer face, or the face whose hull area is still growing), should count as occluder. This is an inference; no source found.

### Gaps
- No quantitative study found comparing persistence, median and hysteresis for occluder masks in face swap; values need tuning on the project's own contact clips (task #6 in the current plan).

---

## Q3. Tracking faces through partial or total occlusion and detector dropouts; keeping the swap painted

### Takeaway
Treat each face as a track that survives missed detections. Predict its geometry during a gap with a Kalman filter or by flow-carrying the landmarks with Lucas-Kanade (about 1 ms for 68 points). Accept low-score detections for existing tracks, as ByteTrack does with its 0.1 floor. When a later detection exists inside the processing chunk, interpolate between anchors, which is what FaceFusion 3.9 now does. Hold the track for about 30 frames (≈1 s), the default track buffer in ByteTrack and OC-SORT. Keep painting the swap at the predicted geometry instead of reverting; the occluder mask already hides it where the face is covered.

### Cited Findings
- **FaceFusion 3.9.2 now has a face tracker** (`--face-tracker-score`, default 0.0 = off, range 0–0.5). It detects faces in a window of `target_frame_amount` frames (default 2, range 0–10) around the target frame. It links detections into tracks by greedy IoU against each track's last box (above `face_tracker_score`) and fills gaps with `refill_faces`, which linearly interpolates the bounding box and the 5- and 68-point landmarks between the two neighbouring anchor detections: `average_factor = (gap_index − prev)/(next − prev)`. The embedding, scores and attributes come from whichever anchor is nearer. The target face is output if the target frame lies inside a track's first-to-last range — [FaceFusion face_tracker.py / face_creator.py / program.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_tracker.py)
- SORT (Bewley 2016) community defaults are `max_age = 1`, `min_hits = 3`, IoU threshold 0.3, with a constant-velocity Kalman prediction while unmatched — [SORT implementation (ByteTrack repo copy)](https://huggingface.co/spaces/ECCV2022/bytetrack/blob/refs%2Fpr%2F1/yolox/sort_tracker/sort.py). A crowd head-tracking study found that larger `max_age` improved results, in contrast to remarks in the original SORT paper — [arXiv 2103.13516](https://arxiv.org/pdf/2103.13516)
- ByteTrack official code: `track_thresh` 0.5, `track_buffer` 30 frames (scaled by `frame_rate/30`), `match_thresh` 0.8 (first association, IoU distance). The second association uses low-score detections with `0.1 < score < track_thresh` against still-unmatched tracks with IoU-distance threshold 0.5. Unconfirmed tracks match at 0.7. New tracks need `score ≥ track_thresh + 0.1` — [ByteTrack byte_tracker.py / demo_track.py](https://github.com/ifzhang/ByteTrack/blob/main/yolox/tracker/byte_tracker.py)
- OC-SORT official code defaults: `max_age = 30`, `min_hits = 3`, `iou_threshold = 0.3`, `delta_t = 3` (velocity estimated from the observation 3 steps back), `inertia = 0.2` (weight of the direction-consistency cost, OCM), optional `use_byte`. **ORU:** when a lost track is re-associated, the code generates a virtual trajectory by linear interpolation between the last observation and the new one (`dx = (x2−x1)/time_gap`, and so on) and re-runs the Kalman updates along it, removing error built up while the track was lost. **OCR:** a second association matches unmatched detections against each unmatched track's *last observation* box rather than its Kalman prediction. Association alone runs at about 700 FPS on an i9 3.0 GHz CPU — [OC-SORT GitHub ocsort.py / kalmanfilter.py / README](https://github.com/noahcao/OC_SORT); [paper arXiv 2203.14360](https://arxiv.org/abs/2203.14360)
- A critique notes that OC-SORT can still swap IDs when two objects cross or occlude each other for many frames, because it uses no appearance features and assumes linear interpolation — [Labellerr OC-SORT blog](https://www.labellerr.com/blog/oc-sort/) (secondary source)
- **[measured here]** Pyramidal LK for 68 points (win 21, 3 levels) costs 0.6–1.0 ms at 256 px and 1.1–1.2 ms at 512 px.

### Inferences
- **Recipe B: never revert to the original face.**
  1. Each face is a track with state = 5/68 landmarks plus the similarity/affine transform to the aligned crop, and the identity embedding taken from confident frames only.
  2. Per frame: detect with a lowered threshold for existing tracks (about 0.1–0.3, as in ByteTrack's second stage), associating against the predicted box with IoU ≥ 0.5 (IoU distance ≤ 0.5). New tracks still need a high score (≥ 0.6) and 2–3 hits.
  3. If there is no detection, or the occluded fraction of the face (from the occluder mask) exceeds about 30–40%: do not trust fresh landmarks. Propagate the previous landmarks by LK on the *visible* region, using forward-backward LK error below 1–2 px to reject points on the occluder. Fit a similarity transform to the surviving points with `cv2.estimateAffinePartial2D` + RANSAC, and apply it to the full previous landmark set. If fewer than about 6 points survive, fall back to the Kalman or constant-velocity prediction (as in SORT/OC-SORT). Thresholds are my suggestions.
  4. Look-ahead inside the chunk: if the track is re-detected within the chunk, replace the predictions with FaceFusion-style linear interpolation between anchors (ORU-like). That is exact at both ends and avoids the snap when tracking is re-acquired.
  5. Keep painting the swap at the predicted geometry for up to about 30 frames (1 s at 30 fps, matching ByteTrack `track_buffer` and OC-SORT `max_age`). Run the swapper normally on the predicted crop. If the face is fully covered, the occluder mask hides it anyway, and nothing ever reverts to the original. If the track is not re-acquired after the hold, fade the swap alpha out over about 8–10 frames, which is rarely visible because by then the face is usually gone or covered.
  6. Identity guard during a kiss or crossing: do not reassign a track because of an embedding drop while it is occluded. Gate re-ID on embeddings only from frames with occlusion below about 20%, and use OC-SORT's direction-consistency (OCM) and last-observation matching (OCR) to keep two touching faces from swapping IDs.
- Cheaper alternative to running the swapper on predicted frames when the CPU budget is tight: warp the last good swapped crop by the predicted affine and paste it, which holds the last swap. This introduces frozen expressions if it lasts more than a few frames, so limit it to about 3–5 frames. Inference, not sourced.
- FaceFusion's tracker is greedy IoU with no motion model, so fast motion or a hand covering most of the face can break the link. Its windowed gap-fill only covers frames between two detections inside the default window of ±2 frames.

### Gaps
- DeepSORT and appearance-based re-ID details were not researched here (low priority on 2 vCPU).
- No source found on how commercial tools hold the swap during dropouts.
- No reachable source quantified how often RetinaFace/SCRFD drops faces under partial occlusion.

---

## Q4. Landmark stability under occlusion; smoothing filters (One Euro) and their lag

### Takeaway
Use the 1€ filter (Casiez et al., CHI 2012) for causal landmark or transform smoothing. Start at `mincutoff ≈ 1 Hz` and `beta` found by an order-of-magnitude sweep, with `dcutoff = 1 Hz`. Because a 1–2 frame look-ahead is available, a centred (zero-phase) smoother such as [0.25, 0.5, 0.25] over t−1..t+1 removes jitter with no lag at all and is preferable for chunked offline processing. Freeze or downweight landmarks while the face is occluded instead of smoothing garbage.

### Cited Findings
- 1€ filter tuning from the authors: set `beta = 0` and `mincutoff` to about 1 Hz; then move quickly while increasing `beta` (start at 0.001 or 0.0001 and step by ×10) until the lag is acceptable. If slow-speed jitter is the problem, decrease `mincutoff`; if high-speed lag is the problem, increase `beta` — [Casiez 1€ page](https://cristal.univ-lille.fr/~casiez/1euro) (via search snippet); [CHI 2012, HAL](https://hal-lara.archives-ouvertes.fr/CRISTAL-MINT/hal-00670496v1)
- Reference implementation defaults: `freq = 120 Hz`, `mincutoff = 1.0 Hz`, `beta = 0.1`, `dcutoff = 1.0 Hz` — [casiez/OneEuroFilter python](https://github.com/casiez/OneEuroFilter); [PyPI oneeurofilter](https://pypi.org/project/oneeurofilter)
- **[unverified]** Filter maths: the cutoff is `fc = mincutoff + beta·|dx̂/dt|`, the smoothing factor is `α = 1/(1 + τ/Te)` with `τ = 1/(2π·fc)`, and the derivative is itself low-passed at `dcutoff`. These are the standard published equations; I could not fetch the paper in this session.
- Computed from those equations at 30 fps (Te = 1/30 s): fc = 0.5 Hz → α = 0.095 (≈10.5-frame time constant); 1 Hz → 0.173 (≈5.8 frames); 2 Hz → 0.295 (≈3.4 frames); 5 Hz → 0.512 (≈2 frames). This is my own arithmetic.
- Deep-Live-Cam deliberately avoided landmark-based masks and EMA, deriving the paste mask from the swap's own affine transform, because seamlessClone is "hyper-sensitive" to landmark jitter in the mask boundary — [Deep-Live-Cam face_swapper.py](https://github.com/hacksider/Deep-Live-Cam/blob/main/modules/processors/frame/face_swapper.py)
- **[unverified]** Occlusion-aware landmark detectors that output per-landmark visibility exist, for example LUVLi (CVPR 2020: landmark location, uncertainty and visibility likelihood). Detectors such as PIPNet and 3D-model fitters (3DDFA) degrade more gracefully under occlusion. Not checked against sources here.

### Inferences
- At 30 fps, a 1 Hz mincutoff gives an effective lag of several frames for slow motion. The β term raises the cutoff during fast motion, but the first frames of a fast move still lag. For offline chunked processing a centred smoother is strictly better. For landmarks or the affine transform at t, use `x̂_t = 0.25·x_{t−1} + 0.5·x_t + 0.25·x_{t+1}` (zero phase). Or run a 1€ filter forward and backward over the chunk. Reset at cuts.
- Smooth the **alignment transform** (scale, rotation, translation; 4 DOF) rather than 68 independent points. It is cheaper, and per-point noise cancels in the fit. Keep the expression-carrying landmarks (mouth) lightly smoothed or unsmoothed, otherwise lip motion lags.
- Visibility proxy without a new model: per landmark, use the occluder mask value at that landmark's crop position. Points with occluder probability above 0.5 get zero weight in the transform fit and keep their last good or flow-carried value.

### Gaps
- The paper's own recommended values for specific signal types (for example the mouse-pointer example) could not be verified.
- No source measured 1€ lag specifically on face landmarks at 25–30 fps.

---

## Q5. Shot and scene change handling (resetting temporal state)

### Takeaway
No reachable source covered reset logic for face-swap temporal filters specifically. The standard practice is inferred below. Because frames arrive in chunks with look-ahead, a cut at t+1 can be detected before frame t is finalised, so no filter ever blends across it.

### Cited Findings
- ByteTrack scales its lost-track buffer by frame rate (`buffer_size = frame_rate/30 · track_buffer`), so hold durations should be specified in seconds, not frames — [ByteTrack byte_tracker.py](https://github.com/ifzhang/ByteTrack/blob/main/yolox/tracker/byte_tracker.py)
- Deep-Live-Cam resets its interpolation state (`PREVIOUS_FRAME_RESULT = None`) on error or retry, and checks for a shape mismatch before blending. That is a minimal reset — [Deep-Live-Cam core.py / face_swapper.py](https://github.com/hacksider/Deep-Live-Cam/tree/main/modules/processors/frame)
- **[unverified]** PySceneDetect's ContentDetector, an HSV content-difference detector, has a default threshold of 27; histogram-correlation or mean-absolute-difference tests on downscaled frames are the usual cheap cut detectors.

### Inferences
- Reset triggers, any of which should clear the mask envelope, the flow state, smoother history and track predictions, and start new tracks:
  1. a global cut metric on a 64×64 thumbnail (HSV histogram correlation below about 0.5, or mean abs diff above a tuned threshold), tested between t and t+1 using the look-ahead;
  2. per-face: the new detection's IoU with the predicted box is below about 0.1 while a high-score detection exists elsewhere;
  3. flow-validity failure (high forward-backward error or high warped photometric error), which resets only the mask propagation for that face.
- For a gradual dissolve, do not reset; instead skip flow warping (step 2 of Recipe A) while the photometric error is high.
- After a reset, the first frame of a shot has no history. Use the look-ahead frames (t+1, t+2) for the median vote instead of t−1.

### Gaps
- No source found on cut detection inside FaceFusion, Rope or Deep-Live-Cam; thresholds above are untested suggestions.

---

## Cost summary against the 0.1–0.3 s per face per frame budget (2 vCPU)
- DIS ULTRAFAST 256 px: about 0.6 ms; remap for 2 warps (prev state, look-ahead): about 0.3 ms; median-of-3 and envelope (numpy, 256²): **[unverified]** about 1 ms; morphology + connected components: about 0.4 ms; LK on 68 points: about 1 ms; affine fit: under 1 ms. Measured items: [measured here], OpenCV 5.0.0, Xeon 2.1 GHz.
- Total temporal machinery: about 3–5 ms per face, i.e. about 2–5% of the smallest budget. The dominant extra cost is running XSeg on the look-ahead frame, which the chunked pipeline already computes for t+1 anyway, plus running the swapper on predicted-geometry frames during dropouts, which is the same cost as a normal frame.
