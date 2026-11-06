# Geometry under contact: disfigurement, flicker and reverting to the original face (SCRFD + 2d106det + inswapper_128, CPU)

Scope note: network access in this session was limited to GitHub (raw source and READMEs) plus web search snippets. arxiv.org, openaccess.thecvf.com, docs.opencv.org and the 1€ homepage could not be fetched directly, so paper numbers come from authors' GitHub READMEs where possible. Numbers taken only from search snippets are marked "(snippet, not verified against the paper)". Calculations I made myself are marked "(own calculation)".

## 1. Landmarks under occlusion: how wrong do 2D regressors get, and which cheap confidence proxies exist?

### Takeaway
All mainstream 2D landmark regressors always output a full, plausible-looking shape. On the WFLW occlusion subset their mean error rises by only about 18–29% over the full test set, but the tail grows much more: for SPIGA the failure rate doubles (2.08% to 4.48%) and the 99th-percentile error reaches 16.4% of inter-ocular distance. Neither 2d106det nor the InsightFace landmark wrapper outputs a per-point confidence. Practical proxies are: a heatmap-peak score (FaceFusion does this for 2DFAN4), the residual of a similarity fit to a reference shape, and disagreement between SCRFD's 5 keypoints and the 5 points derived from the 106 landmarks.

### Cited Findings
**Occluded vs full-set error (WFLW, NME normalised by inter-ocular distance, %)**
- ADNet README table (WFLW Test / Occlusion): ESR 11.13/13.75; SDM 10.29/13.03; CFSS 9.07/11.76; DVLN 6.08/7.33; LAB 5.27/6.79; Wing 5.11/6.37; DeCaFA 4.62/5.74; AWing 4.36/5.19; ADNet 4.14/5.06. LUVLi is listed at 4.37 on the full set with no subset numbers. — [ADNet README](https://github.com/huangyangyu/ADNet)
- HRNetV2-W18 (9.3M params, 4.3 GFLOPs) on WFLW: test 4.60, pose 7.86, illumination 4.57, occlusion 5.42, blur 5.36, makeup 4.26, expression 4.78. On COFW (heavily occluded): NME 3.45, FR0.1 0.20%. — [HRNet-Facial-Landmark-Detection README](https://github.com/HRNet/HRNet-Facial-Landmark-Detection)
- SPIGA on WFLW full: NME_ioc 4.060, AUC_10 60.56, FR_10 2.08%, P90 6.77, P95 8.20, P99 13.07. On the occlusion subset: NME 4.952, AUC_10 53.31, FR_10 4.484%, P90 8.09, P95 9.93, P99 16.44. Pose subset: NME 7.141, FR_10 11.66%, P99 26.89. COFW68: NME_bbox 2.517, FR_7 0.0%, P99 5.558. — [SPIGA README](https://github.com/andresprados/SPIGA)
- SPIGA's stated design goal is "producing plausible face shapes in presence of strong appearance changes" (a CNN plus a GNN shape prior). — [SPIGA README](https://github.com/andresprados/SPIGA)
- STAR loss (CVPR 2023): about 4.80 NME on the WFLW occlusion subset, with ADNet at 5.06 (snippet, not verified against the paper; the column was inferred from position). The STAR README only claims SOTA on COFW, 300W and WFLW and shows its results as an image. — [Heatmap Regression without Soft-Argmax, arXiv 2508.14929 (snippet)](https://arxiv.org/pdf/2508.14929); [STAR README](https://github.com/ZhenglinZhou/STAR)
- COFW (inter-pupil NME): human 5.60, RCPR 8.50, Wing 5.44, AWing 4.94, ADNet 4.68 (FR10 0.59%). — [ADNet README](https://github.com/huangyangyu/ADNet)
- PFLD: I found no WFLW-occlusion number in an accessible source (the repo READMEs show only images).

**Models that output visibility or uncertainty**
- LUVLi (CVPR 2020) predicts, for each landmark, its location, an uncertainty and a visibility likelihood. It is trained with binary cross-entropy for visibility plus a negative log-likelihood for location and uncertainty. Externally occluded landmarks get higher predicted uncertainty than unoccluded ones on MERL-RAV and on COFW-68 cross-dataset. MERL-RAV (more than 19,000 faces) labels each point as unoccluded, self-occluded or externally occluded. I found no official ONNX or CPU-ready release. — [LUVLi arXiv 2004.02980 (via search snippet)](https://arxiv.org/abs/2004.02980)

**What the target pipeline's landmark model actually does**
- InsightFace `2d106det`: MobileNet-0.5 backbone, 5 MB, 192×192 input, "loose cropped detection bounding-box", direct coordinate regression ("fast coordinate regression"). It is described as "mainly used for static image inference". A video variant, `2d106track`, which takes a landmark-derived box, is listed as "coming soon". — [InsightFace coordinate_reg README](https://github.com/deepinsight/insightface/blob/master/alignment/coordinate_reg/README.md)
- The InsightFace `Landmark.get()` crop is centred on the detector box centre with `_scale = input_size / (max(w,h)*1.5)`. The landmark crop therefore depends entirely on the SCRFD box, and the model returns coordinates only, with no score. — [insightface landmark.py](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/landmark.py)

**Confidence proxies used in practice (FaceFusion)**
- 2DFAN4 score = mean over 68 points of each point's heatmap maximum (`numpy.amax(face_heatmap, axis=(2,3))` then `mean`), linearly mapped from [0, 0.9] to [0, 1]. Peppa_wutz score = mean of its per-point third output channel, mapped from [0, 0.95] to [0, 1]. FaceFusion keeps 2DFAN4 unless `score_2dfan4 < score_peppa − 0.2`. — [FaceFusion face_landmarker.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_landmarker.py)
- Gating: if the 68-point landmarker score exceeds `face_landmarker_score` (default 0.5, range 0–1, step 0.05), FaceFusion builds the 5 alignment points from the 68 landmarks (`'5/68'`). Otherwise it keeps the detector's own 5 keypoints. Embedding and alignment both use `'5/68'`. — [FaceFusion face_creator.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_creator.py); [program.py defaults](https://github.com/facefusion/facefusion/blob/master/facefusion/program.py)

### Inferences
- Unit conversion to the inswapper crop (own calculation). The 128-px inswapper template is the ArcFace 112 template shifted +8 px in x (`diff_x = 8.0*ratio` in `estimate_norm`). The two eye points sit 35.2 px apart, so outer-canthus inter-ocular distance is roughly 48–53 px (estimate, assuming outer-canthal ≈ 1.4–1.5× interpupillary). SPIGA's occlusion-subset errors then come to roughly: mean 4.95% ≈ 2.5 px, P95 9.9% ≈ 5 px, P99 16.4% ≈ 8 px per landmark in crop pixels. ([face_align.py](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py) for the template)
- WFLW's "occlusion" subset is mostly hands, objects and hair on a single face. In kissing or cheek-to-cheek contact, the occluder is another face whose eyes, nose and mouth texture is in-distribution. Coordinate regressors such as 2d106det have no heatmap to show bimodality, and they may snap the hidden points onto the partner's features rather than spread them. The benchmark errors above are therefore a lower bound for contact. This is unverified; I found no benchmark of face-on-face occlusion.
- With no visibility output, three proxies suit a millisecond budget:
  (a) SCRFD-5 versus 106-derived-5 disagreement, normalised by eye distance. These two come from different networks with different crops, so a large disagreement on one point flags that point.
  (b) Per-point residual after a robust similarity fit of the 106 points to a per-track reference shape. Use the track's own median shape from clean frames, not a global mean, so expression is not over-penalised.
  (c) Temporal acceleration of each point relative to the track's smoothed trajectory.
  Each costs well under 0.1 ms in NumPy (estimate).
- Mouth corners and the lower jawline are the points most likely to be hidden in a kiss, and they are exactly the two lower points of the 5-point alignment. See section 2 for how their errors turn into scale and rotation wobble.

### Gaps
- No published NME for 2d106det on WFLW, COFW or any occlusion subset was found. PFLD occlusion numbers are not verified, and the STAR occlusion figure is from a snippet.
- No benchmark of landmark accuracy when the occluder is another face.
- LUVLi visibility accuracy (paper Table 9) could not be retrieved. No CPU or ONNX timings for SPIGA, STAR or LUVLi were found in accessible sources.

## 2. Robust alignment (visible-points fitting, shape priors, 3DMM) and temporal smoothing; what causes "breathing"

### Takeaway
The 5-point least-squares similarity fit (Umeyama via scikit-image `SimilarityTransform.estimate`, as used by InsightFace) is very sensitive to the two mouth points. A 5-px mouth-corner error (about P95 under occlusion) changes crop scale by about 4.6%, and 1 px of independent noise per point gives about 1.9% scale jitter per frame. Weighting or RANSAC that removes the hidden points fixes the bias but increases jitter (2.65% with mouth weight 0.1), so a robust fit must be paired with temporal smoothing. A 3DMM fit (3DDFA_V2) costs about 2.25 ms per face on 2 CPU threads and is a reasonable hard-frame-only fallback.

### Cited Findings
- InsightFace alignment for inswapper: `norm_crop2(img, target_face.kps, 128)` → `estimate_norm`, which does `SimilarityTransform().estimate(lmk, dst)` on exactly 5 points with equal weights. The template for 128 px is `arcface_dst*1.0 + [8,0]`. — [face_align.py](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py); [inswapper.py](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/inswapper.py)
- OpenCV `estimateAffinePartial2D` (4-DoF similarity: rotation, uniform scale, translation) defaults: `method=RANSAC`, `ransacReprojThreshold=3`, `maxIters=2000`, `confidence=0.99`, `refineIters=10` (Levenberg–Marquardt refinement on inliers). `LMEDS` is available and needs no threshold. — [OpenCV calib3d.hpp](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/include/opencv2/calib3d.hpp)
- 3DDFA_V2 (MobileNet-V1, 120×120 input, 3.27M params, 183.5M MACs) regresses 3DMM parameters in 4.4 ms with 1 thread, 2.25 ms with 2 threads and 1.35 ms with 4 threads in onnxruntime 1.5.1 on an i5-8259U. MobileNet ×0.5 (0.85M params): 1.37 / 0.7 / 0.5 ms. The repo states 3DMM regression is "about 1~2ms for one face" and dense reconstruction (38,365 points) about 1 ms. It notes that "tracking applications may benefit… since detection is not needed for every frame". — [3DDFA_V2 readme](https://github.com/cleardusk/3DDFA_V2/blob/master/readme.md)
- 3DDFA_V2's video smoothing demo uses a centred moving average with `n_pre=1`, `n_next=1` by default (a 3-frame window, look-ahead of 1 frame). — [3DDFA_V2 demo_video_smooth.py](https://github.com/cleardusk/3DDFA_V2/blob/master/demo_video_smooth.py)
- 1€ filter reference Python implementation defaults: `mincutoff=1.0` Hz, `beta=0.0`, `dcutoff=1.0` Hz. `freq` is required (the frame rate when there are no timestamps). Lower `mincutoff` removes more jitter, and `beta` reduces lag at speed. — [casiez/OneEuroFilter python](https://github.com/casiez/OneEuroFilter); paper: Casiez, Roussel, Vogel, CHI 2012, [doi:10.1145/2207676.2208639](https://doi.org/10.1145/2207676.2208639)
- 1€ tuning procedure from the author: set `beta=0` and `mincutoff` to about 1 Hz, adjust `mincutoff` while the signal is still or slow until jitter is acceptable, then move fast and raise `beta` (in ×10 steps) until lag is acceptable. "If high speed lag is a problem, increase beta; if slow speed jitter is a problem, decrease fcmin." — [1€ author page (via search snippet)](https://cristal.univ-lille.fr/~casiez/1euro); [OneEuroFilterArduino README](https://github.com/casiez/OneEuroFilterArduino/)
- Disney's high-resolution neural face swapping (Naruniec et al., CGF 2020) credits its temporal stability to a refinement step added to face-landmark stabilisation, not to the generator. — [Disney Research page](https://studios.disneyresearch.com/2020/06/29/high-resolution-neural-face-swapping-for-visual-effects/)
- Deep-Live-Cam's comments say its Poisson-blend mask "tracks the swapped face exactly per-frame — no landmark jitter, no EMA, no lag". Its optional "interpolation" is a whole-frame blend: `final = prev_result*(1−w) + current*w` with default `interpolation_weight=0.2`, which is 80% previous output (prone to ghosting). — [Deep-Live-Cam face_swapper.py](https://github.com/hacksider/Deep-Live-Cam/blob/main/modules/processors/frame/face_swapper.py)

### Inferences (own calculations on the 128-px inswapper template unless noted)
- Sensitivity of the 5-point similarity fit. Centred template spread S = Σ|p_i|² ≈ 2702 px² (√S ≈ 52 px), and the mouth-corner radius is about 25.1 px.
  - A 5-px radial error on one mouth corner gives a scale error of +4.6%.
  - A 5-px tangential error gives a rotation error of 2.7°.
  - Independent 1-px Gaussian noise on all 5 points gives a scale standard deviation of 1.9% per frame. Generally, scale std ≈ σ/√S and rotation std ≈ σ/√S rad.
  - With mouth weights 0.1, the 5-px mouth error falls to +1.5% scale and 0.16°, but noise jitter rises to a 2.65% scale std, because the fit now relies on 3 points spaced about 35 px apart.
- Cause of "breathing" and scale wobble. The similarity scale comes mostly from the eye–mouth vertical span, and the mouth points move with expression (kissing purses and protrudes the lips) and with occlusion. Mouth error goes straight into crop scale, and the pasted face then pulses even when the head is still. Remedies:
  - estimate scale from rigid points (eye corners, nose bridge and tip from the 106 set) and only translation and rotation from the full set;
  - smooth log-scale with a lower cutoff than translation;
  - optionally lock scale per track (median over clean frames) and let it vary only slowly.
- Recommended robust fit (ms-class).
  1. Build correspondences from more than 5 points. Use about 20–30 rigid 106-point indices (eyes, brows, nose) mapped to a per-track reference shape captured on clean frames, plus SCRFD's 5 keypoints.
  2. Fit with `cv2.estimateAffinePartial2D(..., method=cv2.RANSAC, ransacReprojThreshold≈2–3% of eye distance)`, or with weighted Umeyama (closed form: weighted centroids, weighted cross-covariance, SVD).
  3. Convert the inlier fit into the equivalent 5-point target via the reference shape, so inswapper still gets its canonical template.
  With 25–30 points, RANSAC with 2000 iterations costs well under 1 ms (estimate). The extra 106-point inference costs about one MobileNet-0.5 at 192 px, a few ms on CPU (estimate, not measured).
- Mean-shape or PCA prior as a cheaper alternative to a 3DMM: project the 106 points onto the first k shape modes of the track's own clean frames, after similarity normalisation. Points with large residual to the projection are outliers. Re-project with them down-weighted (an iteratively reweighted fit, 2–3 iterations, sub-ms). This is the same idea as SPIGA's "plausible shape" GNN, but done in closed form.
- Hard-frame fallback: run 3DDFA_V2 (about 2.25 ms per face at 2 threads, MobileNet; 0.7 ms for ×0.5) only when the proxies in section 1 flag a frame. Use its projected 68 points, which are always anatomically consistent, for the 5-point targets. 3DDFA_V2 still takes a box as input, so a merged box will still mislead it. Box hygiene (section 3) comes first.
- Smoothing choice. For offline video, use a centred window, not a causal filter: lag is the main visual artefact of causal filters, and the look-ahead removes it. FaceFusion (centred interpolation, section 3) and 3DDFA_V2 (centred mean, n_pre=n_next=1) both work this way. For live or causal use, a 1€ filter on (tx, ty, log s, θ) starting at mincutoff 1 Hz, beta 0, then tuned, is the standard choice. Smooth the transform parameters, not the 106 points independently.
- Do not use whole-frame temporal blending (the Deep-Live-Cam style) as a geometry stabiliser. It ghosts both faces during motion and does not fix wrong shapes.

### Gaps
- No published measurement of jitter or "breathing" reduction from RANSAC or weighted fits in face-swap pipelines was found. The numbers above are analytical only.
- No CPU timing for 2d106det was found. The 3DDFA_V2 timings are on a 4-core laptop i5 with onnxruntime 1.5.1, not a 2-vCPU server.
- Naruniec et al.'s landmark refinement details could not be retrieved (the PDF host was blocked).

## 3. Detection during contact: NMS, merged or split boxes, plausibility rejection, keeping tracks alive

### Takeaway
SCRFD in InsightFace uses hard greedy NMS at IoU 0.4 with a 0.5 score threshold. When two heads overlap enough (for example a kiss in profile, one head partly behind the other), the lower-scoring face can be deleted outright, and a merged box can appear. To keep both faces, raise the IoU threshold (for example 0.5–0.6) or use Soft-NMS. Lowering it suppresses more. Then deduplicate per track using keypoint geometry. FaceFusion's practice is to associate faces by IoU and linearly interpolate geometry across missed frames in a centred window. DeepFaceLab's practice is to re-extract only the failed frames manually.

### Cited Findings
- InsightFace SCRFD: `self.nms_thresh = 0.4`, `self.det_thresh = 0.5`. The NMS is standard greedy hard NMS: sort by score, keep the top box, drop any remaining box with IoU > 0.4. — [insightface scrfd.py](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/scrfd.py)
- FaceFusion runs a second NMS (`cv2.dnn.NMSBoxes`) over all detections with `score_threshold=face_detector_score` (default 0.5) and an IoU threshold from `get_nms_threshold`:
  - 0.4 normally;
  - 0.3, 0.2 or 0.1 when 2, 3 or 4 rotation angles are merged;
  - 0.1 for the "many" (multi-detector) mode.
  These choices remove duplicates aggressively and risk suppressing a second, overlapping face. — [FaceFusion face_helper.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_helper.py); [face_creator.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_creator.py)
- Score-specific NMS paper: with standard NMS, "in the case of overlapped two faces, this simple and crude method will cause missed detection". Its proposed NMS "preserves overlapped faces". — [Score-specific NMS and Coexistence Prior for Multi-scale Face Detection](https://dongl-group.github.io/pdf/Score-specific_Non-maximum_Suppression_and_Coexistence_Prior_for_Multi-scale_Face_Detection.pdf)
- Soft-NMS decays overlapping scores instead of deleting the boxes (linear or Gaussian weighting; detections below 0.001 are discarded). On COCO, R-FCN went from 33.9% to 34.8% (linear) and 35.1% (Gaussian) mAP; Faster R-CNN went from 24.4% to 25.5%. — [Soft-NMS repo README](https://github.com/bharatsingh430/soft-nms)
- Adaptive NMS raises the suppression IoU threshold in dense regions to recover overlapping true boxes (via search summary; primary paper not fetched). — [Adaptive NMS, arXiv 1904.03629](https://arxiv.org/abs/1904.03629)
- FaceFusion face tracker (current master):
  - `create_face_tracks` assigns each detected face to the track whose last face has the highest bounding-box IoU above `face_tracker_score`. The default is 0.0 (range 0–0.5, step 0.05), so any positive overlap links.
  - `track_faces` takes a list of frames and produces faces for the middle frame (`target_index = len//2`), a centred offline window.
  - `refill_faces` fills each gap between two detections by linear interpolation of the bounding box and all landmark sets. Each filled face is marked `origin='refill'` and takes its embedding and scores from the nearer anchor (`factor < 0.5` → first anchor).
  — [face_tracker.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_tracker.py); [face_creator.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_creator.py); [program.py](https://github.com/facefusion/facefusion/blob/master/facefusion/program.py); [choices.py](https://github.com/facefusion/facefusion/blob/master/facefusion/choices.py)
- FaceFusion reference-face matching: `reference_face_distance` default 0.3. The distance is `1 − dot(embedding_norm_a, embedding_norm_b)`, range [0, 2], mapped to [0, 1]. A face matches if the result is < 0.3, that is, raw cosine distance < 0.6, or cosine similarity > 0.4. — [face_selector.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_selector.py); [program.py](https://github.com/facefusion/facefusion/blob/master/facefusion/program.py)
- FaceFusion can also run the detector on rotated frames (`face_detector_angles` 0/90/180/270) and map boxes and keypoints back. — [face_detector.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_detector.py)
- DeepFaceLab extractor: `--manual-fix` "Enables manual extract only frames where faces were not recognized". `--manual-output-debug-fix` "Performs manual reextract input-dir frames which were deleted from [output_dir]_debug dir". Manual window size defaults to 1368. Detector choices are s3fd or manual. — [DeepFaceLab main.py](https://github.com/iperov/DeepFaceLab/blob/master/main.py)
- Deep-Live-Cam caches detections on Apple Silicon (`DETECTION_INTERVAL`, `FACE_DETECTION_CACHE`) rather than tracking. I found no IoU tracking in its face analyser. — [Deep-Live-Cam face_swapper.py](https://github.com/hacksider/Deep-Live-Cam/blob/main/modules/processors/frame/face_swapper.py)

### Inferences
- Threshold direction. In InsightFace and OpenCV, `nms_thresh` is the IoU above which the weaker box is deleted. Keeping two overlapping faces needs a higher value (for example 0.5–0.6), Soft-NMS, or NMS skipped for boxes already claimed by different tracks. "Lowering the NMS threshold" makes contact worse. Unverified on kiss footage; measure it on the test bench.
- Track-aware NMS (cheap and recommended).
  1. Run SCRFD with `nms_thresh` around 0.6 (or keep 0.4 and also take the raw pre-NMS candidates).
  2. Predict each live track's box and 5 keypoints from its motion model.
  3. Assign candidates to tracks by keypoint distance normalised by eye distance (more discriminative than IoU when heads overlap), then keep the best per track.
  4. Only treat a candidate as a new face if it matches no track.
- Merged-box and implausible-layout checks (thresholds to calibrate per track; not from a source):
  - box area or aspect ratio more than about 1.5× the track's running median, or a sudden jump in box centre;
  - eye distance / box width outside the track's median ±30%;
  - the nose keypoint outside the eye–mouth quadrilateral;
  - left and right eyes swapped (negative x-order after de-rotation);
  - 5-point similarity-fit residual to the template more than about 4 MAD above the track's clean-frame median.
  A merged box usually shows an inflated box with keypoints taken from one face. Trust the keypoints over the box and rebuild the box from the keypoints and the track scale.
- The 2d106det crop is centred on the box at 1.5× its larger side. A merged box therefore centres the 106-landmark crop between the two faces, which can produce a hybrid shape. Always re-crop for 2d106det from a box rebuilt from the track's keypoints, not from the raw SCRFD box. This is the main way geometry "disfigures" under contact.
- Keeping a track alive when detection drops. Offline, use FaceFusion-style centred interpolation between the last and next good detections. For long gaps, interpolate the similarity parameters (tx, ty, log s, θ) rather than raw points. Causally, extrapolate with constant velocity, damped toward zero velocity over about 5–10 frames, then hold position. Practical frame counts are not published (gap).
- FaceFusion's tracker uses IoU with threshold 0.0 and only the last face of each track, so two faces in contact (high mutual IoU) can be cross-linked. Using keypoints plus embedding gating (section 4) is safer than copying it.

### Gaps
- No published measurement of SCRFD or RetinaFace recall on face-on-face overlap (kissing, cheek-to-cheek) at different NMS thresholds was found.
- The window length FaceFusion passes to `track_faces` could not be located (the caller was not found in accessible files).
- No published "max frames to predict before dropping" numbers from face-swap tools were found.

## 4. Swap quality gating: detecting a bad swapped crop and what to paint instead

### Takeaway
inswapper_128 is a single-frame model with no temporal state. Its identity input is the source embedding (`latent = normed_embedding @ emap`, L2-normalised), so per-frame flicker and disfigurement come from the target-side geometry and crop, not from the identity vector. Cheap gates are: embed the swapped crop with the same ArcFace model and compare it to the source embedding, check landmark consistency between the swapped crop and the target crop, and check the alignment transform jump. On failure, warp the last good swapped crop to the current (smoothed) geometry rather than showing the original.

### Cited Findings
- inswapper inputs: `aimg, M = norm_crop2(img, target_face.kps, 128)` and `latent = normed_embedding.reshape(1,-1) @ emap`, then `latent /= norm(latent)`. The output is a 128×128 BGR crop. `paste_back=False` returns `(bgr_fake, M)`, which allows custom compositing. — [inswapper.py](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/inswapper.py)
- inswapper's default paste mask:
  - The whole warped square crop (`img_white`), thresholded at more than 20 → 255, eroded with kernel `max(mask_size//10, 10)` and Gaussian-blurred with kernel `2*max(mask_size//20, 5)+1`, where `mask_size = sqrt(h*w)` of the warped square.
  - A `fake_diff` map (difference threshold 10) is computed, but the final mask is the square-based `img_white` (the line `#img_mask = fake_diff` is commented out).
  — [inswapper.py](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/inswapper.py)
- UniFaceGAN (TIP 2021) attributes flicker in facial video editing to per-frame processing. It fixes it with a 3D temporal loss and region-aware normalisation, which needs a trained model, so it cannot be applied to a frozen inswapper. — [UniFaceGAN arXiv 2108.05650](https://arxiv.org/abs/2108.05650)
- Recent diffusion-based video face swapping (for example VFace, training-free, with temporal smoothing and attention regularisation across neighbouring frames) targets flicker from per-frame image swappers. It is not CPU-viable here. — [VFace summary (emergentmind, secondary source)](https://www.emergentmind.com/topics/vface)

### Inferences
- inswapper has no latent or temporal state to smooth, and the source embedding is constant. Temporal methods that are usable with it are those acting on geometry (smoothed M), on the mask (temporally smoothed occlusion and parse masks), and on output caching (reuse the last good crop). Latent smoothing could still apply if a per-frame averaged source embedding were used, but that does not address contact-induced disfigurement.
- Identity gate (estimate): run the ArcFace recogniser (w600k_r50 in buffalo_l, about tens of ms on 2 vCPU; not measured here) on the swapped 128 crop, re-aligned to 112. Compare the cosine similarity with the source embedding against the track's running median on clean frames; flag a drop of more than about 0.15. Running the recogniser every frame may exceed the ms budget, so run it on frames already flagged by the geometry proxies, or every Nth frame. FaceFusion's own matching threshold corresponds to cosine distance < 0.6 (section 3), which is far too loose for quality gating.
- Geometry consistency gate (ms-class): run SCRFD keypoints or 2d106det on the swapped output crop and compare them with the target crop's landmarks in crop coordinates. inswapper preserves target pose and expression, so a large deviation means the output is malformed. A cheaper version compares only the 5 SCRFD points on a 128 crop (estimate).
- Transform-jump gate: if |Δlog s| > 3–5% or |Δθ| > 3° frame-to-frame while the smoothed trajectory is calm, treat the raw geometry as untrusted and use the carried (predicted or smoothed) geometry. These thresholds come from section 2's sensitivity numbers (about 4.6% per 5-px mouth error).
- Fallback order when a frame fails:
  1. Re-run inswapper on a crop built from the carried geometry (the identity is the same source embedding, so the output stays on-identity).
  2. If that also fails, warp the last good swapped 128 crop with M_current·M_last⁻¹ and composite it under the current occlusion mask.
  3. Never composite the raw target crop.
  Use a face-shaped or parse-based mask instead of inswapper's square-based mask during contact. The default square, eroded by about 10% and feathered, can spill swapped pixels over the partner's face when crops overlap.

### Gaps
- No published identity-similarity distributions for inswapper_128 outputs, so no source-backed thresholds; calibrate on the bench.
- No paper found that adds temporal consistency to inswapper specifically. No CPU timing for the ArcFace recogniser on 2 vCPU was retrieved.

## 5. "Never revert" policies: hold durations, stop conditions, fade-out

### Takeaway
None of the inspected tools (FaceFusion, Deep-Live-Cam, InsightFace) has an explicit "keep painting the swap through lost detections" policy. FaceFusion comes closest with centred interpolation inside a window. When nothing is detected, every tool simply leaves the original face, which is the "revert" seen during contact. A never-revert design has to be built: a track state machine with carried geometry, bounded hold, and an explicit end-of-track fade.

### Cited Findings
- FaceFusion fills only gaps between two detections of the same track inside its window (`refill_faces` interpolates between `anchor_index_previous` and the next detected index). If the target frame lies outside a track's first–last range, that track contributes no face for that frame, and the frame is left unswapped. — [face_tracker.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_tracker.py); [face_creator.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_creator.py)
- Deep-Live-Cam's only temporal carry-over is the optional whole-frame blend (`interpolation_weight` 0.2 on the current frame), which is not a per-face hold. — [Deep-Live-Cam face_swapper.py](https://github.com/hacksider/Deep-Live-Cam/blob/main/modules/processors/frame/face_swapper.py)
- DeepFaceLab's answer to frames where detection fails is manual re-extraction of exactly those frames (`--manual-fix`, `--manual-output-debug-fix`), which guarantees geometry for every frame in offline work. — [DeepFaceLab main.py](https://github.com/iperov/DeepFaceLab/blob/master/main.py)

### Inferences (proposed policy; numbers are engineering estimates to validate on the kiss bench, not sourced)
- Track states:
  - CONFIRMED: fresh, plausible geometry. Paint with fresh geometry, smoothed.
  - CARRIED: detection is missing, implausible, or fails the gates. Paint with predicted or interpolated geometry and the same source identity.
  - ENDING: fade out.
- Decision rule for each frame, carried versus fresh geometry: use fresh geometry only if all of these hold:
  - the box and layout are plausible (section 3);
  - the 106-vs-5 disagreement and reference-fit residual are within track norms (section 1);
  - the transform jump is within the gate (section 4).
  Otherwise use carried geometry. Offline, carried geometry is the interpolation between the bracketing good frames (FaceFusion-style), done in (tx, ty, log s, θ). Causally, it is a damped constant-velocity prediction.
- Hold duration (estimate): during contact, hold CARRIED for as long as the track's predicted region still contains skin or face pixels and has not been claimed by another track. Cap it at about 1–2 s (25–50 frames at 25 fps) without any confirming evidence. Offline, the cap matters less, because the next good detection usually brackets the gap.
- Stop conditions (only these end painting):
  - a shot cut (frame-difference or histogram jump);
  - the predicted box leaving the frame by more than about 50% of its area;
  - the hold cap reached with no partial evidence;
  - the identity gate on the target region showing a different, confirmed person for several consecutive frames.
  Do not stop for low detector score, a merged box, or landmark failure alone. Those trigger CARRIED, not ENDING.
- Ending (estimate): fade the swap alpha to 0 over about 6–10 frames, about 0.25–0.4 s at 25 fps, with geometry frozen. On a shot cut, end immediately with no fade, because a fade across a cut is visible as a ghost. Offline, if a track reappears within the hold window, retro-fill the gap by interpolation instead of fading.

### Gaps
- No published hold durations, fade lengths or stop rules from face-swap tools or papers were found. All numbers in this section are engineering estimates and should be set by testing on contact clips.
