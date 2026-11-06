# Keeping face identity bound to the right person through contact (kiss, cheek to cheek, hug)

Scope note: the target pipeline is CPU-only (InsightFace SCRFD + 106-pt landmarks + ArcFace w600k_r50 512-d, per-person slot with a locked reference embedding, Hungarian assignment per frame).

Method note: arxiv.org and openaccess.thecvf.com could not be fetched in this session (network policy). Paper-level facts from those hosts come from search-engine summaries and are marked as such. All tracker defaults were read directly from the official source code (shallow git clones of the official repos), which is the most reliable part of these notes.

---

## 1. Embedding behaviour: how ArcFace / w600k_r50 similarity reacts to occlusion, yaw, blur, low resolution and two-face crops; typical thresholds

### Takeaway
No published curve of "ArcFace cosine vs. occluded-area fraction" was found. What exists: (a) occlusion such as masks moves the *genuine* score distribution down toward the impostor distribution, while impostor scores barely move; (b) yaw beyond 30 degrees costs a few points of accuracy for ArcFace; (c) a crop that blends two identities can match *both* identities (face-morphing literature), so during a kiss the appearance score is unreliable in both directions. No official cosine threshold exists for w600k_r50; values from 0.25 to 0.5 are used in practice and must be calibrated on your own footage.

### Cited Findings

**Model facts (w600k_r50 / buffalo_l)**
- buffalo_l = SCRFD-10GF detector + ResNet50 trained on WebFace600K recognition + 2d106 and 3d68 landmarks. Reported accuracy: LFW 99.83, CFP-FP (frontal vs profile) 99.33, AgeDB-30 98.23, IJB-C (TAR at FAR=1e-4) 97.25, MR-ALL 91.25. buffalo_m has the same recognition accuracy. buffalo_s (MobileFaceNet) is much weaker: IJB-C 95.02, MR-ALL 71.87. — [InsightFace model_zoo README](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md)
- InsightFace's `Face` object exposes the raw `embedding`, `embedding_norm` (L2 norm of the raw vector) and `normed_embedding = embedding / embedding_norm`. The raw norm is therefore available at zero extra cost for use as a quality proxy. — [insightface/app/common.py](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/app/common.py)
- The ArcFace ONNX wrapper aligns the face with `face_align.norm_crop(img, landmark=face.kps, image_size=...)`, i.e. a similarity transform from the **5 SCRFD keypoints**, then normalizes with mean 127.5 / std 127.5. So the crop is entirely determined by 5 points. If the eye or mouth keypoints land on the other person during contact, the 112x112 crop will contain the other face. — [insightface/model_zoo/arcface_onnx.py](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/arcface_onnx.py)

**Thresholds (no official value)**
- I found no official InsightFace same-person threshold for buffalo_l or w600k_r50. — (search across InsightFace docs and model zoo, see the README above)
- Community pipeline defaults: cosine 0.25 for buffalo_l and 0.35 for antelopev2 (unverified, a third-party Hugging Face pipeline). — [AXERA-TECH Insightface pipeline](https://huggingface.co/AXERA-TECH/Insightface/blob/main/insightface_pipeline.py)
- Other practitioner write-ups use "0.5 or higher is often the same person". Attendance-system samples also default to 0.5 (unverified, blog-level). — [zenn.dev write-up](https://zenn.dev/taku_sid/articles/20250511_face_sim_metric?locale=en)
- ArcFace thresholds range from 0.328 to 0.823 on IJB-B still-image identification as FAR moves from 1e-1 to 1e-3 (footnote in AdvHat; a different ArcFace model and protocol). This shows how strongly the threshold depends on the operating point. — [AdvHat, arXiv 1908.08705](https://arxiv.org/pdf/1908.08705)
- IJB-C low-FMR thresholds do not transfer to other data, because IJB-C templates average embeddings from many images. A per-model threshold should come from your own impostor score quantiles. — [Breaking High Confidence, arXiv 2608.20884 (search summary)](https://arxiv.org/pdf/2608.20884)
- Face-specific MOT reference value: BoT-FaceSORT (SCRFD-10G + AdaFace IR-101 WebFace12M) accepts an appearance match only when halved cosine distance <= 0.3, i.e. **cosine similarity >= 0.4**. — [BoT-FaceSORT config botfacesort.yaml](https://github.com/bellhyeon/BoT-FaceSORT/blob/master/tracker/configs/botfacesort.yaml); [bot_facesort.py](https://github.com/bellhyeon/BoT-FaceSORT/blob/master/tracker/trackers/botfacesort/bot_facesort.py)

**Occlusion (masks as the best-studied proxy)**
- NIST IR 8311 (July 2020): every pre-COVID algorithm had higher false non-match rates on masked probes. The best algorithms went from about 0.3% to about 5% FNMR. Many otherwise competent algorithms failed 20-50% of the time. Accuracy got worse with more nose coverage (low, medium, high coverage levels). Black masks were worse than surgical blue, and round masks gave lower error. False positives stayed stable or fell slightly. — [NIST news release](https://www.nist.gov/news-events/news/2020/07/nist-launches-studies-masks-effect-face-recognition-software); [NIST.IR.8311 PDF](https://nvlpubs.nist.gov/nistpubs/ir/2020/NIST.IR.8311.pdf); [World Privacy Forum summary](https://worldprivacyforum.org/posts/face-recognition-and-face-masks-accuracy-of-face-recognition-plummets-when-applied-to-mask-wearers-nist-report/)
- Damer et al. (IET Biometrics 2021, includes ArcFace): with masked probes the genuine score distribution shifts toward the impostor distribution, and the impostor distribution stays largely unchanged for the stronger systems (ArcFace among them). Real masks shift genuine scores more than simulated masks. — [Damer et al., IET Biometrics 2021](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/bme2.12044); [PubMed 34221363](https://pubmed.ncbi.nlm.nih.gov/34221363/)
- One occlusion study reports ArcFace accuracy by occlusion type: mask 88.9%, hat 87.3%, sunglasses 81.4%, random occlusion 78.6%, 82.4% overall. Coverage is not graded by percentage. — [Sensors, doi 10.3390/s26133977 (search summary)](https://doi.org/10.3390/s26133977)

**Yaw / pose, blur, resolution**
- Forensic evaluation: ArcFace accuracy was 95.6% at baseline, about 93.8% with head rotation over 30 degrees, 91.4% with dark glasses and/or masks, and 88.4% with glasses or masks plus rotation plus low light combined. ArcFace degraded less than FaceNet. — [An Evaluation of Forensic Facial Recognition, arXiv 2311.06145 (search summary)](https://arxiv.org/pdf/2311.06145)
- Filtering out low-quality inputs raised the mean ArcFace genuine cosine similarity from 0.66 to 0.76 in a real-time screening study. Quality strongly shifts the genuine score distribution. — [Lightweight FIQA framework, arXiv 2507.15961 (search summary)](https://arxiv.org/html/2507.15961v1)
- CFP-FP 99.33 for buffalo_l shows that frontal-vs-profile verification remains strong on curated pairs. — [InsightFace model_zoo README](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md)

**Crops that mix two faces (closest literature: face morphing)**
- Morphing-attack metric MMPMR: a morph succeeds when its *minimum* similarity to images of **both** contributing subjects is above the verification threshold. Studies set that threshold at FMR 0.1% or 1%. One study reports MMPMR 91.30% at FMR 1%, meaning 91.3% of morphs verified against *both* identities. — [Deep embeddings for morphing pair selection, arXiv 2305.18216](https://arxiv.org/html/2305.18216); [PLOS ONE version](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0304610)
- ArcFace (ResNet-100) is the standard recognizer under test in morphing studies. Morphs made from similar-looking pairs (pre-selected by ArcFace or MagFace embeddings) give higher similarity scores to both contributors. — [arXiv 2305.18216](https://arxiv.org/html/2305.18216)
- Landmark-based image-level blending "highly preserves the characteristic of both bona fide identities". — [Optimal-Landmark-Guided Image Blending, arXiv 2401.16722](https://arxiv.org/pdf/2401.16722)
- Studies conflict on whether more accurate recognizers are more or less vulnerable to morphs. ArcFace and AdaFace show lower MMPMR in a later benchmark, while an earlier study found higher vulnerability for more accurate systems. — [MorFacing benchmark (ResearchGate)](https://www.researchgate.net/publication/385728910_MorFacing_A_Benchmark_for_Estimation_Face_Recognition_Robustness_to_Face_Morphing_Attacks); [arXiv 2305.18216](https://arxiv.org/html/2305.18216)

### Inferences
- Occlusion mostly lowers similarity to the *correct* identity and leaves similarity to *other* identities near the impostor level. So, for a slot whose face is half hidden behind the partner, expect a lower genuine score, not a higher impostor score. The danger case is a **mixed crop**, not a merely occluded one. A crop that holds cheek/nose/mouth parts of both people behaves like a crude morph and can score above threshold against *both* slots. A kiss crop's embedding should be treated as "between" the identities and given little weight, not trusted.
- Because the crop comes from 5 keypoints, a cheap contact flag is any of: the two faces' SCRFD boxes overlap (IoU > ~0.05-0.1), one face's 106-landmark convex hull intersects the other's box, or one face's 5 keypoints fall inside the other face's box. Inside that state, down-weight or ignore appearance for re-binding. (Numeric values are design suggestions, not sourced.)
- Practical threshold plan for w600k_r50 (inferred, needs calibration on your own clips). Take same-person and different-person cosines from clean frames of the two actors. Set "accept" near the 1st percentile of the genuine scores, typically somewhere in 0.3-0.5. During contact, do not relax the accept threshold to keep a match alive. Instead stop using appearance for *switching* decisions and keep the existing binding unless strong evidence accumulates (Q3/Q4).
- What counts in two-slot assignment is the *difference* between the two people's similarities, not each absolute value. Two different actors usually score far below 0.3 against each other on clean frames, but mixed crops compress that gap. Log `s(slotA,det) - s(slotB,det)` per frame as the main diagnostic.

### Gaps
- No source found that plots ArcFace cosine against occluded-area fraction (e.g., 10-50% synthetic occlusion). OCFR 2022 (structure-aware synthetic occlusions) may contain it, but it could not be fetched: [OCFR 2022, arXiv 2208.02760](https://arxiv.org/pdf/2208.02760).
- No study found on embeddings of naturally "two-faces-in-one-crop" images (kisses, cheek to cheek). The morphing literature is an imperfect proxy: morphs are pixel blends of aligned faces, not partial overlaps.
- NIST IR 8311 coverage-specific FNMR tables (e.g., per nose-coverage level) were not retrieved, and specific coverage percentages are unverified.
- No published w600k_r50-specific yaw/blur/resolution vs cosine curves found. Must be measured on the target footage.

---

## 2. Template update policy: when to update, freeze or blend; quality-aware aggregation

### Takeaway
All major trackers keep an EMA template with momentum 0.9-0.95 on the *old* template. The best ones make the update confidence-dependent: Deep OC-SORT ignores the new embedding entirely at the detection threshold, and Hybrid-SORT weights by relative detection confidence. A face-specific tracker (BoT-FaceSORT) keeps a *memory bank* of all past smoothed templates and matches against the closest one. For contact, the safe policy is: never update during contact, gate updates by quality, and keep a frozen pre-contact snapshot. Your locked user-chosen reference already acts as an anchor that cannot be poisoned.

### Cited Findings

**EMA momentum defaults (read from official code)**
- BoT-SORT: `smooth_feat = 0.9*smooth_feat + 0.1*feat`, then re-L2-normalized. Alpha is hard-coded at 0.9. — [BoT-SORT tracker/bot_sort.py](https://github.com/NirAharon/BoT-SORT/blob/main/tracker/bot_sort.py)
- StrongSORT: `--EMA_alpha` default 0.9. With EMA on, `nn_budget = 1` (only the smoothed feature is kept). Without EMA, a gallery of 100 features is kept. — [StrongSORT opts.py](https://github.com/dyhBUPT/StrongSORT/blob/master/opts.py)
- DeepSORT (original): no EMA. It keeps a gallery of past features (`--nn_budget` default None = unlimited) and uses the *minimum* cosine distance to the gallery. `--max_cosine_distance` default 0.2. — [deep_sort_app.py](https://github.com/nwojke/deep_sort/blob/master/deep_sort_app.py)
- Deep OC-SORT ("dynamic appearance"): `trust = (s_det - det_thresh)/(1 - det_thresh)`, `alpha = alpha_fixed_emb + (1 - alpha_fixed_emb)*(1 - trust)`, with `alpha_fixed_emb = 0.95`. At s_det = det_thresh, alpha = 1 and the new embedding is ignored. At s_det = 1, alpha = 0.95. The paper presents this as adding no new hyperparameters. — [Deep-OC-SORT ocsort.py lines 417-420](https://github.com/GerardMaggiolino/Deep-OC-SORT/blob/main/trackers/integrated_ocsort_embedding/ocsort.py); [Deep OC-SORT paper, arXiv 2302.11813 (search summary)](https://arxiv.org/pdf/2302.11813); [Ultralytics docs](https://docs.ultralytics.com/reference/trackers/deep_oc_sort)
- Hybrid-SORT (`adapfs` option): `pre_w = alpha*c_prev/(c_prev + s)`, `cur_w = (1-alpha)*s/(c_prev + s)`, both renormalized to sum to 1, then `smooth = pre_w*smooth + cur_w*feat` and L2-normalize. The default `--alpha` is 0.8. It also keeps a long-term feature bank with `--longterm_bank_length` default 30. — [HybridSORT hybrid_sort_reid.py](https://github.com/ymzis69/HybridSORT/blob/master/trackers/hybrid_sort_tracker/hybrid_sort_reid.py); [HybridSORT utils/args.py](https://github.com/ymzis69/HybridSORT/blob/master/utils/args.py)
- BoT-FaceSORT (face-specific): alpha 0.9 EMA as in BoT-SORT. It also pushes every smoothed template, tagged with the track id, into a "shared memory" deque (maxlen 1000) and a per-track `face_features` deque (feat_history 1000). The appearance cost is the **minimum cosine distance over all of the track's stored templates**, halved. — [bot_facesort.py](https://github.com/bellhyeon/BoT-FaceSORT/blob/master/tracker/trackers/botfacesort/bot_facesort.py); [tracker/utils/matching.py `embedding_distance_modified`](https://github.com/bellhyeon/BoT-FaceSORT/blob/master/tracker/utils/matching.py)
- boxmot (a widely used re-implementation) ships tuned configs: StrongSORT `ema_alpha 0.9`, `max_cos_dist 0.4`; Deep OC-SORT `alpha_fixed_emb 0.95`, `w_association_emb 0.75`, `aw_param 0.5`; BoT-SORT `max_lost_frames 40`, plus several hyperparameter-searched values. — [boxmot configs/trackers](https://github.com/mikel-brostrom/boxmot/tree/master/boxmot/configs/trackers)

**Face image quality signals**
- MagFace: the embedding magnitude (`np.linalg.norm`) is the quality score, learned without quality labels. The paper reports that this generalizes across datasets and features, and that it beat generic IQA (BRISQUE, NIQE, PIQE). It was compared against SER-FIQ and FaceQNet. — [MagFace repo](https://github.com/IrvingMeng/MagFace); [MagFace paper, CVPR 2021 (search summary)](https://arxiv.org/pdf/2103.06627)
- AdaFace: shows that the feature norm correlates with image quality (BRISQUE) for both ArcFace-trained and AdaFace-trained models (1,534 sampled MS1MV2 images), and uses the norm as a quality proxy. No Pearson r found in the summaries. AdaFace also argues that MagFace's "quality" is really recognizability or sample difficulty. — [AdaFace, arXiv 2204.00964](https://arxiv.org/pdf/2204.00964); [AdaFace supplemental](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Kim_AdaFace_Quality_Adaptive_CVPR_2022_supplemental.pdf)
- CR-FIQA: a learned quality regressor (iResNet100 "L" / iResNet50 "S" backbones on MS1MV2). It needs its own forward pass on an aligned RGB crop scaled to [-1, 1]. — [CR-FIQA repo](https://github.com/fdbtrs/CR-FIQA)
- In surveillance video, using only higher-quality frames beat using all frames. A properly chosen quality threshold improved accuracy significantly. — [PMC10938076](https://pmc.ncbi.nlm.nih.gov/articles/PMC10938076/)

### Inferences
- CPU-cheap quality gate for "may this frame update or vote" (design suggestion, thresholds unverified): det_score >= 0.6-0.7 (cf. BoT-SORT `new_track_thresh` 0.7 and StrongSORT `min_confidence` 0.6), |yaw| <= ~30-35 deg from the 106/3d68 landmarks (cf. the 30-degree degradation above), inter-ocular distance >= ~30-40 px, Laplacian variance above a clip-relative percentile, and **raw ArcFace embedding norm** above a clip-relative percentile (free via `embedding_norm`). Plus the hard rule: **not in contact state**.
- Recommended structure (inferred):
  1. Locked reference R (user frame, never changes).
  2. Rolling template T with Deep OC-SORT-style confidence-scaled EMA (alpha in [0.95, 1]), updated only when the quality gate passes and the face is not in contact.
  3. Snapshot T_pre frozen at contact entry.
  4. Optional small bank of the K best-quality past embeddings per slot (BoT-FaceSORT/DeepSORT gallery idea, K ~ 10-30), scored by max similarity.
  The appearance score is then max(sim to R, sim to T_pre, max over bank). Taking the max over a bank tolerates pose changes, because the best-matching stored pose wins.
- Template poisoning is what happens when a mixed (kiss) crop gets EMA-blended into T, so that T drifts toward the partner. With alpha 0.9, ten consecutive bad frames already give the poisoned content about 65% of the weight (1 - 0.9^10 = 0.65). Freezing during contact is therefore essential, not optional. (Arithmetic inference.)

### Gaps
- No direct comparison of MagFace / SER-FIQ / CR-FIQA for *video frame gating* was found. Whether w600k_r50's raw norm is a good quality proxy is supported only indirectly (AdaFace shows the norm-quality correlation for ArcFace-margin training). Validate on target clips.
- SER-FIQ needs multiple stochastic forward passes with dropout. Its CPU cost was not quantified in the sources found, and it is likely too slow for per-frame use (unverified).

---

## 3. Association: multi-cue costs, weights, gates, merged detections, and preventing swaps between stable tracks

### Takeaway
Modern trackers mix IoU/motion with appearance in three main ways: a weighted sum (Deep OC-SORT: IoU + 0.2 x direction consistency + 0.75-1.25 x cosine), an appearance-dominant sum with a motion gate (StrongSORT: 0.98 x appearance + 0.02 x Mahalanobis, gated at chi-square 9.4877), or a min-fusion with gates (BoT-SORT: appearance used only if IoU >= 0.5 and cosine >= 0.5). Deep OC-SORT's adaptive weighting raises the appearance weight only when the top-1 vs top-2 similarity gap is large. That is exactly the right behaviour for two touching faces, where the gap collapses. None of the generic trackers has an explicit "crossing lock". That logic has to be added as hysteresis, using the gating and min-hits ideas they do provide.

### Cited Findings

**Deep OC-SORT (official code)**
- Cost: `final_cost = -(IoU + angle_diff_cost + emb_cost * W)`, solved with linear assignment. `angle_diff_cost` is OC-SORT's velocity-direction consistency (OCM), weighted by `inertia` (default 0.2) and multiplied by detection score. — [association.py](https://github.com/GerardMaggiolino/Deep-OC-SORT/blob/main/trackers/integrated_ocsort_embedding/association.py)
- Adaptive weighting, `compute_aw_new_metric`: W starts at `w_assoc_emb = 0.75`. For each detection row it adds `min(top1 - top2, 0.5)/2`, and likewise for each track column, so W lies in [0.75, 1.25]. The appearance weight rises when the best candidate clearly beats the second. — [association.py lines 260-280](https://github.com/GerardMaggiolino/Deep-OC-SORT/blob/main/trackers/integrated_ocsort_embedding/association.py); [main.py defaults `--w_assoc_emb 0.75`, `--aw_param 0.5`](https://github.com/GerardMaggiolino/Deep-OC-SORT/blob/main/main.py)
- Shortcut: if the IoU > iou_threshold matrix is already one-to-one (every row and column has at most one hit), matching is done by **IoU alone and appearance is skipped**. After Hungarian assignment, any pair with IoU < iou_threshold (0.3) is rejected, so appearance cannot create a match without overlap. — [association.py lines 354-387](https://github.com/GerardMaggiolino/Deep-OC-SORT/blob/main/trackers/integrated_ocsort_embedding/association.py)
- Paper description: adaptive weighting "adaptively reduces the weight of the appearance cost for individual track and detection matching candidates based on the similarity of appearance embeddings" (max vs second max). — [CMU MSCV project page](https://mscvprojects.ri.cmu.edu/2022team14/?p=17)

**BoT-SORT (official code)**
- `ious_dists_mask = ious_dists > proximity_thresh` (default 0.5). `emb_dists = cosine_distance/2`. `emb_dists[emb_dists > appearance_thresh] = 1` (default 0.25, i.e. cosine >= 0.5 required). `emb_dists[ious_mask] = 1`. Then **`dists = min(ious_dists, emb_dists)`**. — [bot_sort.py lines 304-314](https://github.com/NirAharon/BoT-SORT/blob/main/tracker/bot_sort.py); [tools/track.py defaults](https://github.com/NirAharon/BoT-SORT/blob/main/tools/track.py)
- Other defaults: track_high_thresh 0.6, track_low_thresh 0.1, new_track_thresh 0.7, track_buffer 30 frames (scaled by fps/30), match_thresh 0.8. — [tools/track.py](https://github.com/NirAharon/BoT-SORT/blob/main/tools/track.py)

**StrongSORT / DeepSORT (official code)**
- StrongSORT cost: `cost = MC_lambda*appearance + (1 - MC_lambda)*mahalanobis`, with MC_lambda 0.98. Entries whose Mahalanobis distance exceeds `chi2inv95[4] = 9.4877` are set to INFTY (1e5). — [StrongSORT linear_assignment.py](https://github.com/dyhBUPT/StrongSORT/blob/master/deep_sort/linear_assignment.py); [kalman_filter.py chi2inv95](https://github.com/dyhBUPT/StrongSORT/blob/master/deep_sort/kalman_filter.py)
- `max_cosine_distance` 0.3 (simple CNN) or 0.4 (BoT ReID), plus 0.05 when motion-cost blending is on. `min_confidence` 0.6. Tracker `max_age` 30, `n_init` 3 (a track must be hit 3 times before it is confirmed), `max_iou_distance` 0.7. — [StrongSORT opts.py](https://github.com/dyhBUPT/StrongSORT/blob/master/opts.py); [StrongSORT tracker.py](https://github.com/dyhBUPT/StrongSORT/blob/master/deep_sort/tracker.py)
- DeepSORT: `max_cosine_distance` 0.2, `nms_max_overlap` 1.0, `nn_budget` None. — [deep_sort_app.py](https://github.com/nwojke/deep_sort/blob/master/deep_sort_app.py)

**OC-SORT / Hybrid-SORT (official code)**
- OC-SORT defaults: track_thresh 0.6, iou_thresh 0.3, min_hits 3, inertia (VDC/OCM weight) 0.2, deltat 3 (frames used to estimate direction), use_byte off. — [OC_SORT utils/args.py](https://github.com/noahcao/OC_SORT/blob/master/utils/args.py)
- OC-SORT mechanisms: OCM adds direction consistency to the cost. OCR is a second association round that matches each lost track's **last observation** (not its Kalman prediction) to leftover detections. ORU builds a virtual trajectory between the last observation before loss and the re-found observation, then re-updates the Kalman filter so errors from the occlusion period do not accumulate. — [OC-SORT, arXiv 2203.14360 (search summary)](https://arxiv.org/pdf/2203.14360); [CVPR 2023 version](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_Observation-Centric_SORT_Rethinking_SORT_for_Robust_Multi-Object_Tracking_CVPR_2023_paper.pdf)
- Hybrid-SORT ReID configs for DanceTrack (a dataset of look-alike dancers who cross often): iou_thresh 0.15, TCM (track confidence modelling) weight 1.0 (val) / 1.5 (test), appearance (EG) weight 4.0 / 4.4 (val, high / low score) and 2.8 / 1.4 (test). — [yolox_dancetrack_test_hybrid_sort_reid.py](https://github.com/ymzis69/HybridSORT/blob/master/exps/example/mot/yolox_dancetrack_test_hybrid_sort_reid.py); [val config](https://github.com/ymzis69/HybridSORT/blob/master/exps/example/mot/yolox_dancetrack_val_hybrid_sort_reid.py)

**BoT-FaceSORT (face-specific, official code)**
- First association: proximity_thresh 0.35, so appearance only counts where IoU >= 0.65. face_appearance_thresh 0.3 on halved cosine distance (cosine >= 0.4). `dists = min(iou_dist, emb_dist)`, match_thresh 0.8, track_buffer 100 frames. **On a detected shot change the cost becomes appearance-only.** The shot change is detected from colour histograms of adjacent frames, with `--shot-change-threshold` 0.4. — [botfacesort.yaml](https://github.com/bellhyeon/BoT-FaceSORT/blob/master/tracker/configs/botfacesort.yaml); [bot_facesort.py lines 299-308](https://github.com/bellhyeon/BoT-FaceSORT/blob/master/tracker/trackers/botfacesort/bot_facesort.py); [track.py](https://github.com/bellhyeon/BoT-FaceSORT/blob/master/track.py)
- Second association (unmatched tracked tracks and unconfirmed tracks vs leftover detections): appearance only, same 0.3 gate, match_thresh 0.8. — [bot_facesort.py lines 328-345](https://github.com/bellhyeon/BoT-FaceSORT/blob/master/tracker/trackers/botfacesort/bot_facesort.py)
- Detector and embedder defaults: `weights/scrfd_10g_gnkps.onnx` (same family as buffalo_l's detector) and `resnet101_ir_webface12m` AdaFace. — [track.py](https://github.com/bellhyeon/BoT-FaceSORT/blob/master/track.py)

### Inferences
- **Min-fusion is dangerous during contact.** In BoT-SORT/BoT-FaceSORT, `min(iou_dist, emb_dist)` lets a large IoU override contrary appearance. When two faces overlap, track A's predicted box can have high IoU with B's detection, and IoU alone wins. Likewise, Deep OC-SORT's "unique IoU match, skip appearance" shortcut can lock in a crossing swap. For contact frames, do **not** let IoU alone decide. (Inference from the code paths above.)
- **For exactly two slots, compare the two hypotheses directly.** Straight is A-d1, B-d2. Crossed is A-d2, B-d1. Score each as `S = sum of w_app*cos + w_mot*motion_term`, and Hungarian then picks the larger. Add hysteresis by requiring `S_cross - S_straight > margin` to switch away from the previous binding, and keep the previous binding otherwise. Suggested margin: about 0.10-0.15 on the summed cosines while not in contact, raised to about 0.2-0.3 or made switch-forbidden while in contact. (Design suggestion, unverified; it mirrors the AW top1-top2 logic and the n_init=3 confirmation idea.)
- **Gates to apply (CPU-trivial):**
  - Motion gate: centre jump <= ~0.5-1.0 x face width per frame, or a Kalman Mahalanobis gate at 9.4877 as in DeepSORT/StrongSORT.
  - Appearance accept: cosine >= calibrated threshold (~0.3-0.4 to R or T).
  - Size gate: area ratio within ~0.5-2x of the last value.
  A detection failing every slot's gates is left unassigned instead of being forced. (Values partly inferred.)
- **Two faces merged into one detection, or one face missing:**
  - Assign the single detection to the slot with the best combined score, but only if its appearance margin over the other slot is clear. Otherwise assign by motion/last observation (OCR-style) and mark the assignment "low confidence: do not update templates".
  - Put the other slot into a "hidden" state: hold its last box, skip rendering the swap for it (or fade), and do not run its Kalman prediction for long. Re-find it with OCR (last observation) and repair the trajectory ORU-style.
  - Typical buffers before a lost track is dropped: 30 frames (DeepSORT/StrongSORT/OC-SORT/BoT-SORT) to 100 frames (BoT-FaceSORT). For a two-person kiss you can keep both slots alive indefinitely, because the user has defined them.
- **K consecutive frames before a re-bind**: generic trackers use n_init/min_hits = 3 for confirming *new* tracks. For an identity *switch* of an established slot, a stricter rule such as K = 5-10 consecutive good-quality, non-contact frames all agreeing on the cross hypothesis by the margin is a reasonable extension (inferred, unverified).

### Gaps
- No tracker found with a published explicit "crossing lock" or switch-hysteresis rule. The K-frame and margin values above are engineering suggestions, not sourced numbers.
- Ablation numbers (IDF1/IDSW change) for Deep OC-SORT's AW and DA, and for OC-SORT's OCR/ORU, could not be retrieved because arXiv was blocked. Only the qualitative descriptions are confirmed.

---

## 4. Re-identification after separation: deciding which emerging face is which

### Takeaway
Trackers resolve post-occlusion identity by combining (1) appearance against *pre-occlusion* templates (memory or bank, minimum distance), (2) last-observation and motion consistency (OC-SORT OCR/OCM, then ORU to repair the Kalman state), and (3) confidence-weighted, margin-aware appearance weighting (Deep OC-SORT AW). For two people the decision can be made robust by deferring it: keep the pre-contact binding, accumulate evidence over several clean frames, and switch only on a consistent margin.

### Cited Findings
- OCR re-associates lost tracks via their last observation. ORU then re-updates the Kalman filter along a virtual trajectory after the track is found again. — [OC-SORT, arXiv 2203.14360 (search summary)](https://arxiv.org/pdf/2203.14360)
- OCM direction consistency uses `deltat = 3` frames to estimate each track's motion direction, weighted by `inertia = 0.2`. — [OC_SORT utils/args.py](https://github.com/noahcao/OC_SORT/blob/master/utils/args.py); [Deep-OC-SORT association.py](https://github.com/GerardMaggiolino/Deep-OC-SORT/blob/main/trackers/integrated_ocsort_embedding/association.py)
- Deep OC-SORT AW boosts appearance weight by up to +0.25 (row) and +0.25 (column) when the top1-top2 cosine gap is up to 0.5. A small gap leaves the weight at the 0.75 base. — [association.py](https://github.com/GerardMaggiolino/Deep-OC-SORT/blob/main/trackers/integrated_ocsort_embedding/association.py)
- BoT-FaceSORT matches re-appearing faces against *every* stored template of a track (min cosine distance over its memory), keeps lost tracks 100 frames, and switches to appearance-only association after shot changes. — [bot_facesort.py](https://github.com/bellhyeon/BoT-FaceSORT/blob/master/tracker/trackers/botfacesort/bot_facesort.py); [matching.py](https://github.com/bellhyeon/BoT-FaceSORT/blob/master/tracker/utils/matching.py)
- Hybrid-SORT can use long-term ReID features (bank of 30) for association correction (`--with_longterm_reid_correction`). — [HybridSORT utils/args.py](https://github.com/ymzis69/HybridSORT/blob/master/utils/args.py)

### Inferences
Concrete separation protocol for the two-slot pipeline (inferred design; numbers to be tuned on kiss clips):
1. **Exit condition.** The two face boxes have IoU 0 and centre distance > ~1.0 x mean face width for >= 2-3 frames, and both detections pass the quality gate.
2. **Evidence per frame.** `Delta = [s(A,d1) + s(B,d2)] - [s(A,d2) + s(B,d1)]`, where `s(X,d) = max(cos(R_X,d), cos(T_pre_X,d), max over bank_X)`. Add a motion term: which detection is consistent with each slot's *pre-contact* velocity and side of the frame (OCM-style: who was left stays left unless motion says otherwise), plus the last observation before contact (OCR-style).
3. **Decision.** Keep the pre-contact binding by default. Switch only if `Delta < -m` (cross hypothesis wins) for K consecutive qualifying frames. Suggested m ≈ 0.10-0.15 summed cosine (≈ 0.05-0.08 per face) and K ≈ 5 frames at 25-30 fps. If |Delta| < m, stay undecided and keep the default binding while continuing to accumulate. Then use a running sum of Delta over the window (sequential evidence) rather than a single frame.
4. **Retroactive fix.** If the switch is confirmed, relabel the K buffered frames, and optionally the contact span back to the last confident frame, since the pipeline can buffer output. Repair the trajectories ORU-style.
5. **Template update** resumes only after the binding is confirmed and quality passes.

### Gaps
- No published value found for the post-occlusion top1-top2 margin or the number of frames that face trackers require before re-binding. The margins and K above are unverified engineering defaults.
- No source quantifies how many frames after separation ArcFace similarity takes to recover to its pre-contact level (motion blur and partial occlusion while pulling apart).

---

## 5. Face-tracking-specific work: ID-switch behaviour and the cues that helped

### Takeaway
Multi-face tracking has its own small benchmarks: 8 YouTube music videos from Zhang et al. (ECCV 2016), MovieShot (BoT-FaceSORT, ACCV 2024), and ChokePoint. The recurring lessons are: Kalman motion breaks on shot cuts and fast camera motion; appearance memory (many stored templates per track) plus shot-change resets reduce identity errors; and in movies and TV, body/person tracks help when faces are occluded.

### Cited Findings
- Zhang et al., "Tracking Persons-of-Interest via Adaptive Discriminative Features" (ECCV 2016): a dataset of 8 YouTube music videos with 3,845 face tracklets and 117,598 face detections. Challenges are frequent shot changes and rapid camera motion. The method adapts discriminative features to the video and reported fewer identity switches than competing methods. — [ECCV 2016 paper PDF](https://faculty.ucmerced.edu/mhyang/papers/eccv16_face_tracking.pdf); [Springer](https://link.springer.com/chapter/10.1007/978-3-319-46454-1_26)
- Lin and Hung, "A Prior-Less Method for Multi-Face Tracking in Unconstrained Videos" (CVPR 2018): evaluated on the same 8 music videos plus 4 body-worn-camera videos. — [CVPR 2018 paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Lin_A_Prior-Less_Method_CVPR_2018_paper.pdf)
- An online multi-face tracker for unconstrained videos (IEEE) combines CNN face detection, correlation-filter trackers and ResNet identity labels, and reports MOTA/MOTP/IDS on the music videos. — [IEEE 8706169](https://ieeexplore.ieee.org/document/8706169/)
- BoT-FaceSORT (ACCV 2024): KF-based trackers struggle with shot changes, occlusions and appearance variation. The paper adds shot-change detection that resets the KF, a shared feature memory, and adaptive cascade matching. It reports HOTA 47.39 on MovieShot and gains of more than 5% HOTA and 6% IDF1 there. It also evaluates on Music and ChokePoint with TrackEval (HOTA, MOTA, IDF1). — [BoT-FaceSORT repo README](https://github.com/bellhyeon/BoT-FaceSORT); [ACCV 2024 page](https://openaccess.thecvf.com/content/ACCV2024/html/Kim_BoT-FaceSORT_Bag-of-Tricks_for_Robust_Multi-Face_Tracking_in_Unconstrained_Videos_ACCV_2024_paper.html) (paper numbers from search summary)
- FaceQSORT / FaceSORT (2025): a multi-face tracker that combines biometric (face recognition) features with general appearance features. Its search summary mentions ArcFace trained on MS-Celeb-1M. — [FaceQSORT, arXiv 2501.11741](https://arxiv.org/html/2501.11741v1)
- TV-series character identification (BMVC 2015): person bodies stay recognizable when faces are occluded, so face and person tracks are fused into one track node. — [BMVC 2015 abstract](https://bmva-archive.org.uk/bmvc/2015/papers/paper119/abstract119.pdf)
- Face-track clustering on TV (Buffy, Idiap): occlusion, pose and lighting are the main sources of within-person variation. — [Khoury et al., Idiap RR-31-2013](https://infoscience.epfl.ch/record/192473/files/Khoury_Idiap-RR-31-2013.pdf)
- Video face clustering with an unknown number of clusters (ICCV 2019) clusters all face tracks, including minor characters. — [arXiv 1908.03381 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1908.03381)
- Cast grouping in movies aims for clean single-identity tracklets, ideally one per character appearance. — [arXiv 1709.02458](https://arxiv.org/pdf/1709.02458)
- An egocentric face-vs-body tracking dataset for human-robot interaction (2026) compares face and body tracking. Not read in detail. — [arXiv 2606.03694](https://arxiv.org/html/2606.03694v1)

### Inferences
- For a kiss/hug clip, the face-tracking literature points to three cheap, transferable cues:
  1. **Shot-cut detection**: a colour-histogram difference with threshold ~0.4 as in BoT-FaceSORT. Reset motion and let appearance plus the locked reference decide after a cut.
  2. **Per-slot template memory** (min-distance over a bank) instead of a single EMA.
  3. **Context beyond the face**: hair, clothing or upper-body colour histograms taken from a region below or around each face box. These stay distinct when faces touch, as in the BMVC 2015 face+person fusion. A 16-32-bin HSV histogram of the torso/hair region costs well under 1 ms on CPU and stays discriminative during a kiss, when the face crops become mixed. (Cost estimate and bin counts are inferred, unverified.)
- Since the user locks each slot's reference from a chosen frame, the problem is closer to "persons-of-interest tracking" (Zhang et al. 2016) than to open-set MOT. Assignment can always be closed-set over the 2 slots plus "none", which removes new-track creation as a failure mode.

### Gaps
- Per-paper IDS counts on the music-video benchmark, and the BoT-FaceSORT ablation tables (per-trick IDF1/IDSW), were not retrieved because the PDFs were blocked. Only the headline numbers above are available.
- No face-tracking paper found that specifically reports ID-switch rates *during physical contact* (kissing or hugging) as a separate category. The literature treats it under general "occlusion".
