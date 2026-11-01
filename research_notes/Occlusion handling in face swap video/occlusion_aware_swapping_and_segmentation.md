# Occlusion-aware face swapping and segmentation of touching / overlapping faces (CPU-practical)

> Research constraints, read first. In this cloud session **arxiv.org, ar5iv.labs.arxiv.org, openaccess.thecvf.com and docs.ultralytics.com could not be reached** (DNS blocked by the session network policy). Paper content was therefore taken from search-engine summaries of those pages and from GitHub READMEs, which could be fetched. Items marked **[unverified, model knowledge]** come from my prior knowledge of the papers and were NOT re-checked in this session. The report writer should present them as such, or drop them. Date of research: 2026-11-01.
>
> Context from this repo (local, not a web source): the engine already uses XSeg `xseg_2` (256x256) as a face-occluder inside the landmark outline, restricted to the mouth/chin zone, with 100% scoop detection and 0.00% clean-face cut on its synthetic tests (`/home/user/ClaudeCode/SDOS-091_OcclusionNet.md`).

## Q1. Occlusion handling in face-swapping methods (FaceShifter/HEAR-Net, FSGAN, 2020-2026 follow-ups) and pretrained-model availability

### Takeaway
There are two families of methods. The first is **self-supervised "what the generator could not reconstruct"** (FaceShifter HEAR-Net, and FaceDancer's attention fusion). These need a second network trained jointly with the swapper, and none are released as ONNX for inswapper_128. The second is **segmentation-gated compositing** (Nirkin 2018, FSGAN, E4S, RAFSwap, and every production tool such as DeepFaceLab XSeg and FaceFusion). Here a visible-face mask decides where the swap is pasted. Only the second family drops into a CPU pipeline built on inswapper_128. As late as 2024 a survey still calls extreme occlusion unsolved.

### Cited Findings
- **FaceShifter (Li et al., arXiv 1912.13457, 2019/2020).** It is a two-stage framework. Stage 2 is HEAR-Net, which fixes occlusion errors left by the stage-1 generator AEI-Net and is trained **without manual annotations**. — [arXiv abs](https://arxiv.org/abs/1912.13457v3) (via search summary)
  - **How it works:** feed the same face to the trained AEI-Net as both source and target. The reconstruction drops the occlusions, so the difference between input and reconstruction (the "heuristic error") locates them. The heuristic error and the stage-1 swap are fed to a U-Net, which outputs the refined result. No occlusion masks are used, not even synthetic ones. — [arXiv PDF](https://arxiv.org/pdf/1912.13457) (via search summary)
  - **Motivation:** earlier methods depend on segmentation networks trained on manually annotated occlusion masks, which may not generalise to unseen occluder types. — [arXiv PDF](https://arxiv.org/pdf/1912.13457)
  - **Reported quality:** identity retrieval of 97.38% against 82.41% for the next-best method (FaceForensics++ protocol). — [The Batch summary](https://www.deeplearning.ai/the-batch/trading-faces)
  - **Availability [unverified, model knowledge]:** Microsoft released no official code or weights. Only unofficial re-implementations exist (e.g. on GitHub), and most implement AEI-Net only. I know of no public HEAR-Net weights.
- **FaceDancer (Rosberg et al., WACV 2023, arXiv 2210.10473).** A single-stage swapper with Adaptive Feature Fusion Attention (AFFA) and Interpreted Feature Similarity Regularization (IFSR). It is claimed to keep the target's occlusions **without a separate facial segmentation step**. — [deepai summary](https://deepai.org/publication/facedancer-pose-and-occlusion-aware-high-fidelity-face-swapping); [arXiv 2210.10473](https://arxiv.org/abs/2210.10473v1)
- **FSGAN (Nirkin et al., ICCV 2019; FSGANv2 TPAMI 2022).** "Occluded face regions are handled by a face completion network". A separate blending network then preserves the target's skin colour and lighting, using a Poisson-blending loss plus a perceptual loss. **Pretrained models are available:** run `python download_fsgan_models.py` after install. The wiki has "Face Swapping Inference" and "Paper Models Inference" guides, and there is a Colab. — [fsgan GitHub](https://github.com/YuvalNirkin/fsgan)
  - **[unverified, model knowledge]** The FSGAN pipeline is: reenactment generator, then a segmentation net (U-Net, classes background/face/hair), then an inpainting net that fills the source face's missing regions, then a blending net, all at 256x256. Occluders in the target (hands etc.) are excluded because the target segmentation marks them as non-face, and the swapped face is pasted only where the target's visible face is. The weights are PyTorch, with no official ONNX, and the download form is gated (non-commercial).
- **Nirkin et al., "On Face Segmentation, Face Swapping, and Face Perception" (FG 2018, arXiv 1704.06729).**
  - A **standard FCN** gives fast, accurate face segmentation without a specially tailored model, and that is enough for swaps between unconstrained, arbitrarily paired images.
  - The LFW examples include occlusions.
  - The released Caffe model is **not** the network used in the paper, and the authors note "some performance drop" from converting it.
  - — [project page](https://www.openu.ac.il/home/hassner/projects/faceswap/); [OpenU CRIS](https://cris.openu.ac.il/en/publications/on-face-segmentation-face-swapping-and-face-perception-2/)
  - **[unverified, model knowledge]** The FCN-8s was trained on semi-automatically labelled video frames (motion cues) and on images augmented with synthetic hand occluders pasted from EgoHands, and was evaluated on COFW occlusion masks.
- **E4S (CVPR 2023; extended arXiv 2310.15081).**
  - It treats swapping as regional GAN inversion: a mask-guided encoder projects the texture of each face component into regional StyleGAN style codes.
  - The authors argue that "the facial occlusion challenge can be naturally addressed by facial masks", because a face parser labels each face region.
  - Code is public, and the target's face-parsing mask is an optional input.
  - — [E4S arXiv html](https://arxiv.org/html/2310.15081); [CVPR paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Fine-Grained_Face_Swapping_via_Regional_GAN_Inversion_CVPR_2023_paper.pdf); [GitHub e4s2022/e4s](https://github.com/e4s2022/e4s)
  - **Heavy:** it is StyleGAN-based at 1024 px [unverified latency], so it is not CPU-practical.
- **BlendFace (ICCV 2023).** An identity encoder, not an occlusion method. It notes that ArcFace-based swappers produce inconsistent face boundaries, and its training uses a predicted mask supervised by BCE against a face-parser mask. — [BlendFace paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Shiohara_BlendFace_Re-designing_Identity_Encoders_for_Face-Swapping_ICCV_2023_paper.pdf); [project page](https://mapooon.github.io/BlendFacePage/)
- **Region-Aware Face Swapping (RAFSwap, CVPR 2022)** and **HifiFace (2021)** both use semantic/parsing-derived masks inside the generator. — [RAFSwap arXiv](https://arxiv.org/pdf/2203.04564); [HifiFace arXiv](https://arxiv.org/pdf/2106.09965) (titles from search, details not fetched)
- **ReliableSwap (arXiv 2306.05356).** It is about **supervision**, not occlusion: it synthesises "cycle triplets" (reenactment, multi-band blending, reshaping) so that real images can act as pixel-level supervision. The README makes no occlusion claims. Licence: research/non-commercial. — [GitHub ReliableSwap](https://github.com/ygtxr1997/ReliableSwap)
- **Survey (Deepfake Generation and Detection: A Benchmark and Survey, arXiv 2403.17881, 2024):** these methods "have not thoroughly addressed the issue of artifacts in generated images under extreme occlusion conditions." — [survey PDF](https://arxiv.org/pdf/2403.17881) (via search summary)
- **No paper on mutual occlusion between two faces:** a targeted search for two faces kissing, mutual occlusion and instance face parsing found no face-swap paper on this case. — (search, 2026-11-01; see Gaps)
- **Production practice:** FaceFusion ships XSeg occluders `xseg_1/2/3` (from DeepFaceLab, 256x256 input, `/255` scaling, no mean/std). With the `'many'` option all three run, and their masks are combined with `numpy.minimum.reduce`, so any model flagging an occluder wins. — [facefusion face_masker.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_masker.py)

### Inferences
- With inswapper_128 fixed, HEAR-Net cannot be reused (it needs AEI-Net's reconstruction). A cheap **analogue of the "heuristic error"** is possible: compare the target crop with inswapper's output when the target's own embedding is the source (a self-swap). Pixels inswapper cannot reproduce (hands, food, the other person's cheek) show large error. This costs a second inswapper pass, roughly the same cost as the swap itself [estimate]. It is noisy on texture and lighting, so treat it as a **secondary vote** next to XSeg, not as the main mask.
- The segmentation-gated family (XSeg plus parser, already in the pipeline) is the only realistic option at 0.1-0.3 s per face on 2 cores. The research gap is in **instance separation and depth ordering for two faces**, not in single-face occluders.

### Gaps
- No HEAR-Net weights, and I could not confirm whether any unofficial HEAR-Net is trained well enough to use.
- FSGAN/FSGANv2 segmentation classes and occlusion results could not be read (arXiv and CVF blocked). The details above are marked unverified.
- DiffSwap (CVPR 2023) and later diffusion swappers (2024-2026) could not be checked. They are almost certainly far outside the CPU budget (multi-step diffusion) [unverified].
- No quantitative occlusion benchmark (e.g. a swap evaluated on an occluded test set) was found that compares these methods head to head.

## Q2. Face parsing robust to occlusion: models, datasets, and which labels exist for occlusion, hand or other person

### Takeaway
The standard BiSeNet/CelebAMask-HQ parser has **no occluder, hand or other-person class**. An occluder simply gets whatever face label it resembles, and another person's skin is labelled "skin". Occlusion-specific segmenters (Nirkin FCN, XSeg, and the Voo et al. 2022 models trained on NatOcc/RandOcc) are **binary: visible face vs everything else**, which treats hands and objects as background. None of them distinguishes "the other person's face" as a class.

### Cited Findings
- **Voo, Jiang, Loy, "Delving into High-Quality Synthetic Face Occlusion Segmentation Datasets" (CVPRW 2022, pp. 4711-4720).**
  - **Task:** occlusion-aware face segmentation, with **occluders treated as background**.
  - **Datasets:** NatOcc generates naturalistic occluded faces from CelebAMask-HQ using colour transfer and image harmonisation. RandOcc overlays random shapes and textures. They also release two real, high-resolution occluded-face test sets with fine annotations, **RealOcc** and **RealOcc-Wild**.
  - **Hands:** these are called hard because their colour is close to skin. The authors hand-annotated 200 hand images from 11k Hands.
  - — [CVF html](https://openaccess.thecvf.com/content/CVPR2022W/VDU/html/Voo_Delving_Into_High-Quality_Synthetic_Face_Occlusion_Segmentation_Datasets_CVPRW_2022_paper.html); [arXiv 2205.06218](https://arxiv.org/pdf/2205.06218) (via search summary)
  - **Availability:** the datasets were released 04/2022 (Google Drive), and the NatOcc/RandOcc generation code 06/2022. The README text lists no pretrained weights or numbers; the results are only in images. — [GitHub kennyvoo/face-occlusion-generation](https://github.com/kennyvoo/face-occlusion-generation)
- **Hand-over-face:** VLM-HandOverFace (UTA dissertation) has 4,384 Kinect frames with hand-over-face labels. It is a separate work from Voo et al. — [UTA repository](https://uta-ir.tdl.org/uta-ir/handle/10106/29383)
- **Nirkin 2018 FCN:** the released Caffe model differs from the paper model, with a reported drop. — [project page](https://www.openu.ac.il/home/hassner/projects/faceswap/)
- **FaceFusion parsers:** `bisenet_resnet_18` and `bisenet_resnet_34`, both 512x512, MIT. Preprocessing is BGR to RGB, `/255`, then ImageNet mean/std. Region selection is `argmax(0)` followed by `numpy.isin` on the selected label ids. — [facefusion face_masker.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_masker.py)
- **[unverified, model knowledge] CelebAMask-HQ / BiSeNet label set** (zllrunning face-parsing.PyTorch ordering, 19 classes):
  - 0 background, 1 skin, 2 l_brow, 3 r_brow, 4 l_eye, 5 r_eye, 6 eye_g (glasses), 7 l_ear, 8 r_ear, 9 ear_r (earring), 10 nose, 11 mouth, 12 u_lip, 13 l_lip, 14 neck, 15 neck_l (necklace), 16 cloth, 17 hair, 18 hat.
  - The only "occluder-like" classes are eye_g, hat, hair and cloth. There is no hand, object or other-person class.
  - Source repo: [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) (not fetched).
- **[unverified, model knowledge] Other occlusion datasets:**
  - **COFW:** 1,852 faces with landmark-level occlusion flags. Nirkin 2018 used it with occlusion masks.
  - **EgoHands:** about 4,800 annotated hand frames, used as a source of hand occluders for augmentation.
  - **11k Hands:** used by Voo et al. as above.
  - **"Hand over Face" (HOF):** about 300 images (Khan et al., 2018).
  - **XSeg:** DeepFaceLab's model, a binary face/obstruction mask trained on community-labelled faces. Its training data is unpublished, so how it behaves on a *second face* is undocumented.

### Inferences
- **Two touching faces:** inside face A's crop, BiSeNet labels face B's cheek or lips as skin or lips of A. So **parsing alone cannot separate them**. XSeg might flag B's face as an occluder (it is "not A's face" geometrically), but this is undocumented and must be measured on contact clips (task #6).
- **A practical label scheme:** `visible_A = parser_face_classes(A_crop) ∧ ¬occluder(A_crop) ∧ ¬owned_by_B`. The last term needs an instance or ordering step (Q3/Q4), because no public dataset or model provides an "other person" class.
- **Fine-tuning:** NatOcc-style synthesis can generate "other face pasted over face" training pairs, by pasting a CelebAMask-HQ face crop with harmonisation as the occluder. That is the most direct way to train a binary "own face vs other face" head if fine-tuning ever becomes an option.

### Gaps
- RealOcc and RealOcc-Wild image counts, and the Voo et al. mIoU numbers, could not be extracted (arXiv/CVF blocked; README results are image-only).
- XSeg's behaviour on a second, adjacent face: no source found.
- No public dataset with per-instance face masks *and* front/behind order for two faces in contact (kissing, cheek to cheek) was found.

## Q3. Instance separation of two touching faces: per-face crops, promptable segmenters (SAM family), person instance segmentation and matting, with CPU cost

### Takeaway
- **Per-face crop + parser** gives each face's region but cannot decide ownership of the overlap.
- **Promptable SAM-family decoders prompted with each face's box and landmarks**, with negative points on the other face, are the cleanest way to get two separate instance masks. EdgeSAM and MobileSAM have ONNX encoder/decoder exports. On CPU the 1024x1024 image encoder is the bottleneck: about **22-38 GFLOPs**, and MobileSAM reports about 3 s for the whole pipeline on a Mac i5. Run it **only on frames where face boxes overlap**, on a crop of the union of the two faces, with the encoder possibly reused across a few frames.
- **YOLO-seg nano** gives person instances at about 50-100 ms CPU ONNX at 640, but its masks are coarse and separate *persons*, not faces.

### Cited Findings
- **MobileSAM:**
  - Image encoder 5M params, 8 ms; mask decoder 3.876M params, 4 ms; whole pipeline 9.66M params, 12 ms on an unnamed GPU. SAM ViT-H is 611M params and 452 ms.
  - **CPU:** "On our own Mac i5 CPU, it takes around 3s."
  - **ONNX export** via `scripts/export_onnx_model.py`, tested with onnx 1.12.0 / onnxruntime 1.13.1.
  - Note: the README's parameter totals do not add up (5 + 3.876 ≠ 9.66).
  - — [MobileSAM GitHub](https://github.com/ChaoningZhang/MobileSAM)
- **EdgeSAM:**
  - **Speed (encoder + decoder, one prompt):** 38.7 FPS on iPhone 14 and 164.3 FPS on a 2080 Ti. MobileSAM gets 4.9 FPS on iPhone 14 and SAM 4.3 FPS on a 2080 Ti.
  - **Compute:** EdgeSAM 22.1 GFLOPs and 9.6M params; MobileSAM 38.2 GFLOPs and 9.8M; SAM 2734.8 GFLOPs and 641M.
  - **Accuracy:** COCO mask AP with ViTDet-H box prompts is 42.1-43.0, against MobileSAM 39.4 and SAM 46.1. mIoU is +2.3 (COCO) and +3.2 (LVIS) over MobileSAM.
  - **ONNX:** encoder and decoder are exported separately (`scripts/export_onnx_model.py`, with `--decoder --use-stability-score`), and pre-exported ONNX files are listed. Tested with onnxruntime 1.16.3 CPU.
  - Inconsistency: the README claims "14x faster than MobileSAM on edge", but its own table gives about 7.9x.
  - — [EdgeSAM GitHub](https://github.com/chongzhou96/EdgeSAM)
- **EfficientSAM (Ti/S):** an ONNX version with separate encoder and decoder was published on its HF Space (news, 12 Jan 2024), with `export_to_onnx.py` in the repo. The README gives no params or latency. — [EfficientSAM GitHub](https://github.com/yformer/EfficientSAM)
- **SAM 2.1** (A100, PyTorch 2.5.1; no CPU numbers or ONNX in the README):

  | Model | Params | FPS | SA-V J&F |
  |---|---|---|---|
  | hiera_tiny | 38.9M | 91.2 | 76.5 |
  | hiera_small | 46M | 84.8 | 76.6 |

  SAM2's memory-based video propagation would track each face's mask through contact, but with 39M params it is roughly 4-8x heavier than EdgeSAM [inference from param counts]. — [SAM2 GitHub](https://github.com/facebookresearch/sam2)
- **Ultralytics YOLO26-seg** (COCO, 640 px, CPU ONNX on an EC2 P4d instance whose CPU model is unnamed):

  | Model | Mask mAP | CPU ONNX | Params | FLOPs |
  |---|---|---|---|---|
  | YOLO26n-seg | 33.9 | 53.3 ms | 2.7M | 9.3B |
  | YOLO26s-seg | 40.0 | 118.4 ms | 10.4M | 34.5B |

  — [Ultralytics GitHub README](https://github.com/ultralytics/ultralytics)
  - **[unverified, model knowledge]** YOLO11n-seg is about 65.9 ms CPU ONNX with mask mAP about 32.0 and 2.9M params. YOLOv8n-seg is about 96 ms with mask mAP about 30.5. The docs page could not be fetched.
  - YOLO masks are produced at 1/4 of the input resolution (160x160 at 640) [unverified], so edges are coarse.
- **RobustVideoMatting:** the official speeds are GPU-only (RTX 3090 FP16: 172 FPS at HD; GTX 1080 Ti FP32: 104 FPS at HD), tensor throughput only. ONNX models (opset 12) are tested on ORT CPU and CUDA, with no CPU timings. — [RVM GitHub](https://github.com/PeterL1n/RobustVideoMatting)
- **MODNet:** no CPU timing source was found. FastDeploy supports it on CPU. — [FastDeploy docs](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/matting.html)
- **Matting gives no instances:** RVM and MODNet output a single foreground alpha for *all* people. They cannot separate two hugging people [inference from the model design described in the README].

### Inferences

**Practical recipe for separating two touching faces (A and B):**

1. **Trigger.** Run the step only when SCRFD boxes overlap (IoU > 0 after expanding each box by about 10%) or the 106-landmark hulls intersect. The rest of the time keep the current per-face path, so the cost is zero on normal frames.
2. **Crop.** Take the union box of A and B, plus a margin, as the crop. Resize so that the long side is 1024, as the SAM-family encoder requires.
3. **Encode.** Run the EdgeSAM ONNX encoder once on that crop. EdgeSAM is 22.1 GFLOPs, against 38.2 for MobileSAM. **Estimated CPU latency on 2 cores is about 0.3-0.8 s** (unverified; benchmark it). That is over budget per frame, so:
   - run it every 2-4 frames during contact, and warp the masks in between with landmark motion (affine from the 106 points), or
   - accept the cost only on contact frames.
4. **Decode.** Run the decoder twice:
   - **A:** box A, positive points on about 10 of A's landmarks (nose bridge, eye centres, mouth corners), negative points on B's nose, eyes and mouth.
   - **B:** the mirror of A's prompts.

   The decoder is about 4 ms on GPU (MobileSAM figure); it is light on CPU too (about 4M params) [estimate: tens of ms].
5. **Exclusive ownership.** Where both masks claim a pixel, give it to the face with the higher decoder logit, or apply the front/behind rule from Q4.
6. **Compose.**
   - `final_A = parser_face(A) ∧ ¬xseg_occluder(A) ∧ (SAM_A ∨ ¬SAM_B)`, with SAM_B cut away only where B is in front.
   - `final_B` is the mirror of `final_A`.

**A cheaper alternative with no SAM, roughly 0 extra models:**
1. Parse each face in its own crop with BiSeNet.
2. Build each face's *geometric* support from its 106 landmarks: the convex hull of the jaw and brows, extended to the forehead.
3. In the overlap of the two supports, decide ownership by the Q4 order cue.
4. Assign the overlap pixels to the front face, and remove them from the behind face's mask.

This is crude for profiles (kissing is usually two profiles) but costs less than 5 ms.

**Notes:**
- The SAM decoder's multi-mask output plus its stability score gives a confidence value per frame, which can be used to fall back to the cheap path.
- Avoid YOLO-seg for face separation. Its person masks are coarse and include hair, neck and body, so they only help as a whole-person depth or ownership prior in a hug.

### Gaps
- No published CPU ONNX latency for the EdgeSAM, EfficientSAM or SAM2-tiny encoders on x86 (only GPU and iPhone figures, plus MobileSAM's "3 s on a Mac i5"). It must be measured on the target 2-core machine.
- I found no source for whether SAM-family encoders accept non-1024 inputs in their ONNX exports. A 512 input would cut the cost about 4x, but the positional embeddings are tied to 64x64 tokens [unverified].
- No paper evaluates SAM-prompted instance masks on two faces in contact.

## Q4. Depth (front/behind) ordering of two overlapping faces

### Takeaway
No face-swap paper addresses mutual face ordering (none found). The workable approach is a **vote of cheap geometric cues**: whose landmarks fall inside the other's visible segment, which parser or decoder claims the overlap with higher confidence, and detection-score drop and box scale. A **monocular depth model** can serve as an optional tie-breaker. Depth Anything V2-Small has public ONNX ports but **no x86 CPU latency was found**.

### Cited Findings
- **Depth Anything V2:**
  - The ViT-S encoder at 518x518 averages 13.3 ms on an RTX 4080 (CUDA EP); CPU timings were not reported.
  - Qualcomm lists 77.3 ms ONNX on Snapdragon X Elite, and a Qualcomm developer reports 17 ms at 256x256 (setups differ).
  - Core ML runs at 17.4 ms on the iPhone 15 Pro Max Neural Engine.
  - — [Qualcomm HF model card](https://huggingface.co/qualcomm/Depth-Anything-V2); [HF discussion 256x256](https://huggingface.co/qualcomm/Depth-Anything-V2/discussions/1); [depth-anything-onnx](https://awesome.ecosyste.ms/projects/github.com%2Ffabio-sim%2Fdepth-anything-onnx); [CoreML small](https://huggingface.co/f-labor/coreml-depth-anything-v2-small)
- **MiDaS small:** no CPU timing source was found (search returned nothing).
- **[unverified, model knowledge]** Depth Anything V2-Small has about 24.8M params. MiDaS v2.1 small takes 256x256 input with an EfficientNet-Lite3 backbone, about 21M params.
- **[unverified, model knowledge]** insightface's 2d106det outputs only coordinates, with no per-point visibility or confidence. The SCRFD detection score typically drops when a face is heavily occluded.

### Inferences

**Ordering vote, computed only on contact frames.** A is behind if:

1. **Landmark intrusion.** Count A's 106 landmarks that fall inside B's *exclusive* visible mask (B's parser face classes minus B's occluder mask), and the reverse. The face with more of its own landmarks inside the other's visible face is **behind**: its landmark regressor "hallucinated" points under the occluding face. This is the strongest cue for kissing and cheek-to-cheek [reasoning; not validated].
2. **Parser agreement in the overlap.** In the overlap O, check face A's crop parser, and likewise B's. If A's parser labels O as skin, nose or lips *and* XSeg in A's crop does **not** flag it, while XSeg in B's crop flags O as an occluder, then A is in front.
3. **Landmark temporal consistency.** The behind face's occluded landmarks jitter more (higher frame-to-frame residual after a similarity fit), so a jitter term can be added as a vote.
4. **Scale.** A larger inter-ocular distance means nearer. This is weak in kissing (similar distances) but useful when one person is clearly closer to the camera.
5. **Depth tie-breaker (optional).**
   - Run Depth Anything V2-Small at a low resolution (e.g. 252 or 308 px, a multiple of 14) on the union crop, once every N frames.
   - Compare the median relative depth in the overlap O with each face's exclusive region, and assign O to the face whose own region has the more similar depth.
   - **Estimated CPU cost on 2 cores is about 0.15-0.4 s at about 300 px** [unverified estimate; benchmark it].

**Temporal smoothing:**
- Hysteresis on the order decision, e.g. switch only after 3-5 consistent frames.
- Per-pixel ownership smoothing in the overlap, to avoid flicker.

### Gaps
- No published method, dataset or benchmark for front/behind ordering of two faces in contact was found.
- No CPU latency source for MiDaS small or Depth Anything V2-Small on x86 was found.
- Whether relative monocular depth resolves the few-centimetre difference at a kiss boundary is unknown. Likely poorly [unverified].

## Q5. Edge quality: feathering, guided filter and matting for clean mask edges

### Takeaway
Production tools use a **blur-then-threshold-rescale feather** (FaceFusion: Gaussian σ=5, clip to [0.5, 1], rescale). It is cheap and gives a narrow soft band, but it ignores image edges. Edge-aware refinement adds about 1-5 ms per face on CPU [estimate]:
- a **guided filter** using the frame as guide, which is O(N) and independent of radius, or
- a **fast guided filter** at 1/2-1/4 scale.

Learned matting (RVM, MODNet) is person-level and too coarse or costly to run per face.

### Cited Findings
- **FaceFusion feathering:**
  - **Occlusion, region and area masks:** `(cv2.GaussianBlur(mask.clip(0,1),(0,0),5).clip(0.5,1) - 0.5) * 2`. This is a Gaussian blur with sigma 5, then a clip to [0.5, 1] and a rescale to [0, 1], so the transition band is narrow and values below 0.5 become 0. Because the result starts at 0.5, the mask is effectively *eroded* by about half the band before feathering.
  - **Box mask:** blur with `blur_amount = int(crop_w * 0.5 * face_mask_blur)` and sigma `blur_amount*0.25`. Each edge is zeroed by a padding percentage, at least `blur_area` wide.
  - **Combining:** multiple occluder masks are combined with `minimum.reduce`.
  - — [facefusion face_masker.py](https://github.com/facefusion/facefusion/blob/master/facefusion/face_masker.py)
- **[unverified, model knowledge] Guided filter and related:**
  - The guided filter (He, Sun, Tang, ECCV 2010 / TPAMI 2013) runs in O(N) with box filters, independent of radius. It is available as `cv2.ximgproc.guidedFilter` (opencv-contrib).
  - The fast guided filter (He & Sun, arXiv 1505.00996, 2015) subsamples by a factor s, cutting the cost to about O(N/s²) with little quality loss.
  - Deep Guided Filter (Wu et al., CVPR 2018) is a learnable version used to upsample low-res masks.
  - Closed-form alpha matting (Levin et al., 2008) solves a sparse linear system: seconds per megapixel, so it is unsuitable per frame.
- **RVM:** GPU-only speed figures, person-level alpha, ONNX available. — [RVM GitHub](https://github.com/PeterL1n/RobustVideoMatting)

### Inferences
**Recipe for clean edges, about 2-6 ms per face on CPU [estimate]:**
1. **Combine at crop resolution.** Combine the masks (parser, XSeg, instance ownership) at the crop resolution (256 or 512) as soft probabilities, not hard argmax.
2. **Upsample.** Bring the combined mask up to the paste resolution.
3. **Refine edges.**
   - Apply `cv2.ximgproc.guidedFilter(guide=target_frame_crop_gray_or_rgb, src=mask, radius≈max(2, 0.01*face_w), eps≈1e-3..1e-2 on [0,1] data)`. The mask edge then snaps to real image edges: hair, the hand contour, the other person's cheek line.
   - For faces over about 400 px, use the fast-guided variant (filter at 1/2 scale, upsample the coefficients a and b).
4. **Feather on top.** Apply a narrow FaceFusion-style feather: a small sigma (about 0.5-1% of face width) inside the face, and **no feather into an occluder**. Erode the face mask 1-2 px at occluder boundaries before feathering, so no swapped pixels bleed onto the hand, object or other face. This matches the current engine's "object stays solid" goal.
5. **Smooth over time.** Use an EMA of the soft mask with motion compensation (warp by the landmark affine), and reset at shot cuts. This is consistent with the SDOS-093 lesson about not blending across cuts.

Use separate feather widths for the two kinds of edge:
- **swap boundary** (jaw, hairline): wider, so colour mismatch is hidden;
- **occluder boundary** (object or other face in front): very narrow and guided-filtered, because occluders have hard physical edges.

**Budget check (2 cores, per face per frame) [estimates, need benchmarking]:**

| Step | Cost |
|---|---|
| inswapper_128 | ~40-80 ms |
| XSeg 256 | ~10-25 ms |
| BiSeNet-r34 512 | ~60-120 ms |
| guided filter | ~2-5 ms |
| landmark ordering vote | <2 ms |

That is about 0.1-0.25 s, within budget. Adding the EdgeSAM encoder on *contact frames only*, amortised every 2-4 frames, and Depth Anything-S every N frames as an optional tie-breaker pushes contact frames to roughly 0.3-0.5 s, unless they are subsampled.

### Gaps
- No measured CPU latency source was found for the guided filter at face-crop sizes. The O(N) complexity is from model knowledge.
- No published study compares feathering vs guided filter vs matting specifically for face-swap compositing quality.
