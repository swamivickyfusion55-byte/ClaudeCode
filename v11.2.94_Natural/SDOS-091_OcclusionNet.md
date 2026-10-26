# SDOS-091 — OcclusionNet: food in front of the mouth stays solid, the lips go around it; a lighter "Cinematic"

**Pack:** v11.2.91 "OcclusionNet" (v11.2.90 MotionKeys + a trained occlusion model + a new Cinematic enhancer). Engine `aequus-1.2.91-occlusionnet`.
**Reported:** "On eating, the lower lip comes over the object instead of encircling under it." And: "change the cinematic face enhancer to more subtle refinements (quicker as well): subtle shiny skin, naturally beautiful, a slight bit of HDR, a bit of vibrancy."

I have not seen your footage (I only work with synthetic media and your logs). Every number below is a synthetic scoop on real photographs, through the real detector, swapper and compositor.

---

## 1. Eating: why every earlier attempt failed, and what is different now

Versions 11.2.82–87 tried to find the object with hand-made colour/novelty rules. On real faces they could not tell a skin-coloured scoop from lips (8–76 % found, depending on colour), and on your footage they cut 82–91 % of all face-frames. That is why I rolled them back.

The standard answer in professional face-swap tools is a **trained face-occlusion model**. FaceFusion ships one ("face occluder", XSeg, from the DeepFaceLab line): 256×256 face crop in, "face / not face" per pixel out, trained on real hands, food, cups and hair in front of faces. I tested the three published versions on 12 real faces before building anything:

| xseg_2 (chosen), aligned real-face crops | result |
|---|---|
| Realistic scoop at the mouth / lips / cheek, 6 colours incl. strawberry, caramel, mango | **100 % of the object found in every case** |
| Clean faces (normal, dim, dark + noise, side light, blur), 60 cases, with the rules below | **0.00 % of the face cut** |
| Face right next to the object wrongly cut (2–7 px ring) | 0.4–2.1 % |

Two rules keep the rest of the swap exactly as it was:
- **only inside the face's own landmark outline** (eroded 8 %), so the swap's edge (hairline, jaw, ears) is never re-shaped;
- **only pieces that reach the mouth/chin zone**, because the model rightly calls hair falling over a cheek an occluder and that must not cut the face (without this rule a long-haired face lost a strip of cheek, up to 3.7 %);
- and nothing that touches the part of the crop lying **outside the picture** (a face at the frame edge): there the crop is filled with stretched edge pixels, which the model reads as an occluder.

A cut appears on the frame the object appears; when it goes, the face comes back over a few frames.

## 2. Results through the real pipeline

Share of the scoop that the swap still painted over (pixels inside it changed by more than 12 grey levels against the footage), 100-frame clips, a face about 320 px wide, Balanced:

| Scoop | v11.2.90 | **v11.2.91** |
|---|---|---|
| strawberry-pink, at the mouth | 44.4 % | **2.3 %** |
| caramel, at the mouth | 42.9 % | **0.8 %** |
| vanilla, at the mouth | 50.4 % | **0.5 %** |
| strawberry-pink, left cheek | 35.8 % | **0.4 %** |
| caramel, right cheek | 36.0 % | **1.7 %** |
| the cone below it | 8–15 % | 3–6 % |

In v11.2.90 the swapped mouth and lips were drawn over the scoop: your report. In v11.2.91 the scoop is solid and the swapped face is painted around it. Before the scoop arrives the output is the same as v11.2.90.

**Clean footage** (net on against net off, same pack):
- fast head turns and rolls: the real face is **never cut** (0 of 70 frames). The cuts in that clip's log are all on a second "face" the detector finds in the background of my synthetic clip, which is not a face.
- two faces touching (cheek to cheek): cuts only during the contact frames, where the other face really is in front of this one (2.5 % of the crop on average).

## 3. The new "Cinematic" enhancer

The old one was a bilateral filter (smoothing), local contrast (CLAHE) and an unsharp mask: it made faces waxy and over-crisp (fine-detail energy ×2.2), deepened the shadows, and the bilateral filter was slow. The new one makes three light touches on the swapped face:
1. **soft glow on lit skin**: a screen blend with a blurred copy, weighted by brightness, so highlights get a gentle sheen and shadows are left alone;
2. **gentle HDR**: the large-scale light is compressed 12 % toward its mean and fine detail lifted 12 % (shadows open a little, highlights calm a little, no halos);
3. **vibrance**: muted colours lifted up to 12 %, strong colours untouched, so skin does not turn orange.

| 4 real faces, 256-px crop | old Cinematic | new Cinematic |
|---|---|---|
| average change to the picture | 6.0–9.7 grey levels | 4.9–5.7 |
| dark end (5th percentile of brightness) | pushed down 4–7 | unchanged (±0.4) |
| fine-detail energy | ×2.1–2.4 (over-sharpened) | ×1.4–1.5 |
| colour | unchanged | +6 % (vibrance) |
| time per face crop: 128 / 256 / 512 px | 0.8 / 7.4 / 38 ms | **0.6 / 2.9 / 16 ms** |

Same name in the menu ("Cinematic (clarity + smooth)"), so the app keeps working unchanged. Strengths in `config.py` → `ENHANCE_CINEMATIC` (glow 0.18, hdr 0.12, vibrance 0.12; 0 switches a touch off).

## 4. Read your log

```
occlusion net ready: xseg_2.onnx (70 MB) · 1.5s
OCCLUSION-NET: something in front of the mouth/chin was cut on 61 of 198 face-frames (mean 9.4% of the face crop); 142 ms per face-frame
```
The model (70 MB) is downloaded once from FaceFusion's GitHub release at the first job and checked against its SHA-256; it is kept in `/data/phoenix_models` if the Space has storage, otherwise it is fetched again after a restart. If the download fails, the line says so and the job runs exactly as v11.2.90. **On a clip with no food, the number should be near 0**: if it is high, send me the line and the time of a frame that looks wrong.

## 5. What this does NOT do / costs

- **Time:** the model costs about 0.15 s per face per frame on the Space's 2 CPUs (about +10–15 % on a 2-face job). Off: `occ_net` = 0 in `config.py`.
- It acts at the mouth and chin only. A hand over the eyes or hair over a cheek is not cut (deliberately; see §1).
- The swap network still draws its own mouth; where lips close around a spoon or a straw, the lips are the swapped ones and the object is the original. Thin objects (a straw, a spoon handle) were not tested.
- **Licence:** XSeg comes from DeepFaceLab (GPL-3.0) and is published as ONNX by FaceFusion. Fine for your own Spaces; if you ever distribute the app or Space commercially, check that licence.
- Real eating footage is still untested by me.

## 6. Verification

The 13 unit checks of 11.2.90 pass on this pack; the model is downloaded from GitHub, checked (SHA-256) and loaded by the pack's own code (1.5 s here); the model-level tests of §1; five food clips and three clean clips through the real pipeline (§2); the enhancer comparison (§3); `occ_net` = 0 gives v11.2.90's masks exactly (only the enhancer differs).

## 7. Install

Upload the 8 files (`app.py config.py core_pipeline.py phoenix_api_adapter.py swap_engine.py packages.txt requirements.txt README.md`) to the root of the Space, commit, restart (or "Push to all Spaces"). Log: `Phoenix v11.2.91`, `swap_engine aequus-1.2.91-occlusionnet`, then `occlusion net ready` at the first job. Do not restart a Space while a job is running.
