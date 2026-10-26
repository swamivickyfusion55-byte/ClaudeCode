# SDOS-092 — FastRestore: GPEN face restore, clean side views, one mouth while eating, updates that wait for running jobs

**Pack:** v11.2.92 "FastRestore" (v11.2.91 OcclusionNet + four changes). Engine `aequus-1.2.92-fastrestore`.
**App:** Phoenix Mobile 1.2.24 (`Phoenix_APK_Builder_1.2.24.zip`), versionCode 144, installs over 1.2.23.

**Reported / asked:**
1. "Is there a light version of GFPGAN that runs properly on CPU? Please add." / "What's the time difference between Cinematic and GPEN? If not much, replace its slot with GPEN."
2. "Is it possible to push a patch update which does not remove my running jobs?"
3. "When the face is extreme sideways with little part visible, the pasting is not clean."
4. "On eating it appears as if another lip moves above the lips."

I have not seen your footage (I only work with synthetic media, public test photographs and your logs).

---

## 1. GPEN: a light face restorer that runs on the Spaces' 2 CPUs

### 1.1 What I measured (2 CPU cores, like the Spaces)

| Restorer (ONNX, FaceFusion's published exports) | works at | size | time per face |
|---|---|---|---|
| **GPEN-BFR-256** | 256 px | 76 MB | **0.21 s** alone, **0.23-0.28 s** inside a real job |
| GFPGAN 1.4 | 512 px | 340 MB | 2.0 s, plus ~1 s for its own face detector and parser in your current pack |
| GPEN-BFR-512 | 512 px | 284 MB | 2.5 s |
| (the face swap itself, for scale) | 128 px | | 1.15 s |

GPEN-BFR-256 takes the same face layout the swapper produces (arcface 128), so the swapped face goes in directly: no second face detector, no parsing net, no paste of its own. That is where GFPGAN's extra second goes.

**Cinematic vs GPEN, the time you asked about:** Cinematic costs about 2 ms per face, GPEN about 0.25 s. On your jobs (about one swap per face per frame, 2.4-4 s per frame) that is **about +7-16 % job time** (about 10-20 minutes on a 2½-hour job). On soft footage it costs nothing (see 1.2).

### 1.2 Built so that it does not look pasted on

A GAN restorer at full strength looks plastic, invents pores and lashes anew on every frame (shimmer) and can make the face sharper than the shot around it. Three rules:

1. **Mixed at 60 %** with the plain upscale (`ENHANCE_GPEN["blend"]`).
2. **Never sharper than the footage**: the footage is measured at 256 px straight from the frame; the mix is lowered until the face's fine detail is at most 1.25 x the footage's. When the footage is no sharper than the swap already is, the network is **skipped** (no time spent).
3. **Steady detail**: the restored detail is mixed half-and-half with the previous frame's of the same face (`steady`), so it does not boil.

### 1.3 Results (real pipeline, 2-face job, Ultra; synthetic clips from public photos)

Sharp footage (a 494-px face at the photo's native resolution, nothing upscaled):

| | Cinematic | **GPEN + Cinematic** |
|---|---|---|
| sharpness of the face vs the footage (1 = as sharp as the shot) | 0.24 | **0.39** |
| shimmer (fine detail change per frame) | 1.26 | 1.25 |
| flicker vs the footage | x1.65 | x1.63 |
| brightness pumping | 0.17 | 0.11 |
| identity (ArcFace match to the replacement photo) | 0.776 | 0.768 |
| time, 60 frames | 123 s | 142 s |

With Cinematic the swapped face is visibly softer than the hair and background next to it; with GPEN eyes, lashes, lips and nose are crisp and match the shot. Shimmer and flicker are unchanged.

Soft footage (my other clips are small photos scaled up 3x, and a motion-blurred sweep): GPEN holds itself back (mean strength 0.00-0.04) and the result equals Cinematic. Without rule 2, on the blurred sweep it made the face **5x sharper than the shot** and doubled the shimmer (measured: 4.2 vs 2.2), which is why the rule exists.

### 1.4 In the menu

- App 1.2.24: **"GPEN + Cinematic"** takes Cinematic's place as the default (GPEN, then the subtle glow / HDR / vibrance of v11.2.91). "Cinematic (clarity + smooth)" stays as the fastest choice. **CodeFormer is removed**: on your Spaces it never loaded (the GFPGAN library has no CodeFormer model type) and silently ran "Soft polish"; on the Space it now runs GPEN.
- An older Space receiving "GPEN + Cinematic" runs plain Cinematic (no error).
- The model (76 MB) is downloaded from FaceFusion's GitHub release at the first job that asks for it and checked against its SHA-256; kept in `/data/phoenix_models`.
- Log (example): `RESTORE: GPEN (fast restore) on 1240 swapped face crops; 251 ms each; mean strength 0.55 ...; 12 crops skipped`.
- Licence: GPEN is published for academic / non-commercial use. Fine for your own Spaces; check it before any commercial use.

## 2. Updates that do not lose running jobs

Hugging Face restarts a Space on every commit; nothing in the Space can prevent that, and a job running there is lost. So the app now **waits** instead:

- "Push to all Spaces" first asks each Space whether it is rendering (new `/phoenix/busy` route in v11.2.92; for Spaces still on an older pack the app checks the jobs it sent there through `/phoenix/status`). Idle Spaces are updated at once.
- Busy Spaces are listed under the button ("Waiting for their job to end: SwamitechGradio3 ...") and updated by themselves when their job ends: checked every 2 minutes while Phoenix is open, and again when you next open it (the pack is kept on the phone until then). "Cancel waiting updates" drops them.
- If a Space cannot be reached and the phone knows of no job there, it is updated (a sleeping Space has nothing to lose); if the phone sent a job there and the Space does not answer, it waits.

Tested: the app's real network code against the Space's real routes (v11.2.92 and v11.2.91 adapters, served locally): idle / running / finished / failed jobs, the older route, an unknown job, an unreachable Space - 12 of 12 checks pass.

**This round still needs one restart per Space** to install v11.2.92 (the waiting is in the new app). Push with app 1.2.24: Spaces that are rendering will be updated when their job ends.

## 3. Extreme side views: the paste is now cut to the visible face

### 3.1 What was wrong

The paint mask is a fixed oval placed by the face's 5 key points. On a side view those points crowd into the front of the head, so the oval sits half **in front of the face** (air, background) and leaves the back cheek out. Measured on public photos with true head angles (AFLW2000 samples; real detector, swapper and compositor; face area from a separate face-parsing model):

| Photo (true yaw) | paint off the face, v11.2.91 | **v11.2.92** | face covered, v11.2.91 → v11.2.92 |
|---|---|---|---|
| 55 deg | 48.6 % | **0.0 %** | 82.6 → 74.6 % |
| 55 deg (mirror) | 49.2 % | **0.0 %** | 76.3 → 67.1 % |
| 66 deg | 43.8 % | **0.0 %** | 84.8 → 81.8 % |
| 66 deg (mirror) | 40.4 % | **0.0 %** | 80.2 → 76.5 % |
| 71 deg | 37.4 % | **0.2 %** | 74.1 → 60.9 % |
| 71 deg (mirror) | 34.6 % | **0.3 %** | 78.8 → 60.9 % |

("Paint off the face" is the swap painted over background and hair, as a share of the face's own area: the halo / smear you saw.) The face left unpainted is the top of the forehead under the hairline and, on one photo, sunglasses: the occlusion model keeps them as filmed, which is right.

### 3.2 The change

From about 45 deg of yaw (measured as the nose's sideways offset from the eye midpoint, calibrated on the photos' true angles: 0.25 at 18 deg, 0.45 at 45 deg, 0.76 and more from 55 deg), the mask moves over to the **visible face's own outline**: the occlusion model's face map (XSeg, already loaded since v11.2.91; it segments faces in profile too) times the 106-point landmark outline, with a short feather. Full from about 55-60 deg; smoothed over frames so a turn does not jump.

Every face below about 45 deg keeps the v11.2.91 mask **exactly** (checked bit for bit on 43 faces: the only differences are side views, a small start at 38-45 deg, and the three cases of §4.3).

### 3.3 In motion

The same photos as 60-frame clips (camera drift, zoom, sensor noise), real pipeline; each frame's paint weight mapped back onto the photo:

| Clip | paint off the face, v11.2.91 → v11.2.92 | on the face | mask change per frame (steadiness) |
|---|---|---|---|
| 71 deg | 46.8 % → **0.8 %** | 79.2 → 61.8 % | 0.49 → 0.74 % (worst 1.9 → 2.1 %) |
| 55 deg | 51.6 % → **0.0 %** | 82.1 → 74.7 % | 0.30 → 0.32 % |

In the frames: v11.2.91 has a hazy light band along the nose and lips, a ghost over the eye and see-through sunglasses; v11.2.92 has a clean outline, the glasses as filmed, no halo. The mask moves a little more from frame to frame at 71 deg (the face map is re-read every frame), still under 1 % on average.

## 4. Eating: one mouth instead of two lips

### 4.1 What was wrong

When food touches the lips, the occlusion model cuts it out and the footage shows through there, **including the footage's own lower lip next to the food**, while the swap model draws its own lips (it cannot see the lower lip) a little higher. Result: the swapped lips plus the real lip edge, and as the food moves the cut moves: "another lip moves above the lips". Your log shows how much this happens: one eating job (4529A5A1) had the occlusion cut on 1725 of 2315 face-frames, 17 % of the face.

### 4.2 The change

While an object is at the mouth **and reaches outside the lips** (food, a spoon, a cup), the lips are shown as filmed together with the object: one mouth, the real lips going around the food, blended into the swapped face. It comes on within 2-3 frames and goes away over about half a second (no flash). An open mouth, teeth or a tongue stay inside the lips and do not trigger it.

### 4.3 Results

Synthetic bite: a scoop pressed against the lower lip of a real photo (both lips visible), real pipeline; every lip pixel not covered by the scoop classed as the footage's lip or the swapped lip:

| Scoop | | footage's lip | swapped lip |
|---|---|---|---|
| strawberry | v11.2.91 | 26 % | 61 % (both at once: two lips) |
| | **v11.2.92** | **73 %** | 7 % |
| caramel | v11.2.91 | 31 % | 58 % |
| | **v11.2.92** | **92 %** | **0 %** |

False triggers: on 42 real faces with nothing at the mouth, closed or with a synthetic open mouth (dark inside + teeth), it never fired. It did fire on hair falling across the lips (a real occluder; keeping the real lips under the hair is the right call) and on two "faces" the detector found in leaves and in a shirt (not faces; the job's identity checks drop those). On the no-food clip it never fired.

Cost: the lips during eating are the footage's lips (their shape and colour), not the replacement's. The rest of the face stays swapped.

## 5. Read your log (new lines; the numbers are an example)

```
SIDE-VIEW: outline mask used on 214 face-frames (mean weight 0.81); EATING: the footage's own lips shown around an object on 96 face-frames (mean weight 0.84)
RESTORE: GPEN (fast restore) on 1240 swapped face crops; 251 ms each; mean strength 0.55 (lowered where the footage is soft); 12 crops skipped
```

## 6. Dials (`config.py`)

- `ENHANCE_GPEN = {"blend": 0.6, "sharp_cap": 1.25, "steady": 0.5}`: strength, how sharp relative to the footage, how steady.
- `ENGINE_TUNABLES`: `profile_mask` (1; 0 = v11.2.91 side views), `profile_lo` 0.45 / `profile_hi` 0.70; `eat_mouth` (1; 0 = v11.2.91 mouth), `eat_lo` / `eat_hi` (how much must stick out past the lips), `eat_attack` 0.6 / `eat_release` 0.12.

## 7. What was NOT tested

- **Real footage**: none of it. Side views are still photos with camera motion, not a head turning in 3-D; the bite is a still face with a scoop, not a chewing mouth.
- GPEN on real phone footage: the test clip is one sharp photo. If faces look too smooth or too crisp, change `blend` (0.4-0.8).
- The app: compiled (0 errors) and audited (43/43); the push waiting was tested against the Space's real routes, not on a phone. No APK built here.

## 8. Verification

`verification/fastrestore/`: `unit92.py` (21 checks: side-view weight, outline mask, eating rule incl. open mouth and fade, GPEN strength / skipping / sharpness cap / steady detail, enhancer names), `profile_test.py` + `yaw_calib.py` (AFLW2000 photos), `mask_equiv.py` (v11.2.91 masks unchanged below 45 deg), `make_bite.py` + `bitemetrics.py`, `mouth_false.py`, `make_sharpclip.py` + `restore_metrics.py`, `make_profclip.py` + `profclip_metrics.py`. App: `mobile/verification/BusyTest.kt` + `busy_server.py`. The 13 checks of 11.2.90 (`verification/motionkeys/unit90.py`) still pass.

## 9. Install

1. Space: upload the 8 files (`app.py config.py core_pipeline.py phoenix_api_adapter.py swap_engine.py packages.txt requirements.txt README.md`) to every Space, or build app 1.2.24 first and use its "Push to all Spaces" (it holds back the Spaces that are rendering). Log: `Phoenix v11.2.92`, `swap_engine aequus-1.2.92-fastrestore`; `GPEN restorer ready` at the first job that uses it.
2. App: `Phoenix_APK_Builder_1.2.24.zip` to the APK Builder Space (6 files), Build APK, install over 1.2.23.

