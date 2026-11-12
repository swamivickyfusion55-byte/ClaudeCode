# v11.2.92 FastRestore vs v11.2.205 SkinTone — what is the same, what changed, measured side by side

All tests use synthetic clips made from public sample photos (two people stepping, kissing, crossing, one face hiding behind
the other). No user footage was looked at.

## 1. Is 11.2.205's swap built on 11.2.92? Yes.

The line is: 11.2.92 FastRestore → your 11.2.175 / 11.2.202 (on 11.2.92) → 11.2.176–179 contact fixes → 11.2.203 →
11.2.204 → 11.2.205. Compared function by function with the 11.2.92 pack:

| File | 11.2.92 functions unchanged in 11.2.205 | Changed | New |
|---|---|---|---|
| `core_pipeline.py` (detection, swap, masks, compositing, restoration, the job) | **191 of 203** | 12 | 27 |
| `swap_engine.py` (the swap model wrapper, compositor, tracker) | **80 of 88** | 8 | 8 |

Byte-identical to 11.2.92:
- the face detector and its settings;
- the swap network and how it is called;
- the occlusion model (XSeg) and every mask: the landmark outline, the side-view outline, the eating rule;
- the colour match, the blending and the compositor;
- GPEN restoration and the Cinematic finish.

What changed is only:

| Area | What changed | Version |
|---|---|---|
| Two people (tracker) | which face is which person when they touch, step fast or cross; front / behind when one slides behind the other | 176, 177 |
| The job loop | your saved inputs for 24 hours; encrypted (AES-256-GCM, Space secret `PHOENIX_STORE_KEY`) | 202, 203 |
| Optimized only | every frame swapped in contact or side poses, otherwise every 2nd frame; the ArcFace skip | 203, 204 |
| Ultra | nothing in the swap. Ultra swaps every frame exactly as 11.2.92 does | – |
| New features | manual patch removal (your 175), the Preview endpoints (178, 179), skin tone (205, off unless used) | 175–205 |

## 2. Measured side by side (Ultra, the same clips, the same machine at the same time, 2 CPU threads each)

Five synthetic two-person clips, Ultra (a swap on every frame), the real pipeline (detector, swap model, compositor), scored
against the clips' ground truth (`identity_check.py`, `contact_lag.py`). "Weak" = the swapped face painted at under 90 % of its
own strength on that frame; "revert" = the original face shows; "mix-up" = the other person's replacement on this person;
placement = how far the painted face sits from the true face (px); flicker = frame-to-frame change against the footage's own.

| Clip | | Mix-ups | Reverts | Weak frames | Placement mean, person A / B | Worst, A / B | Flicker A / B |
|---|---|---|---|---|---|---|---|
| **Harsh step** (two people, a fast step) | 11.2.92 | 0 | 0 | 0 | 4.1 / 1.5 px | **50.2** / 2.7 px | 1.01 / 0.98 |
| | 11.2.205 | 0 | 0 | 0 | **1.6** / 1.5 px | **4.8** / 2.7 px | 1.00 / 0.98 |
| **Side-view kiss** | 11.2.92 | **22** | **48** | 20 | 29.6 / 16.1 px | 62.0 / 33.0 px | 1.04 / 0.97 |
| | 11.2.205 | **0** | **7** | 27 | **7.9 / 3.7** px | **16.3 / 7.7** px | 1.02 / 0.97 |
| **Crossing** | 11.2.92 | 0 | 0 | 9 | 24.2 / 0.7 px | 114.6 / 1.8 px | 1.01 / 1.00 |
| | 11.2.205 | 0 | 0 | 12 | 23.9 / 0.7 px | 104.9 / 1.6 px | 1.01 / 1.00 |
| **One face hides behind the other** | 11.2.92 | 0 | 0 | **30** | 38.7 / 7.2 px | 65.6 / 10.6 px | 1.00 / 1.01 |
| | 11.2.205 | 0 | 0 | **3** | 39.0 / 7.2 px | 66.3 / 10.6 px | 1.00 / 1.01 |
| **Smooch** | 11.2.92 | 0 | 0 | 45 | 17.0 / 3.9 px | 30.4 / 6.5 px | 1.01 / 1.01 |
| | 11.2.205 | 1 (frame 34) | 0 | 46 | 17.0 / 3.9 px | 30.4 / 6.5 px | 1.01 / 1.01 |

**Speed (Ultra), each version alone on the same two CPU threads, one after the other:** harsh step, 120 frames —
11.2.92 **296.7 s**, 11.2.205 **295.7 s**. The same: the swap step costs the same (1742 / 1738 ms per call), the detector step
10 ms more per call (the fast-step rescue of 11.2.176). Both did the same work on every clip (the same number of detector
passes, swaps and ArcFace passes). (The five clips above were run with both versions at the same time, on different CPU
cores, so their times are not comparable and are left out.)

## 3. Is it better "in all respects"?

Better where it matters, the same elsewhere, and not better in every single number:

- **Clearly better:** the side-view kiss (mix-ups 22 → 0, the original face showing 48 → 7 frames, bad frames in total 68 → 34,
  placement error a quarter), the fast step (worst placement 50 px → 5 px), one face hiding behind the other (weak frames 30 → 3).
- **The same:** speed in Ultra; flicker; everything about occlusion, masks, colour and restoration (the same code, §1).
- **Not better, honestly:**
  - the crossing clip is a draw (3 more weak frames, a 10 px smaller worst error);
  - the kiss has 7 more *weak* frames (27 vs 20) — most of the frames where 11.2.92 lost the face entirely (48 reverts) are now
    painted, some of them at reduced strength;
  - the smooch has **one** frame (34) where the other person's replacement shows for a frame (11.2.92: none) — it came with
    the 11.2.176 contact rules and is the one measured regression;
  - a few more frames read "unclear" by the identity check on the crossing and hiding clips (3-5 of about 200).
- Optimized and Balanced also changed (SDOS-203/204: every frame in contact or side poses, otherwise every 2nd), which these
  Ultra runs do not cover; SDOS-204 compares Optimized with Ultra.

These are synthetic clips with still-photo faces: they measure placement, identity and coverage, not expression.

## 4. Occlusion smoothness: what is left to gain

The occlusion logic of 11.2.205 is 11.2.92's, unchanged (§1). Reading it again, there is real room to make it smoother. In
order of expected gain:

| # | Opportunity | What you would see | Cost per face, per frame |
|---|---|---|---|
| 1 | **Instant cut when an object arrives.** A new cut opens at 45 % per frame (`occ_open`), so a hand or cup that appears is partly see-through for its first 2-3 frames. Open at once where the occlusion model is sure; keep the slow give-back (22 % per frame) so nothing flashes when it leaves. | no see-through leading edge on a fast hand, spoon or cup | none |
| 2 | **Move the previous cut with the object.** The cut is smoothed over time in the face's own frame, so an object sliding across the face leaves a fading trail where it was. Moving last frame's cut along the object's motion (optical flow inside the face crop) before smoothing removes the trail. | no ghost edge behind a sliding hand or hair strand | about 5 ms |
| 3 | **Sharper cut edges.** The occlusion model works at 256 px and its mask is blurred (sigma 3), which at full size is a soft band of several pixels around the object. Fitting that edge to the picture's own edges (a guided filter, already used by the skin tone) gives a clean cut. | crisp edge where a finger or cup crosses the face | about 5 ms |
| 4 | **The occlusion model over the whole face.** The model's cut is used only in the mouth-and-chin zone; elsewhere (eyes, forehead, cheeks) only the colour test cuts, which is weaker on skin-coloured hands. Using the model's cut over the whole face, with the existing guard against false cuts on turning heads, covers hands over the eyes or a cheek. | hands over eyes / cheeks cut cleanly | none (the model already runs) |
| 5 | **Cut-to-cut between two faces in contact.** Where two swapped faces overlap (a kiss, cheek to cheek), front / behind is decided per face by the tracker. 11.2.205 already has a per-person segmenter (MagicTouch, used by the skin tone); using it there would give a pixel-exact boundary between the two faces. | a clean line where two faces meet | about 85 ms per person, only on contact frames |

These are proposals, not yet built or measured. 1-4 are small, local changes to the mask code (Ultra and every other tier
benefit); 5 is larger. Each would be checked on the object and contact clips (`make_objects.py`, `obj_metrics.py`, the kiss
clips) against 11.2.205 before shipping.
