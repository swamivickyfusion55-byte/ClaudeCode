# SDOS-208 — RefFallback: a Detect picture that matches nobody no longer leaves the person original

**Pack:** v11.2.208, which is v11.2.207 (SDOS-207) with the Detect-picture fallback switched on. Engine
`aequus-1.2.208-reffallback`. Everything else as 11.2.207: the front/behind fix, the skin tone libraries in `packages.txt`,
and the IDENTITY warning.

**Why:** in your jobs C77E081B and 3717965D (the same video, part 4) person 2 stayed original for the first 32 seconds when
Detect pictures were used. Without them, person 2 was swapped from the first shot cut (frame 85). With a picture, that
picture was the only way to find the person, and it did not match how they looked in this part.

All tests use synthetic clips made from insightface's public sample photo. No user footage was looked at.

## 1. What changes

A person whose Detect picture matches no face for 45 detector calls in a row (about a second) is found by elimination, as if
there were no picture, only when all of these hold:
- a face that no one else has is on screen;
- there are no more faces on screen than people to swap (with bystanders around, the picture must still decide);
- no other person's picture matches that face (0.30), and it does not look like a person already found (0.30);
- the replacement face's gender matches, when it can be read (the same rule as without pictures).

The person is swapped from that frame on. The frames before it stay original: they are not painted with the person's later
position, which a first version did (§2).

The log says it once per person: `person 2: the Detect picture matched no face for 45 detector calls - found by elimination
instead`, and at the end: `IDENTITY: N face(s) given to a person by elimination because their Detect picture matched no face for
a while`. Off: `trk_ref_stale_calls = 0` in `swap_engine.py` (that is 11.2.207).

## 2. Verification

**The fallback** (harsh-step clip, two people, 120 frames, Ultra, both swapped; `bench_pipeline.py refs=...`):

| Detect pictures | | Person A | Person B |
|---|---|---|---|
| A right, B a picture of someone else | 11.2.206 / 207 | 120 correct, 0 mix-ups | **never swapped** (0 % painted; the C77E081B pattern) |
| | fallback on (first version) | 120 correct, 0 mix-ups | swapped after about a second; 88 correct, 32 unclear, **0 mix-ups** |
| A and B right | 11.2.206 | 120 / 120 | 120 / 120 |
| | fallback on | identical; the fallback never fired | identical |

**The first version painted B's frames before the fallback with B's later position:** about 1 s, up to 14 px off (placement
mean 9.1 px vs 1.5 px with the right picture). Those frames now stay original. **That change is being re-measured now; the
result follows.**

**Contact clips** (harsh step, side-view kiss, hide; no Detect pictures, so the fallback cannot fire there; they check the
11.2.207 front/behind fix): running now; results follow. The smooch and the crossing were measured on 11.2.207
(SDOS-207 §3: smooch mix-ups 1 → 0).

**Unit checks on this pack:** `unit176` 12, `unit176_pipe` 14, `unit204` 7, all pass.

## 3. Limits

1. **The cost of the fallback is a short delay.** A person whose picture does not match stays original for about a second
   before being found.
2. **It cannot help with crowds.** With more faces on screen than people to swap, the picture still decides, so a bystander is
   never chosen by elimination.
3. **6332D118-type refusals are not changed.** There both people were already found, and later faces matched neither of them.
   The IDENTITY warning (SDOS-207) says when it happens.

## 4. Install

Upload the pack's files to the Space, including `packages.txt` (the Space rebuilds once). Changed against 11.2.207:
`swap_engine.py`, `core_pipeline.py`, `config.py`. The log at start reads `Phoenix v11.2.208`.
