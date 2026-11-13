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

All runs are on this exact code (file checksums compared).

**The fallback** (harsh-step clip, two people, 120 frames, Ultra, both swapped; `bench_pipeline.py refs=...`). Per frame,
`c` = clearly the right replacement, `u` = unclear (for person B on frames 0-45: not swapped yet):

| Detect pictures | | Person A | Person B |
|---|---|---|---|
| A right, B a picture of someone else | 11.2.206 / 207 | 120 of 120 correct | **never swapped** (the C77E081B pattern) |
| | **11.2.208** | 120 of 120 correct | swapped from frame 46 to the end: **74 correct, 0 mix-ups**; frames 0-45 original (the wait) |
| A and B right | 11.2.206 | 120 / 120 | 120 / 120 |
| | **11.2.208** | identical | identical (the fallback never fired) |

Person B's painted position with the wrong picture is the same as with the right one: 0.0 px apart on average, 0.7 px at most
(`place_vs.py`). Person A has one frame 47.5 px off in the wrong-picture runs; 11.2.206 has the same frame. A's average went
from 1.8 px to 0.9 px.

**Two defects found and fixed while testing:**
1. **Backward hold.** The first version filled B's frames before the fallback with B's later position: about 1 s, up to 14 px
   off. Those frames now stay original.
2. **Noise resets.** After the fallback, an 11.2.176 safeguard reset B's identity: "the lock looks more like another person's
   picture" (`own=-0.03 other=0.04 - reset`). With a picture that matches nobody, that comparison is noise, and B was lost
   again from frame 95 to the end. Now, once a person's picture has been given up, only a real likeness to another picture
   (0.35) resets them, and if lost they are found again at once.

**Contact clips, Ultra, no Detect pictures** (the fallback cannot fire there; they check the 11.2.207 front/behind fix):

| Clip | 11.2.206 identity correct / unclear | 11.2.208 | Everything else |
|---|---|---|---|
| Harsh step | 240 / 0 | 240 / 0 | identical |
| Side-view kiss | 136 / 104 | 135 / 105 | mix-ups 0, reverts 7, weak 27, placement: identical |
| Hide | 164 / 3 | **166 / 1** | identical |
| Smooch (11.2.207) | 168 / 2, 1 mix-up | **170 / 1, 0 mix-ups** | |
| Crossing (11.2.207) | 192 / 4, weak 12 | **193 / 3, weak 10** | |

The side-view kiss and hide runs used the code from before the two tracker fixes above. Those fixes act only with Detect
pictures, and these runs have none.

**Unit checks on this pack:** `unit176` 12, `unit176_pipe` 14, `unit204` 7, all pass.

## 3. Limits

1. **The cost of the fallback is a short delay.** A person whose picture does not match stays original until it fires: 45
   detector calls, about 1.5 s in the test (frames 0-45). After that, the same person is found again at once.
2. **It cannot help with crowds.** With more faces on screen than people to swap, the picture still decides, so a bystander is
   never chosen by elimination.
3. **6332D118-type refusals are not changed.** There both people were already found, and later faces matched neither of them.
   The IDENTITY warning (SDOS-207) says when it happens.

## 4. Install

Upload the pack's files to the Space, including `packages.txt` (the Space rebuilds once). Changed against 11.2.207:
`swap_engine.py`, `core_pipeline.py`, `config.py`. The log at start reads `Phoenix v11.2.208`.
