# SDOS-177 — ContactStep: the front / behind decision when one face slides behind the other

**Pack:** v11.2.177, which is v11.2.176 plus one change in `core_pipeline._front_relations`. Engine
`aequus-1.2.177-contactstep`. Everything in SDOS-176 holds. This document adds only the change and its measurements.

All tests use synthetic clips made from public sample photos. No user footage was looked at.

## 1. What 176 got wrong

When two faces first touch, who is in front is decided once, by box size. If the sizes are within 10 % (a near tie), the
decision may change on the first clear (10 %) difference.

In the **cross** clip (A walks behind B), A's box is 9 % *bigger* on the first touching frame, so A is put in front. A then
shrinks steadily as it is covered: 1.09, 1.07, 1.06, 1.03, 1.00, 0.98, 0.94, 0.93, 0.91 of B's size. It is hidden before it
is ever 10 % smaller, so A stayed in front on frames 40-49. B, the face truly in front, was cut on 3 frames (revert) and
showed A's replacement on 2. On 175, whose per-frame size rule switched at frame 46, B was clean.

## 2. The change

The near-tie first decision can also switch once on a steady shrink. When the face put in front has lost 8 % of its size
against the other since that decision, and is now the smaller one, two frames in a row, the other face goes in front. After
any first switch, the steady rule (10 % for 4 frames) applies, as in 176.

The two-frames-in-a-row condition is what keeps the side-view kiss steady. Detector jitter that alternates every frame
(±6 % area) never meets it.

## 3. Verification (from the pack's own final files)

**Unit checks:** 26 of 26 pass (`unit176.py` 12, `unit176_pipe.py` 14). The two new ones are:
- "first touch a near tie, then the face in front shrinks steadily: it goes behind within 2 frames of becoming the
  smaller" — switches at index 7. **It fails on 176**, which never switches.
- "detector jitter that alternates every frame (±6 % area) never switches it" — 0 changes.

**The real job on 177's files** (the same bench as SDOS-176):

| Clip | v11.2.175 | v11.2.176 | v11.2.177 |
|---|---|---|---|
| Cross, B (in front) | revert 0, mix-up 0, weak 2 | **revert 3, mix-up 2**, weak 5 | revert 0, mix-up 0, weak 4 |
| Cross, A (passes behind) | weak 7; worst 115 px off, jump 25 px | weak 8; worst 105 px, jump 5.7 px | weak 8; worst 105 px, **jump 5.7 px** |
| Hug, B (in front) | weak 9, revert 0 | not run | weak 11, revert 0 |
| Hug, A | weak 0; worst 143 px off, jump 43 px | not run | weak 0; worst 131 px, jump 22 px |

The 177 log on the cross clip: `FRONT/BEHIND: ... by sight on 25, by size (held) on 19; changes 1`, and `UNPAINTED person
1: 0 frames none`, `UNPAINTED person 2: 0 frames none`.

**Still running at delivery** on 177's files: the side-view kiss, smooch, hide and the three step clips. Their results will
follow. From SDOS-176:
- The step clips have no overlapping faces, so this rule does not act there.
- The side-view kiss is decided by identity on 78 of 81 face-pair frames. The new rule acts only on a near-tie first
  decision made by size.

## 4. Limits (in addition to SDOS-176 section 5)

- **Hug:** B is weak on 2 more frames than in 175 (40-41). A is 6-7 % bigger yet behind when the two first touch. The trend
  needs two frames to switch, so the 11.2.92 per-frame rule switches 2 frames sooner here. It would also flip on the
  side-view kiss.
- The size cue cannot know who is in front while the face going behind is still the bigger one (hug frames 32-39). That is
  the same on 175.

## 5. Correction to SDOS-176

In the side-view kiss table, "the other person's paint on a face" is **at most 4 %** over all 120 frames, not 1 %. 175: up
to 51 %, and 5 % or more on 128 face-frames.

## 6. Install

Upload the pack's files. Only `core_pipeline.py`, `swap_engine.py` (version string) and `config.py` (version) differ from
176. The log at start reads `Phoenix v11.2.177` and `swap_engine aequus-1.2.177-contactstep`. The app (1.2.49) and the APK
Builder need no change. Do not restart a Space while a job runs on it.
