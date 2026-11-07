# Phoenix v11.2.147 — KissFollow (diagnostic, log only)

- No change to how faces are painted (replays and unit checks identical to v11.2.146).
- New log lines: `CONTACT-PATH slot N` (per frame in contact: r rel placement, f follow, x faded out, h held, . not in contact);
  `ID-WHY refusals per slot` (why each contact detection was refused: not_touching, reads_nobody_else, margin, is_partner_det,
  far_from_partner, duplicate_points, no_partner_ref, no_other_id); `ID-WHY first refusals` (the first 60 while touching).
- Engine `aequus-1.2.147-kissfollow`.

# Phoenix v11.2.146 — KissFollow (kiss, partly hidden face)

- **A face partly hidden by its partner in a kiss is taken back where it touches the partner**, when its crop reads as the
  partner (nose and upper face behind the other's face). It is not admitted when its points duplicate the partner's own, nor when
  it reads as nobody (a sliver).
- **A face the detector loses in contact is placed against its partner**, by where the two were last seen touching, when the
  partner moved too far for its own motion to be followed (a sudden approach). It no longer stays at the place from before the kiss.
- Log: `contact_rel=` (frames placed against the partner) and `partner_side=` (partly hidden faces taken back).
- Engine `aequus-1.2.146-kissfollow`. Synthetic tests: unit144 11/11, unit144_pipe 14/14, unit146 3/3 (a face reappearing 120 px
  away, partly hidden: 0 of 12 in v11.2.145, 12 of 12 now); hide and profile-kiss replays unchanged. Not re-benched on video.

# Phoenix v11.2.145 — KissFollow (fix on v11.2.144)

- **A face turned sideways in a kiss is taken back by where it was last really seen when its crop reads as the partner by 0.10**
  (was 0.25). In the field log of v11.2.151 the man, sideways, was rejected 87 times by the identity gate and painted on a
  predicted position for 27 frames: the hazy, lagging copy. A sliver behind the other still stays out (its own similarity is
  higher than the partner's). Engine `aequus-1.2.145-kissfollow`. Synthetic tests: unit checks 11/11; the real-detection replay of
  the profile-kiss and hide clips unchanged. Not re-benched on video.

# Phoenix v11.2.144 — KissFollow

- **A kiss in profile no longer loses the face behind.** In profile the partner fills half of a face's identity crop, so the face
  behind reads as the partner (0.80-0.88); two rules meant for two boxes on one face then removed it, and it was painted on a
  guess and faded to the original. Two people the tracker sees in contact are now never merged, and in contact a crop that reads
  as the partner is given back to its own person by geometry (where it was last really seen, its own facial points).
- **Front / behind from the faces' own identity:** the face whose crop reads as the other person is behind. In profile the old
  outline cue named the face behind as in front, and its replacement took the partner's nose and lips.
- **In contact a missed face moves with its partner** (tracker and paint), instead of drifting on at its own speed or standing
  still; a face that is only a sliver behind the other keeps the v11.2.141 rules.
- Profile kiss test clip, v11.2.141 -> v11.2.144: the face behind reverted 56 -> 0 frames, carried the other person's replacement
  on 37 -> 0 frames, worst lag 62 -> 21 px. The six frontal contact clips: no new revert or mix-up (SDOS-144). Synthetic tests only.
- Belly clean: as in v11.2.142.

# Phoenix v11.2.142 — BellyAnchor

- Belly clean: Preview belly outlines the marks with a **pentagon** (five corners, drawn in yellow over the cleaned frame) that the
  app lets you drag, move and stretch; only the marks inside it are filled, each whole or not at all. The pentagon is kept on the
  belly: never below the line 0.10 above the hip line (unchanged), never on the chest.
- **The video takes what the preview took:** the preview returns each mark with its place on the body and whether it was taken; the
  job gives each mark that same decision on every key frame (`belly_pentagon`, `belly_marks`; app 1.2.45).
- **The cover is held on the body:** the job reads the shoulders and hips (MediaPipe Pose, every 2nd frame) and checks the belly
  cover against them on every frame. Where the image tracker loses the patch or hangs, the body places it and the patch is found
  again in the image from there (coarse to fine). Stress test, frames 20-40 of the fast clip: tracker frozen 82.6 % -> 1.7 % of the
  marks visible, tracker lost 95.2 % -> 0.0 %. Where the tracker is right nothing changes (SDOS-142). Synthetic tests only.

# Phoenix v11.2.141 — ContactOrder

- Kiss / hug / one person passing behind the other: a face nobody can see on a frame (lost behind the other, held at a guessed
  position) is never ranked in front of the face that is seen. On the cross and hug test clips the front person lost its paint for
  11 and 8 frames while the other was hidden behind it; now 0.
- Gender for placing a person who joins late comes from the TARGET person (the app's M / F under each Detect face, or learned from
  the person's own clean frames), never from the replacement photo. A soft cue only, never a reason to drop a face.
- Tested on six synthetic contact clips (two new ones where the people end on the other side): no identity mix-up, no revert, the
  person behind fully painted except as a thin sliver (SDOS-141). Research: reports/Face swap contact identity binding.md.

# Phoenix v11.2.140 — BellyTattoo

- Belly clean (job setting `clean_belly`, preview `phoenix_clean_belly_preview`) covers the **whole belly**: from just under the
  chest (0.56 of the way from the hip joints to the shoulders) down to 0.10 above the hip line, parallel to it (the bikini line, a
  hard limit: nothing below it changes). Tattoos and marks above, beside and below the navel are removed; 118 found 0 % of a
  tattoo above or beside the navel.
- The navel is recognised wherever it sits on the middle line (also with a tattoo stroke touching it); shaded skin at the waist's
  sides is no longer taken for ink (a non-skin piece must be clearly darker than its skin, or far from its colour).
- Built on v11.2.118 (changes 119-139 are not in it). App 1.2.44 works unchanged. Synthetic tests only (SDOS-140).

# Phoenix v11.2.118 — LockFinder

- BodyLock no longer switches off the mark finder for painted dark marks (moles, tattoos). Fast motion: a mole 36 % -> 16 % still
  visible, a tattoo 17 % -> 1 %. Hair and the one-go belly clean are unchanged (the one-go positions keep the v11.2.117 behaviour).
- Known: a fast-moving mole still loses its hold on ~24 of 150 frames; the v11.2.112 base holds it (SDOS-118).

# Phoenix v11.2.117 — OneGoBelly

- The one-go belly clean (job setting `clean_belly`) works on the Space: MediaPipe Pose Tasks (0.10.35), the band from just below the
  navel to the hip line, the navel and clothing left alone (`belly_clear.py`). Before, it failed on `mp.solutions` and painted nothing.
- New route `phoenix_clean_belly_preview`: one frame -> the frame with the marks removed, the band outlined, and the info.
- `requirements.txt`: `mediapipe==0.10.35`; `packages.txt`: `libgles2`, `libegl1`.
- Tested on synthetic skin with MediaPipe's test photo (SDOS-117). Fast motion under BodyLock is not yet tested on this pack.

# Phoenix v11.2.112 — BodyLock

The paint is placed from the surface **around** the stroke, then smoothed. It is no longer driven by the dark texture inside the stroke, which is what made a removal sit still or flicker while the body moved.

- Dark-on-skin strokes (body hair and similar patches) are detected on the anchor frame.
- Their motion is an ECC fit of the surrounding ring, with the stroke excluded.
- That pose is smoothed on a wider window than the global track.
- The per-frame "find the dark mark and snap to it" step is off for these strokes, so strands cannot yank the paint.
- The filled area is grown by 5 px so the soft fringe is covered.
- Other marks (moles, tattoos, ink) still use the existing local fit and mark finder.

Synthetic check: a dark patch translating 3 px / 2 px per frame on a textured plate stayed within 2.9 px of the true position (mean 0.8 px), with frame-to-frame residual jitter of 0.5 px mean.
