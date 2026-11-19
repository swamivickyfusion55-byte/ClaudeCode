# SDOS-212 — TruePerson: a person's replacement never goes on a face of the other gender

**Pack:** v11.2.212, built on v11.2.211. Engine `aequus-1.2.212-trueperson`.

**Asked:** "man's face pasted on women and vice versa in few parts of the video.. Please strengthen the identity logic.. A girls
face pasted on man with stubble as well."

No footage was looked at. Every number below is from synthetic clips of a man and a woman (public sample photo; made by
`make_mw.py`: a kiss, a side kiss, a crossing, a hug, and a clip that starts in the middle of the crossing, as a later part of a
split video does), compared with **11.2.211 at Ultra**, the whole job (`bench_pipeline.py`), and who was painted on whom read
back from the output video (`identity_check.py`).

---

## 1. What was wrong

### 1.1 The genders you confirm in Detect never reached the engine

The app sends each person's M / F from the Detect step (the letter under each face, which you can tap to correct) as
`person_gender` with every job, and every part of a split. v11.2.141 used it. When the engine was rebuilt on 11.2.92 it was lost
twice over: the Space's API kept only a fixed list of settings and `person_gender` was not on it, and the job had no place for it.
In 11.2.211 the app's genders were thrown away on arrival.

### 1.2 The tracker took the gender of the replacement PHOTO for the person's

Without them, the gender a person was looked for by was the gender the model reads on that person's replacement photo. That is
the person's own gender only when the swap is woman-to-woman and man-to-man and the photo is read right. Where it is not - a
woman given a man's face, or the reverse, or a photo the model misreads (short hair, a cap, a side view, low light) - the rules
that use it point the wrong way:

- **the first binding of a person without a match to their Detect picture** (11.2.209's "evidence" rule: a confident front view
  "of this person's gender") binds the person to a face of the photo's gender - the other person;
- **a person who joins late** (comes into the picture, or was hidden) is offered first the face of the photo's gender.

- and where no Detect picture decides, **the first person bound in a part simply takes the face that is there** (the
  leftmost), with no look at gender at all.

Each binds for the rest of the part. In a split video every part starts with nobody bound, so it happens in some parts and not
others: "in few parts of the video".

### 1.3 Nothing checked the gender of a face once a person was bound

The identity rules compare a face with the person's Detect picture and their own learnt look (ArcFace). When the Detect picture
is poor for a part (another angle, other light, a face half hidden on the Detect frame), and the first binding in a part was the
other person, nothing in 11.2.211 could see it was wrong: the learnt look is then the wrong person's, and every later frame agrees
with it.

### 1.4 What the gender model can and cannot be trusted with (measured, `gender_read.py`)

The model's own certainty (insightface keeps only its yes / no; 11.2.212 keeps the score) on every detection of the four
man-woman clips (five people of the sample photo: three women, two men), each also as a phone-quality copy (camera noise, blur,
H.264 at crf 30) - 8 clips, each person matched by their true points. "Wrong" means the model gave the other gender this
probability or more:

| view of the face | | face-frames | wrong at 0.88+ | wrong at 0.70+ | unsure (0.30-0.70) | lowest P of the true gender |
|---|---|---|---|---|---|---|
| front, not touching the other face | woman | 417 | **0** | **0** | 5 | 0.37 |
| | man | 562 | **0** | **0** | 0 | 0.88 |
| turned (nose 0.34+ eye distances off centre), not touching | man (no turned woman in the clips) | 112 | 0 | 0 | 0 | 0.97 |
| touching the other face | woman | 275 | **8** (hug: 4 of 34, its phone copy 3 of 34; kiss 1 of 29) | 31 | 50 | 0.06 |
| | man | 275 | 0 | 0 | 0 | 0.71 |

(`verification/trueperson212/gender_read.txt`, per clip.)

So: on a clean view - front, not touching the other face, a firm detection of a fair size - it was never confidently wrong. While
two faces touch it is: the woman read as a man with 0.88 or more on up to 4 of 34 frames, because the man's face is in her crop.
11.2.212 reads gender only on clean views.

### 1.5 A mistake of mine the bench caught (fixed before this pack)

The first build of 11.2.212 took "the face where the bound person is expected" as that person's own face. In the hug clip the
woman goes behind the man and is not detected; the only face where she was is his - alone, front, a man. That build counted it
against her, let her binding go, and she was unpainted for 3 frames until her Detect picture found her again. Now a face counts as
the person's own only if it also looks like them (ArcFace 0.40 to their learnt look, and more than like anyone else bound).
`unit212.py` has the case; it fails on the first build and passes on this one.

---

## 2. What 11.2.212 changes

| | Change | Switch (`swap_engine._P`; 0 = 11.2.211) |
|---|---|---|
| 1 | **The app's genders reach the job again** (API, job settings, tracker); the log says which were given. | - |
| 2 | **A person is looked for by their own gender** (the one you confirmed in Detect), not the replacement photo's. Where the app gave none, the photo's steers as in 11.2.211 - never as a gate, a swap across genders is allowed. | - |
| 3 | **The gender gate.** A face that reads as the other gender on a clean view with 0.88 or more is not that person: no path of the identity logic (the picture match, the learnt look, a late join, elimination, the uncrossing) gives it their replacement. A face that looks like (ArcFace 0.45) one that did is refused too, unless it reads as the right gender now (0.70) - so the gate holds through the frames where the face is turned or touching. | `id_gender_gate` 1, `id_gender_bind` 0.88, `id_gender_yaw` 0.34, `id_gender_det` 0.50, `id_gender_minw` 48, `id_gender_mem_sim` 0.45, `id_gender_agree` 0.70 |
| 4 | **A bound person whose face reads as the other gender is let go.** Evidence adds up over clean views (decayed log-odds); at about 3 clean frames against, the person's binding and learnt look are dropped and they are found again. On each such frame their replacement is not painted on that face. | `id_gender_decay` 0.90, `id_gender_unbind` 9.0, `id_gender_own_sim` 0.40 (a face counts as theirs only if it also looks like them, §1.5) |
| 5 | **Found by gender.** When the person is the only one of their gender, and exactly one face on screen reads as that gender (0.95, clean view, 3 detections in a row), not someone already found and not unlike their own Detect picture, that face is theirs - at once, instead of after the ~45 sightings of the elimination rule. | `id_gender_found` 0.95, `id_gender_found_calls` 3, `id_gender_found_ref` 0.10 |
| 6 | **The gender model's certainty is kept** on each face (same computation as insightface; anything unexpected falls back to the original). | - |

Not changed: Ultra's and Optimised's quality settings, the contact and occlusion work of 11.2.206-211, skin tone, the swap
network. With no genders from the app (an older app, or a person left unknown) the job is 11.2.211's (§3.4).

---

## 3. Verification

### 3.1 Who is painted on whom - the whole job, Ultra, 11.2.211 against 11.2.212

`bench_pipeline.py` (the real job) on each clip, then `identity_check.py` on the output video: each person's face cut at their
true position, ArcFace against both replacement photos, counted on the frames where at least 70 % of the face is visible. Scored
against the replacement each person **should** get (`IDC_INTENDED`). The app's genders given (woman = slot 1, man = slot 2) in
every run but the last.

| clip | Detect pictures | photos | 11.2.211 | 11.2.212 |
|---|---|---|---|---|
| **a later part of a split: opens in the middle of a crossing, the woman 22 % visible** | none | woman's on her, man's on him | **wrong on every frame**: she has the man's replacement (36 of 36), he the woman's (68 of 68) | **right on every frame** (36 of 36, 68 of 68) |
| the same part, photos across genders (she gets the man's face, he the woman's) | none | across | **wrong on every frame** (36 of 36, 68 of 68) | **right on every frame** (36 of 36, 68 of 68) |
| the same part | both | as above | right on every frame | right on every frame |
| side kiss | none | woman's on her, man's on him | right (57 and 120 frames) | **the same video, pixel for pixel** |
| side kiss | both | as above | right | **the same video** |
| side kiss | none | across | right | **the same video** |
| the whole crossing | both | woman's on her, man's on him | right (72 and 120 frames) | **the same video** |
| hug (she goes behind him) | both | as above | right; 3 frames of her unpainted at the deepest point | **the same video** |
| two people kissing (`smooch`), **no genders from the app** | - | as the older bench | right | **the same video** |

The first two rows are your report: the man with the woman's face and the woman with the man's, for a whole part. In 11.2.211 a
part that opens with one of them barely visible binds the first person to the face that is there (no Detect picture to say
otherwise), and every later frame agrees with that binding. In 11.2.212 the man is never given the woman's replacement on a
clean front view, so she is the one found.

"The same video": every frame of the output and every paste mask equal to 11.2.211's (largest pixel difference 0), on all
seven clips where 11.2.211 was right. 11.2.212 changes the result only where 11.2.211 had the wrong people, and nothing at all
when the app sends no genders.

### 3.2 The tracker alone (`unit212.py`, stand-in faces, no models)

17 checks, all pass. Among them, with 11.2.211 for comparison where it differs:

| case | 11.2.211 | 11.2.212 |
|---|---|---|
| no Detect pictures, the man on the left | the woman's replacement on him, 120 of 120 | 0 wrong, 120 right |
| a part opening on the woman alone, her picture poor (she reads more like the man's) | the man's replacement on her, 80 calls | 0 wrong; found by her gender on call 2 |
| bound to the wrong people (as after a swap in contact) | stays wrong | not painted wrong from the first clean front view; both found again within 2 calls |
| faces touching, the woman misread as a man on 1 frame in 3 | - | she keeps hers on every frame |
| a side view of the woman misread as a man | - | she keeps hers |
| two women on screen, a poor picture | - | nobody "found by gender" |
| a hug: she hidden, his face where she was | - | not let go; hers again on the first frame back |
| no genders from the app / `id_gender_gate = 0` | - | every decision equal to 11.2.211 |

### 3.3 Nothing else moved

Every earlier suite on the 11.2.212 files: `unit211` 23, `unit210` 19, `unit209` 13, `unit206` 4 + 4 (two clips), `unit205` 30,
`unit204` 7, `unit203` 18, `unit176` 12, `unit176_pipe` 14, `nofalse206` 1, and `unit212` 17 - **162 checks, all pass**
(`verification/trueperson212/unit_suites.txt`). The occlusion, contact, eating and skin-tone paths are untouched by the change
(it is in the identity step only).

The gender model's readings (§1.4): 8 clips, 1,641 face-frames.

## 4. What the log says now

At the start: `people's genders from the app (0=F 1=M): {0: 0, 1: 1} · gender gate on · replacement photos: {...}`. When the
gate acts: `person N: a face that reads (or looked) like a man was not given their replacement`, `person N: bound to a face
that reads as a man on clean views (P 0.97) - let go, to be found again`, `person N: found by gender - the only woman ...`. At the
end: `GENDER: faces refused ... N; bound people let go for it N`.

## 5. Limits

1. **A wrong M / F in Detect blocks the right person.** The gate trusts the letter you confirm. If the app shows the wrong
   letter under a face and it is not corrected, that person's own face is refused on every clean view and stays unswapped.
   Check the letters in Detect (tap to correct) before you start; a person left unknown is not gated.
2. **Two people of the same gender** get no help from the gate: it only tells a man from a woman. Their identity is the
   Detect picture and the learnt look, as in 11.2.211.
3. **Contact and turned faces are not read.** The gate reads only front views apart from the other face. A part that is all
   contact or all profile gets no gender evidence; what the gate learnt on clean frames still holds there (the look memory), but
   a part that starts in contact and never parts is as in 11.2.211.
4. **Faces the model gets wrong when clean** (strong make-up on a man, a woman with very short hair and a strong jaw, a child)
   were not in the tests: 0 confident errors in 979 clean face-frames of five people is not proof for every face. The
   threshold (0.88, and 0.95 to find someone) is set so a borderline reading does nothing.
5. Synthetic clips only; no footage was looked at. Whether your report is fully covered can only be seen on your videos.

## 6. Install

Upload the pack's files to the Space. Changed against 11.2.211: `swap_engine.py`, `core_pipeline.py`,
`phoenix_api_adapter.py`, `config.py`. No change to `packages.txt`. **No app change**: app 1.2.42 and later (1.2.63 included)
already send the genders. The log at start reads `Phoenix v11.2.212`.

`id_gender_gate = 0` turns the gate off (the app's genders then only steer, as in 11.2.141).
