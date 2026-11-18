# SDOS-209 — SureID: the other person's replacement is never put on a face on a guess

**Pack:** v11.2.209, built on v11.2.208. Engine `aequus-1.2.209-sureid`.

**Asked:** "check and revert some bad changes of last 3-4 versions, the swap is missed or male is swapped on female in few
frames" (the Space log of two v11.2.208 jobs). Only the log was read; no footage. All tests use stand-in data or synthetic
clips made from insightface's public sample photo.

---

## 1. What was wrong

### 1.1 The 11.2.208 fallback fired without real evidence (job DFA984A1)

11.2.208 let a person whose Detect picture matches nobody be found "by elimination". It counted every look at the video in
which *some* face was left unclaimed, and took a face that reads unlike the people already found as someone else. That is
wrong: the same person in a side view reads 0.1-0.3 against their own first sighting, so a found person alone on screen can
be that "unclaimed, unlike" face.

In job DFA984A1 (a Detect picture for person 1, the male replacement, only):
- For the first 341 frames only person 2 (the female replacement) is painted. She is unpainted on frames 32-42, 108-123 and
  327-341: 42 frames. The fallback needed 45 looks with an unclaimed face. A second person on screen beside her would have
  given it 45 within two seconds; it took until frame 342. So the count was almost certainly made of her own refused frames.
- At frame 342 the log says `person 1: the Detect picture matched no face ... found by elimination`, and person 1 is first
  painted on that frame.
- **The log cannot say which face was taken at that moment.** The tracker test shows what 11.2.208 does when the face is the
  woman's, in a view her own slot refuses: it puts the male replacement on her (`unit209.py`, first check, on the 11.2.208
  pack).
- After it fired the two identities were confused: 370 faces were refused for looking equally like both people
  (`rel_no_margin` 219 + 151; under 30 in every earlier job), and only 55 % of frames were painted.

So this fault can cause both things reported: the male replacement on the woman, and missed swaps.

### 1.2 The same weakness in the old first-binding rule (11.2.92 to 11.2.208)

Without Detect pictures, a person not found yet took any face that read under 0.45 against the people already found. The
woman alone on screen in an unfamiliar view was such a face, and the man's replacement was put on her. From then on it stayed
on her in that view. `unit209.py` reproduces this on the 11.2.205 pack (second and third checks).

### 1.3 The 11.2.207 front / behind rule changes on detector noise

11.2.207 switched which face is in front as soon as it shrank on two frames in a row (5 % in all). It removed one wrong frame
of a synthetic clip, whose boxes are free of noise. A real detector's boxes jitter. Two same-size faces overlapping for 300
frames with nothing really changing (`front_jitter.py`, the real `_front_relations`, 400 trials):

| Box-size jitter | 11.2.176-206 rule: contacts where the face in front changed on noise | 11.2.207 rule |
|---|---|---|
| 1 % | 0 of 400 | 147 of 400 |
| 2 % | 109 of 400 | 393 of 400 |
| 3 % | 261 of 400 | 345 of 400 |

A wrong face in front paints one person's replacement over the other where they overlap.

## 2. What 11.2.209 changes

| | Change | Switch (in `swap_engine._P` / `core_pipeline`) |
|---|---|---|
| 1 | **A person not found yet is given a face only on evidence** (below). This covers the 11.2.208 fallback and the old first-binding rule. | `trk_virgin_guard` 1 (0 = the old rule) |
| 2 | **The 11.2.207 front / behind rule is off.** The 11.2.176 rule decides, as in 11.2.206. | `FRONT_TREND = False` |
| 3 | The end-of-job IDENTITY lines say what happened (§4). The 11.2.207 "person N was refused X times" warning is removed: its count included every correct refusal of the other person's face. | - |

**The evidence (change 1).** A face that matches no picture of the not-yet-found person is given to them only when:
- **nobody else has been found yet** (the first binding of the job, as always); or
- **every person already found is identified in the same frame on another face**: then this face can only be someone else; or
- **it is a confident front view whose gender is the replacement's**, and not that of a found person missing from the frame, on
  12 looks in a row. Side views are not trusted for gender.

One face alone on screen, in a view the person already found does not match, is left original. It is never given the other
person's replacement.

**The Detect-picture fallback** (11.2.208) is kept, on the same evidence, and stricter:
- exactly one face left over, a confident detection (score 0.5 or more), at least 0.4 of the size of the other faces;
- unlike everyone found and every other picture;
- 45 sightings of the same face, each with that evidence, no more than a second apart (it counted any 45 looks before);
- a person found this way who later looks like the other person is undone, and not found this way again in that job.

**Kept as they are:** the 11.2.206 occlusion mask (§3.4), the 11.2.207 skin-tone libraries, the rest of 11.2.208.

## 3. Verification

### 3.1 The tracker (`unit209.py`, stand-in faces, the real `MultiFaceTracker.assign`)

| Case | 11.2.205 | 11.2.208 | 11.2.209 |
|---|---|---|---|
| The woman alone, the man's picture matching nobody (job DFA984A1) | never on her | **male replacement on her** | never on her |
| The woman alone, no pictures | **male replacement on her** | **male replacement on her** | never on her |
| The same, her side view read as male | **on her** | **on her** | never on her |
| Both on screen, no pictures: the man found at once | yes | yes | yes |
| Both on screen, his picture matching nobody: found by elimination | never found | after 45 looks | after 45 sightings |
| The same, his gender read wrong | never found | found | found |
| A weak detection (score 0.35) taken this way | no | **yes** | no |
| A third face on screen | not taken | not taken | not taken |
| The man alone after the woman left, front view, read as male | at once | at once | after 12 looks |
| The man alone in a side view, or read as female | at once (a guess) | at once (a guess) | waits; found at once when the woman is back |
| One-person job; first person of a two-person job | at once | at once | at once |

All 13 checks pass on 11.2.209.

### 3.2 The full job, Ultra, against 11.2.206

Scored against the clips' ground truth (`identity_check.py`, `contact_lag.py`).

**Two people in contact, no Detect pictures** (does the new rule or the reverted one lose anything?):

| Clip | | Identity correct / unclear | Mix-ups | Reverts | Weak frames | Placement mean A / B | Worst A / B |
|---|---|---|---|---|---|---|---|
| Harsh step | 11.2.206 | 240 / 0 | 0 | 0 | 0 | 1.6 / 1.5 px | 4.8 / 2.7 px |
| | 11.2.209 | 240 / 0 | 0 | 0 | 0 | 1.6 / 1.5 px | 4.8 / 2.7 px |
| Side-view kiss | 11.2.206 | 136 / 104 | 0 | 7 | 27 | 7.9 / 3.7 px | 16.3 / 7.7 px |
| | 11.2.209 | 137 / 103 | 0 | 7 | 27 | 7.9 / 3.7 px | 16.3 / 7.7 px |
| Crossing | 11.2.206 | 192 / 4 | 0 | 0 | 12 | 23.9 / 0.7 px | 104.9 / 1.6 px |
| | 11.2.209 | 192 / 4 | 0 | 0 | 12 | 23.9 / 0.7 px | 104.9 / 1.6 px |
| One face hides behind the other | 11.2.206 | 164 / 3 | 0 | 0 | 3 | 39.0 / 7.2 px | 66.3 / 10.6 px |
| | 11.2.209 | 164 / 3 | 0 | 0 | 3 | 39.0 / 7.2 px | 66.3 / 10.6 px |
| Smooch | 11.2.206 | 168 / 2 | 1 (frame 34) | 0 | 45 | 17.0 / 3.9 px | 30.4 / 6.5 px |
| | 11.2.209 | 168 / 2 | 1 (frame 34) | 0 | 45 | 17.0 / 3.9 px | 30.4 / 6.5 px |

The same as 11.2.206 on every clip (one frame better on the kiss). Flicker is identical too.

**Detect pictures** (harsh-step clip, 120 frames):

| Detect pictures | | Person A | Person B |
|---|---|---|---|
| A right, B a picture of someone else | 11.2.206 / 207 | 120 of 120 correct | never swapped |
| | **11.2.209** | 120 of 120 correct | found by elimination at frame 52 (45 sightings with A identified); **68 of 68 frames correct, 0 mix-ups**; placed 0.1 px from where the right picture puts it |
| A and B right | 11.2.206 | 120 / 120 | 120 / 120 |
| | **11.2.209** | identical | identical (the fallback never fired) |

A first 11.2.209 build counted the 45 sightings strictly in a row. On this clip person A makes a fast step and is not
identified for a few frames; the count restarted and never reached 45. The count now tolerates up to a second without
evidence. The two Detect-picture runs above, the harsh-step, hide and smooch runs and every unit suite ran on the final
files (checksums compared). The side-view kiss and crossing runs ran on the build just before that last change; it touches
only the Detect-picture path, which those two runs do not use.

### 3.3 Earlier checks on this pack

On the final files: `unit209` 13, `unit176` 12, `unit176_pipe` 14, `unit203` 18, `unit204` 7, `unit205` 30, all pass.
`unit206` passes on the eye and cup clips (the occlusion results are unchanged).

### 3.4 The 11.2.206 occlusion mask under real-footage conditions

Because §1.3 showed a rule that passed on clean synthetic data and failed with noise, the 11.2.206 mask was tried the same
way (`nofalse206.py` with `DEGRADE=1`): six faces with nothing in front, each drifting and shaking, as they are and with
camera noise, blur, darker, brighter and JPEG quality 25. In all 36 runs the mask cuts **0.00 %** more of any face than
11.2.205. It is kept.

## 4. What the log says now

- `IDENTITY: N face(s) given to a person by elimination because their Detect picture matched no face, with the other person
  identified in the same frame`
- `IDENTITY: N time(s) a person not found yet was NOT given a face that could have been someone already found` (change 1 at
  work: those frames stay original instead of risking the wrong replacement).
- A warning when faces were found but none was used on 10 % or more of the looks at the video: `IDENTITY: on N of M looks ...
  faces were found but none was close enough ... to be sure who it is; those frames stay original`.

## 5. Limits

1. **Missed swaps that are not from §1.1 remain.** A person in a view that reads unlike their Detect picture and first sighting
   (strong side views, a new scene) is refused by the identity check, and that frame stays original. This rule is 11.2.92's and
   was not changed by the last versions. 11.2.209 chooses a missed frame over a wrong face. In the 11.2.206 job 6332D118 (46 %
   painted, no fallback) this was the cause.
2. **Change 1 can delay a real person.** A man alone on screen in a side view, after the woman was found, now waits until
   she is identified in the same frame. Before, he was taken at once, on the same guess that put him on the woman.
3. **The synthetic smooch clip has its one wrong frame again** (frame 34; it had it from 11.2.176 to 11.2.206). The 11.2.207
   rule that removed it is the one §1.3 shows to be unsafe on real detections.
4. **The log cannot prove which fault you saw in which video.** In job DFA984A1 the fallback fired without real evidence
   (§1.1); which face it took cannot be read from the log. For the other job (CF76E96E) the log shows nothing wrong: person 1
   is painted on every frame, and person 2 from frame 223 on, apart from two stretches.
5. Synthetic and stand-in tests only; no footage was looked at.

## 6. Install

Upload the pack's files to the Space. Changed against 11.2.208: `swap_engine.py`, `core_pipeline.py`, `config.py`. No change to
`packages.txt` or the app. The log at start reads `Phoenix v11.2.209`.

For parts where a person looks very different from the Detect frame: capture Detect from that part, or run it without Detect
pictures.
