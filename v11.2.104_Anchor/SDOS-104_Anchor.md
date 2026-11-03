# SDOS-104 — Anchor: "double lips" with an object in the mouth fixed (the rule never ran); a mark that is not found says why, and the search has a time limit

**Pack:** v11.2.104 "Anchor". Engine `aequus-1.2.104-anchor`. **App: no change** (1.2.29 stays).
Contains everything of v11.2.101-103 (limits, planner, skin paste, sticking paint, FACE-EVENTS).

**Reported (with a Space log, text only):**
1. "Around 46-48 s, when an object is in the mouth, the lips position was not proper, it looked like double lips."
2. "The problem occurred in this part, from 23 s to 28 s" (the kiss).
3. The Space log of the job (v11.2.100): patch removal found nothing.

All tests use synthetic clips (public photos, procedural skin); no user footage. The log text was read; the footage was not.

---

## 1. What the log says

- **The Space ran v11.2.100** (`Swamitech Phoenix v11.2.100`). Nothing of v11.2.101-103 was in it: engine cap of 4 marks (6 positions were sent, 4 taken, 2 dropped without a word), the follow-everything planner, the tracker, the finder.
- **Patch removal did nothing**: `patch 1..4: neither the painted frame nor the painted area was found in this video - skipped`, after **946 s** of planning. The anchor search (section 3) is the first thing to look at, before any tracking.
- **Faces:** painted on 1473 of 1589 frames (93 %). `TIMELINE #################..bbbbbbbb` (2 s per character): both people until 34 s, **nobody 34-38 s** (that is the `REVERTS=115`, 3.8 s), then only person 2 (`b`). Person 1 is not painted after 34 s (`recorded` 998 of 1589).
- **The kiss (23-28 s):** `OCCLUSION-PRO ... two faces kept apart on 318 face-frames` is 159 frames, 5.3 s, which matches 23-28 s; `contact_hold=605/119` says the contact hold was working (605 held face-frames).
- **`EATING: ... on 0 face-frames`** in a job with an object in the mouth. This is the finding that matters for the double lips (section 2).
- Not in the log: *when* inside the 23-28 s a face was off or faint. The 2 s timeline is too coarse for a few frames. `FACE-EVENTS` (v11.2.103) is the line that gives seconds; it is in this build.

## 2. Double lips with an object in the mouth: the rule never ran

**Cause.** Since v11.2.92 there is a rule for exactly this: when an object at the mouth reaches outside the lips (food, a spoon, a cup), the footage's own lips are shown around it, so there is one set of lips instead of the swapped lips next to the real ones (`eat_mouth`, SDOS-092 section 4). It only runs on a face that is **not** flagged `predicted`. At render time nearly every output frame is composited from the stored swap by `_reuse_one`, and that function built the face with the default flag `predicted = True` — for every frame, including those on a real detection. So the rule **never ran in the real render path**: `EATING ... 0 face-frames` in your log, and 0 on my own bite clips (the occlusion net cut the object on 67 of 100 face-frames, and the rule fired on none).

**Fix.** `_reuse_one` now sets the flag from the geometry record: a frame on a real detection, or between two real detections, is not predicted; only a held or extrapolated one is.

**Measured** (bite clips of a strawberry and a caramel at the lips, 100 frames; "two lips" = share of the lip area where both the footage's lip and the swapped lip are visible; "ghost" = a see-through mix of the two):

| Clip | EATING face-frames | two lips (mean / worst) | ghost |
|---|---|---|---|
| strawberry, v11.2.103 | 0 | **24 %** / 28 % | 27.7 % |
| strawberry, **v11.2.104** | **72** (weight 0.72) | **3 %** / 6 % | 27.1 % |
| caramel, v11.2.103 | 0 | **30 %** / 32 % | 36.0 % |
| caramel, **v11.2.104** | **72** (weight 0.82) | **0 %** / 3 % | 9.9 % |
| no food (control) | 0 | - | - |

The control clip stays at 0: nothing triggers without an object.

**Limits.** The rule needs the occlusion net to cut something *outside* the lips (at least 0.2 % of the face crop); an object inside the mouth line (an open mouth, a straw held between the lips) does not trigger it, by design. On held or extrapolated frames (no real detection near) it is still off. During eating the lips are the footage's, not the replacement's. If double lips still show at some seconds, the `EATING` count and the `FACE-EVENTS` lines will tell.

## 3. A mark that is not found: why, and a time limit

The 946 s and the four "skipped" marks are in the anchor search, before any tracking. When the painted frame is not in the video the engine searches the frames for the painted *area* (ORB and template matching, every third frame) — once per mark, one after the other. That cost about 1.8 s per sampled frame and mark, with no limit and no word on why it failed.

**Now:**
- One search for the painted area stops after `find_budget_s` = 90 s, and all searches of a job together after `find_total_s` = 240 s (the worst case was 946 s). When the time is used up the log says so.
- A skipped mark now gives its numbers: the app frame's and the video's size, how many size matches, **how close the best whole frame came** (it needs 0.85 on the 160 px picture and 0.60 after alignment), how many frames the area search tried and how long it took, and its best score (it needs 0.8 and 0.7). Example: `... skipped (app frame 1280x720 px, video 1280x720 px, 2 size match(es); best whole-frame match 0.19 at frame 69, after alignment 0.37; area search: 48 frames tried in 2 s, best 0.11)`.

**What I do not know.** Why *this* job did not find its marks. The app sends every painted position to every part of a split job; a part that does not contain the painted frame falls back to the area search, and a mark that is out of the picture, covered or too changed is not found. If this was a single, whole video, the numbers above are what will say whether the frame came from a different video, a different shape, or just did not match. Please send that line (text only).

## 4. The kiss (23-28 s)

Not fixed, not reproduced: 5 synthetic kisses and 3 turn clips paint the face on every frame with `REVERTS=0` (SDOS-103). The `FACE-EVENTS` lines (v11.2.103, in this build) list, in seconds, when a person is unpainted while present, faint, or painted on a predicted geometry only. With them I can tell, for 23-28 s, which of those it was.

## 5. Verification (from the zip's own files)

- `unit104.py` (new): the normal case still anchors; another video's frame is skipped inside the budget with numbers in the log; a frame of another shape is reported as "0 size match(es)"; with the total time used up nothing is searched and the log says so; the render path flags a face on a real detection as not predicted and a held one as predicted.
- Earlier suites on the pack: `unit103` 12/12, `unit102` 28/28, `unit101` 21/21, `unit98` 11/11, `unit97` 18/18, `unit96` 18/18.
- Bite clips before and after (table above).
- Not clean: `verification/fastrestore/unit92.py` stops with `ValueError: too many values to unpack` after its first 16 checks (the side-view and eating checks, which all pass). It stops the same way on v11.2.101 and v11.2.95, so it is not from this build; I did not investigate it.

__JOBS__

## 6. Install

- **Space:** upload the 8 files (or "Push to all Spaces", which skips busy Spaces). Log: `Phoenix v11.2.104`, `swap_engine aequus-1.2.104-anchor`. **Update every Space that will run a job**: the log you sent came from v11.2.100.
- **App:** nothing to install; 1.2.29 works with this Space.
- Send, from a job where something still looks wrong (text only): the `FACE-EVENTS` lines, the `Done in ...REVERTS=...` line, the `EATING:` count, and the `PATCH-REMOVAL` lines.
