# SDOS-105 — PatchMode: a hang on fast motion fixed; the earliest patcher as a switch; a log of when and why a mark showed

**Pack:** v11.2.105 "PatchMode". Engine `aequus-1.2.105-patchmode`. **App: no change** (1.2.29 stays).
Contains everything of v11.2.101-104.

**Reported:**
1. "The patch is not properly done, it keeps moving, and then the original mark comes back as the body moves even slightly. Please do the pasting on the basis of the body part identification. The earliest version of the patcher was better."
2. "If the mark is one, shall I apply patch removal multiple times as the body position changes, or only one time? E.g. a big tattoo below the navel."

All tests use synthetic clips (procedural skin, public photos); no user footage.

---

## 1. What I measured

**The paint sticking to the body** (`stick_test.py`, four settings, five clips: gentle, hard turn/zoom/perspective, jitter, fast 72 px/frame, birthmark). Wobble = how much the centre of the painted area shakes beyond the body's own motion; area flicker = how much its size jumps from frame to frame:

| Setting | wobble | area flicker |
|---|---|---|
| current (finds the mark on every frame) | 0.03-0.13 px | 0.9-5.9 % |
| pose only (the paint carried by the body's tracked position, like the earliest versions) | 0.03-0.13 px | 0.9-8.0 % |

No setting shakes on these clips. **A new measure, the ring just outside the mark's outline** (where a blurred or soft-edged mark would show when the body moves): 0.0-0.2 % of the ring still dark on the mole and birthmark clips in every setting, including the earliest version's edge bands.

**The earliest patcher's behaviour against the current one**, now selectable (section 3). Mark left / frames where it showed (of 150):

| Clip | **classic** (earliest) | **auto** (current) |
|---|---|---|
| mole, gentle motion | 3.8 % / 4 | 1.8 % / 1 |
| tattoo, gentle motion | **0.2 % / 0** | 1.5 % / 0 |
| photo mole | **-1.8 % / 1** | 2.8 % / 1 |
| mole / tattoo shifting against the skin | 0.1 % / 0, 0.3 % / 0 | (0.5 % / 0, 0.1 % / 0) |
| mole, turning, zooming, bending, shadow | 25.9 % / **36** | 0.3 % / 0 |
| birthmark, same | 50.7 % / **74** | 1.0 % / 0 |
| tattoo, same | 27.1 % / **26** | -0.7 % / 0 |
| photo mole, same | 23.2 % / **35** | -12.6 % / 0 |
| mole moving 45 / 72 / 108 px per frame | 60-71 % / **90-108** | -0.9-1.8 % / 0-2 |
| tattoo moving 72 px per frame | 46.8 % / **62** | 3.7 % / 1 |

On gentle footage the two are equal (classic slightly better on the tattoo). As soon as the body turns, bends or moves faster, classic loses the mark and it shows — which is the effect you describe. So **`auto` stays the default**; `classic` is there for you to compare on your own footage.

**What I could not reproduce:** the paint moving and the mark coming back with *slight* movement. Something in your footage is not in my clips. The most likely candidate for a big tattoo on the belly: the engine moves the painted area as a flat patch (move, turn, scale, shear); **belly skin also bends, folds and stretches** (breathing, sitting, twisting), and that is not modelled — the further the body gets from the painted frame, the more the edge of the paint slides. More painted positions of the same mark fix that (section 5). The `PATCH-EVENTS` line (section 4) will say where and why it showed.

## 2. A job could hang on fast motion (fixed)

Found while testing: when the tracker loses the area for a few frames, the next position is predicted from the last motion, compounded frame by frame. A wrong motion compounded 12 times can shrink the predicted patch to nothing or blow it up to infinity, and **`cv2.warpAffine` with an infinite matrix never returns**: a test job ran for over an hour on one frame. In this build every predicted or candidate pose is checked first (finite, scale between 0.25 and 4); a broken one is replaced by the last good pose, or the search uses the whole frame. In place since v11.2.103's immediate re-detection; seen only with the classic settings on my clips, but possible in any mode on real footage. After the fix: the same clip plans in 14 s; the auto results are unchanged (table above).

## 3. PATCH_MODE

In **`config.py`**, one line:

```
PATCH_MODE = "auto"        # or "classic"
```

- `auto`: the current engine (finds the mark itself on every frame, opaque skin paste, texture from the skin beside it, follows fast moves).
- `classic`: the earliest versions' behaviour (v11.2.96-98): the plain brightness tracker, the paint carried on the tracked position only, the edge bands that cover a rim of the mark peeking out, the v11.2.98 object gate, no texture copy, short gap bridging. The fast-motion tracking of v11.2.103 and the hang fix stay.
- The job log says which mode ran: `PATCH-REMOVAL mode: auto`.

## 4. PATCH-EVENTS: when and why the mark showed

New job log line, one per mark, text only:

```
PATCH-EVENTS mark 1: removed on 1380 frames; SHOWED on 23 frames: 12.3-12.8 s refused: position doubtful or covered, 15 fr; 31.0-31.3 s not followed, 8 fr
```

Reasons: **not followed** (the area was not found on that frame), **refused** (the position does not fit the mark, or an object covers it), **held position, mark not seen**, **out of the picture / too small**. Only the time between the first and the last frame the mark was removed counts. `never showed` when there is nothing to report. With these seconds I can tell which part fails on your footage.

## 5. One mark, several positions (your question)

Paint the **same mark several times, as positions of that mark** (not as new marks), one for each clearly different body position. Each painted position becomes the reference for its own stretch of the video; on every frame the nearest painted position in time is used.
- Add a position at the first clear frame, then whenever the torso turns noticeably, sits, bends or stretches, the tattoo becomes much bigger or smaller in the picture, after a cut, and when it comes back after being covered. For a belly tattoo in a moving scene, 4-8 positions.
- Sharp, well-lit frames; paint a few pixels past the edge; if the waistband or a hand covers part of it, paint only the visible part.
- Split jobs: paint at least one position inside each part's time range (a part without one has to search for the tattoo: slower, and it can fail).
- Limits: 10 positions per mark, 20 in all, 10 marks. Extra positions cost little (each is followed only through its own stretch).

## 6. Verification (from the zip's own files)

- `unit105.py` 14/14: the two modes (classic sets the earliest settings, auto restores every one, switching back and forth loses nothing, an unknown name means auto); the event codes (painted, painted around an object, refused, held, not followed) and the log lines; the pose check (inf, nan, collapsed or exploded scale refused), a template search around a broken pose returns at once, a step with a runaway velocity does not hang.
- Earlier suites on the pack: `unit104` 12/12, `unit103` 12/12, `unit102` 28/28, `unit101` 21/21, `unit98` 11/11, `unit97` 18/18, `unit96` 18/18.
- `stick_test.py` (wobble, area flicker) and `video_test.py` (ring residue) extended.

**Full jobs from the extracted zip** (the real render path, 720p, Balanced; scored against the clean clip):

| Job | Result |
|---|---|
| hard mole, `PATCH_MODE = "auto"` | removed on 150 of 150 frames, 3.5 % left after the final encode, 0 missed; `PATCH-EVENTS mark 1: removed on 150 frames; never showed between its first and last frame` |
| the same clip, `PATCH_MODE = "classic"` (set in `config.py`) | `PATCH-REMOVAL mode: classic`; removed on 92 frames, 42 % left, 59 missed; `PATCH-EVENTS mark 1: removed on 92 frames; SHOWED on 39 frames: 1.7-3.0 s not followed, 39 fr; not removed after 4.4 s (19 fr) - the mark may be out of the picture there, or it was not found` (the line names 58 of the 59 frames the scoring finds) |
| mole moving 72 px per frame, auto | removed on 150 of 150, 4.5 % left, 1 missed; never showed |
| 10 marks on one frame, auto | removed on 1496 of 1500 frames, 3.2 % left, 0 missed; mark 5: `SHOWED on 4 frames: 3.4-3.5 s refused` |
| kiss, the detector made blind for 40 frames of contact | both faces painted 120/120; `REVERTS=0 faint=0`; `FACE-EVENTS none` |

## 7. Limits

- Synthetic tests only; the slight-movement problem you see is not reproduced (section 1).
- A large patch of skin that bends or folds is followed as a flat patch; more positions are the remedy.
- Classic mode loses the mark whenever the body turns, bends or moves fast (table in section 1).

## 8. Install

- **Space:** upload the 8 files to every Space (or "Push to all Spaces", which skips busy Spaces). Log: `Phoenix v11.2.105`, `swap_engine aequus-1.2.105-patchmode`, `PATCH-REMOVAL mode: auto`.
- To try the earliest behaviour: change the `PATCH_MODE` line in `config.py` to `"classic"` on one Space and run the same video there.
- **App:** nothing to install.
- Send the `PATCH-EVENTS` and `PATCH-REMOVAL` lines (text only) of a job where the mark still shows, with the seconds where you see it.
