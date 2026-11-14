---
title: APKBuilder
emoji: 🤖
colorFrom: blue
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Phoenix APK Builder 1.2.62 (Space control: status, Pause, Restart / Resume for each Space)

Upload these six files: `Dockerfile`, `app.py`, `build.sh`, `gradle.properties`, `README.md`, `source.zip`. The page title must
say Phoenix Mobile 1.2.62. versionCode is **182**.

**1.2.62 - each Space's status, and Pause / Restart in the app:**
- **Status everywhere.** The dot on each Space number at the top shows its state on Hugging Face:
  - green: running;
  - blue: one of your jobs is running there;
  - amber: paused;
  - red: error;
  - light blue: starting or building;
  - grey: asleep, or not read yet.

  Under the selected Space's name (Connection card) the state is in words. It is read when the app opens and every 5 minutes.
- **Connection - "Pause / restart Spaces":** one row per Space with its state (and the error text when it has one), **Pause**
  and **Restart** (**Resume** when paused), a colour key and **Check status**.
- **Asked first, every time.** The app asks the Space whether a job is running. If one is, or the Space does not answer, it
  says in red that pausing or restarting stops that job for good. A restart is followed until the Space runs again (or fails).
- **A paused Space gets no jobs:** a split skips it, and a job sent to it says to resume it first.
- **Needs the saved token to have write access to the Spaces** (the same as "Push to all Spaces"). Without it, the app says
  so. It uses Hugging Face's own Space API: `/api/spaces/{owner}/{name}/runtime`, `/pause`, `/restart`.
- Everything else as 1.2.61.

## 1.2.61 (One upload: Detect sends the frame, not the video)

**1.2.61 - the video goes up once:**
- **Detect** ("Show frame & detect faces") and the **skin-tone Preview** send only the picked frame, as a photo. They used to
  upload the video (the selected range) first, so a split sent the video twice: once for Detect, then again in parts.
  Detect now takes seconds. The frame is the one the app shows at "FRAME n %" (n % of the whole video). With a trim set, the
  Space used to take n % of the trimmed range instead, a different frame from the one on screen.
- **The progress bar only moves forward.** Upload 0-35 %, faces 35-40 % ("Video sent - sending faces"), the Space's work
  40-96 %, download 96-100 %. It used to drop back to 4 % after the video upload, and again to the Space's own 8 %, which
  looked like a second upload.
- Works with every Phoenix Space version (the same Space routes). If the phone cannot decode the frame, the video is uploaded as
  before.
- Everything else as 1.2.60 (the copy of the picked video) and 1.2.59 (three uploads at once, stalls resume).

## 1.2.60 (Local copy: Phoenix no longer closed with the storage provider)

**1.2.60 - "Phoenix was closed because a system part it needs stopped (depends on provider
com.android.externalstorage ... in dying proc)":**
- A video picked from the gallery or Files is read through Android's storage provider. Phoenix kept that connection open
  for the whole upload (a single video) or the whole cutting (a split). When Android stopped the provider's process, it
  closed Phoenix with it - that is the message.
- Now Phoenix first copies the video into its own storage (a few seconds; the status line says "Copying the video into
  Phoenix (N MB)"), then cuts and uploads from the copy. The copy is deleted when the parts are cut, or when the upload ends.
- When there is too little free space for the copy (the video's size plus 400 MB), or the copy fails, Phoenix reads the
  picked video as before.
- Everything else as 1.2.59 (three uploads at once, a stalled upload steps aside and resumes).

## 1.2.59 (Upload queue: no more parts stuck at "Uploading video 2 %")

**1.2.59 - split parts no longer sit at "Uploading video 2 %":**
- **Why it happened:** the parts of a split went up one at a time, through a single upload slot, and every part waiting
  for that slot already read "Uploading video 2 %". When one part's Space stopped taking the video (asleep, restarting,
  overloaded), that part kept the slot for up to 15 minutes, then sent the whole file again in one request for up to
  15 more: every other part waited behind it at 2 %. Redo part put the part back in the same queue.
- **Now:** three parts upload at once (each to its own Space), so one Space that does not answer cannot hold up the rest.
- A part whose Space accepts nothing for **3 minutes** steps out of the upload slot, waits 20 seconds and tries again,
  resuming at the last byte the Space has (up to 5 times; then the card says Failed - check this Space, then tap Redo part).
  It no longer falls back to sending the whole file in one request after a stall.
- A part waiting for a slot says so: **WAITING TO UPLOAD**, "Queued - 2 parts ahead (3 upload at a time)".
- After the app is closed or updated, cards whose video never reached the Space show **INTERRUPTED** - "tap Redo part"
  (they used to stay at "Uploading video 2 %" for ever, with nothing sending them).

## 1.2.58 (Azure: new look)

**1.2.58 - new look ("Azure"):** a calm, professional light theme. Dark text on light surfaces everywhere: near-black body text
(16 : 1 contrast on white) and dark-grey secondary text (7.6 : 1) instead of the lighter grey; one azure-blue accent (#0F6CBD)
instead of burgundy and orange. The dark header, the black selected tabs, the dark "Video ready" tiles and the dark Patch
Removal / Belly editor are now light with dark text; white text remains only on the solid blue buttons (5.2 : 1). Nothing else
changed (Redo part, Skin tone and the small download as in 1.2.57).

## 1.2.57 (Redo part)

Upload every file in this pack, including `source.zip`.
The page title must say Phoenix Mobile 1.2.57.
versionCode is **177**.

**1.2.57 - Redo part works again:**
- A part the Space lost (the Space restarted and kept nothing) no longer says "not visible now - still checking" for ever:
  after about 3 minutes the card shows **LOST** and "Tap Redo part".
- **Redo part** did nothing because the part's clip was deleted right after its upload. The clip is now kept until the part
  is done. When it is gone (cards made by 1.2.56 or older), Redo part cuts the part again: from the original video the card
  remembers, or for older cards from the video selected in step 01 with the same trim as the split. Its messages now show on
  the card, not only in the status line.
- Put the same replacement faces in the slots first (Redo part says so on the card if they are missing).

**1.2.56 - "package appears to be invalid" fixed:** the 1.2.55 APK built fine, but the phone fetched it through Gradio's
file link, which broke off again and again (the Space log shows one 200 and then six 206 partial answers); the phone then
had a cut-off file. Now:
- the APK is for 64-bit ARM phones only and its native libraries are stored compressed: a far smaller download
  (it was 92 MB, mostly three unused copies of ONNX Runtime);
- it is offered only through the **Download** button (the direct, resumable `/phoenix-apk` link), never through Gradio's file box;
- the page shows the exact size: install only a file of exactly that size, never one marked "Download failed".

**Skin tone (since 1.2.55, needs Space v11.2.205):** step 04 **Skin tone**: a Darker / Brighter slider per replacement face (that
person's face, neck, arms and legs; everyone else stays as filmed), **Preview at N %** on the Space with Before | After and a
line to drag, and the setting sent with the job. Built on 1.2.54 (free-shape patch tap kept unchanged).

**1.2.54:** Tap draws a free shape on the dark patch, at most 4 square inches. Needs Space v11.2.193.

**Builder fix (1.2.24):** a build that was stopped half way used to leave a cut-off APK that made the next build
fail at the very end ("Unexpected end of ZLIB input stream"). Packaging now always starts fresh, and a failed
build is retried once from a clean tree.

**1.2.46**: the belly screen states the three rules: inside the fence, dark strands only, cluster at least 0.18 of the ear width. After Preview belly the measured fence is shown. Needs Space v11.2.137.

**1.2.42**: after Preview belly, drag the yellow box with one finger. Drag a corner or pinch to stretch it. One frame is enough for the whole video. Needs Space v11.2.122.

**1.2.32**: the **Belly clean** buttons work with Space v11.2.117 (one-go belly clean): **Preview belly** sends the frame to the Space as PNG (lossless) and shows the cleaned frame with the band outlined; **Use for this video** sends the one-go setting with the job. Needs Space v11.2.117 or newer.

**1.2.31**: **Belly clean** in the Patch Removal frame screen: a second mode next to "Paint a mark" with the on / off switch for the video, **Length** and **Breadth** of the band (50-150 %), **Preview belly** on the chosen frame (the cleaned skin, with the band outlined), Before / After and the outline on / off. Needs Space v11.2.111 or newer for the preview and the length / breadth.

**1.2.30**: a **Lower belly: clear skin** switch in the job options: every tattoo or mark on the uncovered skin from below the navel to the waistline is removed and the skin evened (Space v11.2.110 or newer; off by default).

**1.2.29**: a second job can be started while a job is split across Spaces: it uses an idle Space (checked with the Space itself), and a split uses as many idle Spaces as there are. Use with Space v11.2.101.

**1.2.28**: one Patch Removal mark can be painted at up to **10 positions** (was 6), up to 10 marks (was 4) and 20 positions in all (was 12). Use with Space v11.2.100.

**1.2.27**: the Patch Removal editor keeps every button on the screen (it is now drawn inside the app's own window,
inside the system bars, with a margin) and works in portrait and in landscape. Use with Space v11.2.98.

**1.2.26**: Patch Removal can take **several positions of the same mark** (+ Position) for when the person moves or
turns; the editor is landscape-only and no longer closes when the phone turns; controls in a side panel with the
action buttons always visible; zoom buttons and a magnifier under the finger; frame-by-frame stepping. Needs Space v11.2.97.

**1.2.25**: optional **Patch Removal** under the target video. Pick the exact frame, paint over a tattoo, mole or
dark mark with a finger (pinch to zoom), tap Preview to see it erased by the Space, then Freeze. Frozen marks are
removed on every frame of the video by the Space, even as they move. Needs Space v11.2.96.

**1.2.24**: "Push to all Spaces" no longer restarts a Space that is rendering: busy Spaces are updated by
themselves when their job ends (no running job is lost). Face restoration gets "GPEN + Cinematic" (fast face
restore, needs Space v11.2.92) as the default; plain Cinematic stays as the fastest choice; CodeFormer (which never
worked) is gone.

**1.2.23**: video uploads continue when the app is minimised (resumable upload in 4-MB pieces through the
Space's own routes, plus an "uploading" notification that keeps the app running while an upload is in
progress), and replacement faces can be saved with a name and picked from a drop-down with thumbnails.

**Installing:** install the new APK *over* the one on the phone - do not uninstall. Same app id, same signing
key and a higher version number, so Android updates in place and keeps your token, Spaces and History, and the
jobs keep running on the Spaces. The build log shows `Signing key (SHA-256): e2564a9b...`; it must be the same
on every build.

**1.2.42**: the belly mark is an eight-corner box. Detect shows M or F under each face; tap the letter if it is wrong, then send that face to the slot. Needs Space v11.2.130.

**1.2.46**: the paint brush is one step finer. S is half the old S, M is the old S, L is the old M.
