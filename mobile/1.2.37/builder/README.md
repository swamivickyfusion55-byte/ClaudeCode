---
title: APKBuilder
emoji: 🤖
colorFrom: blue
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Phoenix APK Builder 1.2.37

Upload every file in this pack, including `source.zip`, replacing the old ones.
Then open the Space and press Build APK.

**Builder fix (1.2.24):** a build that was stopped half way used to leave a cut-off APK that made the next build
fail at the very end ("Unexpected end of ZLIB input stream"). Packaging now always starts fresh, and a failed
build is retried once from a clean tree.

**1.2.37**: the useful parts of the 1.2.36 pack (the Secure Folder network fix, upload retries with a single-request fallback, clearer failure messages, a one-tap **Automatic belly clean** switch under the target video) on top of 1.2.32's belly preview, length and breadth; and a calmer, more finished look. Needs Space v11.2.117 or newer for the belly clean. Installs over 1.2.32 or 1.2.36.

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
