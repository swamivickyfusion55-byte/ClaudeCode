# v1.2.23 - uploads keep going in the background; saved faces with names

- **Uploads continue when the app is minimised.** Two causes, both fixed:
  1. The video went up in ONE request through Gradio's upload, so any pause (phone in the background, a
     network change, a stall) threw the whole transfer away. The app now uses the Space's resumable upload
     (Space v11.2.80 and later): 4-MB pieces, and after any interruption it continues from the last byte
     the Space holds. Tested against the Space's own upload code through a connection that was cut 6 times
     and stalled once (40 MB arrived byte-identical), and after the app was killed half way (a restart went
     on from 50 %). On an older Space the app falls back to the old upload by itself.
  2. Nothing kept the app running once it left the screen. While an upload runs, Phoenix now shows an
     "uploading" notification (a foreground service, started at the moment you tap Start, with a wake lock
     so it also continues with the screen off), and removes it when the upload ends. It is started as a
     plain service that promotes itself, not with startForegroundService() from the background, which is
     what made 1.2.19/1.2.20 vanish.
- **Saved faces.** Under every replacement-face slot: "Saved" opens a drop-down of your saved faces, each
  with its name and a small thumbnail; picking one puts it in that slot. "Save" keeps the picture in a slot
  under a name. Rename or delete from the same drop-down. The faces are stored in the app's private folder
  (removed with the app) and sent to a Space only as the replacement face of a job, as before.
- Same app id, same signing key, versionCode 143: installs over 1.2.22 and keeps token, Spaces and History.

# v1.2.22 - crash guard

- **Restarts itself.** If the app crashes while it is on screen it starts again at once and reconnects to the
  running jobs (at most twice in two minutes), instead of vanishing.
- **Safe mode.** Three crashes in two minutes: the app opens, shows what happened and does not touch the
  running jobs until "Resume jobs" is tapped. Breaks any crash loop; the jobs keep running on the Spaces.
- **Background work cannot end the app.** Every background task has an error handler; network callbacks
  are wrapped. An error is recorded and shown as a message and the app carries on.
- **Start-up order fixed.** The part of the ViewModel that restarts job watchers ran before several of the
  fields it uses had been created. It now runs last.
- **History is never lost to one bad entry.** One damaged history entry is skipped, an unreadable history
  is kept aside (`items_unreadable_backup`) instead of being overwritten.
- **More memory for video work** (`largeHeap`), so a large frame does not run the app out of memory.
- **The note after a crash now actually shows** (in 1.2.21 it was set before the field existed and was lost).
  A log of the last 6 errors is kept in the app's private folder.
- Same app id, same signing key, versionCode 142: installs over 1.2.19 / 1.2.20 / 1.2.21 and keeps the token,
  Spaces and History.

# v1.2.21 - the app no longer vanishes when it goes to the background

- 1.2.19 and 1.2.20 started a keep-alive foreground service whenever the app went to the background with a
  job running. On the phone this made the app disappear with no message, again and again (the same loop an
  earlier build had to remove). The service is no longer started. The render runs on the Space, so closing
  the app does not stop a job; opening the app reconnects to every tracked job.
- If the app closes unexpectedly, the next launch shows a yellow note with the reason Android recorded
  (crash, low memory, killed by the system) and, for a crash, the first lines of the error.
- Same signing key as 1.2.19 / 1.2.20, so it installs over them and keeps History, token and Spaces.

# v1.2.20 - installs on the phone, and the builder checks the APK before it hands it over

- "Something went wrong" when tapping the APK on the phone: the 1.2.19 pack signs every build with one fixed
  key (`phoenix-debug.keystore`). Android refuses to update an app that is already installed if the new APK
  is signed with a different key, so a Phoenix installed from an earlier builder run has to be uninstalled
  once. Every later build from this pack is signed with the same key and updates in place.
- build.sh now checks the finished APK before offering it: `apksigner verify`, then package id, versionCode,
  SDK levels, ABIs, test-only / debuggable flags and the signing-key SHA-256 are printed and shown in the
  builder. A broken or test-only APK stops the build instead of failing on the phone.
- Brings back the v1.2.10 fix for "Push to all Spaces" (HTTP 400 ... value.summary) that the 1.2.19 line did
  not carry. Nothing else in the app changed.

# v1.2.9 — Spaces 2–20, pack push, frame under the video

- Added SwamitechGradio16 through SwamitechGradio20. Gradio 2–15 and the two UAT spaces stay.
- Connection settings can push the unzipped Phoenix files to every configured Space in one step.
- The frame slider sits directly under the video, so the picture you are picking stays in view.
- Landscape uses a shorter preview, tighter cards, and a side-by-side video / settings layout.

## v1.1.3 — World-class UI refresh

- Reworked the Android visual system into a polished Phoenix studio-style interface.
- Added stronger hierarchy, premium dark header, compact navigation and refined cards.
- Kept Processing Space selection prominent at the top of the workflow.
- Improved multi-Space status visibility without allowing active jobs to make the header grow indefinitely.
- Refined the primary action bar and state messaging.
- Preserved existing functionality and per-Space parallel job routing.

## Phoenix Mobile 1.1.2 — independent Space startup + top Space selector

- Replaced the global 2-permit upload semaphore with one upload gate per configured HF Space. Jobs targeting different Spaces no longer wait behind each other during upload/startup.
- Kept uploads serialized per individual Space to avoid saturating a single remote Space/network path.
- Kept the remote job monitor fully independent per Space.
- Moved the Processing Space selector to an always-visible bar directly below the tabs, so Space selection is available before starting a job without scrolling to Connection settings.
- Kept Connection settings lower in the page for configuration/token management.

## Phoenix Mobile 1.1.0 — SDOS speed optimization

- Removed the forced Gradle `clean` from the APK build path.
- Enabled Gradle build/configuration cache and bounded workers.
- Fixed the Kotlin `imageQuality` setter JVM signature clash.
- Added 30-minute caching for HF/Gradio discovery, endpoint lists and signatures.
- Preserved existing network retry and DNS fallback behaviour.

## Phoenix HF v11.1.0 — SDOS speed optimization

- Skipped source frames now use `VideoCapture.grab()` and are not fully decoded.
- Retained frames are resized once directly to the requested output resolution.
- CPU inference/OpenCV thread counts are explicitly bounded to avoid oversubscription.
- Swap worker count is hardware-aware and conservative.
- Job state persistence is throttled to a 3-second checkpoint interval while terminal states remain immediate.
- Frame/result queues are capped at a smaller memory-aware depth.
- Face detection cadence adapts to motion while preserving every-frame detection for multi-face safety.
- High-resolution detector probes have a cooldown to prevent repeated expensive recovery passes.
- FFmpeg thread count is capped at six; quality CRF/preset mappings are unchanged.

# Changelog

## Dual-space cloud jobs
- Added two configurable Hugging Face Phoenix Space slots with a processing-space dropdown.
- Added up to two concurrent cloud job monitors.
- Each job captures its selected Space URL/name at submission time.
- History now tracks running, queued, processing, downloading, completed and failed jobs independently, including Space, remote job ID, progress and ETA.
- Added per-job Stop monitoring control.
- Persisted the two Space configurations and selected Space locally.
- Added automatic status-poll resume for tracked remote jobs after app restart.
- Kept the existing SwamitechUAT Space as Space 1 by default.


## Phoenix Mobile 1.0.0 — UAT/API/UI enhancement build

- Fixed Android Phoenix Space URL handling by trimming whitespace and trailing slashes.
- Added explicit `UnknownHostException` / DNS diagnostics for SwamitechUAT.
- Preserved the production Phoenix endpoint and Gradio queue protocol.
- Added Image workspace with target-image upload, face detection, replacement-face slots and image output download.
- Added Video workspace with file-only video input/output controls; no embedded video thumbnail or player.
- Added frame-position inspection with annotated frame preview and face capture into replacement slots.
- Added HF-aligned presets: Speed, Balanced, Mobile HQ, Quality and HQ.
- Added duration, FPS, resolution, quality, swap cadence, re-detection cadence, enhancer, face mode, enhancement scope, compute, trim and encrypted-ZIP password controls.
- Added local History tab with refresh, delete and clear-history actions.
- Added local history persistence so completed outputs remain visible after reopening the app.
- Added local stop/monitoring control; remote jobs already queued on the server may continue processing.
- Kept the M3 local-engine guard: the app never claims successful local video processing when the verified local engine is unavailable.


## Speed V2 — 2026-08-25
- Streams untrimmed gallery content directly from ContentResolver to OkHttp; avoids a full cache copy before upload.
- Keeps concurrent face uploads capped at three per Space.
- Retains local trimming for selected ranges, where it materially reduces network transfer.
