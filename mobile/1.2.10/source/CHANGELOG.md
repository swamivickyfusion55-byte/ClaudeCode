# v1.2.10 — Push to all Spaces works against the real Hugging Face commit API

- Fixed "Updated 0, failed 21 … HTTP 400 … Invalid input: expected string, received undefined → at
  value.summary". The commit request went out with `Content-Type: application/x-ndjson; charset=utf-8`
  (OkHttp adds the charset when a body is built from a String). Hugging Face's own clients send exactly
  `application/x-ndjson`, and the Hub chooses between its NDJSON and plain-JSON body formats by that
  header, so the NDJSON body was not read as NDJSON. The body is now sent as bytes with the header
  exactly as the official clients send it, and the lines are written the way `JSON.stringify` writes them.
- If the Hub still answers 400 mentioning `summary`, the commit is retried once in the Hub's plain-JSON
  form, and both answers are shown.
- The failure text shows the FIRST error in full (it showed the last one cut at 120 characters).

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
