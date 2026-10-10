# Phoenix Mobile 1.2.9 — Update Spaces from the phone · SG16–SG20 · frame picker under the video

- **Update Space code** (new card under Connection): pick a Phoenix Space package — the shipped
  `.zip`, or its files — tick the Spaces, tap Update. The app commits the Space files
  (`app.py`, `config.py`, `core_pipeline.py`, `phoenix_api_adapter.py`, `swap_engine.py`,
  `packages.txt`, `requirements.txt`, `README.md`) to every ticked Space through the Hugging Face
  commit API — the same protocol, byte-for-byte, as the official `huggingface_hub` client.
  - A partial package is refused (a Space must never run two versions at once).
  - SDOS notes and `tools/` in the zip are never pushed.
  - Files a Space already has are skipped; a Space that is already up to date is not restarted.
  - `README.md` (the Space settings header) keeps each Space's own title; a README without a valid
    Space header is never pushed.
  - Spaces with a job from this phone still running are skipped (the restart would kill it).
  - Needs a token with WRITE access: the main token, or a separate "update token" saved encrypted
    in the Android Keystore. A read-only token is detected and refused before anything is sent.
  - "Check Space status" shows each Space's stage (BUILDING, RUNNING, RUNTIME_ERROR …).
- **Spaces SG16–SG20** added (`SwamitechGradio16…20`, `https://swamivicky-swamitechgradioN.hf.space`).
  21 Spaces in total. The selected Space is now remembered by its address, so inserting Spaces can
  never silently re-route the next job (an install that had SG_UAT2 selected keeps SG_UAT2).
- **Frame picker** (slider + "Show frame & detect faces") moved directly under the video preview,
  which follows the slider, so the frame being picked is always in view. Card 02 now shows the
  detected frame and faces.

# v1.1.3 — World-class UI refresh

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
