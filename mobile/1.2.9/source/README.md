# Swamitech Phoenix Mobile 2

Phoenix Mobile 2 v1.1.0 is the Android client for the Swamitech Phoenix Hugging Face Space.

## UAT configuration

- Slots 1–19: SwamitechGradio2 … SwamitechGradio20 (`https://swamivicky-swamitechgradioN.hf.space`)
- Slot 20: `https://swamivicky-sg-uat2.hf.space` (SG_UAT2)
- Slot 21: `https://swamivicky-swamitechuat2.hf.space` (SwamitechUAT2)
- Video endpoint: `phoenix_submit_video`
- Protocol: Gradio queue API
- Token storage: Android Keystore-backed secure storage

## Mobile UI

The mobile UI mirrors the functional workflow of the Phoenix Hugging Face app while using a mobile-first card layout:

- Image tab: target image → detect faces → replacement faces → quality → output.
- Video tab: target video file → frame-position inspection → replacement faces → presets and advanced controls → processing.
- History tab: persistent local completed-job history with refresh, delete and clear actions.

Video input and video output are intentionally file-only. The Android client does not embed a video player or automatically render a video thumbnail. The explicit **Show frame** action renders the selected video frame because that frame is required for face inspection/capture.

Image input/output are also file-based without automatic result thumbnails. Detected face crops and the selected frame may be shown where needed for one-to-one face assignment.

## Update Space code from the phone (1.2.9)

Connection → **Update Space code**: choose the Phoenix Space package (`Phoenix_v11.2.x_….zip`, or its
eight files), tick the Spaces, tap **Update**. Each ticked Space receives one commit with the files that
changed and then rebuilds by itself. Needs a Hugging Face token with **write** access (save it as the
separate update token if your main token is read-only). Spaces already on that version are not
restarted; Spaces with a job from this phone running are skipped. See `CHANGELOG.md`.

## Connection diagnostics

The client sanitizes the Space URL and reports DNS/`UnknownHostException` failures separately from HTTP/API failures. This avoids misdiagnosing an Android DNS failure as a Phoenix API or route failure.

## Build

Use the included Android builder workflow. The source passes the bundled static audit (`38/38`).

### v1.1.0 speed optimizations
- Incremental Gradle build: no forced `clean`, build/configuration cache enabled.
- Bounded Gradle workers and 2 GB JVM heap for memory-constrained builders.
- Fixed the `imageQuality` JVM setter signature clash.
- Cached Hugging Face/Gradio discovery metadata for 30 minutes.
- Phoenix backend v11.1.0: skipped video frames use codec `grab()` instead of full decode, requested output resolution is preserved, CPU inference threads are bounded, detection cadence adapts to motion, high-resolution detector probes are cooled down, queues are bounded, and job persistence is coalesced.


## Background processing reliability (2026-08-26)
- Added a foreground data-sync service so long-running HF jobs and result downloads do not depend on the Activity remaining open.
- The service is started while remote jobs are active and stopped automatically when all tracked jobs finish.
- Android 14/15 foreground-service declarations are included.
