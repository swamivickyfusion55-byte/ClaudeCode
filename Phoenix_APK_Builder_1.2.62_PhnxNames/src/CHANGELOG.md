# v1.2.62 - each Space's status; Pause and Restart / Resume per Space

- **Asked:** "add option of pause and restart space in the frontend app itself, pause and restart button against each space",
  then "add a display of status as well, pause, running, error".
- **HfApi.kt:** `spaceRuntime`, `pauseSpace`, `restartSpace`. These are GET `/api/spaces/{repo}/runtime`, POST `/pause` and
  POST `/restart`, as `huggingface_hub` 0.27.1 calls them (`get_space_runtime`, `pause_space`, `restart_space`; restart
  resumes a paused Space). The token is sent as `Authorization: Bearer`. The reply's `stage`, `hardware.current` and
  `errorMessage` are read. HTTP 401/403 says the token needs write access; 404 says to check the Space's name. The repository
  name comes from the slot as for "Push to all Spaces" (`repoIdOf`).
- **PhoenixViewModel.kt:**
  - `spaceStages` / `spaceErrors` per Space, read when the app opens and every 5 minutes, and by Check status.
  - `askSpaceAction` asks the Space (or this phone) whether a job is running; no answer counts as running.
  - `confirmSpaceAction` pauses or restarts, then follows a restart until RUNNING or an error (8 minutes at most).
  - Paused Spaces are left out of `idleSpaces()` (splits), and `startRemote` refuses to send a job to one.
- **MainActivity.kt:**
  - the Space chips' dot is the Space's status;
  - the Connection card shows the selected Space's status in words;
  - a "Pause / restart Spaces" panel: rows, a colour key and Check status;
  - a confirmation dialog, red when a job would be lost.
- **Checked here:**
  - All sources compile with no new error except calls of Android types the sandbox has no definitions for.
  - The real `HfApi` calls against a local stand-in for the Hub: the right method, path and token on each call; stage read;
    paused, then resumed; 403, 404, no token and an unreachable Hub each give a plain message, no crash.
  - The real status words and colours for all 14 stage cases.
- **Not checked here:** a Gradle / Android build, the app on a phone, and the real Hugging Face API (huggingface.co is not
  reachable from this sandbox; the calls follow the official library's source).

# v1.2.61 - the video goes up once; the progress bar only moves forward

- **Asked:** "Why does upload appear to happen 2 times"; then build both changes.
- **Detect / skin-tone Preview:** `pickedFrameJpeg` takes the frame shown at "FRAME n %" (n % of the whole video, the same as the
  thumbnail and the Patch editor; `PatchFrames.frameAt`, exact frame, at most 1920 px) and sends it as a JPEG. It goes to the same
  routes as before (`phoenix_detect_video_frame`, `phoenix_skin_preview`, both take any file) at position 0: a photo is a
  one-frame video for OpenCV on the Space. The Space's "Frame 0/1 - " is cut from the message. If the frame cannot be decoded,
  the video is uploaded as before.
- **Progress:** `PROGRESS_FACES` .35, `PROGRESS_SUBMIT` .40, `PROGRESS_DOWNLOAD` .96. The Space's 0-100 % maps to 40-96 % and
  never goes back (`maxOf`).
- **Checked here:**
  - All sources compile with no new error except calls of Android types the sandbox has no definitions for.
  - The real `pickedFrameJpeg` and progress code against stand-ins: 0 / 50 / 100 % pick 0 / 5000 / 9950 ms of a 10 s video;
    an undecodable frame or no duration gives null (the old path); the message is cleaned; the bar never goes back.
  - The Space's own code (`api_detect_video_frame`, `api_skin_preview` of v11.2.208) given the video at 40 % and given only
    that frame as a JPEG at 0 %: the same 2 faces (crops 1-2 grey levels apart, the JPEG step), the same people and amounts
    in the skin preview (0.9 grey levels apart).
- **Not checked here:** a Gradle / Android build and the app on a phone.

# v1.2.60 - the picked video is copied into Phoenix first

- **Reported:** "Phoenix had a problem last time - Phoenix was closed because a system part it needs stopped (depends on provider
  com.android.externalstorage/.ExternalStorageProvider in dying proc com.android.externalstorage (adj -10000))".
- **Cause:** the picked video (a content:// document of the external-storage provider) was read through the provider for the whole
  upload of a single video (`sendVideo` holds a ParcelFileDescriptor) and for the whole cutting of a split (`VideoTrimmer.trim` per
  part). An app holding an open file from a provider is killed by Android when the provider's process dies.
- **Changed (PhoenixViewModel.kt):** `localVideoCopy` copies the video once into the app's cache (when the free space is at least the
  video's size + 400 MB; the size Android reports must match the copy). A split cuts its parts from the copy and deletes it when
  the parts are cut; a single video uploads from the copy, which is deleted when the upload ends, also when it fails (a retry makes
  a new one). Otherwise the picked video is read as before.
- **Checked here:** all sources compile with no new error except calls of Android types the sandbox has no definitions for; the real
  `localVideoCopy` (its text lifted from the source) against stand-ins for those types: a byte-for-byte copy; a file:// part is
  not copied; a copy that breaks off, a short copy and too little space each give null and leave no file behind.
- **Not checked here:** a Gradle / Android build and the app on a phone.

# v1.2.59 - split parts no longer stuck at "Uploading video 2 %"

- **Reported:** many parts of a split sat at "UPLOADING VIDEO 2 %" for 30 minutes and more; Redo part did not help.
- **Cause:** one upload slot for all parts (`videoUploadSlots = Semaphore(1)`); a part whose Space stopped answering held it for up
  to 15 minutes (the resumable upload's stall limit), then sent the whole file in one request (15-minute write timeout). The parts
  behind it showed "Uploading video 2 %" while they waited. Redo part restarted the part at the back of the same queue.
- **Changed (PhoenixViewModel.kt, net/HfApi.kt):**
  - `VIDEO_UPLOADS_AT_ONCE = 3` video uploads at a time (each part to its own Space; the per-Space gate stays);
  - the video's resumable upload gives up after `VIDEO_STALL_MS` = 3 minutes without a byte accepted, with a new
    `UploadStalledException`; the job then leaves the slot, waits 20 s (or for the network) and tries again, resuming at the Space's
    offset, up to `VIDEO_STALL_TRIES` = 5 times, then Failed ("check this Space, then tap Redo part"). No whole-file request after a
    stall (still used when the Space has no resumable routes or the source cannot be read at an offset);
  - a part waiting for a slot shows "Waiting to upload" and its place in the queue;
  - at start, Video cards without a job id that are not finished are marked "Interrupted" (tap Redo part): nothing sends them after
    the app run that started them has ended.
- **Checked here:** all app sources compile with the Kotlin 2.0.21 / Compose compiler against the same API surface as 1.2.58 with no
  new error (deliberate mistakes in the new code were caught, so the check is real). The real `HfApi.uploadResumable` against a
  local fake Space: a working Space returns the path; a Space answering only 503, and one that stops after two chunks, both end
  with `UploadStalledException` at the stall limit.
- **Not checked here:** a Gradle / Android build (no Android SDK in the sandbox) and the app on a phone.

# v1.2.37 - the good parts of the 1.2.36 pack, and a calmer, more finished look

- **Asked:** check whether 1.2.32 has the features of the attached 1.2.36 ("AutoBelly Button") pack, take only the good ones, and give the
  front end a more beautiful feel.
- **Compared:** 1.2.36 is an older line plus a one-button belly switch and three fixes. 1.2.32 already did more of the belly work
  (switch in the job options, the Belly clean mode with a frame preview, band length and breadth, PNG frame upload). Taken from 1.2.36:
  - **Secure Folder:** the app declares `ACCESS_NETWORK_STATE`, and the network check can no longer abort a job (parts sat at 8 % with
    `ConnectivityService: Neither user ... nor current process has android.permission.ACCESS_NETWORK_STATE`).
  - **Uploads:** if the resumable upload dies part way (a proxy page, a restart, a short read), the video goes up again as one request; a
    proxy answering with HTML or an empty page is retried instead of crashing. 1.2.36 left the two status calls (offset, complete) unguarded;
    they are guarded here too. A read that keeps returning 0 bytes no longer spins.
  - **Messages:** a failure shows its root cause (`Stopped: ...`), and a job that cannot start or a split that stops says so
    (`Could not start: ...`, `Split stopped before upload: ...`) instead of doing nothing. The new catch blocks pass coroutine
    cancellation through (1.2.36's swallowed it).
  - **One-tap belly control:** an "Automatic belly clean" switch under the target video, on the main screen. It is the same setting as the
    job-options switch and the editor's "Use for this video" (1.2.36 kept a separate switch, unconnected to the preview, length and
    breadth). It sends `clean_belly` with the job (Space v11.2.117+).
  - Not taken: 1.2.36's separate belly state, and its removal of the belly preview, length and breadth.
- **Design (the shared building blocks, so every screen follows):**
  - one palette (brand burgundy, a warm ember gradient only for the main buttons and badges, hairline borders, a quiet field colour),
    one set of corner radii and a defined type scale for text that sets none of its own;
  - cards: no more four-colour border and strip; a hairline, a soft shadow and a round numbered badge;
  - top bar: a segmented Image / Video / History control, a gradient Start button, a thicker progress bar;
  - main buttons are gradient with a soft glow; secondary buttons are outlined; fields, dropdowns, presets and rows share one look;
  - the upload zone has a dashed outline and an icon badge; face tiles are rounder with a lighter shadow; split-part pills are quieter;
  - the page has a soft gradient and a little more space between cards.
- **Checked here:** the Kotlin and Compose compiler (2.0.21) compiles the real source of the thirteen restyled composables, the new
  belly card, the theme and root layout, and `HfApi.kt` with the upload fixes (against OkHttp 4.12 and Compose 1.7.3), and the real
  `sendVideo` / `hasInternet`; the other files were checked for new errors against 1.2.32. The builder's static audit passes.
- **Not checked here:** a full Gradle / Android build (no Android SDK in the sandbox) and the app on a phone or an emulator. The design was
  not seen rendered. The APK Builder compiles it for real.
- versionCode 157, versionName 1.2.37 (above 154 of the 1.2.36 pack: install over it or over 1.2.32).

# v1.2.32 - Belly clean in one go: the buttons work with Space v11.2.117

- **Asked:** a front-end button for the one-go belly clean from the Grok pack (Phoenix_v11.2.116_ContactHold), and a preview on a frame.
- **Preview belly** (Patch Removal, Belly clean) now calls `phoenix_clean_belly_preview` of Space v11.2.117: the frame, the
  cleaned frame and the band outlined, from the same steps as the job's one-go clean. Length / Breadth apply to it too.
- The frame for the preview is sent as **PNG** (lossless), not JPEG: on the test body the tattoo was 19 % left after a JPEG
  frame and 1 % after the same frame as PNG (JPEG's colour sub-sampling blurred the mark colour the Space looks for).
- **Use for this video** and the job-options switch "Lower belly: clear skin" now send `clean_belly: true` (the one-go clean of
  Space v11.2.117) as well as `belly_clear` (older Spaces). The length and breadth go with the job as before.
- Not compiled here: the Kotlin toolchain and the Android SDK are not available in this sandbox. The changes are small: a PNG
  save, the endpoint name, the `clean_belly` key and the content type of the frame upload. Built and checked in the APK Builder.
- Versions: versionCode 152, versionName 1.2.32.

# v1.2.31 - Belly clean in the Patch Removal frame screen: preview, length and breadth (Space v11.2.111)

- **Asked:** "Add an option of preview frames in belly clean, and extend it (enhance or reduce length and breadth). Add belly clean
  as an option in the frame screen of Patch Removal."
- The Patch Removal screen (now "Patch Removal / Belly clean") has two modes at the top: **Paint a mark** (as before) and
  **Belly clean**. In Belly clean:
  - choose the frame with the same slider and frame steps;
  - **Belly clean for this video** on / off (the same setting as the switch in the job options);
  - **Length** and **Breadth** of the band, 50-150 % (100 % = as found from the body), in 5 % steps; Reset 100 %;
  - **Preview belly** sends the frame to the Space and shows it cleaned, with the band outlined; Show before / after,
    Hide / Show outline; a line says how many marks were removed, or that no body was found on the frame;
  - **Use for this video** turns it on from the same screen.
- The job sends `belly_length` / `belly_width` with `belly_clear`. Spaces older than v11.2.111 ignore them (and have no preview).
- Compile check: 0 errors (kotlinc 2.0.21 + Compose plugin). Not run on a phone.

# v1.2.30 - lower belly: clear skin (Space v11.2.110)

- **Asked:** "All open (uncovered) skin below the navel area, down to the lowest part of the waist, becomes smooth and flawless -
  I am bothered by any tattoo or mark in that area."
- **New switch** in the job options, under Motion smoothing: **Lower belly: clear skin** (off by default). It sends
  `"belly_clear": true` with the job. A Space v11.2.110 or newer then finds the band from just below the navel to the
  waistline on the person's body, removes every tattoo, mole or mark on the uncovered skin there, and evens the skin. Clothing
  and the navel are left alone. Older Spaces ignore the setting.
- Nothing else changed. Compile check: 0 errors (kotlinc 2.0.21 + Compose plugin). Not run on a phone.

# v1.2.29 - a second job starts while another is split across Spaces (Space v11.2.101)

- **Reported:** 20 Spaces, one job split into 10 parts running, and no other job could be started.
- **Why:** a new job always went to the *selected* Space - which the split had already used for part 1 - and a split asked
  for "N free Spaces" by looking only at the phone's own History, where an old entry in a state such as "Cancelled" or
  "Error" kept a Space "busy" for ever. A split that was still cutting and uploading its later parts also left their
  Spaces looking free to a new job.
- **Now:**
  - Before a job starts, every Space is asked whether it is really idle (Spaces v11.2.92+ answer; the phone's own
    knowledge is used when one does not). A stale History entry no longer blocks a Space.
  - A single job uses the selected Space if it is idle, otherwise the first idle one, and says so ("Space 1 is busy -
    using Space 11").
  - A split uses as many idle Spaces as it needs; if fewer are idle it splits into that many parts and says so (with only
    one idle Space it runs as one job). It says "Every Space is busy" only when none is idle.
  - The Spaces a split has picked are held for it until each of its parts has started, so a second job cannot take them.
- Terminal states for a Space's History entries now also include Cancelled, Error, Stopped, Expired, Interrupted and
  Downloading (the render is finished; only the file is being fetched).
- Works with any Space; the idle check uses `/phoenix/busy` (v11.2.92+). Patch Removal's "stuck at 11%" is fixed in Space v11.2.101.

# v1.2.28 - Patch Removal: one mark can be painted at up to 10 positions (Space v11.2.100)

- One mark can now be painted at up to **10 positions** (10 frames where the person is somewhere else): was 6. Use
  "+ Position" on each frame, and Freeze on the last one.
- A video can have up to **10 marks** (was 4) and **20 painted positions** in all (was 12). The Patch Removal section
  counts them ("3 mark(s), 7 painted position(s)"); at the limit the add button greys out and a line says why.
- Needs Space v11.2.100. A Space on v11.2.96-v11.2.99 refuses a job with more than 4 marks or 12 positions
  ("at most 12 painted positions per job") before anything is processed; within those limits it works as before.
- More positions and marks make a job slower: each mark is followed through the whole video and painted on every frame.

# v1.2.27 - Patch Removal editor: every button always on screen, portrait and landscape (Space v11.2.98)

- The editor is now a full screen inside the app's own window instead of a separate dialog window. A dialog window
  does not get reliable system-bar insets on every phone, so in landscape (navigation bar on the right) the Freeze
  button could end up under it. Now the whole screen sits inside the safe drawing area with a 12 dp margin, and the
  Preview / + Position / Freeze row is never inside a scroll area.
- Portrait and landscape both work: portrait = picture on top, controls below, buttons in one row at the bottom;
  landscape = picture on the left, controls on the right. Turning the phone keeps everything (state in the
  ViewModel; the activity handles the rotation itself). The landscape lock of 1.2.26 is gone.
- Rows of small buttons scroll sideways on narrow screens instead of being cut off. The back button closes the editor.
- Compile check 0 errors; builder static audit passes.

# v1.2.26 - Patch Removal: several positions per mark, landscape editor, zoom (needs Space v11.2.97)

- The same mark can now be painted at several positions: after painting it on one frame, tap "+ Position", move
  to a frame where the person has moved or turned, and paint the same mark there (up to 6 positions per mark, 12 in
  all). The Space follows the mark from every position and uses the nearest one on each frame.
- The editor is landscape-only and survives rotation: the activity now handles orientation changes itself
  (configChanges), and the editor's whole state lives in the ViewModel (it used to close and drop back to the main
  screen when the phone turned).
- Controls moved to a side panel; Preview, + Position and Freeze are fixed at its bottom, inside the safe area
  (they could fall under the navigation bar before).
- Zoom: + / - / Fit buttons besides pinch, a zoom badge, and a 3x magnifier showing what is under the finger while
  painting. Frame stepping: one frame / one second back and forward.
- Compile check 0 errors; builder static audit passes.

# v1.2.25 - Patch Removal (needs Space v11.2.96)

- New optional "Patch Removal" button under the target video. It opens a full-screen editor:
  - choose the exact frame (not the nearest key frame);
  - paint over a tattoo, mole or dark mark with one finger (brush S / M / L, Undo, Clear); pinch with two fingers to
    zoom and pan for small marks, "Fit" to reset;
  - "Preview" sends the frame and the painted mask to the Space (`phoenix_patch_preview`) and shows the frame with
    the mark removed - "Show before / Show after" to compare;
  - "Freeze" keeps the mark (up to 4 per video; each can be removed again from the list).
- Frozen marks travel with every job on that video (settings.patches: frame + mask as base64, and the frame's time in
  the uploaded file - corrected for a local trim, whose cut starts on a key frame). The Space finds each mark on its
  frame, follows the skin through the whole video and removes it on every frame, before the face swap.
- Choosing another video clears the marks. A Space older than v11.2.96 has no preview route; the editor says so.
- Compile check: kotlinc 2.0.21 + Compose plugin against Android 15 classes, 0 errors (a missing
  ViewModelProvider stub was added to the verification stubs). Builder static audit passes.

# v1.2.24 - updates wait for running jobs; GPEN face restore

- **"Push to all Spaces" no longer loses a running job.** An update restarts the Space, so a job rendering there
  was lost. The app now asks every Space first (`/phoenix/busy`, Space v11.2.92; for older Spaces it checks the
  jobs this phone sent there). Idle Spaces are updated at once; busy ones are listed under the button and updated
  by themselves when their job ends (checked every 2 minutes while Phoenix is open, and again the next time it is
  opened - the pack is kept on the phone until then). "Cancel waiting updates" drops them.
- **Face restoration: "GPEN + Cinematic"** (the new default): GPEN-BFR-256 restores eyes, lips and skin on the
  swapped face, never sharper than the footage, then the subtle Cinematic glow / HDR / vibrance. About 0.2-0.3 s
  per face on the Space (GFPGAN: about 3 s). Needs Space v11.2.92; an older Space treats it as plain Cinematic.
  "Cinematic (clarity + smooth)" stays as the fastest choice. CodeFormer is removed: it never loaded on the Space
  and silently ran a soft polish.
- Same app id, same signing key, versionCode 144: installs over 1.2.23 and keeps token, Spaces and History.

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
