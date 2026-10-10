# Phoenix Mobile 1.2.9 — Update Spaces from the phone · SG16–SG20 · frame picker under the video

Built on **1.2.8** (`versionCode 128`), which is the later of the two packs you sent: `PartNames_1` is
1.2.7. 1.2.9 is `versionCode 129`. Changed files: `MainActivity.kt`, `PhoenixViewModel.kt`,
`security/SecureStore.kt`, `app/build.gradle.kts`, `CHANGELOG.md`, `README.md`.
New: `net/HfDeploy.kt`.

## 1. Update Space code (new card under Connection)

1. Open **Update Space code → Update Spaces from a package**.
2. **Choose package** and pick the zip I send (e.g. `Phoenix_v11.2.80_Resume.zip`) as it is, or its 8 files.
3. Tick the Spaces. **All free Spaces** ticks every configured Space with no job from this phone running.
4. Tap **Update N Spaces** and confirm. Each Space shows its own result line.
5. **Check Space status** shows BUILDING, RUNNING or RUNTIME_ERROR for each ticked Space.

What it does and refuses:
- **Files pushed:** only the Space files (`app.py`, `config.py`, `core_pipeline.py`,
  `phoenix_api_adapter.py`, `swap_engine.py`, `packages.txt`, `requirements.txt`, `README.md`).
  SDOS notes and `tools/` in the zip are never pushed.
- **Partial packages:** refused, naming the missing files. A Space must never run two versions at once.
- **Unchanged files:** skipped. A Space that is already on that version is **not restarted**.
- **README.md:** this is the Space settings header. Each Space keeps its **own title**, not the pack's
  "SG UAT2". A README without a valid header (`sdk:` and `app_file:`) is never pushed, because it would
  break the Space. You can switch README off.
- **Busy Spaces:** a Space with a job from this phone running is skipped, because the restart would
  kill the job. A job started from the website on that Space is lost.

**Token:** changing a Space needs a Hugging Face token with **Write** access. Either save a separate
**update token** in the card (stored encrypted in the Android Keystore, like the main one) or use a main
token that has write access. A read-only token is detected and refused before anything is sent.
Create one on huggingface.co under Settings → Access Tokens: a "Write" token, or a fine-grained token
with write access to those Spaces.

**Free-account limit:** every updated Space restarts. On a free account Hugging Face only lets a limited
number of Spaces run at once, so the rest show the cpu-basic quota 403 until others are paused. Tick only
the Spaces you use, or pause the extras afterwards.

## 2. Spaces SG16–SG20

`SwamitechGradio16…20` → `https://swamivicky-swamitechgradioN.hf.space`, same naming as the others, for
21 Spaces in total. The selected Space is now remembered by its address. Without that, inserting five
Spaces would have silently moved the next job: an install with SG_UAT2 selected would have pointed at
SG16. Any address you edited by hand is kept.

## 3. Frame picker under the video

The slider and **Show frame & detect faces** now sit directly under the video preview, which follows
the slider ("FRAME n%"). You see the frame you are picking without scrolling. Card 02 is now
"Detected faces".

## How it was verified, and what was not

- **Not built into an APK here.** Google's Maven repository (Android Gradle plugin and AndroidX) is
  blocked from my sandbox. Build it with your usual APK builder.
- **Compile:** every Kotlin file of 1.2.9 compiles as one module with **0 errors and no new warnings**
  (Kotlin 2.0.21 with the Compose compiler). It was compiled against the real Android 15 framework
  classes (Robolectric `android-all`), Compose 1.7 / Material3 (JetBrains build, same API), OkHttp 4.12
  and coroutines 1.9. Small stand-ins were used for `androidx.activity`, `lifecycle` and `core`, which
  are only on Google's repository. Unchanged 1.2.8 compiles clean under the same setup, so the setup is
  sound.
- **Deploy module tests:** 37/37 checks against a mock Hugging Face Hub that enforces the commit
  protocol. They cover:
  - reading the real v11.2.80 zip;
  - partial, duplicate and oversize packages;
  - resolving all 21 Spaces through paged listings;
  - 20 Spaces updated plus 1 with no write permission reported;
  - re-running changes nothing;
  - a one-file change sends one file;
  - README title kept;
  - read-only token, LFS-only file, no network, wrong token, no token.
- **Conformance:** for the same files, the app's preupload and commit requests are **identical** to those
  of Hugging Face's own `huggingface_hub` 0.27.1 client.
- **Audit:** the pack's own audit (`tools/audit/m3_audit.py`) passes 41/41.
- **Not tested:** against the real huggingface.co (blocked from here) and on a device. First real run:
  tick **one** Space, update it, and check the result line and the Space's startup log before doing all
  21.
