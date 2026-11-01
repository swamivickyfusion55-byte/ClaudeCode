---
title: APKBuilder
emoji: 🤖
colorFrom: blue
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Phoenix APK Builder 1.2.24

Upload every file in this pack, including `source.zip`, replacing the old ones.
Then open the Space and press Build APK.

**Builder fix (1.2.24):** a build that was stopped half way used to leave a cut-off APK that made the next build
fail at the very end ("Unexpected end of ZLIB input stream"). Packaging now always starts fresh, and a failed
build is retried once from a clean tree.

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
