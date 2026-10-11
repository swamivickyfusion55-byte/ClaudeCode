---
title: APKBuilder
emoji: 🤖
colorFrom: blue
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Phoenix APK Builder 1.2.22

Upload every file in this pack, including `source.zip`, replacing the old ones.
Then open the Space and press Build APK.

**1.2.22** adds a crash guard to the app: if it crashes while on screen it restarts itself and reconnects to
the running jobs; repeated crashes open it in a safe mode that leaves the jobs alone; errors in background work
can no longer end the app; and the next launch shows what happened.

**Installing:** install the new APK *over* the one on the phone - do not uninstall. Same app id, same signing
key and a higher version number, so Android updates in place and keeps your token, Spaces and History, and the
jobs keep running on the Spaces. The build log shows `Signing key (SHA-256): e2564a9b...`; it must be the same
on every build.
