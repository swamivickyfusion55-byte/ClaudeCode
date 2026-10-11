---
title: APKBuilder
emoji: 🤖
colorFrom: blue
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Phoenix APK Builder 1.2.21

Upload every file in this pack, including `source.zip`, replacing the old ones.
Then open the Space and press Build APK.

**1.2.21** stops the app vanishing when it goes to the background (the keep-alive service is no longer
started) and shows, on the next launch, why the app closed. It also checks the finished APK (signature,
package, test-only flag) before handing it over.

**Installing:** install the new APK *over* the one on the phone - do not uninstall first, that deletes History
and the numbers of running jobs. The signing key is fixed, so the build log always shows
`Signing key (SHA-256): e2564a9b...`; if it ever differs, Android will refuse to update.
