---
title: APKBuilder
emoji: 🤖
colorFrom: blue
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Phoenix APK Builder 1.2.20

Upload every file in this pack, including `source.zip`, replacing the old ones.
Then open the Space and press Build APK.

**1.2.20** checks the finished APK (signature, package, test-only flag) before it hands it over, and shows
the signing key so you can compare builds.

**Installing:** if the phone says "Something went wrong" or "App not installed", an older Phoenix on the phone
was signed with a different key. Uninstall it once (Settings > Apps > Swamitech Phoenix 2), then install the new
APK. The key in this pack is fixed, so later builds update in place. The build log shows
`Signing key (SHA-256): e2564a9b...` - it must be the same on every build.
