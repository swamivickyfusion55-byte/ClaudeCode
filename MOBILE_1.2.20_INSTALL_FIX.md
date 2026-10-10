# Phoenix APK Builder 1.2.20 — "Something went wrong" when tapping the APK

**Reported:** the APK from the 1.2.19 builder (made by another assistant) shows "Something went wrong" on the phone's installer.

## What I found

I could not build an APK here (Google's Android tools are blocked from this sandbox), so I checked everything around the build instead:

| Check | Result |
|---|---|
| 1.2.19 Kotlin source, compiled as one module (kotlinc 2.0.21 + Compose plugin, AndroidX stubs) | compiles, 0 errors |
| `tools/audit/m3_audit.py` | 40/40 |
| `phoenix-debug.keystore` | valid PKCS12, alias `phoenix`, password as in `build.gradle.kts`, RSA 2048, valid to 2054 |
| Dockerfile / `app.py` | same as the 1.2.10 builder that built before, plus a build-tools pin |
| Source contains the text "Something went wrong" | no — the message comes from the phone, not from Phoenix |

So nothing in the pack stops a build. A message on the phone's installer, with an APK that built, points to the **signature**:

- 1.2.19 signs with one fixed key (`phoenix-debug.keystore`). That is a good change.
- Android will not update an installed app with an APK signed by a different key. If the Phoenix on your phone came from an earlier builder run, its key is probably different: the builder points Gradle's key folder at the Space's own disk (`ANDROID_USER_HOME=/workspace/.android`), so by my reading each fresh container made a new random debug key. The installer then refuses the new one. Some phones word that as "App not installed", others as "Something went wrong".

**I cannot see your phone, so this is the most likely cause, not a proven one.** The fix for it is one step on your side (below).

## What 1.2.20 changes

1. **The build now checks the APK before offering it** (`build.sh`): `apksigner verify`, then package id, versionCode, SDK levels, ABIs, the test-only and debuggable flags, and the signing key's SHA-256. A broken, unsigned or test-only APK stops the build with a clear error instead of failing on the phone. The result is shown in the builder next to the download.
2. **The builder shows the install tip** right under the download.
3. **Version** 1.2.20, versionCode 140 (the 1.2.19 pack was 139), so it is never treated as a downgrade of that pack.
4. **Brings back the 1.2.10 "Push to all Spaces" fix.** The 1.2.19 line did not carry it. Commit requests go out as `application/x-ndjson` exactly as Hugging Face's own clients send them, with one automatic retry in the plain-JSON form.

The signing key, the Dockerfile and the rest of the app are unchanged from 1.2.19.

## What you do

1. Upload all 6 files to the builder Space (replace the old ones, including `source.zip`), press **Build APK**.
2. In the build log / status, look for `Signing key (SHA-256): e2564a9bd48bf5bf8df8a41c28fdd02d53837de8545ef1722b0639968ec46ef4`. It should be the same on **every** build. If it differs, tell me.
3. **Uninstall the old Phoenix once:** Settings → Apps → Swamitech Phoenix 2 → Uninstall. You will need to enter your Hugging Face token and Spaces again (the app does not back them up).
4. Install the new APK. Later builds from this pack install over it without uninstalling.

## If it still says "Something went wrong"

Send me a screenshot of the message, your phone model and Android version, and the lines under `--- install info ---` in the build log. That tells me whether it is the key, the Android version, or something else.

## Verified / not verified

| Verified here | |
|---|---|
| `build.sh` against a fake SDK in 4 cases: tools present (passes, info written), `apksigner` fails (exit 3, no APK offered), test-only APK (exit 3), tools missing (warns, still builds) | pass |
| The builder's `build()` run end to end through that script | pass |
| `source.zip` unpacked with the Dockerfile's own commands, then audited | 40/40 |
| Push fix against a mock Hub: NDJSON form, byte-identical body, JSON fallback, 403/404 messages | 11 pass. (2 more checks in that suite describe the 1.2.9b code's old failure; they do not apply to 1.2.19's different, plain-JSON request, so I do not count them either way) |
| Whole 1.2.20 source compiles | 0 errors |

| **Not verified** | |
|---|---|
| A real Gradle/AGP build and a real install | not possible here. The Gradle files are exactly 1.2.19's apart from the version, which the other assistant's builds already ran |
| The exact text of real `apksigner` / `aapt2` output | my parsing is tolerant; if a line is not recognised the build still passes and that line is missing from the install info. Only a failing `apksigner verify` or a test-only manifest stops a build |
| The real Hub's acceptance of the push | against a mock modelled on the official clients' requests |
| That a key mismatch is your cause | see above |
