# Phoenix Mobile 1.2.22 — crash guard

**Asked:** make the anti-crash mechanism stronger (the app closed a couple more times, then opened), as an APK that installs **over** the existing app without disturbing running jobs and without re-entering settings or the secret.

## Installing over the existing app

| | |
|---|---|
| App id | `com.swamitech.phoenix2` — unchanged |
| Signing key | the same `phoenix-debug.keystore` as 1.2.19 (bytes compared) |
| Version | 1.2.22, versionCode **142** (above 139 / 140 / 141) |
| Saved data | untouched: settings, Spaces and History live in the same private files; the secret is in the Android Keystore under the same alias. New data only adds new keys |
| Running jobs | run on the Spaces; installing only restarts the app, which reconnects to them |

Install it over the app. **Do not uninstall.** If the phone refuses (key mismatch), tell me; do not uninstall while jobs are running.

## What the guard does

1. **Restarts itself.** If the app crashes **while it is on screen**, it starts again immediately and reconnects to the running jobs — at most twice in two minutes. A crash while the app is in the background is not restarted (Android does not allow it); the next time you open it, it reconnects.
2. **Safe mode.** Three crashes in two minutes stop the restarts. The next launch opens in *safe mode*: the app shows a yellow note, does **not** touch the running jobs, and waits for you to tap **Resume jobs**. This breaks any crash loop. The jobs keep running on the Spaces meanwhile. Safe mode lifts by itself after two quiet minutes.
3. **Background work cannot end the app.** Every background task now has an error handler, and the network callbacks are wrapped. An error becomes a message (and a log line) instead of a crash.
4. **A real bug fixed on the way.** The code that restarts job watchers at start-up ran before several fields it uses had been created. It now runs last.
5. **History cannot be lost to one bad entry.** A damaged entry is skipped; an unreadable history is kept aside instead of being overwritten.
6. **More memory for video work** (`largeHeap`), so a large frame does not run the app out of memory.
7. **A note after trouble**, with the reason Android recorded or the first lines of the error, and a log of the last 6 errors in the app's private folder.

**Correction to 1.2.21:** its "closed unexpectedly" note never showed — it was set before the field it uses had been created, so it was lost. 1.2.22 fixes that. Skip 1.2.21 and install 1.2.22.

## What it cannot do

- If **Android itself kills** the app in the background (memory, battery rules), no app can stop that. The jobs are not affected. If the phone does it often: Settings → Apps → Swamitech Phoenix 2 → Battery → **Unrestricted**, and turn **Manage app if unused** off.
- It cannot know why the earlier closes happened. With 1.2.22, if it happens again the yellow note says why; send me its text.

## Verified / not verified

| Verified here | |
|---|---|
| Decision logic on the JVM — restart limits, safe mode, windowing, log ring, tolerant history load, error text | 25 checks pass |
| Coroutine error handling with the real kotlinx-coroutines: handled errors do not reach the crash handler, the scope survives, cancellation is not an error, and (contrast) the same error without the handler does | 5 checks pass |
| Whole app source compiles (kotlinc 2.0.21, Compose plugin, AndroidX stubs) | 0 errors |
| Audit 42/42; keystore bytes equal 1.2.19's; Dockerfile unchanged; builder against a fake SDK | pass |

| **Not verified** | |
|---|---|
| A real Gradle build, install, or crash on a phone | not possible here. The restart itself (starting the activity from the crash handler, then ending the process) is the standard approach but I could not run it on a device |
| The yellow note and the **Resume jobs** button on a device | compile-checked only |
| That the earlier closes were the keep-alive service | circumstantial; the note will tell us |
