# Phoenix Mobile 1.2.21 — the app vanishes when it goes to the background

**Reported:** "The app is closing again and again", no message. Installed version: **1.2.19** (screenshot).

## Cause (most likely, not proven)

1.2.19 starts a keep-alive **foreground service** every time the app goes to the background with a job running (`onStop` → `onAppBackground` → `startBackgroundGuard`). An earlier build had to remove that exact service because it "was closing the app in a loop on this phone" (comment kept in the 1.2.10 source). A crash that happens while the app is in the background shows no dialog, so the app just disappears — which matches what you see. I cannot see the phone's log, so I cannot prove it.

## Changes

1. **The keep-alive service is no longer started.** The render runs on the Space, so closing the app does not stop a job; opening the app reconnects to every tracked job (History). A service left over from an older install is still stopped.
2. **If the app closes again, the next launch says why**: a yellow note under the top bar with the reason Android recorded (crash, low memory, killed by the system) and, for a crash, the first lines of the error. That answers "what is closing it" without a computer. Android 11 or newer.
3. **Version 1.2.21 (versionCode 141), same signing key** as 1.2.19 and 1.2.20, so it installs **over** the app already on the phone and keeps History, token and Spaces. **Do not uninstall first.**
4. Builder text now says that. Everything else (Dockerfile, build checks, the push fix) is unchanged from 1.2.20.

## Your 2 running jobs

They run on the Spaces and do not depend on the app. Open each Space in the phone's browser: "Active job" shows progress; finished jobs are under "Completed jobs" with a download button (kept 3 h on Space versions before v11.2.80, 12 h on v11.2.80 and later). Do not restart or push updates to a Space while its job runs.

## Also seen in your screenshots (not changed)

- **"Manage app if unused" is on** for Phoenix. After a long time unused, Android removes permissions, stops notifications and archives the app. Turn it off for this app.
- **Notifications are Off.** Phoenix does not need them any more; you can leave it off.
- The app was installed from Opera. That is fine.

## Verified / not verified

| Verified | |
|---|---|
| Whole source compiles (kotlinc 2.0.21, Compose plugin, AndroidX stubs) | 0 errors |
| Audit 40/40; same keystore bytes as 1.2.19; `startForegroundService` no longer present | pass |
| `build.sh` + builder UI against a fake SDK (install info, install tip) | pass |

| Not verified | |
|---|---|
| A real Gradle build and a real phone | not possible here |
| That the foreground service was the cause on your phone | strong circumstantial evidence only; if the app still closes, the yellow note on the next launch tells us what it is — send me its text |
| The "closed unexpectedly" note | compiles; not run on a device |
