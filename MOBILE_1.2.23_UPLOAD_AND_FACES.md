# Phoenix Mobile 1.2.23 — uploads keep going in the background; saved faces with names

**Builder:** `Phoenix_APK_Builder_1.2.23.zip` (upload all 6 files to the APK Builder Space, replacing the old ones, press Build APK).
**Install:** over 1.2.22, do not uninstall: same app id, same signing key, versionCode 143. Token, Spaces and History are kept.
**Space:** works with any Phoenix pack; the resumable upload needs Space v11.2.80 or later (every pack since). With an older pack the app uses the old upload by itself.

## 1. "The upload stops when the app is minimised" — two causes, both fixed

1. **The whole video went up in one request.** The app used Gradio's normal `/upload`: one long request. Any interruption (the app frozen in the background, Wi-Fi to mobile data, a stall) lost the entire transfer. The Space has had resumable upload routes since v11.2.80, but the app never used them. Now it does: the video goes up in 4-MB pieces, and after any interruption it asks the Space how much it already has and continues from there. It gives up only after 15 minutes without a single byte accepted. If the app is killed and the same job is started again, it continues from where it stopped (the upload is identified by the video and the Space).
2. **Nothing kept the app running in the background.** 1.2.21 removed the keep-alive service because starting it when the app went to the background made the app vanish. That was the way it was started (`startForegroundService()` from the background: Android refuses it and ends the app). Now an "uploading" notification is shown **only while an upload runs**, started at the moment you tap Start (the app is on screen), as a plain service that promotes itself; if Android refuses, it just stops, and nothing crashes. A partial wake lock (max 3 h) keeps the upload going with the screen off. The notification goes away when the upload ends; following the render after that needs nothing from the phone, as before.

**Tested** (JVM, the app's real `HfApi` code against the Space's real upload code from v11.2.90 running locally):
- 40-MB file through a connection that was cut 6 times and stalled once: arrived **byte-identical**;
- app killed at 50 %: started again, it continued from 50 %;
- clean connection: byte-identical.

## 2. Saved faces

Under every replacement-face slot:
- **▾ Saved (n)** opens a drop-down of your saved faces: name + small thumbnail. Picking one puts it in that slot.
- **☆ Save** (when the slot holds a picture that is not saved yet) asks for a name and keeps it. The slot then shows the name.
- **Rename or delete saved faces…** at the bottom of the drop-down.

Faces are stored in the app's private folder (`files/faces`, a JPEG of at most 1280 px plus a 160-px thumbnail), are removed with the app, and are sent to a Space only as the replacement face of a job, exactly like a picture from the gallery. Two faces cannot share a name ("Priya", "Priya 2"). A damaged face list is kept aside and rebuilt from the pictures.

Tested on the JVM: list, rename, unique names, delete (entry and both files), damaged-list recovery (11 checks).

## 3. What was not tested

- No APK was built here (the Android SDK is not reachable from this sandbox). The whole app source compiles with kotlinc 2.0.21 + Compose against Android 15 classes (0 errors), and the builder's own static audit passes (43/43). **The builder Space does the real build.**
- Not tested on a phone: the notification, the wake lock, the drop-down's look. If Android still stops the upload on your phone, send me the yellow note the app shows and the phone model.
