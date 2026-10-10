# Phoenix Mobile — off-device verification

Used for 1.2.9. Google's Maven (AGP/AndroidX) is unreachable from the build sandbox, so instead of an
APK build:

- `DeployTestMain.kt` + `mockhub.py`: JVM tests of `net/HfDeploy.kt` against a strict mock of the
  Hugging Face Hub (37 checks). Compile with kotlinc 2.0.21 + okhttp 4.12.0 + okio-jvm 3.6.0 +
  org.json; run `python3 mockhub.py PORT`, then
  `java ... DeployTestMainKt PORT <Phoenix_v11.2.x.zip> <extracted pack dir>`.
- `conformance.py`: after the JVM run, has huggingface_hub 0.27.1 commit the same files to the mock
  and compares its preupload/commit requests with the app's.
- `stubs/`: compile-only stand-ins for androidx.activity / lifecycle / core and Android-only Compose
  locals. Compile all app sources as ONE module with kotlinc 2.0.21 + kotlin-compose-compiler-plugin
  2.0.21 against Robolectric android-all 15-robolectric-12650502, JetBrains Compose desktop 1.7.3,
  kotlinx-coroutines 1.9.0, okhttp(+dnsoverhttps) 4.12.0, okio 3.6.0, with
  `-Xnullability-annotations=@android.annotation:warn`.
