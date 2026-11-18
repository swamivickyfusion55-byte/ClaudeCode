# M3 Handoff

This pack is designed for the next build environment (Hugging Face or another Android CI runner).

## Build
1. Open the project root in an Android Gradle build environment.
2. Use JDK 17 and Android SDK 35.
3. Run `./gradlew assembleDebug` or the CI equivalent.
4. Run `python3 tools/audit/m3_audit.py` before packaging.
5. Install the APK on the target phone.
6. Execute the mandatory device/visual audit in `M3_RELEASE_AUDIT.md`.

## Two modes
- **Hugging Face Pro:** remote Gradio queue API. The app intentionally does not fabricate remote percentage; it displays elapsed time until the server returns a completion event.
- **Fully Local:** ONNX Runtime model-pack loader is operational, but actual face-swap inference is disabled until the selected licensed model graph is supplied and passed Gate B.

## Important
Do not mark the APK "production" merely because Gradle succeeds. The local face engine must produce and validate real output on-device.
