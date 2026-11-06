# Phoenix Mobile M2 — verified model pack

The app supports a model pack stored in private app storage. It is NOT embedded in the APK.

## Required files
- detector.onnx
- recognizer.onnx
- landmarks.onnx
- swapper.onnx
- phoenix-model-manifest.json

The manifest contains `files[]` entries with `name` and SHA-256. The app downloads the pack from a Hugging Face model repository and verifies every file before use.

## Why this is separate from the APK
The current InsightFace model zoo states that its model weights are for non-commercial research use, while the vendor currently offers separate licensing for commercial deployment and for InSwapper-128. Do not redistribute those weights inside this project unless you have the applicable rights.

## Recommended private repository
Create a private Hugging Face model repository containing only models you are licensed to use. The Android app uses a read token and SHA-256 verification.
