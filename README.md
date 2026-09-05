---
title: SG UAT2
emoji: 🔥
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10"
app_file: app.py
pinned: false
---

See README_DEPLOY.md for the actual deployment/architecture notes.
This file's YAML header is what Hugging Face reads to build the Space.

sdk_version is pinned to 4.44.1 to match the gradio==4.44.1 pin in
requirements.txt.

python_version is pinned to 3.10 because requirements.txt's own first
line says "HF Spaces · Python 3.10 · ZeroGPU-capable pack" - without
this field, the platform defaults to whatever its current default is
(3.13 as of this build), which breaks onnxruntime<1.20 (no 3.13 wheel
exists below 1.20) and would break torch==2.1.2 next for the same
reason - that exact version predates Python 3.13's release, so no
3.13 wheel exists for it either.

## V51.1 visibility gate

Skip swap when the face is looking down/away, out of frame, or only predicted for >2 frames. Leaves original pixels instead of pasting a ghost face on the body.

## SoftStable (v11.0.4)

Brightness-only deflicker on Seamless-CPU. **Do not use ProStable hold-everywhere** — it pasted faces onto body when bbox drifted.
