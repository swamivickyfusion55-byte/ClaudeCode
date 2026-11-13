# Hugging Face Space mobile contract

The Android client supports both Gradio 4.x and newer Hugging Face Space API routes.

For this SwamitechUAT Space (Gradio 4.44.1), the client prefers `/config` for connectivity and uses `/upload` + `/call/<endpoint>`; it automatically falls back to the newer `/gradio_api/*` routes when available. A private Space still requires an HF token with READ access.

Recommended long-term endpoint:

`/phoenix_mobile`

Suggested inputs:
1. video FileData
2. source face image(s) or source mapping payload
3. processing mode
4. quality options

Suggested output:
1. output video FileData
2. optional JSON diagnostics (frame count, effective FPS, elapsed time)

The existing server-side pipeline should remain the source of truth for the remote mode. Do not duplicate its Python face-identity logic in the Android UI.
