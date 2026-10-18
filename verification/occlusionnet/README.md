# OcclusionNet verification (SDOS-091)

Synthetic food on real photographs (insightface sample photo). **No user footage.** Paths are those of the sandbox.

| Script | What it does |
|---|---|
| `xseg_test2.py [models]` | the XSeg occluder models on 12 real aligned faces: clean-face false cuts inside the landmark outline, scoop found at mouth / lips / cheek |
| `xseg_zone.py [blur]` | the shipped rule (outline + mouth/chin zone): 0.00 % false cut on 60 clean cases, 100 % of the scoop found |
| `make_food2.py SPEC...`, `run_food2.sh PACK TAG SPEC`, `foodmetrics2.py LABEL...` | scoop clips (L / R cheek, M mouth), real pipeline, share of the scoop still painted over |
| `enh_test.py` | old vs new Cinematic enhancer: time per crop, change, shadows, fine detail, colour |

App tests (`mobile/verification/`): `UploadTest.kt` + `server.py` (the Space adapter's real upload routes) + `flaky.py` (a connection that cuts and stalls); `FaceLibTest.kt` (saved-face storage).
