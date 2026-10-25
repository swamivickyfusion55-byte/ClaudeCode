# FastRestore verification (SDOS-092)

Synthetic clips and public test photographs only (insightface sample photo, AFLW2000 samples from the PRNet repo,
the 3DDFA repo's sample photo). **No user footage.** Paths are those of the sandbox the scripts were written in.

| Script | What it does |
|---|---|
| `unit92.py PACK` | 21 checks: side-view weight, outline mask, eating rule (open mouth, fade in/out), GPEN strength / skipping / sharpness cap / steady detail, enhancer names |
| `yaw_calib.py` | the side-view measure (nose offset from the eye midpoint) against AFLW2000's true yaw |
| `profile_test.py PACK OUT` | single photos, real detector + swapper + compositor: paint off the face / on the face (BiSeNet face area) |
| `make_profclip.py`, `profclip_metrics.py` | the side-view photos as moving clips; paint off the face and mask steadiness per frame |
| `mask_equiv.py` | v11.2.91 and v11.2.92 masks on 43 faces: identical except side views and the mouth rule |
| `make_bite.py`, `bitemetrics.py` | scoop pressed against the lower lip; which lip (footage / swapped) shows |
| `mouth_false.py PACK` | the eating rule on 42 clean faces, closed and with a synthetic open mouth |
| `make_sharpclip.py`, `restore_metrics.py` | sharp footage; Cinematic vs GPEN: sharpness, shimmer, flicker, identity |

App: `mobile/verification/BusyTest.kt` + `busy_server.py` (the app's busy check against the Space's real routes, v11.2.92 and v11.2.91).
