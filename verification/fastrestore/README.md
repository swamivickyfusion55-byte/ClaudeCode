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

## v11.2.141 ContactOrder (SDOS-141) - identity and painting through contact

| Script | What it does |
|---|---|
| `make_cross.py` | two contact clips where the people end on the other side: `cross` (A walks behind B) and `hug` (cheek to cheek, half a turn) |
| `identity_check.py OUT CLIP GT SRC_A SRC_B [MASK_DUMP]` | per person and frame: which replacement is on the face (ArcFace on the person's own visible pixels at the true 5 points; mix-up), and how much paint covers the visible face (the engine's paste mask over the true 106-point outline; revert / weak) |
| `geom_offset.py GT DUMP...` | how far each run's paint crop sits from the person's true face, frame by frame |
| `unit141.py PACK` | the ordering rule (a face not seen on a frame is never in front) and the app's per-person gender values |

`bench_pipeline.py` takes `pg=0,1` (the app's M / F per slot). Models for the bench: InsightFace buffalo_l (GitHub release v0.7),
xseg_2 and inswapper_128 from FaceFusion's GitHub assets (huggingface.co was not reachable from this sandbox).

`src_a.jpg`, `src_b.jpg`: the bench's two replacement faces - faces f1 and f4 of InsightFace's own sample photo (`insightface/data/images/t1.jpg`), cropped and enlarged 3x; the contact clips use faces f2 and f3 of the same photo.
