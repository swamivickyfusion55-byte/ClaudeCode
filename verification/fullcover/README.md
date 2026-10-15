# FullCover verification (SDOS-085)

Synthetic objects on real photographs (the six faces of insightface's sample photo) and synthetic clips from it. **No user footage.**
Scripts carry the sandbox's absolute paths (`/tmp/claude-0/...`); change them to your checkout. Run end-to-end jobs one at a time.

| Script | What it does |
|---|---|
| `food.py` | the realistic scoop (shaded, textured, highlight, cone); `ang` = cone direction, `parts=True` also returns the scoop alone |
| `harness.py PACK SIZE NFACES FOODS` | `occluder_keep_final` on real-face crops; `POS="cx,cy,angle"` puts the object at a cheek (`0.27,0.60,125` left, `0.73,0.60,55` right, `0.5,0.76,65` mouth); `OVR="key=value,..."` overrides dials (all new dials at 0 = 11.2.85) |
| `series.py`, `viz3.py`, `miss.py`, `stages.py` | per-frame cut share, pictures with the whole-object outline, where the missed pixels are, per-stage coverage (not skin / new / both) |
| `ghost.py`, `ghost2-4.py` | the face must come back within a few frames after an object vanishes; what stays cut and why (model drift) |
| `rb85.py` | false-cut battery on the final keep map (`SEEDS="31,32,33"`) |
| `rollback86s.py` | with the new dials off the mask equals 11.2.85 bit for bit (needs a copy of 11.2.85 with the graph-cut seed line) |
| `unit86.py` | extension limit, nearest-colour test, held statistics, learning hold, frozen model, attached pieces |
| `cost86.py`, `moving_ovr.py` | cost per call; the moving-scoop test with dial overrides |
| `make_food2.py`, `foodmetrics2.py`, `run_food.sh` | end-to-end clips (`name_L/R/M`) at a realistic face size and the change inside the scoop against the input |
