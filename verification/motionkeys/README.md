# MotionKeys verification (SDOS-090)

Synthetic clips from real photographs. **No user footage.** Paths are those of the sandbox the scripts were written in.

| Script | What it does |
|---|---|
| `unit90.py [PACK]` | 13 checks of `_FaceMotionKeys` (still, drift, sweep, turn, travel, gap, cap, held geometry, degenerate input) |
| `make_turn.py` | fast sweep / roll / squeeze clips with the true key points |
| `run_turn.sh`, `run_gen.sh` | run a clip through the real pipeline (2-face job; `OVR=` tunables, `XARGS=one=1` for one slot) |
| `turnmetrics.py`, `flicker.py` | placement against the truth, missing swaps, sharpness; flicker of the pasted face in its own frame |
| `lowlight_det.py` | detector recall and landmark error in synthetic low light, with and without an analysis-only enhanced copy (the SDOS-089 F-04 test) |
