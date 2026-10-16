# Steady verification (SDOS-088)

Synthetic clips from real photographs. **No user footage.** Paths are those of the sandbox the scripts were written in.

| Script | What it does |
|---|---|
| `make_turn.py` | one face that sweeps, rolls and squeezes (fake yaw) fast, with the true key points (`turn_<name>_gt.npz`) |
| `run_turn.sh PACK TAG CLIP QUALITY` | runs a clip through the real pipeline (`MASK_DUMP=1`; env `SM` = smooth motion, `OVR` = tunables) |
| `turnmetrics.py LABEL` | placement error against the true geometry, missing swaps, sharpness |
| `flicker.py LABEL` | flicker of the pasted face measured in its own aligned frame (head motion removed), per quality |
| `turns.py PACK` | false cut of the occluder gate while an aligned real face squeezes (turns) over a plain uncovered background (`FILL=wall_bright|wall_dark|green|blue|skin|red`, `YAWS=`) |
