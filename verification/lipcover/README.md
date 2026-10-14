# LipCover / ObjectLock / RealFaces verification (SDOS-081, 082, 083)

Synthetic faces only (`face_syn.py`); no user footage. Run each script with the pack directory as the first argument
(`diag*.py`, `neg_seed.py`, `onset.py`); `t_occ_final.py` has the pack path on its line 2 (`sys.path.insert`).
Scripts load `face_syn.py` from the current directory.

| Script | What it shows |
|---|---|
| `t_occ_final.py` | SDOS-074 regression: 8 must-keep sequences x 3 seeds must stay at 0.0 % cut |
| `diag.py`, `diag2.py` | where the old gate cuts food at the lips (large / small scoops, 5 colours) |
| `diag3.py` | big scoop at the mouth: lower-lip cut before/after (the reported symptom) |
| `onset.py` | a small scoop sits at the lips, then the whole scoop arrives (learned-at-onset hole) |
| `neg_seed.py` | an unrelated object is present while the mouth talks: lips must not be cut |
| `realgate*.py`, `realdbg.py`, `sigma.py` | the gate on REAL aligned crops of the insightface sample photo (needs insightface + buffalo_l); shows the skin-spread finding in SDOS-081 §5 |
| `moving.py`, `moving_sweep.py` | a scoop moves to the mouth, holds, leaves: face opacity left over its inside / edge band / next to it (SDOS-082) |
| `leak.py` | 7 colours x 3 sizes: part of the object cut, the ring around it, the face far away (no leak) |
| `blink.py` | cost of a one-frame false detection vs the hole opening speed |
| `viz_moving.py` | writes `moving_alpha.png`, the face opacity over the moving scoop as a heat map |

## RealFaces (SDOS-083): the gate on real faces

All of these load the six faces of the insightface sample photo (needs insightface + buffalo_l; run with plain `python3`, from this directory, with the pack directory as the first argument). They need `realset.py` (face loader, `jitter`, `scoop`, `COLS`).

| Script | What it shows |
|---|---|
| `realset.py` | loader of the six aligned real-face crops, the synthetic scoop and the colour list |
| `realdetect.py` | 7 object colours x 6 real faces: share of the object cut + false cut on clean sequences (11.2.83: 0 of 42 combinations; 11.2.84: 32 of 42) |
| `realbattery.py`, `realbattery_hard2.py` | calibration sweeps of the dials (`occ_tol`, `occ_open_first`, `occ_blob`...) on perturbation sequences |
| `realbattery_final.py` | the final check: 3 new seeds x 12 faces (6 + mirror images) x 9 perturbation tests + 7 colours |
| `rb_var.py` | the same battery with dial overrides (edit the list) |
| `cost.py` | median time of one `occluder_keep` call at 128/256/512 px, with and without an object (run it once per pack) |

Note: `t_occ_final.py` and the other synthetic scripts print `N FAILURES` at the end; that is the count of must-cut frames reported late or missed (spoon, strawberry...), not must-keep violations. The must-keep rows must all read 0.0.
