# LipCover verification (SDOS-081)

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
