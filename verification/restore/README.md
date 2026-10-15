# Restore verification (SDOS-087)

`cleancmp.py CLIP...` (env `PACKS=pack83,pack88,...`): composites each pack with the real compositor on real detections of synthetic clips (no object) and reports how much each differs from the first pack inside the face boxes. v11.2.88 vs v11.2.83: 0 on every frame; v11.2.84/87 vs v11.2.83 on a kissing clip: up to 21-28 % of the face box changed in the worst frame. Paths are those of the sandbox it was written in.
