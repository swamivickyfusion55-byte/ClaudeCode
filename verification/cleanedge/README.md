# CleanEdge verification (SDOS-084)

Synthetic objects on real photographs (the six faces of insightface's sample photo, `t1.jpg`) and synthetic clips made from that photo. **No user footage.**
The scripts carry the absolute paths of the sandbox they were written in (scratchpad, `/tmp/claude-0/...`); change them to your checkout.
The end-to-end scripts need insightface + buffalo_l + `inswapper_128.onnx` and the Phoenix pack directory as an argument.
**Run end-to-end jobs one at a time**: two `bench_pipeline.py` jobs at once corrupt each other's output video.

## The measurement that found the problem

| Script | What it does |
|---|---|
| `food.py` | A realistic scoop: shaded, textured, specular, anti-aliased, with a waffle cone running to the crop border. The flat discs of SDOS-074..083 hid the problem |
| `harness.py PACK SIZE NFACES FOODS` | The real `AlignedCompositor.occluder_keep_final` (temporal smoothing, snap, steepen) on real-face crops with `food.py` objects: detection, interior face alpha, arrival, edge position error, halo, flicker, over-cut near the object. `OVR="occ_gc=0,occ_cont=0,occ_open_near=0,occ_bias=0"` gives the v11.2.84 behaviour |
| `series.py`, `viz2.py`, `dbgdrop.py` | per-frame cut share, pictures of the keep map with the ground-truth outline, raw-vs-final detection per frame (the on/off flicker) |
| `ghost.py` | the face is fully back within 4 frames after an object vanishes (continuity must not leave a stuck cut) |
| `rb85.py` | the false-cut battery (9 perturbation tests × 12 real faces × seeds) on the final keep map |
| `rollback.py` | with every new dial at its off value the mask is bit-identical to v11.2.84 |
| `unit85.py` | cheek zone, cheek smoothing step response, graph-cut safety rules |
| `cost85.py` | median time per call at 256 and 512 px |

## End to end (real detector, swapper, compositor)

| Script | What it does |
|---|---|
| `make_food.py` | 3x-enlarged crop of the sample photo (face ≈ 320 px) with the `food.py` scoop held at the mouth; writes the ground-truth alpha |
| `foodmetrics.py LABEL` | change inside the object against the input footage, share of visibly changed pixels, edge profile |
| `make_cheeks.py` | the cheeks of the centre face puff out and chew (bulge warp) |
| `cheekmetrics.py`, `jerk.py`, `fid.py` | frame-to-frame jerk of what the swap adds in the cheek band, mask width in the cheek rows, closeness to the Ultra render |
| `cheekdbg.py` | dumps the aligned target, raw swap, toned swap and mask of chosen frames (this is how the dark smudge on the swollen cheek was found in the raw swap output) |
| `bench_pipeline.py`, `runq.sh`, `run_food.sh`, `run_cheek*.sh` | run `core_pipeline._run_job_body` inside a 2-vCPU cgroup; `MASK_DUMP=1` stores every mask; `OVR="key=value,..."` overrides engine tunables |

Note: the synthetic scripts from SDOS-081/082/083 (`../lipcover/`) still apply; `t_occ_final.py`-style "N FAILURES" counts late or missed must-cut frames (spoon, strawberry), not must-keep violations.
