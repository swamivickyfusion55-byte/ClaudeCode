# TrueEdge verification (SDOS-086)

Synthetic objects and synthetic two-person contact clips made from real photographs (insightface's sample photo `t1.jpg`). **No user footage.**
The scripts carry the absolute paths of the sandbox they were written in (scratchpad, `/tmp/claude-0/...`); change them to your checkout.
End-to-end scripts need insightface + buffalo_l + `inswapper_128.onnx` and the Phoenix pack directory as an argument. **Run end-to-end jobs one at a time.**

| Script | What it does |
|---|---|
| `unit87.py [PACK]` | 26 checks: frame-space edge refinement (too-narrow and too-wide cuts, same colours, darker rim inside the object, degenerate input), depth order (key-point rule, jitter, reversal after 6 frames, forgetting, paint order, `_rival_landmarks`), size hold during contact |
| `rollback87.py [SPEC...]` | with `occ_img_refine`, `depth_order`, `contact_scale_lock` at 0 every composited frame equals v11.2.86 bit for bit (real compositor, cached detections of the food clips) |
| `cost87.py` | time of `_refine_keep_img` with and without an object |
| `imgharness.py PACK SPEC [size]` | image-domain profile of the painted face across the object's outline (real TrackState / build_mask / paste_back, a fixed second identity as the swap) |
| `profile.py LABEL...` | the same profile from full pipeline jobs (`bench_pipeline.py`); `../fullcover/foodmetrics2.py` gives the change inside the scoop |
| `make_kiss.py` | five two-person contact clips (`cheek`, `near`, `deep`, `hide`, `flip`) with the true face masks and key points |
| `run_kiss.sh PACK TAG CLIP [quality]`, `kissmetrics.py LABEL...` | run a contact clip through the real pipeline (`MASK_DUMP=1`, `OVR=` overrides); reverted frames, jitter, placement error and size of each person against the truth |
