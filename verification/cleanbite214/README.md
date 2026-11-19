# Evidence for SDOS-214 (v11.2.214 CleanBite)

Synthetic and public test media only (AFLW2000 sample photos; no user footage). Clips are made by `../fastrestore/make_bar.py`
(open mouth, a drawn bar; `PHOTO=image02616 FLIP=1 ... 90` for the 38 deg smile), `../fastrestore/make_bar_side.py` (profile photos,
the bar held to the lips) and `../fastrestore/degrade_clip.py` (phone-quality). Ground truth (`_gt.npz`) comes with each clip.

| file | what |
|---|---|
| `probe3.py`, `probe3_final_table.txt` | the swap network's own output (128-px crop) with the bar in its input / the engine's erase / the true bar erased, against the ideal (its output for the bar-free frame) - SDOS §1.1 |
| `erase_probe.py`, `er_dbg.py`, `sheet_erase_mask_cream_input_overlay_result.png` | which pixels the erase takes (input, red overlay, result) |
| `erase_series.py` | the erase on every frame of a clip: area taken, share of the bar inside the outline covered (0.94-0.95 mean), the open mouth's cavity touched (0.000), frames where the bar is in the mouth and nothing is taken (the first 3 and last 2), false alarms (0 of 150 on the bar-free clip) |
| `final_err.py`, `job_numbers.txt` | the whole job at Ultra, output against the ideal around the mouth, 213 against 214 - SDOS §3.2 |
| `same.py` | pixel comparison of two job outputs (bar-free clip, profile clip, Fast tier: identical to 11.2.213) |
| `erase_cost.py` | time of the erase per face-frame at 720p, 1080p, 4K |
| `scan_open34.py` | the open-mouth photos at a 30-65 deg turn in the public AFLW2000 sample (the 38 deg photo comes from it) |
| `sheet_swap_input_bar_in_or_out.png` | what the swap network draws with the bar in its input, with it taken out, and the ideal (chocolate, cream, strawberry) |
| `sheet_job_*.png` | input / 11.2.213 / 11.2.214 / ideal around the mouth, from the real job at Ultra |
| `sheet_lick_in_213_214.png`, `sheet_handsweep_in_213_214.png` | regression clips of SDOS-210 and SDOS-206 (an ice-cream scoop, a hand sweeping the face) |
| `units214.sh`, `unit_suites.txt` | every unit suite on the 11.2.214 files |
| `streamG.sh` ... `streamL.sh`, `run1.sh` | how the jobs were run (two at a time, 2 cores each) |
