#!/bin/bash
# usage: run_gen.sh PACK TAG CLIPFILE QUALITY  -> label tg_<TAG>_<clip stem>_<Q> (any clip, 2-slot job)
cd /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench
PACK=$1; TAG=$2; CLIPF=$3; Q=$4; STEM=$(basename $CLIPF .mp4)
MASK_DUMP=1 OVR="$OVR" ./runq.sh python3 bench_pipeline.py $PACK tg_${TAG}_${STEM}_${Q} clip=$CLIPF "face_mode=2 faces" quality=$Q smooth=Off > /tmp/claude-0/bench_work_tg_${TAG}_${STEM}_${Q}.log 2>&1
echo d > tg_${TAG}_${STEM}_${Q}.done
