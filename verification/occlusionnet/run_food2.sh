#!/bin/bash
# usage: run_food2.sh PACK TAG SPEC [QUALITY] -> label fd_<TAG>_<SPEC> (2-slot job, like the earlier food runs)
cd /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench
PACK=$1; TAG=$2; SPEC=$3; Q=${4:-Balanced}
PHOENIX_MODEL_DIR=/tmp/claude-0/occmodels MASK_DUMP=1 OVR="$OVR" ./runq.sh python3 bench_pipeline.py $PACK fd_${TAG}_${SPEC} clip=food_${SPEC}.mp4 "face_mode=1 face (fastest)" quality=$Q > /tmp/claude-0/bench_work_fd_${TAG}_${SPEC}.log 2>&1
echo d > fd_${TAG}_${SPEC}.done
