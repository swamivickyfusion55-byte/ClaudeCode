#!/bin/bash
cd /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench
PACK=$1; TAG=$2; Q=$3; C=$4
MASK_DUMP=1 OVR="$OVR" ./runq.sh python3 bench_pipeline.py $PACK ck_${TAG}_${Q}_${C} clip=cheek_$C.mp4 "face_mode=1 face (fastest)" quality=$Q > /tmp/claude-0/bench_work_ck_${TAG}_${Q}_${C}.log 2>&1
echo d > ck_${TAG}_${Q}_${C}.done
