#!/bin/bash
cd /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench
PACK=$1; TAG=$2; shift 2
for c in "$@"; do
  MASK_DUMP=1 OVR="$OVR" ./runq.sh python3 bench_pipeline.py $PACK fd_${TAG}_${c} clip=food_$c.mp4 "face_mode=1 face (fastest)" quality=Balanced > /tmp/claude-0/bench_work_fd_${TAG}_${c}.log 2>&1
done
echo ALLDONE > fd_${TAG}.done
