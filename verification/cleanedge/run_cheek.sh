#!/bin/bash
# usage: run_cheek.sh PACKDIR TAG quality...
cd /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench
PACK=$1; TAG=$2; shift 2
for q in "$@"; do for c in static swell; do
  MASK_DUMP=1 OVR="$OVR" ./runq.sh python3 bench_pipeline.py $PACK ck_${TAG}_${q}_${c} clip=cheek_$c.mp4 "face_mode=1 face (fastest)" quality=$q > /tmp/claude-0/bench_work_ck_${TAG}_${q}_${c}.log 2>&1
done; done
echo ALLDONE > ck_${TAG}.done
