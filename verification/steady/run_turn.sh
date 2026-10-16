#!/bin/bash
# usage: run_turn.sh PACK TAG CLIP QUALITY  -> label ts_<TAG>_<CLIP>_<Q>
cd /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench
PACK=$1; TAG=$2; CLIP=$3; Q=$4
MASK_DUMP=1 OVR="$OVR" ./runq.sh python3 bench_pipeline.py $PACK ts_${TAG}_${CLIP}_${Q} clip=turn_${CLIP}.mp4 "face_mode=1 face (fastest)" quality=$Q smooth="${SM:-Off}" > /tmp/claude-0/bench_work_ts_${TAG}_${CLIP}_${Q}.log 2>&1
echo d > ts_${TAG}_${CLIP}_${Q}.done
