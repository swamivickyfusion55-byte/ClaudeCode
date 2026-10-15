#!/bin/bash
cd /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench
PACK=$1; TAG=$2; CLIP=$3; Q=${4:-Ultra}
MASK_DUMP=1 OVR="$OVR" ./runq.sh python3 bench_pipeline.py $PACK ks_${TAG}_${CLIP} clip=kiss_${CLIP}.mp4 quality=$Q > /tmp/claude-0/bench_work_ks_${TAG}_${CLIP}.log 2>&1
echo d > ks_${TAG}_${CLIP}.done
