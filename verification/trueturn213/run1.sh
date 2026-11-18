#!/bin/bash
# run1.sh PACK LABEL CLIP.mp4 [k=v...]: the real job, one face, at Ultra
PACK=$1; LABEL=$2; CLIP=$3; shift 3
S=/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad
cd /home/user/ClaudeCode/verification/fastrestore
PHOENIX_MODEL_DIR=$S/face/models MASK_DUMP=1 BENCH_UNIQUE_JID=1 $S/face/venv311/bin/python -I bench_pipeline.py $PACK $LABEL clip=$CLIP threads=2 one=1 "face_mode=1 face" "$@" > $S/side213/$LABEL.bench.log 2>&1
echo finished >> $S/side213/$LABEL.bench.log
