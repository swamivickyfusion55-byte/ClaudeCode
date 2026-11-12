#!/bin/bash
# run_one.sh PACK LABEL CLIPNAME [extra key=value ...]  - the real job on a contact clip, then the identity check
PACK=$1; LABEL=$2; CLIP=$3; shift 3
cd /home/user/ClaudeCode/verification/fastrestore
PHOENIX_MODEL_DIR=/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/face/models MASK_DUMP=1 BENCH_UNIQUE_JID=1 /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/face/venv311/bin/python -I bench_pipeline.py $PACK $LABEL clip=/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/kiss_$CLIP.mp4 threads=2 "$@" > /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/$LABEL.bench.log 2>&1
OUT=$(ls -t /tmp/claude-0/bench_work/$LABEL/out/*/final.mp4 /tmp/claude-0/bench_work/$LABEL/out/*.mp4 2>/dev/null | head -1)
[ -z "$OUT" ] && OUT=$(ls -t /tmp/result_BENCH*.mp4 2>/dev/null | head -1)
echo "OUT=$OUT" > /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/$LABEL.id.log
/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/face/venv311/bin/python -I identity_check.py "$OUT" /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/kiss_$CLIP.mp4 /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/kiss_$CLIP\_gt.npz src_a.jpg src_b.jpg /tmp/claude-0/bench_work/$LABEL/mask_dump.npz >> /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/$LABEL.id.log 2>&1
grep -h "REVERTS\|FACE-EVENTS\|Done in" /tmp/claude-0/bench_work/$LABEL/job.log | tail -3 >> /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/$LABEL.id.log
/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/face/venv311/bin/python -I contact_lag.py /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/kiss_$CLIP\_gt.npz /tmp/claude-0/bench_work/$LABEL/mask_dump.npz >> /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/$LABEL.id.log 2>&1
echo finished >> /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/$LABEL.id.log
