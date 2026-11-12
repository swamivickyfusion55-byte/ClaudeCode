#!/bin/bash
export ORDER_DUMP=1
P92=/home/user/ClaudeCode/v11.2.92_FastRestore
P205=/home/user/ClaudeCode/v11.2.205_SkinTone
taskset -c 0,1 bash -c "for c in step2 pkiss cross hide smooch; do /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/run_two.sh /home/user/ClaudeCode/v11.2.92_FastRestore c92_\$c \$c; done" &
taskset -c 2,3 bash -c "for c in step2 pkiss cross hide smooch; do /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/run_two.sh /home/user/ClaudeCode/v11.2.205_SkinTone c205_\$c \$c; done" &
wait
echo done > /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/cmp92.done
