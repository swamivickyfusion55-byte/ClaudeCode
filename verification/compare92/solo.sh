#!/bin/bash
# timing only: each version alone on the same two cores, one after the other (nothing else running)
until [ -f /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/cmp92.done ]; do sleep 20; done
export ORDER_DUMP=1
taskset -c 0,1 /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/run_two.sh /home/user/ClaudeCode/v11.2.92_FastRestore t92_step2 step2
taskset -c 0,1 /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/run_two.sh /home/user/ClaudeCode/v11.2.205_SkinTone t205_step2 step2
echo done > /tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench/solo.done
