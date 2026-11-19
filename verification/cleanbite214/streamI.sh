#!/bin/bash
R=/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bar214; P3=/home/user/ClaudeCode/v11.2.213_TrueTurn; P4=/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/snap214f
taskset -c 0,1 $R/run1.sh $P4 b214_h_choc $R/barh_choc.mp4
taskset -c 0,1 $R/run1.sh $P3 o213opt_none $R/bar_none.mp4 quality=Optimized
taskset -c 0,1 $R/run1.sh $P3 o213opt_h_cream $R/barh_cream.mp4 quality=Optimized
taskset -c 0,1 $R/run1.sh $P4 o214opt_h_cream $R/barh_cream.mp4 quality=Optimized
echo done > $R/sI.done
