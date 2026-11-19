#!/bin/bash
R=/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bar214; P3=/home/user/ClaudeCode/v11.2.213_TrueTurn; P4=/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/snap214f
taskset -c 2,3 $R/run1.sh $P4 b214_none $R/bar_none.mp4
taskset -c 2,3 $R/run1.sh $P4 b214_34_choc $R/bar34_choc.mp4
taskset -c 2,3 $R/run1.sh $P3 o213fast_h_cream $R/barh_cream.mp4 quality=Fast
taskset -c 2,3 $R/run1.sh $P4 o214fast_h_cream $R/barh_cream.mp4 quality=Fast
echo done > $R/sJ.done
