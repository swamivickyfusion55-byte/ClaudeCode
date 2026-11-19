#!/bin/bash
R=/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bar214; B=/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench; P3=/home/user/ClaudeCode/v11.2.213_TrueTurn; P4=/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/snap214e
taskset -c 0,1 $R/run1.sh $P4 b214_h_choc $R/barh_choc.mp4
taskset -c 0,1 $R/run1.sh $P4 b214_h_cream $R/barh_cream.mp4
taskset -c 0,1 $R/run1.sh $P4 b214_34_choc $R/bar34_choc.mp4
taskset -c 0,1 $R/run1.sh $P3 b213_34_choc $R/bar34_choc.mp4
taskset -c 0,1 $R/run1.sh $P3 b213_hs_deg $R/barh_strawberry_deg.mp4
taskset -c 0,1 $R/run1.sh $P4 b214_hs_deg $R/barh_strawberry_deg.mp4
taskset -c 0,1 $R/run1.sh $P4 o214_hand_sweep $B/obj_hand_sweep.mp4
taskset -c 0,1 $R/run1.sh $P3 o213_hand_sweep $B/obj_hand_sweep.mp4
echo done > $R/sG.done
