#!/bin/bash
R=/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bar214; B=/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench; P3=/home/user/ClaudeCode/v11.2.213_TrueTurn; P4=/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/snap214e
taskset -c 2,3 $R/run1.sh $P4 b214_none $R/bar_none.mp4
taskset -c 2,3 $R/run1.sh $P4 b214_h_straw $R/barh_strawberry.mp4
taskset -c 2,3 $R/run1.sh $P4 g214_h_choc $R/barh_choc.mp4 'enh=GPEN + Cinematic'
taskset -c 2,3 $R/run1.sh $P4 b214_34_none $R/bar34_none.mp4
taskset -c 2,3 $R/run1.sh $P3 b213_none_deg $R/bar_none_deg.mp4
taskset -c 2,3 $R/run1.sh $P4 l214_choc $B/lick_choc.mp4
taskset -c 2,3 $R/run1.sh $P4 s214_475_cream $R/side_image00475_cream.mp4
taskset -c 2,3 $R/run1.sh $P4 o214opt_h_cream $R/barh_cream.mp4 quality=Optimized
echo done > $R/sH.done
