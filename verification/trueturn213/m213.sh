#!/bin/bash
S=/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad; R=$S/side213; B=$S/bench; P2=/home/user/ClaudeCode/v11.2.212_TruePerson; P3=/home/user/ClaudeCode/v11.2.213_TrueTurn
(
taskset -c 0,1 $R/run1.sh $P3 t213_turn110 $R/turn110.mp4
taskset -c 0,1 $R/run1.sh $P2 t212_turn110deg $R/turn110deg.mp4
taskset -c 0,1 $R/run1.sh $P3 t213_turn110deg $R/turn110deg.mp4
taskset -c 0,1 $R/run1.sh $P3 p213_02420 $R/prof_image02420.mp4
taskset -c 0,1 $R/run1.sh $P3 p213_00475 $R/prof_image00475.mp4
taskset -c 0,1 $R/run1.sh $P2 l212_choc $B/lick_choc.mp4
taskset -c 0,1 $R/run1.sh $P3 l213_choc $B/lick_choc.mp4
) &
(
taskset -c 2,3 $R/run1.sh $P3 t213_turnL $R/turnL.mp4
taskset -c 2,3 $R/run1.sh $P2 t212_turnLdeg $R/turnLdeg.mp4
taskset -c 2,3 $R/run1.sh $P3 t213_turnLdeg $R/turnLdeg.mp4
taskset -c 2,3 $B/run_two.sh $P2 r212_pkiss pkiss
taskset -c 2,3 $B/run_two.sh $P3 r213_pkiss pkiss
taskset -c 2,3 $B/run_two.sh $P3 g213_smooch smooch
) &
wait; echo done > $R/m213.done
