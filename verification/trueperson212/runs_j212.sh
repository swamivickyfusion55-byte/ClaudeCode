#!/bin/bash
# v11.2.212 (frozen212b) identity runs on the man-woman clips; slot 0 = the woman, pg = the app's genders; swapsrc=1: photos across genders
P1=/home/user/ClaudeCode/v11.2.211_Steady; P2=/home/user/ClaudeCode/v11.2.212_TruePerson; R=/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bench
(
taskset -c 0,1 $R/run_two.sh $P2 g212_mwkissb_noref mw_kissb pg=0,1
taskset -c 0,1 $R/run_two.sh $P2 g212_mwkissb_ref2 mw_kissb pg=0,1 refs=$R/kiss_mw_kissb_refB.jpg,$R/kiss_mw_kissb_refA.jpg
taskset -c 0,1 $R/run_two.sh $P2 g212_mwhug_ref mw_hug pg=0,1 refs=$R/kiss_mw_hug_refA.jpg,$R/kiss_mw_hug_refB.jpg
taskset -c 0,1 $R/run_two.sh $P1 b211_xg_kissb_noref mw_kissb pg=0,1 swapsrc=1
taskset -c 0,1 $R/run_two.sh $P2 g212_xg_kissb_noref mw_kissb pg=0,1 swapsrc=1
taskset -c 0,1 $R/run_two.sh $P1 b211_mwcross_ref mw_cross pg=0,1 refs=$R/kiss_mw_cross_refA.jpg,$R/kiss_mw_cross_refB.jpg
taskset -c 0,1 $R/run_two.sh $P2 g212_mwcross_ref mw_cross pg=0,1 refs=$R/kiss_mw_cross_refA.jpg,$R/kiss_mw_cross_refB.jpg
taskset -c 0,1 $R/run_two.sh $P2 g212_smooch smooch
) &
(
while pgrep -f "b211_mwcrosspart_ref" >/dev/null; do sleep 15; done
taskset -c 2,3 $R/run_two.sh $P2 g212_mwcrosspart_ref mw_crosspart pg=0,1 refs=$R/kiss_mw_cross_refA.jpg,$R/kiss_mw_cross_refB.jpg
taskset -c 2,3 $R/run_two.sh $P1 b211_xg_crosspart_noref mw_crosspart pg=0,1 swapsrc=1
taskset -c 2,3 $R/run_two.sh $P2 g212_xg_crosspart_noref mw_crosspart pg=0,1 swapsrc=1
taskset -c 2,3 $R/run_two.sh $P1 b211_mwcrosspart_noref mw_crosspart pg=0,1
taskset -c 2,3 $R/run_two.sh $P2 g212_mwcrosspart_noref mw_crosspart pg=0,1
taskset -c 2,3 $R/run_two.sh $P1 b211_smooch smooch
) &
wait; echo done > $R/j212.done
