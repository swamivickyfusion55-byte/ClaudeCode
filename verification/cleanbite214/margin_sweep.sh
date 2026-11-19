#!/bin/bash
# margin_sweep.sh DILATE : probe3 on the other bar clips with one erase margin
R=/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/bar214; V=/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad/face/venv311/bin/python; D=$1
cd $R
for spec in "barh_cream.mp4 bar_none.mp4 0" "barh_strawberry.mp4 bar_none.mp4 0" "barv_strawberry.mp4 bar_none.mp4 0" "bar_cream.mp4 bar_none.mp4 0" "bar34_choc.mp4 bar34_none.mp4 9"; do
  set -- $spec
  ERASE_YAW=$( [ "$3" = "9" ] && echo 9 || echo 0.34 ) PJ="{\"obj_erase_dilate\":$D}" nice $V -I $R/probe3.py /home/user/ClaudeCode/v11.2.214_CleanBite $R/$1 $R/$2 45 105 30 2>&1 | grep -E "mean abs|Error|Trace" | sed "s/^/dilate $D: /"
done
