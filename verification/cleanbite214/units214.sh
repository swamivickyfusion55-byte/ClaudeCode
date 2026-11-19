#!/bin/bash
S=/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad; P=/home/user/ClaudeCode/v11.2.214_CleanBite; V=$S/face/venv311/bin/python; X=$S/face/models/xseg_2.onnx; H=/home/user/ClaudeCode
cd $H/verification/fastrestore
{
echo "== unit214 (v11.2.213 against v11.2.214; the real occlusion net on the synthetic bar clips)"; $V -I unit214.py $H/v11.2.213_TrueTurn $P $X $S/bar214 2>&1 | grep -E "^PASS|^FAIL|checks|failed|Error"
echo "== unit213 (v11.2.212 against v11.2.214)"; HEAD=1 $V -I unit213.py $H/v11.2.212_TruePerson $P 2>&1 | grep -E "^PASS|^FAIL|checks|failed"
 echo "== unit212 (v11.2.211 against v11.2.214)"; $V -I unit212.py $H/v11.2.211_Steady $P 2>&1 | grep -E "^PASS|^FAIL|checks|failed"
echo "== unit211 (v11.2.210 against v11.2.214)"; $V -I unit211.py $H/v11.2.210_TrueTouch $P 2>&1 | grep -E "^PASS|^FAIL|checks|failed|Error"
echo "== unit210 (v11.2.209 against v11.2.214)"; $V -I unit210.py $H/v11.2.209_SureID $P $S/bench/lick_choc.mp4 $S/bench/kiss_pkiss.mp4 $X 2>&1 | grep -E "^PASS|^FAIL|checks|failed|Error"
echo "== unit209"; $V -I unit209.py $P 2>&1 | grep -E "^PASS|^FAIL|checks|failed|Error"
for c in hand_eye cup_cheek; do echo "== unit206 $c (v11.2.205 against v11.2.214)"; $V -I unit206.py $H/v11.2.205_SkinTone $P $S/bench/obj_$c.mp4 $X 2>&1 | grep -E "^PASS|^FAIL|checks|failed|Error"; done
echo "== unit205"; rm -rf $S/u205w214; mkdir -p $S/u205w214; $V -I $H/verification/skintone/unit205.py $P $S/u205w214 2>&1 | grep -E "^PASS|^FAIL|checks|failed|Error" | tail -40
for u in unit204 unit176 unit176_pipe; do echo "== $u"; $V -I $u.py $P 2>&1 | grep -E "^PASS|^FAIL|checks|failed|Error"; done
echo "== unit203"; rm -rf $S/u203w214; mkdir -p $S/u203w214; $V -I unit203.py $P $S/u203w214 2>&1 | grep -E "^PASS|^FAIL|checks|failed|Error"
echo "== nofalse206 (v11.2.205 against v11.2.214)"; $V -I nofalse206.py $H/v11.2.205_SkinTone $P $X /usr/local/lib/python3.11/dist-packages/insightface/data/images/t1.jpg 2>&1 | grep -vE "Warning|warn|estimate|Applied providers|find model|model ignore|set det-size"
echo UNITS_DONE
} > $S/units214.txt 2>&1
