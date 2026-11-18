#!/bin/bash
# identity of every man-woman run against the INTENDED mapping (slot 0 = the woman: src_a normally, src_b with swapsrc=1)
S=/tmp/claude-0/-home-user-ClaudeCode/53ae7b21-e3b3-5c92-ac7e-facd0947dd55/scratchpad; B=$S/bench; V=$S/face/venv311/bin/python
cd /home/user/ClaudeCode/verification/fastrestore
for L in "$@"; do
  case $L in
    *xg_kissb*) C=mw_kissb; I=0,1;; *kissb*) C=mw_kissb; I=1,0;;
    *xg_crosspart*) C=mw_crosspart; I=1,0;; *crosspart*) C=mw_crosspart; I=0,1;;
    *xg_cross*) C=mw_cross; I=1,0;; *cross*) C=mw_cross; I=0,1;;
    *hug*) C=mw_hug; I=0,1;; *) continue;;
  esac
  OUT=$(head -1 $B/$L.id.log | sed 's/^OUT=//')
  echo "== $L (intended: A photo $(echo $I | cut -c1 | tr 01 AB), B photo $(echo $I | cut -c3 | tr 01 AB))"
  IDC_INTENDED=$I $V -I identity_check.py "$OUT" $B/kiss_$C.mp4 $B/kiss_${C}_gt.npz src_a.jpg src_b.jpg /tmp/claude-0/bench_work/$L/mask_dump.npz 2>&1 | grep "^A (\|^B (" | sed 's/ | own sim.*//'
done
