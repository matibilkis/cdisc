#!/bin/bash
itraj=$1
cd ~/cdisc
. ~/qenv_bilkis/bin/activate
START=$(date +%s.%N)
python3 mp_run_freq.py --itraj $itraj
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $DIFF
deactivate
