#!/bin/bash
gamma=$1
cd ~/cdisc
. ~/qenv_bilkis/bin/activate
START=$(date +%s.%N)
python3 analysis/main_likelihood.py --gamma $gamma
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $DIFF
deactivate
