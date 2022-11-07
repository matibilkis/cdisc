#!/bin/bash
indgamma=$1
cd ~/cdisc
. ~/qenv_bilkis/bin/activate
START=$(date +%s.%N)
python3 analysis/analysis_freq/main_likelihood.py --Ntraj 10
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $DIFF
deactivate
