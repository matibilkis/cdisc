#!/bin/bash
indgamma=$1
cd ~/cdisc
. ~/qenv_bilkis/bin/activate
START=$(date +%s.%N)
python3 analysis/main_likelihood.py --indgamma $indgamma --Ntraj 30000
echo "likelihoods done"
python3 analysis/main_tests.py --indgamma $indgamma --Ntraj 30000
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $DIFF
deactivate
