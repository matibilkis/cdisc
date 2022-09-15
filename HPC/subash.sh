#!/bin/bash
itraj=$1
cd ~/estimation
. ~/qenv_bilkis/bin/activate
START=$(date +%s.%N)
python3 numerics/integration/integrate.py --itraj $itraj
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $DIFF
python3 numerics/ML/run.py --itraj $itraj
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $DIFF
deactivate
