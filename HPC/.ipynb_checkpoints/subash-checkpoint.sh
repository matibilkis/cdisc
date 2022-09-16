#!/bin/bash
itraj=$1
cd ~/estimation
. ~/qenv_bilkis/bin/activate
START=$(date +%s.%N)
python3 ~/estimation/integration/integrate.py --itraj $itraj
echo "integration done"
python3 ~/estimation/numerics/ML/run.py --itraj $itraj
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $DIFF
deactivate
