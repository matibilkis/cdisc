#!/bin/bash
itraj=$1
cd ~/cdisc
. ~/qenv_bilkis/bin/activate
for pdt in $(seq 1 50 1000)
do
    START=$(date +%s.%N)
    python3 numerics/integration/integrate.py --itraj $itraj --gamma 110. --pdt $pdt
    END=$(date +%s.%N)
    DIFF=$(echo "$END - $START" | bc)
    echo $DIFF 
    echo $pdt
done
deactivate
