#!/bin/bash
itraj=$1
cd ~/cdisc
. ~/qenv_bilkis/bin/activate
for pdt in 1 10 50 100 500 1000 5000
do
    START=$(date +%s.%N)
    python3 numerics/integration/integrate.py --itraj $itraj --gamma 11000. --pdt $pdt --dt 1e-6 --total_time 8.
    END=$(date +%s.%N)
    DIFF=$(echo "$END - $START" | bc)
    echo $DIFF
    echo $pdt
done
deactivate
