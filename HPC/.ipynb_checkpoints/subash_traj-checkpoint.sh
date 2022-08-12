#!/bin/bash
itraj=$1
cd ~/cdisc
. ~/qenv_bilkis/bin/activate
for k in $(seq 0 1 7)
do
    START=$(date +%s.%N)
    python3 mp_run.py --itraj $(($itraj + $k))
    END=$(date +%s.%N)
    DIFF=$(echo "$END - $START" | bc)
    echo $DIFF
done
deactivate
