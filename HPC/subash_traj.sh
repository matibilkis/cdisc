#!/bin/bash
itraj=$1
cd ~/cdisc
. ~/qenv_bilkis/bin/activate
#for k in $(seq 0 1 10)
for gam in $(seq 110 100 110)#10000)
do
    #nitraj=$(($itraj+$k))
    START=$(date +%s.%N)
    python3 numerics/integration/integrate.py --itraj $itraj --gamma $gam --pdt $pdt
    #python3 numerics/integration/integrate.py --itraj $itraj --flip_params 1 --gamma $gam
    END=$(date +%s.%N)
    DIFF=$(echo "$END - $START" | bc)
    echo $DIFF
done
deactivate
