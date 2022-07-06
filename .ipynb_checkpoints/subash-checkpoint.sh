#!/bin/bash
itraj=$1
mode=$2
tim=$3
cd ~/cdisc
. ~/qenv_bilkis/bin/activate
for k in $(seq 0 1 99)
do
    nitraj=$(($itraj+$k))
    start=`date +%s`
    python3 numerics/integration/integrate.py --itraj $nitraj
    end=`date +%s`
    runtime=$((end-start))
    echo $runtime
    python3 numerics/integration/integrate.py --itraj $nitraj --flip_params 1
done
deactivate
