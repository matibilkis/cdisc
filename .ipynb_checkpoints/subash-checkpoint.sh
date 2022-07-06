#!/bin/bash
itraj=$1
mode=$2
tim=$3
cd ~/cdisc_3
. ~/qenv_bilkis/bin/activate
for k in $(seq 0 1 99)
do
    nitraj=$(($itraj+$k))
    python3 numerics/integration/integrate.py --itraj $nitraj
    python3 numerics/integration/integrate.py --itraj $nitraj --flip_params 1
done
deactivate
