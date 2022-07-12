#!/bin/bash
itraj=$1
cd ~/hidden_ou
. ~/qenv_bilkis/bin/activate
for a in $(seq -10.0 .1 10.0)
do
    python3 numerics/integration/integrate.py --itraj $itraj --a $a
done
deactivate