#!/bin/bash
. ~/qenv_bilkis/bin/activate
python3 -c "import multiprocessing as mp; print(mp.cpu_count())"
deactivate
