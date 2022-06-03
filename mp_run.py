import os
import multiprocessing as mp
from numerics.utilities.misc import *


cores = mp.cpu_count()

def int_seed(seed):
    for k in range(Nstep):
        os.system("python3 numerics/integration/integrate.py --itraj {}".format(seed+k))
        os.system("python3 numerics/integration/integrate.py --itraj {} --flip_params 1".format(seed+k))
        print(f"{k}, {seed}, done")

Nstep = 20
int_seed(1)
#
Nstep = cores-1
int_seed(1)
with mp.Pool(cores-1) as p:
    p.map(int_seed, range(1,2000, Nstep))
