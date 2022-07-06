import os
import multiprocessing as mp
from numerics.utilities.misc import *
from datetime import datetime

cores = mp.cpu_count()

def int_seed(seed):
    for k in range(Nstep):
        s1 = datetime.now()
        os.system("python3 numerics/integration/integrate.py --itraj {}".format(seed+k))
        os.system("python3 numerics/integration/integrate.py --itraj {} --flip_params 1".format(seed+k))
        print(f"{k}, {seed}, done")

cores = 20
Nstep = 20
int_seed(1)
with mp.Pool(cores-1) as p:
    p.map(int_seed, range(5000,10000, Nstep))
