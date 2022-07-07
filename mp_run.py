import os
import multiprocessing as mp
from numerics.utilities.misc import *
from datetime import datetime

cores = mp.cpu_count()
a=1

def int_seed(seed):
    for k in range(Nstep):
        s1 = datetime.now()
        os.system("python3 numerics/integration/integrate.py --itraj {} --a {}".format(seed+k, a))
        print(f"{k}, {seed}, done")

cores = 4
Nstep = 4
int_seed(1)
with mp.Pool(cores-1) as p:
    p.map(int_seed, range(1,40, Nstep))
