import os
import multiprocessing as mp
from numerics.utilities.misc import *
from datetime import datetime

cores = mp.cpu_count()

pdts = [1, 10, 100, 1000]
def int_seed(seed):
    for pdt in pdts:
        s1 = datetime.now()
        os.system("python3 numerics/integration/integrate.py --itraj {} --pdt {} --total_time 1.".format(seed, pdt))
        #os.system("python3 numerics/integration/integrate.py --itraj {} --flip_params 1".format(seed+k))
        print(f"{seed} done")
        #print(f" {seed}, done")

int_seed(1)
with mp.Pool(cores-1) as p:
    p.map(int_seed, range(1,100))
