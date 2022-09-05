import os
import multiprocessing as mp
from numerics.utilities.misc import *
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--itraj", type=int, default=1)
args = parser.parse_args()
global itraj
itraj = args.itraj

#cores =  mp.cpu_count()
cores = 6#32
#gammas = np.linspace(110., 10000, 32)
gammas = np.arange(1000,10000,1)
def simu(itraj):
    st = datetime.now()
    os.system("python3 numerics/integration/integrate.py --itraj {} --pdt 1 --dt 1e-4".format(itraj))
    os.system("python3 numerics/integration/integrate.py --itraj {} --flip_params 1 --pdt 1 --dt 1e-4".format(itraj))
    print(itraj)#, cores, gamma, (datetime.now() - st).seconds)

with mp.Pool(cores) as p:
    p.map(simu, gammas)
