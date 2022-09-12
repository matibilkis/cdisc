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
cores = 8#32
gammas = np.linspace(110., 10000, 32) #---> all batch, before 6/11
gammas = np.linspace(gammas[0] + 10, gammas[1]-10, 32) #---> all batch, before 6/11


def simu(gamma):
    st = datetime.now()
    os.system("python3 numerics/integration/integrate.py --itraj {} --gamma {} --pdt 1 --dt 1e-4".format(itraj,gamma))
    os.system("python3 numerics/integration/integrate.py --itraj {} --flip_params 1 --gamma {} --pdt 1 --dt 1e-4".format(itraj, gamma))
    print(itraj, cores, gamma, (datetime.now() - st).seconds)

with mp.Pool(cores) as p:
    p.map(simu, gammas)
