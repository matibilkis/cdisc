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
cores = 8

def simu(itraj):
    st = datetime.now()
    os.system("python3 numerics/integration/integrate.py --itraj {} ".format(itraj+k))
    os.system("python3 numerics/integration/integrate.py --itraj {} --flip_params 1".format(itraj+k))
    print(itraj, "holi - 8 cores //", (datetime.now() - st).seconds, datetime.now())

with mp.Pool(cores) as p:
    p.map(simu, list(range(8)))
