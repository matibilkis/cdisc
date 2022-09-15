import os
import multiprocessing as mp
from numerics.utilities.misc import *
from datetime import datetime
import argparse

cores = 18

trajs = list(range(int(1e3),int(1e4)))
def simu(itraj):
    os.system("python3 numerics/integration/integrate.py --itraj {}".format(itraj))
    print(itraj)
with mp.Pool(cores) as p:
    p.map(simu, trajs)

os.system("python3 analysis/mp_fits.py")
