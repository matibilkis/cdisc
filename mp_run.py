import os
import multiprocessing as mp
from numerics.utilities.misc import *
from datetime import datetime
import argparse

cores = 6

trajs = list(range(1,int(1e4)))
def simu(itraj):
    os.system("python3 numerics/integration/integrate.py --itraj {}".format(itraj))

with mp.Pool(cores) as p:
    p.map(simu, trajs)
