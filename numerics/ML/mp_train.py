import os
import multiprocessing as mp
from numerics.utilities.misc import *
from datetime import datetime
import argparse

cores = 18

trajs = list(range(1,int(1e4)))
def simu(itraj):
    os.system("python3 ML/run.py --itraj {}".format(itraj))

with mp.Pool(cores) as p:
    p.map(simu, trajs)
