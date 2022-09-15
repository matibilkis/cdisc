import os
import sys
sys.path.insert(0, os.getcwd())
import multiprocessing as mp
from numerics.utilities.misc import *
from datetime import datetime
import argparse


cores = 16

trajs = list(range(1,int(1e4)))
def simu(itraj):
    os.system("python3 numerics/ML/run.py --itraj {}".format(itraj))
    print(itraj)
with mp.Pool(cores) as p:
    p.map(simu, trajs)
