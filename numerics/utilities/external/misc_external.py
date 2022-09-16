import numpy as np
import ast
import os
import getpass
#import pyarrow.parquet as pq
#import pyarrow as pa

def give_model():
    return "estimation/external"

def get_def_path():
    """
    mode is changed from branch to branch (from model to model)
    """
    model =give_model()
    user = getpass.getuser()
    if user == "cooper-cooper":
        defpath = '../quantera/trajectories/'
    elif (user =="matias") or (user == "mati"):# or (user=="giq"):
        defpath = '../quantera/trajectories/'
    elif (user=="giq"):
        defpath = "/media/giq/Nuevo vol/quantera/trajectories/"
    else:
        defpath = "/data/uab-giq/scratch/matias/quantera/trajectories/"
    if model[-1] != "/":
        model+="/"
    defpath+=model
    return defpath


def get_path_config(exp_path="", itraj=1, total_time=1, dt=.1, noitraj=False):
    if noitraj == True:
        pp = get_def_path()+ exp_path +"/T_{}_dt_{}/".format(total_time, dt)
    else:
        pp = get_def_path()+ exp_path +"{}itraj/T_{}_dt_{}/".format(itraj, total_time, dt)
    return pp


def load_data(exp_path="", itraj=1, total_time=1, dt=0.1, what="states"):
    path = get_path_config(total_time = total_time, dt= dt, itraj=itraj, exp_path=exp_path)
    states = np.load(path+what,allow_pickle=True,fix_imports=True,encoding='latin1') ### this is \textbf{q}(t)
    return states
