import numpy as np
import ast
import os
import getpass

def give_model():
    return "hidden_ou"

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


def def_params():
    model = give_model()
    a = -1.0
    b = 1.0
    alpha = 1.0
    beta = 1.0
    p = [a, b, alpha, beta]
    return p, str(p)+"/"


def get_path_config(exp_path="", itraj=1, total_time=1, dt=.1):
    pp = get_def_path()+ exp_path +"{}itraj/T_{}_dt_{}/".format(itraj, total_time, dt)
    return pp


def load_data(exp_path="", itraj=1, total_time=1, dt=0.1, what="logliks.npy"):
    path = get_path_config(total_time = total_time, dt= dt, itraj=itraj, exp_path=exp_path)
    logliks = np.load(path+what,allow_pickle=True,fix_imports=True,encoding='latin1') ### this is \textbf{q}(t)
    return logliks


def load_liks(itraj=1, dt=1e-1, total_time=1):
    params, exp_path = def_params(flip=0)
    logliks =load_data(itraj=itraj, total_time = total_time, dt=dt, exp_path = exp_path, what="logliks.npy")
    l_1true  = logliks[:,1] - logliks[:,0]   ### this is l(h1) - l(h0)   be pos --> \inft

    params, exp_path = def_params(flip=1)
    logliks =load_data(itraj=itraj, total_time = total_time, dt=dt, exp_path = exp_path, what="logliks.npy")
    l_0true  = logliks[:,0] - logliks[:,1]    ### this is l(h1) - l(h0)   under hypothesis 0 is true (should be negative --> \inft). It's flipped
    ### because
    return l_1true, l_0true#, tims

def get_timind_indis(total_time, dt, N=1e4, begin=0, rrange=True):
    times = np.arange(0,total_time+dt, dt)
    if len(times)>1e4:
        indis = np.linspace(0,len(times)-1,int(1e4)).astype("int")
    else:
        indis = np.arange(0,len(times))#imtimes[-1],times[1]-times[0]).astype(int)

    indis = [int(k) for k in indis]
    timind = [times[ind] for ind in indis]
    if rrange == True:
        return timind, indis, list(range(len(indis)))
    else:
        return timind, indis
