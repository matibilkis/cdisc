import numpy as np
import ast
import os
import getpass

def give_model():
    return "optical"

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


def def_params(flip =0):
    model = give_model()
    if model == "mechanical":
        gamma0 = gamma1 = 100
        eta0 = eta1 = 1
        kappa0 = kappa1 = 1e6
        n0 = n1 = 1
        omega0, omega1 = 1e4, 1.05e4

        h0 = [gamma0, omega0, n0, eta0, kappa0]
        h1 = [gamma1, omega1, n1, eta1, kappa1]
        if flip == 0:
            p = [h1, h0]
        else:
            p = [h0, h1]

    elif model == "optical":  #genoni's paper
        kappa0 = kappa1 = 1.
        xi0 = xi1 = 0.49*kappa1
        eta0 = eta1 = 1.
        omega0, omega1 = 0.1*kappa1, 0.2*kappa1

        h0 = [kappa0, eta0, omega0, xi0]
        h1 = [kappa1, eta1, omega1, xi1]

        if flip == 0:
            p = [h1, h0]
        else:
            p = [h0, h1]
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
    l1  = logliks[:,0] - logliks[:,1]

    params, exp_path = def_params(flip=1)
    logliks =load_data(itraj=itraj, total_time = total_time, dt=dt, exp_path = exp_path, what="logliks.npy")
    l0  = logliks[:,1] - logliks[:,0]

    return l0, l1#, tims

def int_or_0(x):
    try:
        return int(x)
    except Exception:
        return 0

def get_timind(total_time, dt, N=1e4):
    times = np.arange(0,total_time+dt, dt)
    indis = np.logspace(0,np.log10(len(times)-1), int(N))
    indis = [int(k) for k in indis]
    timind = [times[ind] for ind in indis]
    return timind

def get_timind_indis(total_time, dt, N=1e4):
    times = np.arange(0,total_time+dt, dt)
    indis = np.logspace(0,np.log10(len(times)-1), int(N))
    indis = [int(k) for k in indis]
    timind = [times[ind] for ind in indis]
    return timind, indis
