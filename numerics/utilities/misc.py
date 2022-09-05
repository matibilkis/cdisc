import numpy as np
import ast
import os
import getpass
#import pyarrow.parquet as pq
#import pyarrow as pa

def give_model():
    return "mechanical_damp"

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
    gamma1 = 14*2*np.pi
    gamma0 = 19*2*np.pi #(Hz)
    eta1 = 0.9
    eta0 = 0.9
    n1 = 14.0
    n0 = 14.0
    kappa1 = 2*np.pi*360
    kappa0 = 2*np.pi*360 #(Hz)
    omega0 = omega1 = 0.
    h0 = [gamma0, omega0, n0, eta0, kappa0]
    h1 = [gamma1, omega1, n1, eta1, kappa1]
    if flip == 0:
        p = [h1, h0]
    else:
        p = [h0, h1]
    return p, str(p)+"/"


def get_path_config(exp_path="", itraj=1, total_time=1, dt=.1, noitraj=False):
    if noitraj == True:
        pp = get_def_path()+ exp_path +"/T_{}_dt_{}/".format(total_time, dt)
    else:

        pp = get_def_path()+ exp_path +"{}itraj/T_{}_dt_{}/".format(itraj, total_time, dt)
    return pp


def load_data(exp_path="", itraj=1, total_time=1, dt=0.1, what="logliks"):
    path = get_path_config(total_time = total_time, dt= dt, itraj=itraj, exp_path=exp_path)
    logliks = np.load(path+what,allow_pickle=True,fix_imports=True,encoding='latin1') ### this is \textbf{q}(t)
    return logliks



def load_liks(itraj=1, dt=1e-1, total_time=1):
    params, exp_path = def_params(flip=0)
    logliks =load_data(itraj=itraj, total_time = total_time, dt=dt, exp_path = exp_path, what="logliks")
    l_1true  = logliks[:,1] - logliks[:,0]   ### this is l(h1) - l(h0)   be pos --> \inft

    params, exp_path = def_params(flip=1)
    logliks =load_data(itraj=itraj, total_time = total_time, dt=dt, exp_path = exp_path, what="logliks")
    l_0true  = logliks[:,0] - logliks[:,1]    ### this is l(h1) - l(h0)   under hypothesis 0 is true (should be negative --> \inft). It's flipped
    ### because
    return l_1true, l_0true#, tims

def get_timind_indis(total_time, dt, N=1e4, begin=0, rrange=True):
    times = np.arange(0,total_time+dt, dt)
    if len(times)>1e4:
        indis = np.logspace(begin,np.log10(len(times)-1), int(N))
    else:
        indis = np.arange(begin,len(times))#imtimes[-1],times[1]-times[0]).astype(int)
    indis = [int(k) for k in indis]
    timind = [times[ind] for ind in indis]
    if rrange == True:
        return timind, indis, list(range(len(indis)))
    else:
        return timind, indis



def get_stop_time(ell,b, times, mode_log=True):
    logicals = np.logical_and(ell < b, ell > -b)
    ind_times = np.argmin(logicals)

    if (np.sum(logicals) == 0) or (ind_times==0):
        return np.nan
    else:
        return times[ind_times]
