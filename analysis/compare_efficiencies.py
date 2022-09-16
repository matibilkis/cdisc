import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from numerics.utilities.misc import *
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import LorentzianModel
import multiprocessing as mp
from tqdm import tqdm
import pickle


gamma, omega, n, eta, kappa = [1e1, 1e3, 10., 1., 1e2]#[1e1, 1e3, 1., 1., 1e4]
params = [gamma, omega, n, eta, kappa]
N_periods = 100.
single_period=2*np.pi/omega
total_time = N_periods*single_period
dt = single_period/100.
times = np.arange(0,total_time+dt,dt)
exp_path = str(params)+"/"

timms = np.linspace(100, len(times)-1,10).astype("int")

trajs = np.array(list(range(1,int(1e3),1)))
l=[]
ers=[]
for itraj in tqdm(trajs):
    try:
        ll= np.load(get_def_path()+"lorentzians/{}.npy".format(itraj))
        #if np.min(ll[1:])<0:
        #    ers.append(itaj)
    #        pass
#        else:
        l.append(ll)
    except Exception:
        ers.append(itraj)

lf = []
for k in l:
    op=[]
    for jj in k:
        if jj<0:
            op.append(np.nan)
        else:
            op.append(jj)
    lf.append(op)
lstdbis = np.nanmean( (np.stack(lf) - omega)**2, axis=0)
np.save("lorentzian_fits_errors",lstd)

len(trajs)-len(ers)

fi = []
for itraj in tqdm(trajs):
    states_th = load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="states_th.npy",itraj=itraj)
    fi.append(np.abs(states_th[:,0])**2)

fi = np.stack(fi)

means_der_th_sq = np.mean(fi,axis=0)
fisher = np.cumsum(means_der_th_sq,axis=0)*4*kappa*dt

l=np.stack(l)
lstd = np.mean( (l - omega)**2, axis=0)

ax=plt.subplot()
ax.plot(times,1/fisher, label="Fisher info")
ax.scatter([times[k] for k in timms], lstd, label="Lorentzian fit - Variance")
ax.set_yscale("log")
ax.legend()

exp_path

import numerics.ML.misc as misc_ML
import numerics.ML.model as model_ML

ml_params = {}
erro = []
for itraj in tqdm(trajs):
    ml_params[itraj] = []
    try:

        for train_id in range(9):
                save_dir = misc_ML.get_training_save_dir(exp_path, total_time, dt, itraj,train_id)
                loss = np.load(save_dir+"loss.npy")
                params = np.load(save_dir+"params.npy")
                ml_params[itraj].append(params[np.argmin(loss)])#
    except Exception:
        erro.append(itraj)
        pass

ml_params_ok = []
for v in ml_params.values():
    if len(v) != 0:
        ml_params_ok.append(v)
