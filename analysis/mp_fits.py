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

def fit_lorentzian(signals,dt):
    Period = 2*np.pi/omega
    spectra_signal = np.abs(np.fft.fft(signals))**2
    freqs_signal = np.fft.fftfreq(n = len(spectra_signal), d= dt)*(2*np.pi)

    cutoff = 10*omega
    cond  = np.logical_and(freqs_signal < cutoff, freqs_signal>=0)
    spectra_signal = spectra_signal[cond]
    freqs_signal = freqs_signal[cond]

    mod = LorentzianModel()

    pars = mod.guess(spectra_signal, x=freqs_signal)
    out = mod.fit(spectra_signal, pars, x=freqs_signal)

    return out.values["center"]


gamma, omega, n, eta, kappa = [1e1, 1e3, 10., 1., 1e2]#[1e1, 1e3, 1., 1., 1e4]
params = [gamma, omega, n, eta, kappa]
N_periods = 100.
single_period=2*np.pi/omega
total_time = N_periods*single_period
dt = single_period/100.
times = np.arange(0,total_time+dt,dt)
exp_path = str(params)+"/"

timms = np.linspace(100, len(times)-1,10).astype("int")
#trajs = np.array(list(range(1,int(1e3),1)))
trajs = list(range(int(1e3),int(1e4)))
lorentzians = {itraj:[] for itraj in trajs}

# signals = load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="signals.npy",itraj=1)
# for t in timms:
#     lorentzians[t][itraj] = fit_lorentzian(signals[:t,0],dt)
os.makedirs(get_def_path()+"lorentzians/",exist_ok=True)

def simu(itraj):
    signals = load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="signals.npy",itraj=itraj)
    for t in timms:
        lorentzians[itraj].append(fit_lorentzian(signals[:t,0],dt))
    np.save(get_def_path()+"lorentzians/{}".format(itraj), lorentzians[itraj])
    print(itraj)
with mp.Pool(8) as p:
    p.map(simu, trajs)



#
#
# lv = np.stack(lorentzians.values())
#
# path =
# os.makedirs(path,exist_ok=True)
# np.save(path+"centers_{}".format(len(trajs)), np.stack(list(lorentzians.values())))
#
#
#
#
# path
# fi = []
# for itraj in tqdm(trajs):
#     states_th = load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="states_th.npy",itraj=itraj)
#     fi.append(np.abs(states_th[:,0])**2)
#
# fi = np.stack(fi)
# fisher = np.mean(fi,axis=0)
#
# fisher_physical = 4*kappa*dt*fisher
# ax=plt.subplot()
# ax.plot(times,1/fisher)
# ax.scatter(times[-1], (np.std(lv))**2)
# ax.set_yscale("log")
#
#
#
#
# #
