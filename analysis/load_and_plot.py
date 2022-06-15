import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd())
import matplotlib.pyplot as plt
from numerics.utilities.misc import *
import matplotlib
from scipy.linalg import solve_continuous_are
import pickle
from tqdm import tqdm
from analysis.misc import *


params, exp_path = def_params(flip = 0)

which = mode = give_model()
if which == "mechanical_damp":
    total_time = 4.
    dt = 1e-5

    B = 8.
    dB = .1
    boundsB= np.arange(-B,B+dB,dB)
    bpos = boundsB[boundsB>=0]


elif which == "mechanical_freq":
    omega = .5*(params[1][1] + params[0][1])
    T_param = 150.
    ppg = 1e3
    period = 2*np.pi/omega
    total_time = T_param*period
    dt = period/ppg


    B = 8.
    dB = .1
    boundsB= np.arange(-B,B+dB,dB)
    bpos = boundsB[boundsB>=0]


elif (which == "optical_homodyne") or (which == "optical_heterodyne"):
    damping = params[1][0]
    total_time = 200.*damping
    dt = damping/1e3

timind, indis, indis_range = get_timind_indis(total_time, dt, rrange=True)


Ntraj = 5000


path_data = get_def_path()+"analysis/{}/".format(Ntraj,mode)
path_data_save = get_def_path()+"analysis/{}/done/".format(Ntraj,mode)

os.makedirs(path_data_save, exist_ok=True)


### load (just in case)
timbin1 = np.load(path_data+"timbin.npy")
timbin0 = np.load(path_data+"timbin.npy")#, timbin0)

l0 = np.load(path_data+"l0.npy")
l1 = np.load(path_data+"l1.npy")

cons0 = np.load(path_data+"cons0.npy")
cons1 = np.load(path_data+"cons1.npy")


stoch = np.load(path_data+"times_to_err_stoch.npy")
times_alpha_to_errB = np.load(path_data+"times_to_err_det.npy")


def prob_craft(t, b, mu):
    S= np.sqrt(2*mu)
    div = (np.sqrt(2*np.pi)*S*(t**(3/2)))
    return  abs(b)*np.exp(-((abs(b)-mu*t)**2)/(2*t*(S**2)))/div

muu, _ = np.polyfit(timind, l1,1)

LS, TS = 60, 40
plt.figure(figsize=(20,20))
ax = plt.subplot(111)
indb = 49
LW = 10
# timm =  np.linspace(np.min(timbin1[indb]),np.max(timbin1[indb]),100)
timm =  np.linspace(0,np.max(timbin1[indb]),100)
popo = [prob_craft(tt, bpos[indb] , muu) for tt in timm]
ax.plot(timm,popo, linewidth=LW, label="analytic")
ax.bar(timbin1[indb], cons1[indb], width=timbin1[indb][1]-timbin1[indb][0], color="red", alpha=0.75, edgecolor="black", label="simulations")
ax.set_xlabel(r'$\tau$',size=LS)
ax.set_ylabel(r'$P(\tau)$', size=LS)
ax.tick_params(axis='both', which='major', labelsize=TS)
ax.legend(prop={"size":LS})
plt.savefig(path_data_save+"histogram_stoch.pdf")







fig = plt.figure(figsize=(20,20))
ax = plt.subplot(111)

fromm = 3
too = 50
ax.plot(bpos[fromm:too],(times_alpha_to_errB/stoch)[fromm:too], linewidth=LW)
ax.plot(bpos[fromm:too],4*np.ones(len(stoch[fromm:too])),'--', linewidth=LW)
ax.set_xlabel(r'$\log \frac{1}{\epsilon}$', size=LS)
ax.tick_params(axis='both', which='major', labelsize=TS)
ax.set_ylabel(r'$\frac{t_d}{t_s}$', size=LS)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.savefig(path_data_save+"ratio.pdf")
