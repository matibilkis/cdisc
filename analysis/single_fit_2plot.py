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

    return  list(out.values.values()), freqs_signal, spectra_signal


gamma, omega, n, eta, kappa = [1e1, 1e3, 10., 1., 1e2]#[1e1, 1e3, 1., 1., 1e4]
params = [gamma, omega, n, eta, kappa]
N_periods = 100.
single_period=2*np.pi/omega
total_time = N_periods*single_period
dt = single_period/50.
times = np.arange(0,total_time+dt,dt)
exp_path = str(params)+"/"


timms = np.linspace(100, len(times)-1,10).astype("int")
itraj = 1
signals = load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="signals.npy",itraj=itraj)
states = load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="states.npy",itraj=itraj)

values, freqs_signal, spectra_signal = fit_lorentzian(signals[:-1,0],dt)

def lorentzian_fit(x, out):
    A, x0, sigma, fhwm, height = out#list(out.values.values())
    return (A/np.pi)*(sigma/((x-x0)**2 + sigma**2))

os.makedirs("figures_poster/",exist_ok=True)


np.argmin(np.abs(freqs_signal - omega))

len(freqs_signal)

fit = r'$\frac{1}{\pi}\frac{A \sigma}{(\omega - \omega_0)^2 + \sigma^2)}$'





plt.figure(figsize=(20,10))
ax = plt.subplot(211)
ax.plot(times, states[:,0], alpha=0.75, color="blue")
ax.set_ylabel(r'$x_t$', size=60)
ax.xaxis.set_tick_params(labelsize=0)
ax.yaxis.set_tick_params(labelsize=24)
ax = plt.subplot(212)
ax.plot(times, signals[:,0], alpha=0.75, color="red")
ax.set_xlabel(r'$t$', size=60)
ax.set_ylabel(r'$dy_t$', size=60)
ax.xaxis.set_tick_params(labelsize=24)
ax.yaxis.set_tick_params(labelsize=24)
plt.savefig("figures_poster/evolution.pdf")





error_ml = np.load("analysis/data_comparison/error_ml.npy")
error_lor = np.load("analysis/data_comparison/error_lor.npy")
fisher = np.load("analysis/data_comparison/fisher.npy")
times_compa = np.load("analysis/data_comparison/times.npy")
times_metohd = np.load("analysis/data_comparison/times_method.npy")












from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset







plt.figure(figsize=(15,15))
ax = plt.subplot(111)
a = 5
b = -500
ax.plot(freqs_signal[a:b], spectra_signal[a:b], color="red", linewidth=5, alpha=0.75, label=r'$S(\omega)$')
ax.plot(freqs_signal[a:b], lorentzian_fit(freqs_signal, values)[a:b], '--', color="black", linewidth=5, alpha=0.75, label=fit)
ax.axvline(omega, linestyle='--',linewidth=2, color="blue", label=r'$\omega_{true}$')
aa, bb = 95, 110
axins = ax.inset_axes([0.05, 0.6, 0.3, 0.3])
axins.plot(freqs_signal[aa:bb], spectra_signal[aa:bb], color="red", linewidth=5, alpha=0.75)
x1, x2, y1, y2 = freqs_signal[aa], freqs_signal[bb], np.min(spectra_signal[aa:bb]), 2*np.max(spectra_signal)# spectra_signal[bb]
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
#axins.set_xscale("log")
axins.set_yscale("log")
# axins.set_xticklabels([])
axins.set_yticklabels([])
mark_inset(ax, axins, loc1=1, loc2=2, ec=".1")
axins.axvline(omega, linestyle='--',linewidth=5, color="blue")
axins.axvline(values[1],linestyle='--',linewidth=5, color="black")
ax.set_xlabel(r'$\omega$', size=60)
ax.set_ylabel(r'$S(\omega)$', size=60)
ax.set_xscale("log")
ax.set_yscale("log")
axins.set_xticklabels([])
ax.xaxis.set_tick_params(labelsize=24)
ax.yaxis.set_tick_params(labelsize=24)
ax.legend(prop={"size":25}, loc="upper right")
plt.savefig("figures_poster/spectral_fit.pdf")













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
