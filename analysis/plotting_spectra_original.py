import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import numerics.utilities.external.misc_external as misc_external
from importlib import reload

reload(misc_external)


itraj = 1
gamma, omega, n, eta, kappa = [100., 0., 1., 1., 1e5]
total_time = 20.#N_periods*single_period
dt = total_time*1e-5

id=3
times = np.arange(0,total_time+dt,dt)
params = [gamma, omega, n, eta, kappa]
exp_path = str(params)+"/"
states = misc_external.load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="states.npy",itraj=itraj, id=id)
signals = misc_external.load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="signals.npy",itraj=itraj,id=id)
#states_0 = misc_external.load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="states.npy",itraj=itraj, id=0)

plt.plot(times,states[:,0])
plt.plot(times,signals[:,0])










def give_spectra(signal, dt):
    Period = 2*np.pi/omega
    spectra_signal = np.abs(np.fft.fft(signal))**2
    freqs_signal = np.fft.fftfreq(n = len(spectra_signal), d= dt)*(2*np.pi)
    cutoff = 10*omega
    cond  = np.logical_and(freqs_signal < cutoff, freqs_signal>=0)
    spectra_signal = spectra_signal[cond]
    freqs_signal = freqs_signal[cond]
    return freqs_signal, spectra_signal

def plot(params, states, signals,cut=-1):
    gamma, omega, n, eta, kappa = params
    N_periods = 100.
    single_period=2*np.pi/omega
    total_time = N_periods*single_period
    dt = single_period/100.
    times = np.arange(0,total_time+dt,dt)
    freqs_state,spectra_state = give_spectra(states[:cut,0],dt)
    freqs_signal,spectra_sisg = give_spectra(signals[:cut,0],dt)
    plt.figure(figsize=(15,5))
    ax=plt.subplot(141)
    ax.plot(times, states[:,0], linewidth=.5)
    ax=plt.subplot(142)
    ax.plot(times, signals[:,0], linewidth=1, color="red")
    ax=plt.subplot(143)
    ax.plot(freqs_state,spectra_state**2)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax=plt.subplot(144)
    ax.plot(freqs_signal,spectra_sisg)
    ax.set_yscale("log")
    ax.set_xscale("log")

def load_plot(params,cut=-1, itraj=1):
    gamma, omega, n, eta, kappa = params
    N_periods = 100.
    single_period=2*np.pi/omega
    total_time = N_periods*single_period
    dt = single_period/100.
    times = np.arange(0,total_time+dt,dt)
    params = [gamma, omega, n, eta, kappa]
    exp_path = str(params)+"/"
    states = load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="states.npy",itraj=itraj)
    signals = load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="signals.npy",itraj=itraj)
    plot(params,states,signals,cut=cut)

load_plot([gamma, omega, n, eta, kappa], cut=-1, itraj=9000)


















gamma, omega, n, eta, kappa = [1e1, 1e3, 1., 1., 1e4]
N_periods = 100.
single_period=2*np.pi/omega
total_time = N_periods*single_period
dt = single_period/100.
times = np.arange(0,total_time+dt,dt)
params = [gamma, omega, n, eta, kappa]
exp_path = str(params)+"/"
states = load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="states.npy")

signals = load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="signals.npy")

c,b = np.histogram(states[:,0]/signals[:,0], bins=100)

plt.plot(b[:-1],c)

(states[:,0]/signals[:,0])[:100]





#
