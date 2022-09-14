import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from numerics.utilities.misc import load_data

def give_spectra(signal, dt):
    Period = 2*np.pi/omega
    spectra_signal = np.abs(np.fft.fft(signal))**2
    freqs_signal = np.fft.fftfreq(n = len(spectra_signal), d= dt)*(2*np.pi)
    cutoff = 10*omega
    cond  = np.logical_and(freqs_signal < cutoff, freqs_signal>=0)
    spectra_signal = spectra_signal[cond]
    freqs_signal = freqs_signal[cond]
    return freqs_signal, spectra_signal

def plot(states, signals,cut=-1):
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

def load_plot(params,cut=-1):
    gamma, omega, n, eta, kappa = params
    N_periods = 100.
    single_period=2*np.pi/omega
    total_time = N_periods*single_period
    dt = single_period/100.
    times = np.arange(0,total_time+dt,dt)
    params = [gamma, omega, n, eta, kappa]
    exp_path = str(params)+"/"
    states = load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="states.npy")
    signals = load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="signals.npy")
    plot(states,signals,cut=cut)

gamma, omega, n, eta, kappa = [1e1, 1e3, 1., 1., 1e4]
load_plot([gamma, omega, n, eta, kappa], cut=-1)


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
