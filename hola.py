import numpy as np
import matplotlib.pyplot as plt
from numerics.utilities.misc import load_data


gamma, omega, n, eta, kappa = [1000., 1e2, 10., 1., 1e4]
params = [gamma, omega, n, eta, kappa]
exp_path = str(params)+"/"

N_periods = 1000.
single_period=2*np.pi/omega
total_time = N_periods*single_period
dt = single_period/100.
times = np.arange(0,total_time+dt,dt)


states = load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="states.npy")
signals = load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="signals.npy")


plt.plot(times, signals[:,0], linewidth=.5)

Period = 2*np.pi/omega
# spectra_signal = np.abs(np.fft.fft(states[:,0]))**2
spectra_signal = np.abs(np.fft.fft(signals))**2
freqs_signal = np.fft.fftfreq(n = len(spectra_signal), d= dt)*(2*np.pi)

cutoff = 10*omega
cond  = np.logical_and(freqs_signal < cutoff, freqs_signal>=0)
spectra_signal = spectra_signal[cond]
freqs_signal = freqs_signal[cond]

ax=plt.subplot()
ax.plot(freqs_signal,spectra_signal**2)


#
