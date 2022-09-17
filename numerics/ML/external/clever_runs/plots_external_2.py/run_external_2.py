import os
import sys
sys.path.insert(0, os.getcwd())
import numerics.utilities.external.misc_external as misc_external
import warnings
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from tqdm import tqdm

import os
import getpass
giq = getpass.getuser() == "giq"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if giq != True:
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

import sys
if giq != True:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

if giq != True:
    num_threads = 1
    os.environ["OMP_NUM_THREADS"] = "{}".format(num_threads)
    os.environ["TF_NUM_INTRAOP_THREADS"] = "{}".format(num_threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "{}".format(num_threads)

    tf.config.threading.set_inter_op_parallelism_threads(
        num_threads
    )
    tf.config.threading.set_intra_op_parallelism_threads(
        num_threads
    )
    tf.config.set_soft_device_placement(True)

import tensorflow as tf
from scipy.linalg import solve_continuous_are
import numerics.ML.external.misc_external as misc_ML
import numerics.ML.external.model_external as model_ML
from importlib import reload
import matplotlib.pyplot as plt
reload(misc_ML)
reload(model_ML)
reload(misc_external)

itraj = 1
params = [5., 0., 1e-5, .5, 10]#[5., 0., 1e-5, .01, 1]

total_time = 20.#N_periods*single_period
dt = total_time*1e-4
times = np.arange(0,total_time+dt,dt)
id = train_id=2

gamma, omega, n, eta, kappa = params
gamma, omega, n, eta, kappa = np.array(params).astype("float32")
A = np.array([[-gamma/2, omega],[-omega, -gamma/2]]).astype("float32")
C = np.sqrt(4*eta*kappa)*np.array([[1.,0.],[0., 0.]]).astype("float32")
D = np.diag([gamma*(n+0.5) + kappa]*2).astype("float32")
cov_st = solve_continuous_are( A.T, C.T, D, np.eye(2))
exp_path = str(params)+"/"
states = misc_external.load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="states.npy", itraj=itraj, id=id)
signals = misc_external.load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="signals.npy", itraj=itraj, id=id)

save_path = "analysis/data_external_example_2/"

initial_parameters = [10, 25]
true_parameters = [100, 5]

kk_trained=np.load(save_path+"kk_trained.npy")#, kk_trained)
kk=np.load(save_path+"kk_untrained.npy")#, kk)
timms_trained=np.load(save_path+"timms_trained.npy")#, timms_trained)
timms = np.load(save_path+"timms_untrained.npy")#, timms)
history_params = np.load(save_path+"history_params.npy")#,np.squeeze([history[k]["PARAMS"][0] for k in range(len(history))]))

final_params = history_params[-1]


plt.figure(figsize=(15,15))
ax=plt.subplot(211)
A=.8
LL=15
lo=10
lop=2
ss=4
LSs=15
ax.plot(times, true_parameters[0]*np.exp(-times/true_parameters[1]), '--',color="blue", linewidth=LL, alpha=A, zorder=1, label=r'$f^{\;t}_{true} = A e^{-t/\tau}$')
ax.plot(times, initial_parameters[0]*np.exp(-times/initial_parameters[1]), color="red", linewidth=LL, alpha=A, zorder=2, label=r'${\hat{f^{\;t}}_\theta}$'+" -untrained")
ax.plot(times, final_params[0]*np.exp(-times/final_params[1]), color="black", linewidth=LL, alpha=A, zorder=3, label=r'${\hat{f^{\;t}}_\theta}$'+" -trained")
# ax.legend(prop={"size":20}, loc="upper left")
ax.xaxis.set_tick_params(labelsize=24)
ax.set_ylabel("external control estimate", size=30)
ax.yaxis.set_tick_params(labelsize=24)
axins = ax.inset_axes([.6,.35, 0.35, 0.25])
axins.plot(list(range(len(history_params)))[::ss], history_params[:,0][::ss],  color="black", linewidth=lop, alpha=0.75, label=r'$\hat{A}_{M.L.}$',zorder=2)
axins.plot(np.ones(len(history_params))*true_parameters[0],  linestyle="-", color="blue", linewidth=lo, alpha=0.75, label=r'$A_{true}$',zorder=1)
axins.legend(prop={"size":LSs})
axins.set_xlabel("ML-iteration", size=20)
axins = ax.inset_axes([.6,.65, 0.35, 0.25])
ax.xaxis.set_tick_params(labelsize=0)
axins.set_xticks([])
axins.plot(list(range(len(history_params)))[::ss], history_params[:,1][::ss],  color="black", linewidth=lop, alpha=0.75, label=r'$\hat{\tau}_{M.L.}$',zorder=2)
axins.plot(np.ones(len(history_params))*true_parameters[1],  linestyle="-", color="blue", linewidth=lo, alpha=0.75, label=r'$\tau_{true}$',zorder=1)
axins.legend(prop={"size":LSs})
ax.legend(prop={"size":20}, loc="center",  bbox_to_anchor=(.1, .5, 0.5, 0.5))
ax=plt.subplot(212)
step=10
S=50
ax.plot(times[:len(timms)], states[:len(kk),0] , color="blue", linewidth=10, alpha=1., zorder=1, label=r'$x^t_{true}$')
ax.scatter(timms_trained[::step], kk_trained[::step], s=S,  color="black", alpha=1.,zorder=2, label=r'$\hat{x}^t_{trained}$')
ax.scatter(timms[::step], kk[::step], color="red", s=S,  linewidth=5, alpha=1., label=r'$\hat{x}^t_{untrained}$')
ax.set_xlabel("time", size=30)
#ax.set_ylabel(r'$\hat{x}^t_{M.L.}$', size=60)
ax.set_ylabel("hidden state estimate", size=30)
ax.xaxis.set_tick_params(labelsize=24)
ax.yaxis.set_tick_params(labelsize=24)
ax.legend(prop={"size":35}, loc="upper right")
plt.savefig("figures_poster/external_learn_track_example2.pdf")












# plt.figure(figsize=(15,15))
# ax=plt.subplot(111)
# step=4
# S=300
# ss_untrained = np.squeeze(np.concatenate(preds_in,axis=1))[:,0]
# ss_trained = np.squeeze(np.concatenate(preds_after,axis=1))[:,0]
# ax.plot(times[:len(ss_trained)], signals[:len(ss_trained),0] , color="blue", linewidth=15, alpha=1., zorder=1, label=r'$dy^t_{true}$')
# ax.scatter(times[:len(ss_trained)][::step], ss_trained[::step], s=S,  color="black", alpha=1.,zorder=2, label=r'$\hat{dy}^t_{trained}$')
# ax.scatter(times[:len(ss_trained)][::step], ss_untrained[::step], color="red", s=S,  linewidth=5, alpha=1., label=r'$\hat{dy}^t_{untrained}$')
# ax.set_xlabel("t", size=60)
# ax.set_ylabel(r'$\hat{dy}^t_{M.L.}$', size=60)
# ax.xaxis.set_tick_params(labelsize=24)
# ax.yaxis.set_tick_params(labelsize=24)
# ax.legend(prop={"size":35}, loc="upper left")
# axins = ax.inset_axes([.7,.05, 0.25, 0.25])
# #axins.plot(freqs_signal[aa:bb], spectra_signal[aa:bb], color="red", linewidth=5, alpha=0.75)
# axins.plot(np.squeeze([history[k]["PARAMS"] for k in range(len(history))]),  color="green", linewidth=10, alpha=0.75, label=r'$\hat{f}_{M.L.}$',zorder=2)
# axins.plot(np.ones(len(history))*true_parameters[0],  linestyle="--", color="blue", linewidth=10, alpha=0.75, label=r'$f_{true}$',zorder=1)
# axins.set_xlabel("ML-iteration", size=20)
# axins.set_ylabel(r'$\hat{f}_{M.L.}$', size=20)
# axins.legend(prop={"size":20})
# #plt.savefig("figures_poster/external_learn_signals_ex2.pdf")
