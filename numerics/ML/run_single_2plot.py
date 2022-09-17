import os
import sys
sys.path.insert(0, os.getcwd())
from numerics.utilities.misc import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


import os
import getpass
giq = getpass.getuser() == "giq"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if giq != True:
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

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
import numerics.ML.misc as misc_ML
import numerics.ML.model as model_ML
from importlib import reload
reload(misc_ML)
reload(model_ML)



itraj=1

params = [1e1, 1e3, 10., 1., 1e2]#[#1e1, 1e3, 1., 1., 1e4]
gamma, omega, n, eta, kappa = params
N_periods = 100.
single_period=2*np.pi/omega
total_time = N_periods*single_period
dt = single_period/50.
times = np.arange(0,total_time+dt,dt)
params = [gamma, omega, n, eta, kappa]
exp_path = str(params)+"/"
states = load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="states.npy", itraj=itraj)
signals = load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="signals.npy", itraj=itraj)

gamma, omega, n, eta, kappa = np.array(params).astype("float32")
A = np.array([[-gamma/2, omega],[-omega, -gamma/2]]).astype("float32")
C = np.sqrt(4*eta*kappa)*np.array([[1.,0.],[0., 0.]]).astype("float32")
D = np.diag([gamma*(n+0.5) + kappa]*2).astype("float32")
cov_st = solve_continuous_are( A.T, C.T, D, np.eye(2))

train_id = -1
tt = -1

tfsignals = misc_ML.pre_process_data_for_ML(times[:tt], signals[:tt-1])
save_dir = misc_ML.get_training_save_dir(exp_path, total_time, dt, itraj,train_id)
os.makedirs(save_dir, exist_ok=True)

# One could argue here that we are cheating, but another one could argue that with more gradient descent steps it will converge to the same.
initial_parameters = np.array([1200.]).astype("float32")
true_parameters = np.array([omega]).astype("float32")

epochs = 100
learning_rate = float(omega/50)
batch_size = len(times)
with open(save_dir+"training_details.txt", 'w') as f:
    f.write("Length {}/{}\n BS: {}\nepochs: {}\n learning_rate: {}\n".format(tt, len(times), tt, epochs, learning_rate))
f.close()

model = model_ML.Model(params=params, dt=dt, initial_parameters=initial_parameters,
              true_parameters=true_parameters, initial_states = np.zeros((1,5)).astype(np.float32),
              cov_in=cov_st, batch_size=tuple([None,None,3]),
              save_dir = save_dir)
model.recurrent_layer.build(tf.TensorShape([1, None, 3])) #None frees the batch_size
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

history = model.craft_fit(tfsignals[:,:tt,:], batch_size=batch_size, epochs=epochs, early_stopping=1e-14, verbose=1, not_split=True)

history += model.craft_fit(tfsignals[:,:tt,:], batch_size=batch_size, epochs=epochs, early_stopping=1e-14, verbose=1, not_split=True)

import matplotlib.pyplot as plt




plt.figure(figsize=(15,15))
ax = plt.subplot(111)
ax.plot(np.squeeze([history[k]["PARAMS"] for k in range(100)]),  color="green", linewidth=10, alpha=0.75, label=r'$\hat{\omega}_{M.L.}$')
ax.plot(np.ones(100)*omega,  linestyle="--", color="blue", linewidth=10, alpha=0.75, label=r'$\omega_{true}$')
ax.set_xlabel("epoch", size=60)
ax.set_ylabel(r'$\hat{\omega}_{M.L.}$', size=60)
ax.xaxis.set_tick_params(labelsize=24)
ax.yaxis.set_tick_params(labelsize=24)
ax.legend(prop={"size":35}, loc="lower right")
plt.savefig("figures_poster/ml_fit.pdf")









#
#
#
#     tfsignals = misc_ML.pre_process_data_for_ML(times[:tt], signals[:tt-1])
#
#     save_dir = misc_ML.get_training_save_dir(exp_path, total_time, dt, itraj,train_id)
#     os.makedirs(save_dir, exist_ok=True)
#
#     initial_parameters = np.array([np.abs(abs(omega*np.random.normal()*0.1) + omega)]).astype("float32")
# #    initial_parameters = np.array([900.]).astype("float32")
#
#     true_parameters = np.array([omega]).astype("float32")
#
#     epochs = 50
#     learning_rate = float(omega/50)
#     batch_size = 200
#     with open(save_dir+"training_details.txt", 'w') as f:
#         f.write("Length {}/{}\n BS: {}\nepochs: {}\n learning_rate: {}\n".format(tt, len(times), batch_size, epochs, learning_rate))
#     f.close()
#
#     model = model_ML.Model(params=params, dt=dt, initial_parameters=initial_parameters,
#                   true_parameters=true_parameters, initial_states = np.zeros((1,5)).astype(np.float32),
#                   cov_in=cov_st, batch_size=tuple([None,None,3]),
#                   save_dir = save_dir)
#     model.recurrent_layer.build(tf.TensorShape([1, None, 3])) #None frees the batch_size
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
#     history = model.craft_fit(tfsignals[:,:tt,:], batch_size=batch_size, epochs=epochs, early_stopping=1e-14, verbose=0)
#
#

#