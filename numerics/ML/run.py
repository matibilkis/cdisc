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



import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--itraj", type=int, default=1)
args = parser.parse_args()
itraj = args.itraj

params = [1e1, 1e3, 10., 1., 1e2]#[#1e1, 1e3, 1., 1., 1e4]
gamma, omega, n, eta, kappa = params
N_periods = 100.
single_period=2*np.pi/omega
total_time = N_periods*single_period
dt = single_period/100.
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

timms = np.linspace(100, len(times)-1,10).astype("int")

for train_id, tt in enumerate(timms):
    tfsignals = misc_ML.pre_process_data_for_ML(times[:tt], signals[:tt-1])

    save_dir = misc_ML.get_training_save_dir(exp_path, total_time, dt, itraj,train_id)
    os.makedirs(save_dir, exist_ok=True)

    initial_parameters = np.array([np.abs(2*np.random.uniform()*omega)]).astype("float32")
    true_parameters = np.array([omega]).astype("float32")

    epochs = 50
    learning_rate = 10.
    batch_size = 50
    with open(save_dir+"training_details.txt", 'w') as f:
        f.write("Length {}/{}\n BS: {}\nepochs: {}\n learning_rate: {}\n".format(tt, len(times), batch_size, epochs, learning_rate))
    f.close()

    model = model_ML.Model(params=params, dt=dt, initial_parameters=initial_parameters,
                  true_parameters=true_parameters, initial_states = np.zeros((1,5)).astype(np.float32),
                  cov_in=cov_st, batch_size=tuple([None,None,3]),
                  save_dir = save_dir)
    model.recurrent_layer.build(tf.TensorShape([1, None, 3])) #None frees the batch_size
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    history = model.craft_fit(tfsignals[:,:tt,:], batch_size=batch_size, epochs=epochs, early_stopping=1e-14)



#
