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

total_time = 20.#N_periods*single_period
dt = total_time*1e-4
times = np.arange(0,total_time+dt,dt)
id = train_id=0

params= [5., 0., 1., 1., 1e2]
gamma, omega, n, eta, kappa = params
gamma, omega, n, eta, kappa = np.array(params).astype("float32")
A = np.array([[-gamma/2, omega],[-omega, -gamma/2]]).astype("float32")
C = np.sqrt(4*eta*kappa)*np.array([[1.,0.],[0., 0.]]).astype("float32")
D = np.diag([gamma*(n+0.5) + kappa]*2).astype("float32")
cov_st = solve_continuous_are( A.T, C.T, D, np.eye(2))


params= [5., 0., 1., 1., 1e2]
def give_cov_st(params):
    gamma, omega, n, eta, kappa = params
    A = np.array([[-gamma/2, omega],[-omega, -gamma/2]]).astype("float32")
    C = np.sqrt(4*eta*kappa)*np.eye(2).astype("float32")
    D = np.diag([gamma*(n+0.5) + kappa]*2).astype("float32")
    cov_st = solve_continuous_are( A.T, C.T, D, np.eye(2))
    return (cov_st.dot(C.T))[0,0]

kappas = np.linspace(1, 1000, 100)
kaps = [give_cov_st([5., 0., 1., 1., k]) for k in kappas]
plt.plot(kappas,kaps,'.')



kappas = np.linspace(1, 1000, 100)
kaps = [give_cov_st([500., 0., 1., 1., k]) for k in kappas]
plt.plot(kappas,kaps)

kappas = np.linspace(1, 1000, 100)
kaps = [give_cov_st([5., 0., 1e-5, 1., k]) for k in kappas]
plt.plot(kappas,kaps,'.')


kappas = np.linspace(1, 1000, 100)
kaps = [give_cov_st([5., 0., 1e-5, .01, k]) for k in kappas]
plt.plot(kappas,kaps,'.')







##
