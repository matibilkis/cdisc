import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from numerics.utilities.misc import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import tensorflow as tf
from scipy.linalg import solve_continuous_are
import numerics.ML.misc as misc_ML
import numerics.ML.model as model_ML
from importlib import reload
reload(misc_ML)
reload(model_ML)

train_id = 0
itraj = 1

params = [1e1, 1e3, 1., 1., 1e4]
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

tfsignals = misc_ML.pre_process_data_for_ML(times, signals[:-1])

save_dir = misc_ML.get_training_save_dir(exp_path, total_time, dt, itraj,train_id)
os.makedirs(save_dir, exist_ok=True)

initial_parameters = np.array([omega]).astype("float32")
true_parameters = np.array([omega]).astype("float32")

batch_size = tfsignals.shape[1]
epochs = 10
learning_rate = 1e-2
with open(save_dir+"training_details.txt", 'w') as f:
    f.write("BS: {}\nepochs: {}\n learning_rate: {}\n".format(batch_size, epochs, learning_rate))
f.close()


gamma, omega, n, eta, kappa = np.array(params).astype("float32")
A = np.array([[-gamma/2, omega],[-omega, -gamma/2]]).astype("float32")
C = np.sqrt(4*eta*kappa)*np.array([[1.,0.],[0., 0.]]).astype("float32")
D = np.diag([gamma*(n+0.5) + kappa]*2).astype("float32")
cov_st = solve_continuous_are( A.T, C.T, D, np.eye(2))


BS = 1
batch_shape = [BS, None, 3]
model = model_ML.Model(params=params, dt=dt, initial_parameters=initial_parameters,
              true_parameters=true_parameters, initial_states = np.zeros((1,5)).astype(np.float32),
              cov_in=cov_st, batch_size=tuple([None,None,3]),
              save_dir = save_dir)
L=100
preds = model(tfsignals[:,:L,:])

plt.plot(times[:L],preds[0,:,0])
plt.plot(times[:L], Cxdt[:,0],'--')






Cxdt = np.einsum('ij,bj->bi',C,states[:L])*dt


sts = model.layers[0].cell.ss[0]
output = tf.einsum('ij,bj->bi',model.layers[0].cell.C_matrix, sts)*model.layers[0].cell.dt

A_model = (model.layers[0].cell.training_params[0]*model.layers[0].cell.symp -0.5*model.layers[0].cell.gamma*np.eye(2).astype("float32"))[tf.newaxis]

A_model


cov = model.layers[0].cell.cov_in
XiCov =tf.einsum('bij,jk->bik',cov,model.layers[0].cell.C_matrix.T)
XiCovC = tf.matmul(XiCov,model.layers[0].cell.C_matrix.T)

cov_dt = tf.einsum('bij,bjk->bik',A_model,cov) + tf.einsum('bij,bjk->bik',cov, tf.transpose(A_model, perm=[0,2,1])) + model.layers[0].cell.D_matrix - tf.einsum('bij,bjk->bik',XiCov, tf.transpose(XiCov, perm=[0,2,1]))














model.recurrent_layer.build(tf.TensorShape(batch_shape))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

history = model.craft_fit(tfsignals, batch_size=batch_size, epochs=epochs, early_stopping=1e-8)
