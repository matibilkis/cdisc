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
import numerics.ML.model_states as model_ML
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

gamma, omega, n, eta, kappa = np.array(params).astype("float32")
A = np.array([[-gamma/2, omega],[-omega, -gamma/2]]).astype("float32")
C = np.sqrt(4*eta*kappa)*np.array([[1.,0.],[0., 0.]]).astype("float32")
D = np.diag([gamma*(n+0.5) + kappa]*2).astype("float32")
cov_st = solve_continuous_are( A.T, C.T, D, np.eye(2))


save_dir = misc_ML.get_training_save_dir(exp_path, total_time, dt, itraj,train_id)
os.makedirs(save_dir, exist_ok=True)

initial_parameters = np.array([2*omega]).astype("float32")
true_parameters = np.array([omega]).astype("float32")

batch_size = 100#tfsignals.shape[1]
epochs = 100
learning_rate = 10
with open(save_dir+"training_details.txt", 'w') as f:
    f.write("BS: {}\nepochs: {}\n learning_rate: {}\n".format(batch_size, epochs, learning_rate))
f.close()


model = model_ML.Model(params=params, dt=dt, initial_parameters=initial_parameters,
              true_parameters=true_parameters, initial_states = np.zeros((1,5)).astype(np.float32),
              cov_in=cov_st, batch_size=tuple([None,None,3]),
              save_dir = save_dir)
model.recurrent_layer.build(tf.TensorShape([1, None, 3])) #None frees the batch_size
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

model.grads = []
history = model.craft_fit(tfsignals[:,:100,:], batch_size=100, epochs=epochs, early_stopping=0.)

model.reset_states()

k=0
grads = []
for k in range(10):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        preds = model(tfsignals[:,:k+1,:])
    g =tape.gradient(preds,model.trainable_variables)
    grads.append(g)
gg = np.squeeze(grads)

states_th = load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="states_th.npy")

plt.plot(np.sum(states_th[:10],axis=1))
plt.plot(gg,'--')





#
