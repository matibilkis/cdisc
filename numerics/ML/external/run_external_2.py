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
params= [5., 0., 1., 1., 1e2]

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


tfsignals = misc_ML.pre_process_data_for_ML(times[:], signals[:-1])

save_dir = misc_ML.get_training_save_dir(exp_path, total_time, dt, itraj,train_id)
os.makedirs(save_dir, exist_ok=True)

initial_parameters = np.array([0., 1000.]).astype("float32")
true_parameters = np.array([100., 5.]).astype("float32")

epochs = 10
learning_rate = float(true_parameters[0]/50)
batch_size = 250
with open(save_dir+"training_details.txt", 'w') as f:
    f.write("BS: {}\nepochs: {}\n learning_rate: {}\n".format(len(times), batch_size, epochs, learning_rate))
f.close()


### Predict some ###
def give_batched_data(tfsignals, batch_size):

    if tfsignals.shape[1] < batch_size:
        raise ValueError("Batch size is too big for this amount of data: {} vs {}".format(batch_size, tfsignals.shape[1]))
    ll = tfsignals.shape[1]
    for k in list(range(int(ll/2), ll,1))[::-1]:
        if k%batch_size==0:
            break
    ttfsignals = tfsignals[:,:k,:]
    if tfsignals.shape[1]%batch_size != 0:
        raise ValueError("check your batch_size and training set, i can't split that")
    Ns = ttfsignals.shape[1]/batch_size
    batched_data  = tf.split(ttfsignals, int(Ns), axis=1)
    return batched_data

initial_states = np.array([0.,0., cov_st[0,0], cov_st[1,0], cov_st[1,1]])[np.newaxis]
model = model_ML.Model(params=params, dt=dt, initial_parameters=initial_parameters,
              true_parameters=true_parameters, initial_states = initial_states.astype(np.float32),
              cov_in=cov_st, batch_size=tuple([None,None,3]),
              save_dir = save_dir)

model.recurrent_layer.cell.train_id
model.recurrent_layer.build(tf.TensorShape([1, None, 3])) #None frees the batch_size
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

model.trainable_variables[0][0][1]
history = model.craft_fit(tfsignals[:,:-1,:], batch_size=500, epochs=200, early_stopping=1e-14, verbose=1)





#

batched_data = give_batched_data(tfsignals, batch_size )
model.reset_states()
preds_after=[]
model.recurrent_layer.cell.memory_states=[]
for b in tqdm(batched_data[:3]):
    preds_after.append(model(b))
ints = [model.recurrent_layer.cell.memory_states[k+6][0][0] for k in range(600)]

model.trainable_variables[0].assign(tf.convert_to_tensor([[0.]]))

model.reset_states()
preds_in=[]
model.recurrent_layer.cell.memory_states = []
for b in tqdm(batched_data[:3]):
    preds_in.append(model(b))
ints_initial = [model.recurrent_layer.cell.memory_states[k+6][0][0] for k in range(600)]


plt.plot(np.stack(ints_initial)[:,0])
plt.plot(np.stack(ints)[:,0])


plt.plot(preds_after[0][0,:,0])
plt.plot(preds_in[0][0,:,0])



a=0
b=-1
preds_in = tf.concat(preds_in,axis=1)[0,:,0]
preds_after = tf.concat(preds_after,axis=1)[0,:,0]

a=100
b= 300
plt.plot(preds_in[a:b],color="red")
plt.plot(preds_after[a:b], color="blue")
plt.plot(signals[:,0][:len(preds_in)][a:b], color="black")




ax=plt.subplot(121)
ax.plot(states[:600,0], label="hidden state")
ax.plot(np.stack(ints_initial)[:,0], label="(ML-tracked) hidden state: untrained")
ax.plot(np.stack(ints)[:,0], label="(ML-tracked) hidden state: trained")
ax=plt.subplot(122)
par_hist = np.squeeze([history[k]["PARAMS"] for k in range(len(history))])
ax.plot(par_hist)
ax.plot(true_parameters*np.ones(len(par_hist)), '--')



#
