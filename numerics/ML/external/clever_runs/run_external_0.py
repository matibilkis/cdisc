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

itraj = 2
params = [5., 0., 1e-5, .5, 10]#[5., 0., 1e-5, .01, 1]

total_time = 20.#N_periods*single_period
dt = total_time*1e-4
times = np.arange(0,total_time+dt,dt)
id = train_id=0

gamma, omega, n, eta, kappa = params
gamma, omega, n, eta, kappa = np.array(params).astype("float32")
A = np.array([[-gamma/2, omega],[-omega, -gamma/2]]).astype("float32")
C = np.sqrt(4*eta*kappa)*np.array([[1.,0.],[0., 0.]]).astype("float32")
D = np.diag([gamma*(n+0.5) + kappa]*2).astype("float32")
cov_st = solve_continuous_are( A.T, C.T, D, np.eye(2))
exp_path = str(params)+"/"
states = misc_external.load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="states.npy", itraj=itraj, id=id)
signals = misc_external.load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="signals.npy", itraj=itraj, id=id)

plt.subplot(211)
plt.plot(times,states[:,0])
plt.subplot(212)
plt.plot(times,signals[:,0])




tfsignals = misc_ML.pre_process_data_for_ML(times[:], signals[:-1])

save_dir = misc_ML.get_training_save_dir(exp_path, total_time, dt, itraj,train_id)
os.makedirs(save_dir, exist_ok=True)

initial_parameters = np.array([0.]).astype("float32")
true_parameters = np.array([100.]).astype("float32")

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

model.recurrent_layer.build(tf.TensorShape([1, None, 3])) #None frees the batch_size
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))


batched_data = give_batched_data(tfsignals, 50 )
model.trainable_variables[0].assign(tf.convert_to_tensor([[0.]]))
model.reset_states()
preds_in=[]
model.recurrent_layer.cell.memory_states = []
for b in tqdm(batched_data[:5]):
    preds_in.append(model(b))
ints_initial = [model.recurrent_layer.cell.memory_states[k+6][0][0] for k in range(len(model.recurrent_layer.cell.memory_states)-6)]
kk=[]
timms=[]
for k in range(2,len(model.recurrent_layer.cell.memory_states)):
    try:
        kk.append(np.squeeze(list(model.recurrent_layer.cell.memory_states)[k])[0] )
        timms.append(model.recurrent_layer.cell.times_states[k])
    except Exception:
        pass

# plt.plot(timms, kk)
# plt.plot(times[:len(timms)], states[:len(kk),0])


history = model.craft_fit(tfsignals[:,:-1,:], batch_size=batch_size, epochs=epochs, early_stopping=1e-14, verbose=1)


model.reset_states()
preds_after=[]
model.recurrent_layer.cell.reset_memory()
for b in tqdm(batched_data[:5]):
    preds_after.append(model(b))
ints_initial = [model.recurrent_layer.cell.memory_states[k+6][0][0] for k in range(len(model.recurrent_layer.cell.memory_states)-6)]
kk_trained=[]
timms_trained=[]
for k in range(2,len(model.recurrent_layer.cell.memory_states)):
    try:
        kk_trained.append(np.squeeze(list(model.recurrent_layer.cell.memory_states)[k])[0] )
        timms_trained.append(model.recurrent_layer.cell.times_states[k])
    except Exception:
        pass


plt.figure(figsize=(15,15))
ax=plt.subplot(111)
step=2
S=300
ax.plot(times[:len(timms)], states[:len(kk),0] , color="blue", linewidth=30, alpha=1., zorder=1, label=r'$x^t_{true}$')
ax.scatter(timms_trained[::step], kk_trained[::step], s=S,  color="black", alpha=1.,zorder=2, label=r'$\hat{x}^t_{trained}$')
ax.scatter(timms[::step], kk[::step], color="red", s=S,  linewidth=5, alpha=1., label=r'$\hat{x}^t_{untrained}$')
ax.set_xlabel("t", size=60)
ax.set_ylabel(r'$\hat{x}^t_{M.L.}$', size=60)
ax.xaxis.set_tick_params(labelsize=24)
ax.yaxis.set_tick_params(labelsize=24)
ax.legend(prop={"size":35}, loc="upper left")
axins = ax.inset_axes([.6,.085, 0.35, 0.35])
#axins.plot(freqs_signal[aa:bb], spectra_signal[aa:bb], color="red", linewidth=5, alpha=0.75)
axins.plot(np.squeeze([history[k]["PARAMS"] for k in range(len(history))]),  color="green", linewidth=10, alpha=0.75, label=r'$\hat{f}_{M.L.}$',zorder=2)
axins.plot(np.ones(len(history))*true_parameters[0],  linestyle="--", color="blue", linewidth=10, alpha=0.75, label=r'$f_{true}$',zorder=1)
axins.set_xlabel("ML-iteration", size=20)
axins.set_ylabel(r'$\hat{f}_{M.L.}$', size=20)
axins.legend(prop={"size":20})
plt.savefig("figures_poster/external_learn_track.pdf")



plt.figure(figsize=(15,15))
ax=plt.subplot(111)
step=4
S=300
ss_untrained = np.squeeze(np.concatenate(preds_in,axis=1))[:,0]
ss_trained = np.squeeze(np.concatenate(preds_after,axis=1))[:,0]
ax.plot(times[:len(ss_trained)], signals[:len(ss_trained),0] , color="blue", linewidth=15, alpha=1., zorder=1, label=r'$dy^t_{true}$')
ax.scatter(times[:len(ss_trained)][::step], ss_trained[::step], s=S,  color="black", alpha=1.,zorder=2, label=r'$\hat{dy}^t_{trained}$')
ax.scatter(times[:len(ss_trained)][::step], ss_untrained[::step], color="red", s=S,  linewidth=5, alpha=1., label=r'$\hat{dy}^t_{untrained}$')
ax.set_xlabel("t", size=60)
ax.set_ylabel(r'$\hat{dy}^t_{M.L.}$', size=60)
ax.xaxis.set_tick_params(labelsize=24)
ax.yaxis.set_tick_params(labelsize=24)
ax.legend(prop={"size":35}, loc="upper left")
axins = ax.inset_axes([.7,.05, 0.25, 0.25])
#axins.plot(freqs_signal[aa:bb], spectra_signal[aa:bb], color="red", linewidth=5, alpha=0.75)
axins.plot(np.squeeze([history[k]["PARAMS"] for k in range(len(history))]),  color="green", linewidth=10, alpha=0.75, label=r'$\hat{f}_{M.L.}$',zorder=2)
axins.plot(np.ones(len(history))*true_parameters[0],  linestyle="--", color="blue", linewidth=10, alpha=0.75, label=r'$f_{true}$',zorder=1)
axins.set_xlabel("ML-iteration", size=20)
axins.set_ylabel(r'$\hat{f}_{M.L.}$', size=20)
axins.legend(prop={"size":20})
plt.savefig("figures_poster/external_learn_signals.pdf")