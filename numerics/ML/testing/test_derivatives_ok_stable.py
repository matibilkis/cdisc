import os
import sys
sys.path.insert(0, os.getcwd())
from numerics.utilities.misc import *
import warnings
import matplotlib.pyplot as plt

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



itraj = 1


train_id=0
params = [1e1, 1e3, 10., 1., 1e2]
gamma, omega, n, eta, kappa = params
N_periods = 100.
single_period=2*np.pi/omega
total_time = N_periods*single_period
dt = single_period/100.
times = np.arange(0,total_time+dt,dt)
params = [gamma, omega, n, eta, kappa]
exp_path = str(params)+"/"

gamma, omega, n, eta, kappa = np.array(params).astype("float32")
A = np.array([[-gamma/2, omega],[-omega, -gamma/2]]).astype("float32")
C = np.sqrt(4*eta*kappa)*np.array([[1.,0.],[0., 0.]]).astype("float32")
D = np.diag([gamma*(n+0.5) + kappa]*2).astype("float32")
cov_st = solve_continuous_are( A.T, C.T, D, np.eye(2))
D_th = np.eye(2)*0. ## we estimate \omega
A_th = np.array([[0.,1.],[-1.,0.]])
cov_st_th = solve_continuous_are( (A-np.dot(cov_st,np.dot(C.T,C))).T, np.eye(2)*0., D_th + np.dot(A_th, cov_st) + np.dot(cov_st, A_th.T), np.eye(2))

from scipy.linalg import block_diag

XiCov  = np.dot(cov_st, C.T) #I take G=0.
XiCov_th  = np.dot(cov_st_th, C.T) #I take G = 0.

cth = cov_st_th

cth.dot((A - cov_st.dot(C.dot(C))).T) + (A - cov_st.dot(C.dot(C))).dot(cth) + cov_st.dot(A_th.T) + A_th.dot(cov_st)

cov_st.dot(A.T) + A.dot(cov_st) + D - cov_st.dot(C.dot(C.T).dot(cov_st))
# itraj=53
# states = load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="states.npy", itraj=itraj)
# states_th = load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="states_th.npy", itraj=itraj)
# signals = load_data(exp_path=exp_path,total_time=total_time, dt=dt,what="signals.npy", itraj=itraj)
# plt.plot(states_th[:,0])

class GRCell(tf.keras.layers.Layer):
    def __init__(self,
                units=5,### units = 5 [x,p, Vx, Vp, Cov(x,y)]
                params=[1e1, 1e3, 1., 1., 1e4],
                dt= 1e-4,
                initial_parameters=np.zeros(2).astype(np.float32),
                true_parameters=np.zeros(2).astype(np.float32),
                cov_in = np.zeros((2,2)).astype(np.float32),
                initial_states = np.zeros((1,5)).astype(np.float32), ##this accounts for
                **kwargs):

        self.units = units
        self.state_size = units   ### this means that the internal state is a "units"-dimensional vector
        self.dt = dt

        gamma, omega, n, eta, kappa = np.array(params).astype(np.float32)
        self.gamma, self.omega, self.n, self.eta, self.kappa = np.array(params).astype(np.float32)
        A = np.array([[-gamma/2, omega],[-omega, -gamma/2]]).astype("float32")
        C = np.sqrt(4*eta*kappa)*np.array([[1.,0.],[0., 0.]]).astype("float32")
        D = np.diag([gamma*(n+0.5) + kappa]*2).astype("float32")
        self.A_matrix = A
        self.C_matrix = C
        self.D_matrix = D[tf.newaxis]
        self.symp = np.array([[0,1],[-1,0]]).astype("float32")

        self.cov_in = tf.convert_to_tensor(cov_in.astype(np.float32))[tf.newaxis]
        self.x_signal = tf.convert_to_tensor(np.array([0.,0.]).astype(np.float32))
        self.true_parameters = tf.convert_to_tensor(true_parameters.astype(np.float32))

        self.initial_states = tf.convert_to_tensor(initial_states)
        self.initial_parameters = tf.convert_to_tensor(initial_parameters)
        self.memory_states = []

        super(GRCell, self).__init__(**kwargs)

    def call(self, inputs, states):
        inns = tf.squeeze(inputs)
        time, dy = inns[0], inns[1:][tf.newaxis]
        print("\n")
        sts = states[0][:,:2]
        cc = states[0][0,2:]
        cov=tf.convert_to_tensor(np.array([[cc[0], cc[1]],[cc[1], cc[2]]]))[tf.newaxis]
        self.memory_states.append(states)
        # self.ss.append(sts)
        #cov = self.cov_in

        XiCov =tf.einsum('bij,jk->bik',cov,self.C_matrix.T)
        XiCovC = tf.matmul(XiCov,self.C_matrix.T)

        output = tf.einsum('ij,bj->bi',self.C_matrix, sts)*self.dt
        A_model = (self.training_params[0]*self.symp -0.5*self.gamma*np.eye(2).astype("float32"))[tf.newaxis]
        dx = tf.einsum('bij,bj->bi',A_model - XiCovC, sts)*self.dt + tf.einsum('bij,bj->bi', XiCov, dy)# + self.ext_fun(self.training_params[0], t=time)*self.x_signal*self.dt ##  + params...
        x = sts + dx

        t1 = tf.einsum('bij,bjk->bik',A_model,cov)
        t2 = tf.einsum('bij,bjk->bik',cov, tf.transpose(A_model, perm=[0,2,1]))
        t3 = self.D_matrix
        t4 = - tf.einsum('bij,bjk->bik',XiCov, tf.transpose(XiCov, perm=[0,2,1]))
        # print(t1)
        # print(t2)
        # print(t3)
        # print(t4)
        cov_dt = tf.einsum('bij,bjk->bik',A_model,cov) + tf.einsum('bij,bjk->bik',cov, tf.transpose(A_model, perm=[0,2,1])) + self.D_matrix - tf.einsum('bij,bjk->bik',XiCov, tf.transpose(XiCov, perm=[0,2,1]))
        new_cov = cov + cov_dt*self.dt
        flat_new_cov = tf.squeeze(new_cov)
        new_states = tf.concat([x, tf.convert_to_tensor([flat_new_cov[0,0], flat_new_cov[1,0], flat_new_cov[1,1]])[tf.newaxis]],axis=-1)
        return x, [new_states]####

    def build(self, input_shape):
        self.training_params = self.add_weight(shape=(1, 1),
                                      initializer='uniform',
                                      name='kernel')
        self.training_params[0].assign(self.initial_parameters)
        self.built = True

    def get_initial_state(self,inputs=None, batch_size=1, dtype=np.float32):
        return self.initial_states

    def reset_states(self,inputs=None, batch_size=1, dtype=np.float32):
        return self.initial_states

class Model(tf.keras.Model):
    def __init__(self,stateful=True, params=[], dt=1e-4,
                true_parameters=[1e1, 1e3, 1., 1., 1e4],
                initial_parameters=[],
                cov_in=np.zeros((2,2)),
                initial_states = np.zeros((1,5)).astype(np.float32),
                batch_size=(10), **kwargs):
        super(Model,self).__init__()

        self.recurrent_layer =tf.keras.layers.RNN(GRCell(units=5, params=params, dt=dt, true_parameters=true_parameters,
                                                    initial_parameters=initial_parameters,
                                                    initial_states=initial_states,cov_in = cov_in),
                                                    return_sequences=True, stateful=True,  batch_input_shape=batch_size)

    def call(self, inputs):
        return self.recurrent_layer(inputs)

    def reset_states(self,states=None):
        self.recurrent_layer.states[0].assign(self.recurrent_layer.cell.get_initial_state())
        return self.recurrent_layer.cell.get_initial_state()

    @property
    def metrics(self):
        return [self.total_loss, self.target_params_record, self.gradient_history]

initial_parameters = np.array([omega]).astype("float32")
true_parameters = np.array([omega]).astype("float32")
epochs = 200
learning_rate = 10

initial_states = (np.array([0, 0, cov_st[0,0], cov_st[1,0], cov_st[1,1]])[np.newaxis]).astype(np.float32)
model = Model(params=params, dt=dt, initial_parameters=initial_parameters,
              true_parameters=true_parameters, initial_states = initial_states,
              cov_in=cov_st, batch_size=tuple([None,None,3]))

model.recurrent_layer.build(tf.TensorShape([1, None, 3])) #None frees the batch_size
model.reset_states()

model.recurrent_layer.cell.memory_states=[]

tt=100
tfsignals = misc_ML.pre_process_data_for_ML(times[:tt], signals[:tt-1])
preds = model(tfsignals[:,:tt,:])

model.recurrent_layer.cell.memory_states[-1]

cov_st



t1 = tf.einsum('bij,bjk->bik',A_model,cov)
t2 = tf.einsum('bij,bjk->bik',cov, tf.transpose(A_model, perm=[0,2,1]))
t3 = self.D_matrix
t4 = - tf.einsum('bij,bjk->bik',XiCov, tf.transpose(XiCov, perm=[0,2,1]))


A.dot(cov_st)
cov_st.dot(A.T)

cov_st.dot(C.T).dot((cov_st.dot(C.T)).T)


cov_st


model.recurrent_layer.cell.memory_states[-10]


model.recurrent_layer.cell.memory_states[0]

plt.plot(preds[0,:tt,0])
plt.plot(states[:tt,0])


model.recurrent_layer.cell.memory_states[-1]
model.recurrent_layer.cell.memory_states[0]

cov_st


nsignals = np.squeeze(tfsignals)
# plt.plot(nsignals[:,0], nsignals[:,1],'--')
plt.plot(times[:tt], states[:tt,0])
plt.plot(nsignals[:,0],np.squeeze(preds)[:,0])





model.reset_states()
model.layers[0].return_sequences = True
preds = model(tfsignals[:,:k,:])

with tf.GradientTape(persistent=True) as tape:
    tape.watch(model.trainable_variables)
    preds = model(tfsignals[:,:k,:])

tape.gradient(preds,model.trainable_variables)


from tqdm import tqdm
grads = []
for k in tqdm(range(1,100,1)):

    model.trainable_variables
    model.reset_states()
    model.layers[0].return_sequences = False
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        preds = model(tfsignals[:,:k,:])
    jj = tape.jacobian(preds,model.trainable_variables)
    grads.append(tf.squeeze(jj[0]))

grads_ =np.stack(grads)

tt=99
plt.plot(times[:tt],grads_[:tt,0])
#plt.plot(times[:tt],states_th[:tt,0])





pupu = tf.squeeze(preds)

tape.gradient(pupu,model.trainable_variables)
preds
tape.gradient(preds, model.trainable_variables)

####



gamma, omega, n, eta, kappa = np.array(params).astype("float32")
A = np.array([[-gamma/2, omega],[-omega, -gamma/2]]).astype("float32")
C = np.sqrt(4*eta*kappa)*np.array([[1.,0.],[0., 0.]]).astype("float32")
D = np.diag([gamma*(n+0.5) + kappa]*2).astype("float32")
cov_st = solve_continuous_are( A.T, C.T, D, np.eye(2))

timms = np.linspace(100, len(times)-1,20).astype("int")
for train_id, tt in enumerate(timms):
    tfsignals = misc_ML.pre_process_data_for_ML(times[:tt], signals[:tt-1])

    save_dir = misc_ML.get_training_save_dir(exp_path, total_time, dt, itraj,train_id)
    os.makedirs(save_dir, exist_ok=True)

    initial_parameters = np.array([10*np.random.uniform()*omega]).astype("float32")
    true_parameters = np.array([omega]).astype("float32")

    epochs = 200
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
    history = model.craft_fit(tfsignals[:,:tt,:], batch_size=min(times[tt],100), epochs=epochs, early_stopping=1e-14)




#
