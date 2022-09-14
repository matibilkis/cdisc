from numerics.utilities.misc import *
import tensorflow as tf


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
        #cov_st = solve_continuous_are( A.T, C.T, D, np.eye(2))
        self.A_matrix = A
        self.C_matrix = C
        self.D_matrix = D
        #self.XiCov = np.dot(cov_st, C.T)[tf.newaxis]
        #self.XiCovC = np.dot(self.XiCov,C.T)
        self.symp = np.array([[0,1],[-1,0]]).astype("float32")

        self.cov_in = tf.convert_to_tensor(cov_in.astype(np.float32))[tf.newaxis]
        self.x_signal = tf.convert_to_tensor(np.array([0.,0.]).astype(np.float32))
        self.true_parameters = tf.convert_to_tensor(true_parameters.astype(np.float32))

        self.initial_states = tf.convert_to_tensor(initial_states)
        self.initial_parameters = tf.convert_to_tensor(np.array([omega]))
        self.ss = []

        super(GRCell, self).__init__(**kwargs)
    #
    # def ext_fun(self, params,t):
    #     return params[0]*tf.cos(params[1]*t)

    def call(self, inputs, states):
        inns = tf.squeeze(inputs)
        time, dy = inns[0], inns[1:][tf.newaxis]

        sts = states[0][:,:2]
        self.ss.append(sts)
        cov = self.cov_in

        XiCov =tf.einsum('bij,jk->bik',cov,self.C_matrix.T)
        XiCovC = tf.matmul(XiCov,self.C_matrix.T)

        output = tf.einsum('ij,bj->bi',self.C_matrix, sts)*self.dt
        A_model = (self.training_params[0]*self.symp -0.5*self.gamma*np.eye(2).astype("float32"))[tf.newaxis]
        dx = tf.einsum('bij,bj->bi',A_model - XiCovC, sts)*self.dt + tf.einsum('bij,bj->bi', XiCov, dy)# + self.ext_fun(self.training_params[0], t=time)*self.x_signal*self.dt ##  + params...
        x = sts + dx

        cov_dt = tf.einsum('bij,bjk->bik',A_model,cov) + tf.einsum('bij,bjk->bik',cov, tf.transpose(A_model, perm=[0,2,1])) + self.D_matrix - tf.einsum('bij,bjk->bik',XiCov, tf.transpose(XiCov, perm=[0,2,1]))
        new_cov = cov + cov_dt*self.dt
        new_states = tf.concat([x, tf.zeros((1,3))],axis=-1)
        return output, [new_states]####

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

        save_dir = kwargs.get("save_dir","/")
        self.recurrent_layer =tf.keras.layers.RNN(GRCell(units=5, params=params, dt=dt, true_parameters=true_parameters,
                                                    initial_parameters=initial_parameters,
                                                    initial_states=initial_states,cov_in = cov_in),
                                                    return_sequences=True, stateful=True,  batch_input_shape=batch_size)
        self.total_loss = Metrica(name="LOSS")
        self.target_params_record = Metrica(name="PARAMS")
        self.gradient_history = Metrica(name="GRADS")
        self.save_dir = save_dir

    def call(self, inputs):
        return self.recurrent_layer(inputs)

    def reset_states(self,states=None):
        self.recurrent_layer.states[0].assign(self.recurrent_layer.cell.get_initial_state())
        return self.recurrent_layer.cell.get_initial_state()

    @property
    def metrics(self):
        return [self.total_loss, self.target_params_record, self.gradient_history]

    @tf.function
    def train_step(self, data):
        inputs, times_dys = data
        dys = times_dys[:,:,1:]
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            preds = self(inputs)
            diff = tf.squeeze(preds - dys)
            loss = tf.reduce_sum(tf.einsum('bj,bj->b',diff,diff))

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss.update_state(loss)
        self.target_params_record.update_state(self.trainable_variables[0])
        self.gradient_history.update_state(grads)

        return {k.name:k.result() for k in self.metrics}


    def craft_fit(self, tfsignals, batch_size=50, epochs=10,early_stopping=1e-6):
        if tfsignals.shape[1]%batch_size != 0:
            raise ValueError("check your batch_size and training set, i can't split that")
        Ns = tfsignals.shape[1]/batch_size
        batched_data  = tf.split(tfsignals, int(Ns), axis=1)

        history = []
        for epoch in range(epochs):
            self.reset_states()
            for batch in batched_data:
                bb = self.train_step((batch,batch))
                history+=[bb]
            loss =  np.squeeze(bb["LOSS"].numpy())
            params =  np.squeeze(bb["PARAMS"].numpy())
            grads = np.squeeze(bb["GRADS"].numpy())
            print("\r EPOCH {}/{}   loss:{}    initial_params   {}    params{}    true_params {}    grads{}".format(epoch, epochs,loss,np.round(self.recurrent_layer.cell.initial_parameters,2), np.round(params,2), np.round(self.recurrent_layer.cell.true_parameters,2), np.round(grads,3)),end="")

            loss_save = [history[k]["LOSS"].numpy() for k in range(len(history))]
            grads_save = np.squeeze([history[k]["GRADS"].numpy() for k in range(len(history))])
            params_save = np.squeeze([history[k]["PARAMS"].numpy() for k in range(len(history))])

            for i,j in zip([loss_save, grads_save, params_save], ["loss", "grads", "params"]):
                np.save(self.save_dir+j,i)

            if loss<early_stopping:
                print("Early stopped at loss {}".format(loss))
                break
        return history


class Metrica(tf.keras.metrics.Metric):
    """
    This helps to monitor training (for instance one out of different losses),
    but you can also monitor gradients magnitude for example.
    """
    def __init__(self, name):
        super(Metrica, self).__init__()
        self._name=name
        self.metric_variable = tf.convert_to_tensor(np.zeros((1,2)).astype(np.float32))

    def update_state(self, new_value):
        self.metric_variable = new_value

    def result(self):
        return self.metric_variable

    def reset_states(self):
        self.metric_variable = tf.convert_to_tensor(np.zeros((1,2)).astype(np.float32))
