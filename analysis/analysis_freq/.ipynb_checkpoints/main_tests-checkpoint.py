import os
import sys
sys.path.insert(0, os.getcwd())

import numpy as np
from numerics.utilities.misc import *
import pickle
import numpy as np
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--Ntraj", type=int, default=30000)
args = parser.parse_args()


Ntraj=int(args.Ntraj)
omega0, omega1 = 1e4, 1.05e4
omega_pro = (omega0 + omega1)/2
period = (2*np.pi/omega_pro)
dt = period/100
total_time = 100*period
exp_path = "freqs/{}/".format([omega0,omega1])
save_path = get_path_config(exp_path=exp_path,total_time=total_time, dt=dt, noitraj=True)
os.makedirs(save_path, exist_ok=True)


###########################
# ###### LOAD DATA
times = np.arange(0, total_time+ dt, dt )
indis = np.linspace(0,len(times)-1, int(1e4)).astype(int)
timind = [times[k] for k in indis]
indis_range = list(range(len(indis)))


B = 100
dB = 10
boundsB= np.arange(-B,B+dB,dB)

bpos = boundsB[boundsB>=0]
bneg = boundsB[boundsB<0]


deter, stop = {}, {}
stop["_0"] = {i:[] for i in range(1,Ntraj)}
stop["_1"] = {i:[] for i in range(1,Ntraj)}
deter["h0/h1"] ={indb:[0]*len(indis) for indb in range(len(boundsB))}
deter["h1/h0"] = {indb:[0]*len(indis) for indb in range(len(boundsB))}



def load_traj(itraj, what="logliks.npy", flip_params=0):
    gamma0 = gamma1 = 100
    eta0 = eta1 = 1
    kappa0 = kappa1 = 1e3
    n0 = n1 = 1
    omega0, omega1 = 1e4, 1.05e4

    h0 = [gamma0, omega0, n0, eta0, kappa0]
    h1 = [gamma1, omega1, n1, eta1, kappa1]

    omega_pro = (omega0 + omega1)/2
    period = (2*np.pi/omega_pro)
    dt = period/100
    total_time = 100*period
    if flip_params == 1:
        params = [h0, h1]
    else:
        params = [h1,h0]
    exp_path = str(params)+"/"

    l =load_data(exp_path=exp_path, itraj=itraj, total_time=total_time, dt=dt, what=what)
    return l

l1t, l0t = [],[]
n=1
ers = []
for itraj in range(1,Ntraj):
    try:

        [l0_1,l1_1], [l1_0,l0_0] = load_traj(itraj=itraj,what="logliks.npy", flip_params=0).T, load_traj(itraj=itraj,what="logliks.npy", flip_params=1).T
        log_lik_ratio, log_lik_ratio_swap = l1_1-l0_1, l1_0-l0_0

        #l1t.append(log_lik_ratio[::100])
        #l0t.append(log_lik_ratio_swap[::100])
        
        for indb,b in enumerate(boundsB):
            deter["h0/h1"][indb] += ((log_lik_ratio[indis_range] < b).astype(int)  - deter["h0/h1"][indb])/n
            deter["h1/h0"][indb] += ((log_lik_ratio_swap[indis_range] > b).astype(int)  - deter["h1/h0"][indb])/n
            if b>=0:
                stop["_1"][itraj].append(get_stop_time(log_lik_ratio, b, timind))
                stop["_0"][itraj].append(get_stop_time(log_lik_ratio_swap, b,timind))
        n+=1
    except Exception:
        ers.append(itraj)
        print("error {}".format(itraj))

###########################
# ### type I and II errors

alphas = list(deter["h1/h0"].values())
betas = list(deter["h0/h1"].values())

alphas = np.stack(alphas)
betas = np.stack(betas)

avg_err= lambda b: (1-np.exp(-abs(b)))/(np.exp(abs(b)) - np.exp(-abs(b)))

errs_bound = np.array([avg_err(b) for b in bpos])
tot_err = 0.5*(alphas+betas)

symmetric = tot_err[np.argmin(np.abs(boundsB)),:]
times_to_errs_det = np.array([timind[np.argmin(np.abs(symmetric - bound_err))] for bound_err in errs_bound])




stops0 = [[] for k in range(len(bpos))]
stops1 = [[] for k in range(len(bpos))]

values1 = list(stop["_1"].values())
values0 = list(stop["_0"].values())
for k,val in enumerate(values1):
    if len(val)!=0:
        for indb in range(len(val)):
            if ~np.isnan([values1[k][indb]])[0] == True:
                stops1[indb].append(np.squeeze(values1[k][indb]))#

for k,val in enumerate(values0):
    if len(val)!=0:
        for indb in range(len(val)):
            if ~np.isnan([values0[k][indb]])[0] == True:
                stops0[indb].append(np.squeeze(values0[k][indb]))


### sequential test
avg_times1 = np.array([np.mean(k) for k in stops1])
avg_times0 = np.array([np.mean(k) for k in stops0])

times_sequential = 0.5*(avg_times0 + avg_times1)


path_data = save_path+"B{}_db{}_{}/".format(B,dB,Ntraj)
os.makedirs(path_data,exist_ok=True)

with open(path_data+"stop.pickle","wb") as g:
    pickle.dump(stop, g, protocol=pickle.HIGHEST_PROTOCOL)

with open(path_data+"deter.pickle","wb") as f:
    pickle.dump(deter, f, protocol=pickle.HIGHEST_PROTOCOL)

np.save(path_data+"times_to_err_det",times_to_errs_det)
np.save(path_data+"times_to_err_stoch",times_sequential)

print("data saved in {}\n".format(path_data))
