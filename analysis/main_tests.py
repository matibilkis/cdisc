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
parser.add_argument("--gamma", type=float, default=110.)
parser.add_argument("--Ntraj", type=int, default=30000)
parser.add_argument("--indgamma", type=int, default=0)

args = parser.parse_args()

#gamma = args.gamma
indgamma = args.indgamma
#gammas = np.linspace(110., 10000, 32)
#gamma = gammas[indgamma]
gammas = np.linspace(110., 10000, 32) #---> all batch, before 6/11
gammas = np.linspace(gammas[0] + 10, gammas[1]-10, 32) #---> all batch, before 6/11
gamma = gammas[indgamma]

Ntraj=int(args.Ntraj)

exp_path = "sweep_gamma/{}/".format(gamma)
save_path = get_path_config(exp_path=exp_path,total_time=8., dt=1e-4, noitraj=True)
os.makedirs(save_path, exist_ok=True)


###########################
####### LOAD DATA
total_time = 8.
dt = 1e-4
times = np.arange(0, 8 + dt, dt )
indis = np.linspace(0,len(times)-1, int(1e4)).astype(int)
timind = [times[k] for k in indis]
indis_range = list(range(len(indis)))


B = 6.
dB = .05
boundsB= np.arange(-B,B+dB,dB)

bpos = boundsB[boundsB>=0]
bneg = boundsB[boundsB<0]


deter, stop = {}, {}
stop["_0"] = {i:[] for i in range(1,Ntraj)}
stop["_1"] = {i:[] for i in range(1,Ntraj)}
deter["h0/h1"] ={indb:[0]*len(indis) for indb in range(len(boundsB))}
deter["h1/h0"] = {indb:[0]*len(indis) for indb in range(len(boundsB))}



def load_gamma(gamma, itraj, what="logliks.npy", flip_params=0):
    h0 = gamma0, omega0, n0, eta0, kappa0 = 100., 0., 1., 1., 9
    h1 = gamma1, omega1, n1, eta1, kappa1 = gamma, 0., 1., 1., 9
    if flip_params != 0:
        params = [h0, h1]
    else:
        params = [h1,h0]
    exp_path = str(params)+"/"
    l =load_data(exp_path=exp_path, itraj=itraj, total_time=8., dt=1e-4, what=what)
    return l


n=1
ers = []
for itraj in range(1,Ntraj):
    try:

        [l0_1,l1_1], [l1_0,l0_0] = load_gamma(gamma, itraj=itraj,what="logliks.npy", flip_params=0).T, load_gamma(gamma, itraj=itraj,what="logliks.npy", flip_params=1).T
        log_lik_ratio, log_lik_ratio_swap = l1_1-l0_1, l1_0-l0_0

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
#### type I and II errors

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


#



# cons1, cons0 = [], []
# anals1, anals0 = [], []
# timbin0, timbin1 = [], []
# for indb, b in enumerate(bpos):
#     counts1, bins1 = np.histogram(stops1[indb], 50, normed=True)
#     counts0, bins0 = np.histogram(stops0[indb], 50, normed=True)
#
#     timms1 = np.linspace(0,np.max(bins1), 100)
#     timms0 = np.linspace(0,np.max(bins0), 100)
#
#     timbins1 = .5*(bins1[1:] + bins1[:-1])
#     timbins0 = .5*(bins0[1:] + bins0[:-1])
#
#     cons1.append(counts1)
#     cons0.append(counts0)
#
#     timbin1.append(timbins1)
#     timbin0.append(timbins0)
#
### saving

path_data = save_path+"B{}_db{}_{}/".format(B,dB,Ntraj)
#path_data = get_def_path()+"analysis/{}/".format(Ntraj,mode)
os.makedirs(path_data,exist_ok=True)

with open(path_data+"stop.pickle","wb") as g:
    pickle.dump(stop, g, protocol=pickle.HIGHEST_PROTOCOL)

with open(path_data+"deter.pickle","wb") as f:
    pickle.dump(deter, f, protocol=pickle.HIGHEST_PROTOCOL)

np.save(path_data+"times_to_err_det",times_to_errs_det)
np.save(path_data+"times_to_err_stoch",times_sequential)

# np.save(path_data+"timbin", timbin1)
# np.save(path_data+"deth1h0", deter_data_h1_h0)
# np.save(path_data+"deth0h1", deter_data_h0_h1)
# np.save(path_data+"l0",l0)
# np.save(path_data+"l1",l1)
# np.save(path_data+"cons0",cons0)
# np.save(path_data+"cons1",cons1)

print("data saved in {}\n".format(path_data))
