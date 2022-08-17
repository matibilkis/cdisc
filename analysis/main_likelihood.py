import os
import sys
sys.path.insert(0, os.getcwd())

import numpy as np
from numerics.utilities.misc import *
import numpy as np
from scipy.stats import kstat
from tqdm import tqdm
import argparse

dt = 1e-4
times = np.arange(0, 8 + dt, dt )
indis = np.logspace(0,np.log10(len(times)-1), int(1e4)).astype(int)
timind = [times[k] for k in indis]


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

def get_likelihood_stats(gamma,**kwargs):
    Ntraj = kwargs.get("Ntraj",1000)
    
    ll1 = []
    ll0 = []
    ers=[]
    for itraj in tqdm(range(1,Ntraj)):
        try:
            [l1_1,l0_1], [l0_0,l1_0] = load_gamma(gamma, itraj=itraj,what="logliks.npy", flip_params=0).T, load_gamma(gamma, itraj=itraj,what="logliks.npy", flip_params=1).T
            ll1.append(l1_1-l0_1)
            ll0.append(l1_0-l0_0)    
        except Exception:
            ers.append(itraj)
    #i invert them, since it's actually swapped! (sorry)
    ll0 = np.stack(ll1)
    ll1 = np.stack(ll0)

    print(ll1.shape)

    times = np.arange(0,8. + 1e-4, 1e-4)
    indis = np.linspace(0,len(times)-1, int(1e4)).astype(int)
    timind = [times[k] for k in indis]
    ind_cum = np.linspace(0, ll1.shape[1]-1, 100).astype(int)
    timind_cum = [timind[k] for k in ind_cum]

    cumulants0, cumulants1 = {}, {}

    for k in range(1,5):
        cumulants1[k] = [kstat(ll1[:,t], k) for t in ind_cum]
        cumulants0[k] = [kstat(ll0[:,t], k) for t in ind_cum]

    cum_vals0 = np.stack(cumulants0.values())
    cum_vals1 = np.stack(cumulants1.values())
    k0 = np.concatenate([np.array(timind_cum)[np.newaxis],cum_vals0])
    k1 = np.concatenate([np.array(timind_cum)[np.newaxis],cum_vals1])

    m1 = [np.mean(ll1[:,t]) for t in ind_cum]
    m0 = [np.mean(ll0[:,t]) for t in ind_cum]

    s1 = [np.std(ll1[:,t]) for t in ind_cum]
    s0 = [np.std(ll0[:,t]) for t in ind_cum]

    mm = np.stack([timind_cum,m1, m0, s1,s0])
    
    return k0, k1, mm
                     
    
    
def get_diffS(gamma,**kwargs):
    Ntraj = kwargs.get("Ntraj",1000)
    dfs = []
    ers = []
    for itraj in range(1,Ntraj):
        try:

            st11, st01 = load_gamma(gamma, itraj=itraj,what="states1.npy", flip_params=0).T, load_gamma(gamma, itraj=itraj,what="states0.npy", flip_params=0).T
            diff = st11 - st01
            diffSq = np.einsum('tj,tj->j',diff,diff)
            dfs.append(diffSq)
        except Exception:
            ers.append(itraj)
    dfs = np.stack(dfs)

    cums = {}
    timind_cum = range(0, dfs.shape[1], 100)
    for k in range(1,5):
        cums[k] = [kstat(dfs[:,t], k) for t in timind_cum]

    timcum = np.array([timind[k] for k in timind_cum])[np.newaxis]
    cum_vals = np.stack(cums.values())

    return np.concatenate([np.array(timcum),cum_vals])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gamma", type=float, default=110.)
    parser.add_argument("--Ntraj", type=int, default=1000)
    parser.add_argument("--indgamma", type=int, default=0)

    args = parser.parse_args()

    #gamma = args.gamma
    indgamma = args.indgamma
    gammas = np.linspace(110., 10000, 32)
    gamma = gammas[indgamma]
    Ntraj = int(args.Ntraj)
    exp_path = "sweep_gamma/{}/".format(gamma)

    save_path = get_path_config(exp_path=exp_path,total_time=8., dt=1e-4, noitraj=True)
    os.makedirs(save_path, exist_ok=True)
    
    
    lik_stats1, lik_stats0 = get_likelihood_stats(gamma,Ntraj=Ntraj)
    dif_state_stats = get_diffS(gamma,Ntraj=Ntraj)


    np.save(save_path+"lik1_cum",lik_stats1)
    np.save(save_path+"lik0_cum",lik_stats0)
    np.save(save_path+"st_cum",dif_state_stats)
# 
# 
# 
