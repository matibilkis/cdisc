import os
import sys
sys.path.insert(0, os.getcwd())

import numpy as np
from numerics.utilities.misc import *
import numpy as np
from scipy.stats import kstat
from tqdm import tqdm
import argparse
import pickle
import matplotlib.pyplot as plt 


omega0, omega1 = 1e4, 1.05e4
omega_pro = (omega0 + omega1)/2
period = (2*np.pi/omega_pro)
dt = period/100
total_time = 100*period
times = np.arange(0, total_time+ dt, dt )
indis = np.linspace(0,len(times)-1, int(1e4)).astype(int)
timind = [times[k] for k in indis]

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

def get_stats(gamma,**kwargs):
    Ntraj = kwargs.get("Ntraj",1000)
    
    ll1 = []
    ll0 = []
    ers=[]
    dfs = []
    for itraj in tqdm(range(1,Ntraj)):
        try:
            [l0_1,l1_1], [l1_0,l0_0] = load_traj(itraj=itraj,what="logliks.npy", flip_params=0).T, load_traj(itraj=itraj,what="logliks.npy", flip_params=1).T

            ll1.append(l1_1-l0_1)
            ll0.append(l1_0-l0_0) 
            
            st11, st01 = load_traj(itraj=itraj,what="states1.npy", flip_params=0).T, load_traj(itraj=itraj,what="states0.npy", flip_params=0).T
            diff = st11 - st01
            diffSq = np.einsum('tj,tj->j',diff,diff)
            dfs.append(diffSq)
            
        except Exception:
            ers.append(itraj)
    #i invert them, since it's actually swapped! (sorry)
    ll0 = np.stack(ll0)
    ll1 = np.stack(ll1)
    
    dfs = np.stack(dfs)
    
    print(ll1.shape)

    
    omega0, omega1 = 1e4, 1.05e4
    omega_pro = (omega0 + omega1)/2
    period = (2*np.pi/omega_pro)
    dt = period/100
    total_time = 100*period
    times = np.arange(0, total_time+ dt, dt )
    indis = np.linspace(0,len(times)-1, int(1e4)).astype(int)
    timind = [times[k] for k in indis]
    
    ind_cum = np.linspace(0, ll1.shape[1]-1, 100).astype(int)
    timind_cum = [timind[k] for k in ind_cum]
    ind_st = np.linspace(0, ll1.shape[1]-1, 10).astype(int)
    

    lim = min(Ntraj, int(1e4))
    sll0 = np.stack([ll0[:lim,k] for k in ind_st]).T
    sll1 = np.stack([ll1[:lim,k] for k in ind_st]).T
    sDiff = np.stack([dfs[:lim,k] for k in ind_st]).T

    
    cumulants0, cumulants1, cumdfs = {}, {}, {}

    for k in range(1,5):
        cumulants1[k] = [kstat(ll1[:,t], k) for t in ind_cum]
        cumulants0[k] = [kstat(ll0[:,t], k) for t in ind_cum]
        cumdfs[k] = [kstat(dfs[:,t],k) for t in ind_cum]
        
    cum_vals0 = np.stack(cumulants0.values())
    cum_vals1 = np.stack(cumulants1.values())
    cumdfs_vals = np.stack(cumdfs.values())
    
    k0 = np.concatenate([np.array(timind_cum)[np.newaxis],cum_vals0])
    k1 = np.concatenate([np.array(timind_cum)[np.newaxis],cum_vals1])

    ks = np.concatenate([np.array(timind_cum)[np.newaxis],cumdfs_vals])
    m1 = [np.mean(ll1[:,t]) for t in ind_cum]
    m0 = [np.mean(ll0[:,t]) for t in ind_cum]

    s1 = [np.std(ll1[:,t]) for t in ind_cum]
    s0 = [np.std(ll0[:,t]) for t in ind_cum]
    mm = np.stack([timind_cum,m1, m0, s1,s0])

    
    dm1 = [np.mean(dfs[:,t]) for t in ind_cum]
    ds1 = [np.std(dfs[:,t]) for t in ind_cum]

    mdfs = np.stack([timind_cum,dm1, ds1])
    
    return k0, k1, ks, mm, mdfs, sll0, sll1, sDiff

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--Ntraj", type=int, default=5000)
    args = parser.parse_args()

    Ntraj = int(args.Ntraj)
    omega0, omega1 = 1e4, 1.05e4
    exp_path = "freqs/{}/".format([omega0,omega1])
    
    omega_pro = (omega0 + omega1)/2
    period = (2*np.pi/omega_pro)
    dt = period/100
    total_time = 100*period

    save_path = get_path_config(exp_path=exp_path,total_time=total_time, dt=dt, noitraj=True)
    os.makedirs(save_path, exist_ok=True)
    
    
    cum0, cum1, cumdiff, momlik, momdif, sll0, sll1, sdif = get_stats(gamma,Ntraj=Ntraj)
    
    np.save(save_path+"lik1_cum",cum1)
    np.save(save_path+"lik0_cum",cum0)
    np.save(save_path+"diff_cum",cumdiff)
    
    np.save(save_path+"momlik",momlik)
    np.save(save_path+"momdiff",momdif)

    np.save(save_path+"statsL0",sll0)
    np.save(save_path+"statsL1",sll1)
    np.save(save_path+"statsdiff",sdif)
    
   
    
    print("well done! now i'll plot this!")
    
    path = get_def_path()+exp_path+"T_{}_dt_{}/".format(total_time,dt)

    path = path + "figures/"
    os.makedirs(path,exist_ok=True)
    dif_cumm = cumdiff#np.load(path+"diff_cum.npy")
    statsL0 = sll0#np.load(path+"statsL0.npy")
    statsL1 = sll1#np.load(path+"statsL1.npy")
    statsdiff = sdif#np.load(path+"statsdiff.npy")
    
    
    print("plotting cumulant ell1")
    
    plt.figure()
    ax=plt.subplot(311)
    ax.set_title("kumulant l_1 divided by time",size=20)
    ax.plot(dif_cumm[0], np.abs(cum1[1])/cum1[0])
    ax=plt.subplot(312)
    ax.plot(dif_cumm[0], np.abs(cum1[2])/cum1[0])
    ax=plt.subplot(313)
    ax.plot(dif_cumm[0], np.abs(cum1[3])/cum1[0])
    ax.set_xlabel("time")
    plt.savefig(path+"cuml1_t.pdf")
    
    
    print("plotting cumulant diff States")
    plt.figure()
    ax=plt.subplot(311)
    ax.set_title("kumulant (s_1 - s_0)**2 over time",size=20)
    
    ax.plot(dif_cumm[0], dif_cumm[1])
    ax=plt.subplot(312)
    ax.plot(dif_cumm[0], dif_cumm[2])
    ax=plt.subplot(313)
    ax.plot(dif_cumm[0], dif_cumm[3])
    ax.set_xlabel("time")
    plt.savefig(path+"cum_deltas_t.pdf")
    
    

    B = 8.
    dB = .2
    boundsB= np.arange(-B,B+dB,dB)

    bpos = boundsB[boundsB>=0]
    bneg = boundsB[boundsB<0]
    def get_histogram(stop):
        B = 8.
        dB = .2
        boundsB= np.arange(-B,B+dB,dB)

        bpos = boundsB[boundsB>=0]
        bneg = boundsB[boundsB<0]


        stops0 = [[] for k in range(len(bpos))]
        stops1 = [[] for k in range(len(bpos))]

        values1 = list(stop["_1"].values())
        values0 = list(stop["_0"].values())
        for k,val in enumerate(values1):
            if len(val)!=0:
                for indb in range(len(val)):
                    if ~np.isnan([values1[k][indb]])[0] == True:
                        stops1[indb].append(np.squeeze(values1[k][indb]))

        for k,val in enumerate(values0):
            if len(val)!=0:
                for indb in range(len(val)):
                    if ~np.isnan([values0[k][indb]])[0] == True:
                        stops0[indb].append(np.squeeze(values0[k][indb]))


        cons1, cons0 = [], []
        anals1, anals0 = [], []
        timbin0, timbin1 = [], []
        for indb, b in enumerate(bpos):
            counts1, bins1 = np.histogram(stops1[indb], 50, normed=True)
            counts0, bins0 = np.histogram(stops0[indb], 50, normed=True)

            timms1 = np.linspace(0,np.max(bins1), 100)
            timms0 = np.linspace(0,np.max(bins0), 100)

            timbins1 = .5*(bins1[1:] + bins1[:-1])
            timbins0 = .5*(bins0[1:] + bins0[:-1])

            cons1.append(counts1)
            cons0.append(counts0)

            timbin1.append(timbins1)
            timbin0.append(timbins0)
        return timbin1, timbin0, cons1, cons0

    def fit_2moments(timind,l1_mean, l1_std):

        ini = 10
        fini = -1

        sqrtimind = np.array(np.sqrt(timind))
        timind = np.array(timind)

        mu, oomu = np.polyfit(timind[ini:fini], np.abs(l1_mean)[ini:fini],1)
        sigma, oosig = np.polyfit(np.array(np.sqrt(timind))[ini:fini], l1_std[ini:fini],1)

        return [mu, oomu], [sigma, oosig]
# 
# 
# 
    with open(save_path+"stop.pickle","rb") as f:
       stop = pickle.load(f)#, protocol=pickle.HIGHEST_PROTOCOL)
    timbin1, timbin0, cons1, cons0 = get_histogram(stop)

    def give_me_gauss(b, mu,sigma,xrange):
        gauss = lambda x,m,g: np.exp(-((x-m)**2)/(2*g**2))/np.sqrt(2*np.pi*g**2)
        xx = np.linspace(xrange[0], xrange[1],500)
        return xx, np.array([gauss(x, mu, sigma) for x in xx])
    
    
    
    timstats = [times[k] for k in np.linspace(0,len(times)-1,statsL1.shape[1]).astype(int)]

    [mu1, oomu1], [sigma1, oosig1] = fit_2moments(momlik[0],momlik[1], momlik[3])
    [mu0, oomu0], [sigma0, oosig0] = fit_2moments(momlik[0],momlik[2], momlik[4])

    for indistats in [1, 5, -1]:
        print("plotting indstats {}".format(indistats))

        c1, b1 = np.histogram(statsL1[:,indistats], bins=50, normed=True)
        c0, b0 = np.histogram(statsL0[:,indistats], bins=50, normed=True)



        plt.figure(figsize=(20,10))
        ax=plt.subplot()

        t = timstats[indistats]
        ax.set_title("likelihood distribution at t={}".format(np.round(t,2)),size=20)


        ax.bar(b1[:-1], c1, edgecolor="black", width=b1[1]-b1[0], alpha=0.75)
        ax.bar(b0[:-1], c0, edgecolor="black",width=b0[1]-b0[0], alpha=0.75)

        xrange = (1.2*np.min([b0,b1]), 1.2*np.max([b0,b1]))

        bb,gg = give_me_gauss(b1, -mu1*t,sigma1*np.sqrt(t), xrange)

        ax.plot(bb,gg, color="black", linewidth=3, label=r'$\mu, \sigma$')
        bb,gg = give_me_gauss(b1, -mu1*t,np.sqrt(2*mu1)*np.sqrt(t), xrange)
        ax.plot(bb,gg, '--', color="red",linewidth=3, label="mu only")

        bb,gg = give_me_gauss(b0, mu0*t,sigma0*np.sqrt(t), xrange)
        ax.plot(bb,gg,color="blue", linewidth=3)
        bb,gg = give_me_gauss(b0, mu0*t,np.sqrt(2*mu0)*np.sqrt(t), xrange)
        ax.plot(bb,gg, '--', color="red",linewidth=3)

        ax.legend(prop={"size":20})
        plt.savefig(path+"liks_distrib_indtime_{}.pdf".format(indistats))
        
        
    def prob_craft(t, b, mu, S):
        div = (np.sqrt(2*np.pi)*S*(t**(3/2)))
        return  abs(b)*np.exp(-((abs(b)-mu*t)**2)/(2*t*(S**2)))/div

    #muu = l1[-1]/timind[-1]

    for indb in [5, 15, 20,30, 40]:
        print("plotting b={}".format(indb))
        LS, TS = 30, 20
        plt.figure(figsize=(20,10))
        ax = plt.subplot(121)

        timm =  np.linspace(np.min(timbin1),np.max(timbin1),100)

        [mu0, oomu0], [sigma0, oosig0] = fit_2moments(momlik[0],momlik[2], momlik[4])
        [mu1, oomu1], [sigma1, oosig1] = fit_2moments(momlik[0],momlik[1], momlik[3])


        popo = [prob_craft(tt, bpos[indb] , mu1, np.sqrt(2*mu1)) for tt in timm]
        good = [prob_craft(tt, bpos[indb] , mu1, sigma1) for tt in timm]

        ax.set_title("p(\tau|\h_1) \nb={}".format(np.round(bpos[indb],3)), size=20)
        ax.plot(timm,good, linewidth=4,color="black", label="mu & sigma")
        ax.plot(timm,popo, linewidth=4, color="purple", label="mu only")

        ax.bar(timbin1[indb], cons1[indb], width=timbin1[indb][1]-timbin1[indb][0], color="red", alpha=0.75, edgecolor="black",)#, label="simulations")
        ax.set_xlabel(r'$\tau$',size=LS)
        ax.set_ylabel(r'$P(\tau)$', size=LS)
        ax.tick_params(axis='both', which='major', labelsize=TS)
        ax.legend(prop={"size":25})

        ax=plt.subplot(122)

        popo = [prob_craft(tt, bpos[indb] , abs(mu0), np.sqrt(2*abs(mu0))) for tt in timm]
        good = [prob_craft(tt, bpos[indb] , abs(mu0), abs(sigma0)) for tt in timm]

        ax.set_title("p(\tau|\h_0) \nb={}".format(np.round(bpos[indb],3)), size=20)
        ax.plot(timm,good, linewidth=4,color="black", label="mu & sigma")
        ax.plot(timm,popo, linewidth=4, color="purple", label="mu only")

        ax.bar(timbin0[indb], cons0[indb], width=timbin0[indb][1]-timbin0[indb][0], color="red", alpha=0.75, edgecolor="black",)#, label="simulations")
        ax.set_xlabel(r'$\tau$',size=LS)
        ax.set_ylabel(r'$P(\tau)$', size=LS)
        ax.tick_params(axis='both', which='major', labelsize=TS)
        ax.legend(prop={"size":25})
        plt.savefig(path+"stopping_distrib_{}.pdf".format(indb))
