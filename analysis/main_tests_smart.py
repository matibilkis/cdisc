import os
import sys
sys.path.insert(0, os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
from numerics.utilities.misc import *
import pickle
import numpy as np
from tqdm import tqdm
import argparse
from scipy.special import erf

Ntraj = 1000

###########################
####### LOAD DATA
total_time = 8.
dt = 1e-4
times = np.arange(0, 8 + dt, dt )
indis = np.linspace(0,len(times)-1, int(1e4)).astype(int)
timind = [times[k] for k in indis]
indis_range = list(range(len(indis)))

def load_gamma(gamma, itraj, what="logliks.npy", flip_params=0):
    gamma1 = 14*2*np.pi
    gamma0 = 19*2*np.pi #(Hz)
    eta1 = 0.9
    eta0 = 0.9
    n1 = 14.0
    n0 = 14.0
    kappa1 = 2*np.pi*360
    kappa0 = 2*np.pi*360 #(Hz)
    omega0 = omega1 = 0.



    h0 = [gamma0, omega0, n0, eta0, kappa0]
    h1 = [gamma1, omega1, n1, eta1, kappa1]

    if flip_params != 0:
        params = [h0, h1]
    else:
        params = [h1,h0]
    exp_path = str(params)+"/"
    l =load_data(exp_path=exp_path, itraj=itraj, total_time=8., dt=1e-4, what=what)
    return l

gamma = 1
itraj = 1
[l0_1,l1_1], [l1_0,l0_0] = load_gamma(gamma, itraj=itraj,what="logliks.npy", flip_params=0).T, load_gamma(gamma, itraj=itraj,what="logliks.npy", flip_params=1).T

Ntraj = 5000
B = 4
dB = .2
boundsB= np.arange(-B,B+dB,dB)

bpos = boundsB[boundsB>=0]
bneg = boundsB[boundsB<0]

deter, deter_smart, stop = {}, {}, {}

stop["_0"] = {i:[] for i in range(1,Ntraj)}
stop["_1"] = {i:[] for i in range(1,Ntraj)}
deter["h0/h1"] ={indb:[0]*len(indis) for indb in range(len(boundsB))}
deter["h1/h0"] = {indb:[0]*len(indis) for indb in range(len(boundsB))}

deter_smart["h0/h1"] ={indb:[0]*len(indis) for indb in range(len(boundsB))}
deter_smart["h1/h0"] = {indb:[0]*len(indis) for indb in range(len(boundsB))}


itraj = 1
b=1
[l0_1,l1_1], [l1_0,l0_0] = load_gamma(gamma, itraj=itraj,what="logliks.npy", flip_params=0).T, load_gamma(gamma, itraj=itraj,what="logliks.npy", flip_params=1).T
log_lik_ratio, log_lik_ratio_swap = l1_1-l0_1, l1_0-l0_0


n=1
ers = []
ll1, ll0 = [], []
for itraj in tqdm(range(1,5000)):
    try:

        [l0_1,l1_1], [l1_0,l0_0] = load_gamma(gamma, itraj=itraj,what="logliks.npy", flip_params=0).T, load_gamma(gamma, itraj=itraj,what="logliks.npy", flip_params=1).T
        log_lik_ratio, log_lik_ratio_swap = l1_1-l0_1, l1_0-l0_0
        ll1.append(log_lik_ratio)
        ll0.append(log_lik_ratio_swap)

        for indb,b in enumerate(boundsB):
            deter["h0/h1"][indb] += ((log_lik_ratio[indis_range] < b).astype(int)  - deter["h0/h1"][indb])/n
            deter["h1/h0"][indb] += ((log_lik_ratio_swap[indis_range] > b).astype(int)  - deter["h1/h0"][indb])/n

            deter_smart["h0/h1"][indb] += (np.exp(log_lik_ratio_swap[indis_range])*(log_lik_ratio_swap[indis_range] < b).astype(int)  - deter_smart["h0/h1"][indb])/n
            deter_smart["h1/h0"][indb] += (np.exp(-log_lik_ratio[indis_range])*(log_lik_ratio[indis_range] > b).astype(int)  - deter_smart["h1/h0"][indb])/n
            if b>=0:
                stop["_1"][itraj].append(get_stop_time(log_lik_ratio, b, timind))
                stop["_0"][itraj].append(get_stop_time(log_lik_ratio_swap, b,timind))
        n+=1
    except Exception:
        ers.append(itraj)
        print("error {}".format(itraj))


path_data = "data_smart/"
os.makedirs(path_data, exist_ok=True)

with open(path_data+"stop.pickle","wb") as f:
    pickle.dump(stop, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(path_data+"deter.pickle","wb") as f:
    pickle.dump(deter, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(path_data+"deter_smart.pickle","wb") as f:
    pickle.dump(deter_smart, f, protocol=pickle.HIGHEST_PROTOCOL)


ll1, ll0 = np.stack(ll1), np.stack(ll0)

np.save(path_data+"ll0.npy",ll0)
np.save(path_data+"ll1.npy",ll1)

###########################
#### type I and II errors

len(stop["_0"].keys())
path_data = "data_smart/"

with open(path_data+"stop.pickle","rb") as f:
    stop = pickle.load( f)#, protocol=pickle.HIGHEST_PROTOCOL)

with open(path_data+"deter.pickle","rb") as f:
    deter = pickle.load(f)#, protocol=pickle.HIGHEST_PROTOCOL

with open(path_data+"deter_smart.pickle","rb") as f:
    deter_smart = pickle.load(f)#, protocol=pickle.HIGHEST_PROTOCOL

ll0 = np.load(path_data+"ll0.npy")
ll1 = np.load(path_data+"ll1.npy")

#### convert data to probabilities & get times
alphas = list(deter["h1/h0"].values())
betas = list(deter["h0/h1"].values())

alphas = np.stack(alphas)
betas = np.stack(betas)

alphas_smart = list(deter_smart["h1/h0"].values())
betas_smart = list(deter_smart["h0/h1"].values())

alphas_smart = np.stack(alphas_smart)
betas_smart = np.stack(betas_smart)


ml1 = np.mean(ll1, axis=0)
ml0 = np.mean(ll0, axis=0)

plt.plot(timind,ml1)
plt.plot(timind,ml0)


np.polyfit(timind,ml1,1)
np.polyfit(timind,ml0,1)



len(indis_range)
k= 300
c1, ti1 = np.histogram(ll1[:,indis_range[k]], bins=50)
c0, ti0 = np.histogram(ll0[:,indis_range[k]], bins=50)

ax=plt.subplot(111)
ax.plot(ti0[:-1],c0)
ax.plot(ti1[:-1],c1)
ax.axvline(boundsB[0],color="black")

ind=0#len(bneg)#10+len(bneg)
plt.figure(figsize=(40,10))
ax=plt.subplot()
ax.plot(timind,betas_smart[ind,:],color="black")
ax.plot(timind,betas[ind,:])
ax.axvline(timind[k],color="green",linewidth=5)





stops0 = [[] for k in range(len(bpos))]
stops1 = [[] for k in range(len(bpos))]

len(val)
len(boundsB)
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
##### FIN loading-data


<t>_1 ( error )     ---> beta p(h_0 | h_1)


### Comparing beta with avg_times1
epsilon = lambda b: np.exp(-abs(b))/(1 + np.exp(-abs(b)))  # =alpha = beta SPRT
indb0 = np.argmin(np.abs(boundsB))

dete = [timind[np.argmin(np.abs(betas[indb+len(bneg),:] - epsilon(b)))] for indb,b in enumerate(bpos)]

plt.figure(figsize=(10,5))
ax=plt.subplot(121)
ax.plot(bpos, dete,'.')
ax.plot(bpos, avg_times1,'.')
ax=plt.subplot(122)
ax.plot(dete/avg_times1)
ax.axhline(4,color="black")




indb0 = np.argmin(np.abs(boundsB))
boundsB[indb0]
epsilon = lambda b: np.exp(-abs(b))/(1 + np.exp(-abs(b)))  # =alpha = beta SPRT
dete = [timind[np.argmin(np.abs(betas[indb0,:] - epsilon(b)))] for indb,b in enumerate(bpos)]

plt.figure(figsize=(10,5))
ax=plt.subplot(121)
ax.plot(bpos, dete,'.')
ax.plot(bpos, avg_times1,'.')
ax=plt.subplot(122)
ax.plot(dete/avg_times1)
ax.axhline(4,color="black")


ini = -10
fini = -1
sdet, odet = np.polyfit(bpos[ini:fini], dete[ini:fini],1)
sseq, oseq = np.polyfit(bpos[ini:fini], times_sequential[ini:fini],1)

sdet
sseq

sdet/sseq


counts1, bins1 = np.histogram(ll1, 50, normed=True)
counts0, bins0 = np.histogram(ll0, 50, normed=True)




plt.plot(alphas[indb0,:])
plt.plot(betas[indb0,:])
plt.loglog()

















### Comparing alpha with avg_times1
epsilon = lambda b: np.exp(-abs(b))/(1 + np.exp(-abs(b)))
indb0 = np.argmin(np.abs(boundsB))
dete = [timind[np.argmin(np.abs(alphas[len(bneg)-indb,:] - epsilon(b)))] for indb,b in enumerate(bpos)]

plt.figure(figsize=(10,5))
ax=plt.subplot(121)
ax.plot(bpos, dete,'.')
ax.plot(bpos, avg_times0,'.')
ax=plt.subplot(122)
ax.plot(dete/avg_times0)
ax.axhline(4,color="black")



alphas.shape

ini=0
fini = np.argmin(np.abs(np.array(timind)-2))
ax=plt.subplot()
for indb in range(len(bneg)):
    ax.plot(timind[ini:fini],alphas[indb,:][ini:fini])




## Symmetric (b = 0)
#epsilon = lambda b: np.exp(-b)/(1 + np.exp(-b))
epsilon = lambda o: (1-np.exp(-abs(o)))/(np.exp(abs(o)) - np.exp(-abs(o)))

indb0 = np.argmin(np.abs(boundsB))
symm = (.5*(betas + alphas))[indb0,:]

plt.plot(timind, symm)
for k,b in enumerate([epsilon(b) for b in bpos]):
    plt.scatter(dete[k], b)

dete = [timind[np.argmin(np.abs(symm - epsilon(b)))] for b in bpos]


plt.figure(figsize=(10,5))
ax=plt.subplot(121)
ax.plot(bpos, dete,'.')
ax.plot(bpos, times_sequential,'.')
#ax.plot(bpos[ini:fini], bpos[ini:fini]*sdet + odet)
#ax.plot(bpos[ini:fini], bpos[ini:fini]*sseq + oseq)
ax=plt.subplot(122)
ax.plot(dete/times_sequential)
ax.axhline(4,color="black")




plt.figure(figsize=(10,5))
ax=plt.subplot(121)
ax.plot(bpos, dete,'.')
ax.plot(bpos, times_sequential,'.')
ax.plot(bpos[ini:fini], bpos[ini:fini]*sdet + odet)
ax.plot(bpos[ini:fini], bpos[ini:fini]*sseq + oseq)
ax=plt.subplot(122)
ax.plot(dete/times_sequential)
ax.axhline(4,color="black")

ini = -10
fini = -1
sdet, odet = np.polyfit(bpos[ini:fini], dete[ini:fini],1)
sseq, oseq = np.polyfit(bpos[ini:fini], times_sequential[ini:fini],1)


sdet/sseq


### Deterministic \alpha + \beta
dete = [timind[np.argmin(np.abs((betas[indb,:] + alphas[len(bneg)-indb,:]) - epsilon(b)))] for indb,b in enumerate(bpos)]
plt.figure(figsize=(10,5))
ax=plt.subplot(121)
ax.plot(bpos, dete,'.')
ax.plot(bpos, times_sequential,'.')
ax=plt.subplot(122)
ax.plot(dete/times_sequential)
ax.axhline(4,color="black")



### Deterministic ()\alpha + \beta)/2
dete = [timind[np.argmin(np.abs(.5*(betas[indb,:] + alphas[len(bneg)-indb,:]) - epsilon(b)))] for indb,b in enumerate(bpos)]
plt.figure(figsize=(10,5))
ax=plt.subplot(121)
ax.plot(bpos, dete,'.')
ax.plot(bpos, times_sequential,'.')
ax=plt.subplot(122)
ax.plot(dete/times_sequential)
ax.axhline(4,color="black")




np.polyfit(bpos,times_alpha_to_errB,1)
np.polyfit(bpos,times_sequential,1)









### Check if the probabilities agree.

mu0, o0 = np.polyfit(timind,ml0,1)
mu1, o1 = np.polyfit(timind,ml1,1)


def dete_alpha(t, b, mu):
    inside = (b + mu*t)/(np.sqrt(mu*t)*2)
    return (1 -  erf(inside))/2

k=6
indb = boundsB[k]
ax=plt.subplot()
ini = np.argmin(np.abs(timind-1))
fini = np.argmin(np.abs(timind-3))

ax.plot(timind[ini:fini], alphas[k][ini:fini])
ax.plot(timind[ini:fini], np.array([dete_alpha(t, indb, mu1) for t in timind])[ini:fini], color="black", linewidth=10, alpha=0.5, label="analytical result")
ax.plot(timind[ini:fini], np.array([dete_alpha(t, 0, mu1) for t in timind])[ini:fini], color="black", linewidth=10, alpha=0.5, label="analytical result")
#ax.set_yscale("log")



def dete_beta(t, b, mu):
    inside = (b - mu*t)/(np.sqrt(mu*t)*2)
    return (1 +  erf(inside))/2

### check beta error
plt.plot(timind[:fini], np.array([dete_beta(t, indb, mu1) for t in timind])[:fini])
plt.plot(timind[:fini],betas[k][:fini])



k=len(boundsB)-1
indb = boundsB[k]
ax=plt.subplot()
fini = np.argmin(np.abs(np.array(timind)-3))
ax.plot(timind[:fini], alphas[k][:fini] + betas[k][:fini])
ax.plot(timind[:fini], (np.array([dete_alpha(t, indb, mu1) for t in timind])[:fini] + np.array([dete_beta(t, indb, mu1) for t in timind])[:fini]), color="black", linewidth=10, alpha=0.5, label="analytical result")
#ax.set_yscale("log")











###
