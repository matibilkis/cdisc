import os
import sys
sys.path.insert(0, os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from numerics.utilities.misc import *
import matplotlib
from scipy.linalg import solve_continuous_are
import pickle
from tqdm import tqdm
from analysis.misc import *




params, exp_path = def_params(flip = 0)

which = mode = give_model()
if which == "mechanical_damp":
    total_time = 4.
    dt = 1e-5
elif which == "mechanical_freq":
    omega = .5*(params[1][1] + params[0][1])
    T_param = 150.
    ppg = 1e3
    period = 2*np.pi/omega
    total_time = T_param*period
    dt = period/ppg
elif (which == "optical_homodyne") or (which == "optical_heterodyne"):
    damping = params[1][0]
    total_time = 200.*damping
    dt = damping/1e3

timind, indis, indis_range = get_timind_indis(total_time, dt, rrange=True)


Ntraj = 10000
B = 8.
dB = .1
boundsB= np.arange(-B,B+dB,dB)

stop = {}
stop["_0"] = {i:[] for i in range(1,Ntraj)}
stop["_1"] = {i:[] for i in range(1,Ntraj)}



deter = {}
deter["h0/h1"] ={indb:[0]*len(indis) for indb in range(len(boundsB))}
deter["h1/h0"] = {indb:[0]*len(indis) for indb in range(len(boundsB))}

l0,l1 = np.zeros((2,len(timind)))

intindis = np.arange(0,len(indis))

n=1
ers = []
for itraj in tqdm(range(int(Ntraj/2),Ntraj)):
    try:

        log_lik_ratio, log_lik_ratio_swap = load_liks(itraj=itraj, total_time=total_time, dt=dt)
        for indb,b in enumerate(boundsB):
            if b>=0:
                stop["_1"][itraj].append(get_stop_time(log_lik_ratio, b, timind))
                stop["_0"][itraj].append(get_stop_time(log_lik_ratio_swap, b,timind))

                deter["h0/h1"][indb] += ((log_lik_ratio[indis_range] < b).astype(int)  - deter["h0/h1"][indb])/n
                deter["h1/h0"][indb] += ((log_lik_ratio_swap[indis_range] < b).astype(int)  - deter["h1/h0"][indb])/n

        l1= l1 + log_lik_ratio
        l0 = l0 +log_lik_ratio_swap
        n+=1
    except Exception:
       ers.append(itraj)
       print("error", itraj)
l0/=(Ntraj - len(ers))
l1/=(Ntraj - len(ers))




#### type I and II errors

alphas = list(deter["h0/h1"].values())
betas = list(deter["h1/h0"].values())

alphas = np.stack(alphas)
betas = np.stack(betas)

bpos = boundsB[boundsB>=0]
bneg = boundsB[boundsB<0]

avg_err_alpha = lambda o: (1-np.exp(-abs(o)))/(np.exp(abs(o)) - np.exp(-abs(o)))
avg_err_beta = lambda o :(1-np.exp(-abs(o)))/(np.exp(abs(o)) - np.exp(-abs(o)))

errs = np.array([avg_err_alpha(b) for b in boundsB]) #
tot_err = 0.5*(alphas+betas)#0.5*(alphas + betas)

times_to_errs = [timind[np.argmin(np.abs(tot_err[indb,:] - errs[indb]))] for indb in range(len(bpos))]


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


avg_times1 = np.array([np.mean(k) for k in stops1])
avg_times0 = np.array([np.mean(k) for k in stops0])

avg_times = 0.5*(avg_times0 + avg_times1)

stoch = avg_times

avg_err_alpha = lambda o: (1-np.exp(-o))/(np.exp(o) - np.exp(-o))
errs = [avg_err_alpha(b) for b in bpos]
times_alpha_to_errB = [timind[np.argmin(np.abs(alphas[indb+len(bneg),:]+betas[len(bneg)-indb+1,:] - errs[indb]))] for indb in range(len(bpos))]

##############3 HISTOGRAM



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





### saving

path_data = get_def_path()+"analysis/muDIff/{}/".format(Ntraj,mode)
os.makedirs(path_data,exist_ok=True)

with open(path_data+"stop.pickle","wb") as f:
    pickle.dump(stop, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(path_data+"deter.pickle","wb") as f:
    pickle.dump(deter, f, protocol=pickle.HIGHEST_PROTOCOL)

np.save(path_data+"timbin", timbin1)
# np.save(path_data+"deth1h0", deter_data_h1_h0)
# np.save(path_data+"deth0h1", deter_data_h0_h1)
np.save(path_data+"l0",l0)
np.save(path_data+"l1",l1)
np.save(path_data+"cons0",cons0)
np.save(path_data+"cons1",cons1)

np.save(path_data+"times_to_err_det",times_alpha_to_errB)
np.save(path_data+"times_to_err_stoch",stoch)

print("data saved in {}\n".format(path_data))



#### ploting

def prob_craft(t, b, mu):
    S= np.sqrt(2*mu)
    div = (np.sqrt(2*np.pi)*S*(t**(3/2)))
    return  abs(b)*np.exp(-((abs(b)-mu*t)**2)/(2*t*(S**2)))/div

muu = l1[-1]/timind[-1]

LS, TS = 60, 40
plt.figure(figsize=(10,10))
ax = plt.subplot(111)
indb = -1

timm =  np.linspace(np.min(timbin1),np.max(timbin1),100)

popo = [prob_craft(tt, bpos[indb] , muu) for tt in timm]
ax.plot(timm,popo, linewidth=4)
ax.bar(timbin1[indb], cons1[indb], width=timbin1[indb][1]-timbin1[indb][0], color="red", alpha=0.75, edgecolor="black",)#, label="simulations")
ax.set_xlabel(r'$\tau$',size=LS)
ax.set_ylabel(r'$P(\tau)$', size=LS)
ax.tick_params(axis='both', which='major', labelsize=TS)
plt.savefig(path_data+"stop_time_distrib.pdf")




fig = plt.figure(figsize=(20,20))
ax = plt.subplot(111)
lw=10
TS=30

fromm = 3
ax.plot(bpos[fromm:],(times_alpha_to_errB/stoch)[fromm:], linewidth=5)
ax.plot(bpos[fromm:],4*np.ones(len(stoch[fromm:])),'--', linewidth=5)
ax.set_xlabel(r'$\log \frac{1}{\epsilon}$', size=LS)
ax.tick_params(axis='both', which='major', labelsize=TS)
ax.set_ylabel(r'$\frac{t_d}{t_s}$', size=LS)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.savefig(path_data+"ratio.pdf")


print("Done.  \n all saved in {}".format(path_data))
