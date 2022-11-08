
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

path = "gaussian_convergence/"
ana_alpha = np.load(path+"ana_alpha.npy")#ana_alpha)
ana_beta = np.load(path+"ana_beta.npy")#ana_beta)
wald_1 = np.load(path+"wald_1.npy")# wald_1)
wald_0 = np.load(path+"wald_0.npy")# wald_0)
mu0s = np.load(path+"mu0s.npy")# mu0s)
mu1s = np.load(path+"mu1s.npy")# mu1s)
sig1s = np.load(path+"sig1s.npy")# sig1s)
sig0s = np.load(path+"sig0s.npy")# sig0s)
det_al = np.load(path+"det_al.npy")#det_al)
det_be = np.load(path+"det_be.npy")#det_be)
det = np.load(path+"det.npy")#det)
seq0 = np.load(path+"seq0.npy")#seq0)
seq1 = np.load(path+"seq1.npy")#seq1)
gammas_ok = np.load(path+"gammas_ok.npy")#gammas_ok)



points = [-1]*len(gammas_ok)
for k,g in tqdm(enumerate(gammas_ok)):
    locs = np.where(np.array(det[k]) >= 7)[0]
    if len(locs) >0:
        points[k] = locs[0]

ii=30
dtotal_time = 8.
dt = 1e-4
times = np.arange(0, 8 + dt, dt )
indis = np.linspace(0,len(times)-1, int(1e4)).astype(int)
timind = np.array([times[k] for k in indis])

final = np.argmin(np.abs(timind-3))

choose = lambda x,y: y if x==-1 else x
al_mse = [np.mean(((np.array(det_al[k]) - ana_alpha[k])**2)[ii:choose(points[k], final)]) for k,g in enumerate(gammas_ok)]
bet_mse = [np.mean(((np.array(det_be[k]) - ana_beta[k])**2)[ii:choose(points[k], final)]) for k,g in enumerate(gammas_ok)]
plt.figure(figsize=(10,10))
ax=plt.subplot()
ax.plot(mu0s-mu1s,al_mse)
ax.plot(mu0s-mu1s,bet_mse)
ax.set_ylabel("MSE(over time)")
ax.set_xlabel(r'\mu_0 - \mu_1')
ax.set_yscale("log")




plt.figure(figsize=(100,500))
for i,gamma in enumerate(gammas_ok[:50]):
    ax=plt.subplot2grid((10,5),(i%10,i//10))
    ax.set_title(r'$\gamma =$'+str(np.round(gamma)),size=40)
    ax.plot(det_al[i], linewidth=4, color="red")
    ax.plot(ana_alpha[i], '--',  linewidth=4, color="red")
    ax.plot(det_be[i], linewidth=4, color="blue")
    ax.plot(ana_beta[i], '--',  linewidth=4, color="blue")
    ax.set_yscale("log")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlabel(r'$time$',size=30)
    ax.set_ylabel(r'$error$',size=30)
plt.savefig("assymetric_errors.pdf", dpi=10)





plt.figure(figsize=(10,5))
gamma=gammas_ok[20]
i=20
plt.suptitle("ALPHA: GAUSSIAN MODEL VS NUMERICS")
ax=plt.subplot(121)
ax.set_title(r'$\gamma =$'+str(np.round(gamma)))
ax.plot(timind,det_al[i],linewidth=5)
ax.plot(timind,ana_alpha[i], '--',linewidth=5)
ax=plt.subplot(122)
ax.set_title(r'$\gamma =$'+str(np.round(gamma)))
ax.plot(timind,det_al[i],linewidth=5)
ax.plot(timind,ana_alpha[i], '--',linewidth=5)
ax.set_yscale("log")

times










plt.figure(figsize=(100,50))
for i,gamma in enumerate(gammas_ok[:20]):
    if i<10:
        ax=plt.subplot2grid((10,2),(i,0))
        ax.set_title(r'$\gamma =$'+str(np.round(gamma)))
        ax.plot(det_al[i])
        ax.plot(ana_alpha[i], '--')
    else:
        ax=plt.subplot2grid((10,2),(i-10,1))
        ax.set_title(r'$\gamma =$'+str(np.round(gamma)))
        ax.plot(det_al[i])
        ax.plot(ana_alpha[i], '--')
plt.savefig("alphas_large.pdf")
