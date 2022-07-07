import os
import sys
sys.path.insert(0, os.getcwd())
from numerics.utilities.misc import *
from numerics.integration.steps import Ikpw, Robler_step
import numpy as np
from tqdm import tqdm
import argparse
import ast
from numba import jit
from scipy.linalg import solve_continuous_are


def IntegrationLoop(y0_hidden, y0_exp, times, dt):
    """
    dy = f(y,t) *dt + G(y,t) *dW
    """
    N = len(times)+1
    d = len(y0_hidden)
    m = len(y0_hidden)
    _,I=Ikpw(dW,dt)

    yhidden = np.zeros((N, d))
    yexper = np.zeros((N, len(y0_exp)))

    yhidden[0] = y0_hidden
    yexper[0] = y0_exp
    dys = []

    for ind, t in enumerate(times):
        yhidden[ind+1] = Robler_step(t, yhidden[ind], dW[ind,:], I[ind,:,:], dt, Fhidden, Ghidden, d, m)
        ## measurement outcome
        x1 = yhidden[ind][:2]
        dy=0.
        dys.append(dy)
        yexper[ind+1] = EulerUpdate_x0_logliks(x1, dy, yexper[ind], dt,dW[ind,0])
    return yhidden, yexper, dys

def EulerUpdate_x0_logliks(x1,dy,s, dt,dw):
    """
    this function updates the value of {x0,cov0} (wrong hypothesis) by using the dy
    also updates the log likelihoods l1 and l0
    """
    ### x1 is the hidden state i use to simulate the data

    l0, l1 = s[2:]
    dl0 = dt*np.dot(x1,x1) + x1[0]*dw
    return [0.,0., l0 + dl0, 0. ]


@jit(nopython=True)
def Fhidden(s, t, dt):
    """
    """
    x1 = s[:2]
    x1_dot = np.dot(A1,x1)
    return np.array([x1_dot[0], 0.])

@jit(nopython=True)
def Ghidden():
    return np.array([[1.,0.],[0.,0.]])

def integrate(params, aa, total_time=1, dt=1e-1, itraj=1, exp_path="",**kwargs):
    """
    h1 is the hypothesis i use to get the data. (with al the coefficients gamma1...)
    """
    global proj_C, A0, A1, XiCov0, XiCov1, C0, C1, dW, model_cte, model, a
    model = give_model()
    a = aa

    [gamma1, omega1, n1, eta1, kappa1],[gamma0, omega0, n0, eta0, kappa0],   = params
    model_cte = 1. ### measurement model
    ### XiCov = S C.T + G.T
    #### dx  = (A - XiCov.C )x dt + (XiCov dy) = A x dt + XiCov dW
    #### dy = C x dt + dW
    #### dCov = AS + SA.T + D - xiCov xiCov.T

    def give_matrices(gamma, omega, n, eta, kappa):
        D = C = G = np.zeros((2,2))
        A = np.array([[a, 0.],[0.,0.]])
        return A, C, D,G

    A1, C1, D1, G1 = give_matrices(gamma1, omega1, n1, eta1, kappa1)
    A0, C0, D0, G0 = give_matrices(gamma0, omega0, n0, eta0, kappa0)

    proj_C = np.ones((2,2))#np.linalg.pinv(C1/C1[0,0])
    x1in ,p1in, x0in, p0in, dyxin, dypin, lin0, lin1 = np.zeros(8)

    sst1 = sst0 = np.zeros((2,2))#solve_continuous_are( (A1-np.dot(G1.T,C1)).T, C1.T, D1 - np.dot(G1.T, G1), np.eye(2)) #### A.T because the way it's implemented!

    XiCov1  = np.dot(sst1, C1.T) + G1.T
    XiCov0  = np.dot(sst0, C0.T) + G0.T

    s0_hidden = np.array([x1in, p1in])
    s0_exper = np.array([x0in, p0in, lin0, lin1])

    times = np.arange(0,total_time+dt,dt)

    #### generate long trajectory of noises
    np.random.seed(itraj)
    dW = np.sqrt(dt)*np.random.randn(len(times),2)


    hidden_state, exper_state, signals = IntegrationLoop(s0_hidden, s0_exper,  times, dt)
    states1 = hidden_state[:,0:2]
    states0 = exper_state[:,:2]
    liks = exper_state[:,2:]

    path = get_path_config(total_time=total_time, dt=dt, itraj=itraj, exp_path=exp_path)
    os.makedirs(path, exist_ok=True)

    if len(times)>1e4:
        indis = np.linspace(0,len(times)-1,int(1e4)).astype("int")
    else:
        indis = np.arange(0,len(times))#imtimes[-1],times[1]-times[0]).astype(int)

    timind = [times[ind] for ind in indis]

    logliks_short =  np.array([liks[ii] for ii in indis])
    states1_short =  np.array([states1[ii] for ii in indis])

    np.save(path+"logliks",logliks_short)
    np.save(path+"states1",states1_short)


    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--itraj", type=int, default=1)
    parser.add_argument("--a", type=float,default=200.)

    args = parser.parse_args()

    itraj = args.itraj ###this determines the seed
    a = args.a

    params = [np.zeros(5)]*2
    exp_path = str(a)+"/"
    total_time = 4.
    dt = 1e-5

    integrate(params=params,
                aa=a,
              total_time = total_time,
              dt = dt,
              itraj=itraj,
              exp_path = exp_path)


###
