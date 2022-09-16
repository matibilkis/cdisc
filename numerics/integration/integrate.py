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
from scipy.linalg import block_diag

def IntegrationLoop(y0_hidden, times, dt):
    """
    dy = f(y,t) *dt + G(y,t) *dW
    """
    N = len(times)+1
    d = len(y0_hidden)
    m = len(y0_hidden)
    _,I=Ikpw(dW,dt)

    yhidden = np.zeros((N, d))
    yhidden[0] = y0_hidden
    dys = []

    for ind, t in enumerate(times):
        yhidden[ind+1] = Robler_step(t, yhidden[ind], dW[ind,:], I[ind,:,:], dt, Fhidden, Ghidden, d, m)
        ## measurement outcome
        x = yhidden[ind][:2]
        dy = np.dot(C,x)*dt + np.dot(proj_C, dW[ind,:2])
        dys.append(dy)
    return yhidden, dys

@jit(nopython=True)
def Fhidden(s, t, dt):
    """
    """
    x = s[:2]
    x_th = s[2:4]
    x_dot = np.dot(A,x)
    x_th_dot = np.dot(A - XiCovC, x_th) + np.dot(A_th,x)
    return np.array(list(x_dot) + list(x_th_dot))

@jit(nopython=True)
def Ghidden():
    return big_XiCov

def integrate(params, total_time=1, dt=1e-1, itraj=1, exp_path="",**kwargs):
    """
    h1 is the hypothesis i use to get the data. (with al the coefficients gamma1...)
    """
    global proj_C, A, A_th, XiCov_th, XiCov, C, dW, model, big_XiCov, XiCovC
    model = give_model()
    pdt = kwargs.get("pdt",1)
    dt *=pdt #this is to check accuracy of integration
    times = np.arange(0,total_time+dt,dt)
    dW = np.sqrt(dt)*np.random.randn(len(times),2)
    dW = np.concatenate([dW]*2, axis=1)
    ### XiCov = S C.T + G.T
    #### dx  = (A - XiCov.C )x dt + (XiCov dy) = A x dt + XiCov dW
    #### dy = C x dt + dW
    #### dCov = AS + SA.T + D - xiCov xiCov.T
    gamma, omega, n, eta, kappa = params

    def give_matrices(gamma, omega, n, eta, kappa):
        A = np.array([[-gamma/2, omega],[-omega, -gamma/2]])
        C = np.sqrt(4*eta*kappa)*np.array([[1.,0.],[0., 0.]])#homodyne
        D = np.diag([gamma*(n+0.5) + kappa]*2)
        G = np.zeros((2,2))
        return A, C, D,G

    A, C, D, G = give_matrices(gamma, omega, n, eta, kappa)
    proj_C = np.linalg.pinv(C/C[0,0])
    xin, pin, x_thin, p_thin, dyxin, dypin = np.zeros(6)

    cov_st = solve_continuous_are( (A-np.dot(G.T,C)).T, C.T, D - np.dot(G.T, G), np.eye(2))#### A.T because the way it's implemented! https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_continuous_are.html
    D_th = np.eye(2)*0. ## we estimate \omega
    A_th = np.array([[0.,1.],[-1.,0.]])
    cov_st_th = solve_continuous_are( (A-np.dot(cov_st,np.dot(C.T,C))).T, np.eye(2)*0., D_th + np.dot(A_th, cov_st) + np.dot(cov_st, A_th.T), np.eye(2))

    XiCov  = np.dot(cov_st, C.T) #I take G=0.
    XiCovC = np.dot(XiCov, C)
    XiCov_th  = np.dot(cov_st_th, C.T) #I take G = 0.

    big_XiCov = block_diag(XiCov, XiCov_th)
    s0_hidden = np.array([xin, pin, x_thin, p_thin])
    times = np.arange(0,total_time+dt,dt)#[:(dW.shape[0])]

    #### generate long trajectory of noises
    np.random.seed(itraj)

    hidden_state, signals = IntegrationLoop(s0_hidden, times, dt)
    states = hidden_state[:,:2]
    states_th = hidden_state[:,2:4]

    path = get_path_config(total_time=total_time, dt=dt, itraj=itraj, exp_path=exp_path)
    os.makedirs(path, exist_ok=True)

    THRESHOLD = int(1e8)
    if len(times)>THRESHOLD:
        indis = np.linspace(0,len(times)-1, THRESHOLD).astype(int)
    else:
        indis = np.arange(0,len(times))

    timind = [times[ind] for ind in indis]
    signals_short =  np.array([signals[ii] for ii in indis])
    states_short =  np.array([states[ii] for ii in indis])
    states_th_short =  np.array([states_th[ii] for ii in indis])

    np.save(path+"signals",signals)
    np.save(path+"states_th",states_th_short)
    np.save(path+"states",states_short)

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--itraj", type=int, default=1)
    parser.add_argument("--dt", type=float, default=1e-5)
    parser.add_argument("--pdt", type=int, default=1)
    parser.add_argument("--total_time", type=float, default=8.)

    args = parser.parse_args()

    itraj = args.itraj ###this determines the seed
    dt = args.dt
    pdt = args.pdt
    total_time = args.total_time


    gamma, omega, n, eta, kappa = [1e1, 1e3, 10., 1., 1e2]
    params = [gamma, omega, n, eta, kappa]
    exp_path = str(params)+"/"

    N_periods = 100.
    single_period=2*np.pi/omega
    total_time = N_periods*single_period
    dt = single_period/50.

    integrate(params=params,
              total_time = total_time,
              dt = dt,
              itraj=itraj,
              exp_path = exp_path,
              pdt = pdt)



# import numpy as np
#
# omega = 1e3
# N_periods = 100.
# single_period=2*np.pi/omega
# total_time = N_periods*single_period
# dt = single_period/100.
#
# times = np.arange(0,total_time+dt,dt)
# ###
