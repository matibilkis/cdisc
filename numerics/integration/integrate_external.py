import os
import sys
sys.path.insert(0, os.getcwd())
from numerics.utilities.external.misc_external import *
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
    x_dot = np.dot(A,x)
    if id==0:
        x_dot += 100.*np.array([1.,0.])
    elif id==1:
        x_dot += (100. + np.random.normal())*np.array([1.,0.])
    elif id ==2:
        x_dot += 100*(np.exp(-t/5.))*np.array([1.,0.])
    elif id ==3:
        x_dot += 10.*(np.cos(100*t))*np.array([1.,0.])
    return np.array(list(x_dot))# + list(x_th_dot))

@jit(nopython=True)
def Ghidden():
    return XiCov

def integrate(params, total_time=1, dt=1e-1, itraj=1, exp_path="",**kwargs):
    """
    h1 is the hypothesis i use to get the data. (with al the coefficients gamma1...)
    """
    global proj_C, A, XiCov, C, dW, model, id
    model = give_model()
    pdt = kwargs.get("pdt",1)
    id = kwargs.get("id",0)
    dt *=pdt #this is to check accuracy of integration
    times = np.arange(0,total_time+dt,dt)
    dW = np.sqrt(dt)*np.random.randn(len(times),2)
    ### XiCov = S C.T + G.T
    #### dx  = (A - XiCov.C )x dt + (XiCov dy) = A x dt + XiCov dW
    #### dy = C x dt + dW
    #### dCov = AS + SA.T + D - xiCov xiCov.T
    gamma, omega, n, eta, kappa = params

    def give_matrices(gamma, omega, n, eta, kappa):
        A = np.array([[-gamma/2, omega],[-omega, -gamma/2]])
        C = np.sqrt(4*eta*kappa)*np.eye(2)#homodyne --> rotating frame...
        D = np.diag([gamma*(n+0.5) + kappa]*2)
        G = np.zeros((2,2))
        return A, C, D,G

    A, C, D, G = give_matrices(gamma, omega, n, eta, kappa)
    proj_C = np.linalg.pinv(C/C[0,0])
    xin, pin, x_thin, p_thin, dyxin, dypin = np.zeros(6)

    cov_st = solve_continuous_are( (A-np.dot(G.T,C)).T, C.T, D - np.dot(G.T, G), np.eye(2))#### A.T because the way it's implemented! https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_continuous_are.html

    XiCov  = np.dot(cov_st, C.T) #I take G=0.
    XiCovC = np.dot(XiCov, C)

    s0_hidden = np.array([xin, pin])
    times = np.arange(0,total_time+dt,dt)#[:(dW.shape[0])]

    #### generate long trajectory of noises
    np.random.seed(itraj)

    hidden_state, signals = IntegrationLoop(s0_hidden, times, dt)
    states = hidden_state[:,:2]

    path = get_path_config(total_time=total_time, dt=dt, itraj=itraj, exp_path=exp_path, id=id)
    os.makedirs(path, exist_ok=True)

    THRESHOLD = int(1e8)
    if len(times)>THRESHOLD:
        indis = np.linspace(0,len(times)-1, THRESHOLD).astype(int)
    else:
        indis = np.arange(0,len(times))

    timind = [times[ind] for ind in indis]
    signals_short =  np.array([signals[ii] for ii in indis])
    states_short =  np.array([states[ii] for ii in indis])

    np.save(path+"signals",signals)
    np.save(path+"states",states_short)

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--itraj", type=int, default=1)
    parser.add_argument("--dt", type=float, default=1e-5)
    parser.add_argument("--pdt", type=int, default=1)
    parser.add_argument("--total_time", type=float, default=8.)
    parser.add_argument("--id", type=int, default=1)

    args = parser.parse_args()

    itraj = args.itraj ###this determines the seed
    dt = args.dt
    pdt = args.pdt
    total_time = args.total_time
    id = args.id


    # gamma, omega, n, eta, kappa = [100., 0., 1., 1., .1]
    gamma, omega, n, eta, kappa  = [5., 0., 1e-5, .5, 10]
    params = [gamma, omega, n, eta, kappa]
    exp_path = str(params)+"/"

    #N_periods = 100.
    #single_period=2*np.pi/omega
    total_time = 20.#N_periods*single_period
    dt = total_time*1e-4


    integrate(params=params,
              total_time = total_time,
              dt = dt,
              itraj=itraj,
              exp_path = exp_path,
              pdt = pdt,
              id = id)



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
