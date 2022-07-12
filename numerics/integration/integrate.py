import os
import sys
sys.path.insert(0, os.getcwd())
from numerics.utilities.misc import *
import numpy as np
from tqdm import tqdm
import argparse
import ast
from numba import jit


@jit(nopython=True)
def IntegrationLoop(times, dt):
    """
    dy = f(y,t) *dt + G(y,t) *dW
    """
    N = len(times)+1

    x = np.zeros(N+1).astype(np.float64)
    ell = np.zeros(N+1).astype(np.float64)

    x[0] = 0.
    ell[0] = 0.

    for ind, t in enumerate(times):
        x[ind+1] = a*x[ind] + b*dW[ind]
        ell[ind+1] = alpha*x[ind]**2 + beta*x[ind]*dW[ind]
    return x, ell



def integrate(params, total_time=1, dt=1e-1, itraj=1, exp_path="",**kwargs):
    """
    h1 is the hypothesis i use to get the data. (with al the coefficients gamma1...)
    """
    global a, b, alpha, beta, dW

    a, b, alpha, beta = params
    times = np.arange(0,total_time+dt,dt)

    #### generate long trajectory of noises
    np.random.seed(itraj)
    dW = np.sqrt(dt)*np.random.randn(len(times)).astype(np.float64)

    x, ell = IntegrationLoop(times, dt)

    path = get_path_config(total_time=total_time, dt=dt, itraj=itraj, exp_path=exp_path)
    os.makedirs(path, exist_ok=True)


    if len(times)>1e4:
        indis = np.linspace(0,len(times)-1,int(1e4)).astype("int")
    else:
        indis = np.arange(0,len(times))#imtimes[-1],times[1]-times[0]).astype(int)

    timind = [times[ind] for ind in indis]
    x_short =  np.array([x[ii] for ii in indis])
    ells_short =  np.array([ell[ii] for ii in indis])

    np.save(path+"ell",ells_short)
    np.save(path+"x",x_short)
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--itraj", type=int, default=1)
    parser.add_argument("--a", type=float, default=-1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)

    args = parser.parse_args()

    itraj = args.itraj ###this determines the seed
    a = args.a
    b = args.b
    alpha = args.alpha
    beta = args.beta
    # params, exp_path = def_params()
    p = [a, b, alpha, beta]
    params, exp_path = p, str(p)+"/"

    total_time = 5.
    dt = 1e-6

    integrate(params=params,
              total_time = total_time,
              dt = dt,
              itraj=itraj,
              exp_path = exp_path)


###
