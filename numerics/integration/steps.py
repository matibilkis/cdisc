import numpy as np
from numba import jit

def dot(a, b):
    return np.einsum('ijk,ikl->ijl', a, b)

## Robler - see sdeint ##
def Aterm(N, h, m, k, dW):
    """kth term in the sum of Wiktorsson2001 equation (2.2)"""
    sqrt2h = np.sqrt(2.0/h)
    Xk = np.random.normal(0.0, 1.0, (N, m, 1))
    Yk = np.random.normal(0.0, 1.0, (N, m, 1))
    term1 = dot(Xk, (Yk + sqrt2h*dW).transpose((0, 2, 1)))
    term2 = dot(Yk + sqrt2h*dW, Xk.transpose((0, 2, 1)))
    return (term1 - term2)/k

## Robler - see sdeint ##
def Ikpw(dW, h, n=5):
    """matrix I approximating repeated Ito integrals for each of N time
    intervals, based on the method of Kloeden, Platen and Wright (1992).
    Args:
      dW (array of shape (N, m)): giving m independent Weiner increments for
        each time step N. (You can make this array using sdeint.deltaW())
      h (float): the time step size
      n (int, optional): how many terms to take in the series expansion
    Returns:
      (A, I) where
        A: array of shape (N, m, m) giving the Levy areas that were used.
        I: array of shape (N, m, m) giving an m x m matrix of repeated Ito
        integral values for each of the N time intervals.
    """
    N = dW.shape[0]
    m = dW.shape[1]
    if dW.ndim < 3:
        dW = dW.reshape((N, -1, 1)) # change to array of shape (N, m, 1)
    if dW.shape[2] != 1 or dW.ndim > 3:
        raise(ValueError)
    A = Aterm(N, h, m, 1, dW)
    for k in range(2, n+1):
        A += Aterm(N, h, m, k, dW)
    A = (h/(2.0*np.pi))*A
    I = 0.5*(dot(dW, dW.transpose((0,2,1))) - np.diag(h*np.ones(m))) + A
    dW = dW.reshape((N, -1)) # change back to shape (N, m)
    return (A, I)

@jit(nopython=True)
def Robler_step(t, Yn, Ik, Iij, dt, f,G, d, m):
    """
    https://pypi.org/project/sdeint/
    https://dl.acm.org/doi/abs/10.1007/s10543-005-0039-7
    """
    fnh = f(Yn, t,dt)*dt # shape (d,)
    xicov = Gn = G()
    sum1 = np.dot(Gn, Iij)/np.sqrt(dt) # shape (d, m)

    H20 = Yn + fnh # shape (d,)
    H20b = np.reshape(H20, (d, 1))
    H2 = H20b + sum1 # shape (d, m)

    H30 = Yn
    H3 = H20b - sum1
    fn1h = f(H20, t, dt)*dt
    Yn1 = Yn + 0.5*(fnh + fn1h) + np.dot(xicov, Ik)
    return Yn1
