import numpy as np

def get_stop_time(ell,b, times, mode_log=True):
    logicals = np.logical_and(ell < b, ell > -b)
    ind_times = np.argmin(logicals)

    if (np.sum(logicals) == 0) or (ind_times==0):
        return np.nan
    else:
        return times[ind_times]
