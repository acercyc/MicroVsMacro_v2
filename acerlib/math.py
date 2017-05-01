import numpy as np


def symRange(v, centre=0):
    v -= centre
    if np.abs(np.max(v)) > np.abs(np.min(v)):
        lim = [-np.abs(np.max(v)), np.abs(np.max(v))]
    else:
        lim = [-np.abs(np.min(v)), np.abs(np.min(v))]

    lim += centre
    return lim

