import numpy as np
from numpy import array as npa
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis

import time
from functools import reduce



def genData(dt, fList, x=1, ts=100):
    Y = []
    for t in range(ts):
        fChain = (x,) + fList
        y = reduce(lambda x, y: y(x), fChain)
        Y.append(y)
        x = dt(x)
    Y = np.array(Y)
    return Y





def dt(x):
    return x + 1


def f1(x):
    return (np.sin(x), np.cos(x))


def f2(x):
    return ([x[0], np.sin(x[0])], [x[1], np.cos(x[1])])


class model_seq:
    def __init__(self, funcChain):
        self.funcList = funcChain

    def propagate(self, init):
        y = self.funcList[0](init)
        Y = [y]
        for f in self.funcList[1:]:
            y = f(y)
            Y.append(y)
        return y, Y


    def evolution(self, init, dt, nt):
        ys = []
        Ys = []
        for t in range(nt):
            y, Y = self.propagate(init)
            ys.append(y)
            Ys.append(Y)
            init = dt(init)
        Y = np.array(Y)
        return Y


m = model_seq((f1, f2))
m.propagate(1)





