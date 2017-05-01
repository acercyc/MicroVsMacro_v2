from acerlib.RemoteSession import findX11DisplayPort
findX11DisplayPort()

import numpy as np
from numpy import array as npa
import pandas as pd
import matplotlib.pyplot as plt
import scipy

import os
import glob

from keras.layers import Input, Dense, Lambda, Activation
from keras.layers import LSTM, TimeDistributed, Flatten, Reshape
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv3D
from keras.layers.pooling import MaxPooling3D
from keras.models import Model, Sequential, load_model
from keras import backend
from keras import objectives


from acerlib import Pipeline, timeSeries


def sinGen(fs, f, n):
    x = np.arange(n)  # the points on the x axis for plotting
    y = np.sin(2 * np.pi * x / float(fs / f))
    return y

y = sinGen(500, 20, 50)

fig = plt.figure()
plt.plot(y)
plt.show()