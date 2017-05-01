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
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling3D, MaxPooling1D
from keras.models import Model, Sequential, load_model
from keras import backend
from keras import objectives, optimizers


from acerlib import Pipeline, timeSeries
from SpringMassDataGenerator import genSpringMassData


def genData(nMassSystem=100):
    stoptime = 100
    numpoints = 119
    d_x = []
    d_y = []
    for i in range(nMassSystem):
        d = genSpringMassData(stoptime, numpoints)[:, 1]
        d = d / np.linalg.norm(d)
        d = np.expand_dims(d, 1)
        x, y = timeSeries.timeSeriesBatchGen(d, 50, nPredictTime=5)

        y = x[:, 5:50, :]
        x = x[:, 0:45, :]

        d_x.append(x)
        d_y.append(y)

    X = np.concatenate(d_x, 0)
    Y = np.concatenate(d_y, 0)
    # X = np.squeeze(X)
    # Y = np.squeeze(Y)

    return X, Y


# ---------------------------------------------------------------------------- #
#                                   Pipeline                                   #
# ---------------------------------------------------------------------------- #
p = Pipeline.Pipeline('m_0.2')
p.id = 'try_01_1D_CNN'
p.d_train = genData(1000)
p.d_test = genData(30)
p.d_valid = genData(30)
p.save()

# ---------------------------------------------------------------------------- #
#                                     Model                                    #
# ---------------------------------------------------------------------------- #

m = Sequential()
m.add(Conv1D(8, 3, input_shape=(45, 1)))
# m.add(Conv3D(16, (1, 3, 3), input_shape=(3, 108, 108, 1), data_format="channels_last"))
m.add(Activation('relu'))
m.add(BatchNormalization())
m.add(MaxPooling1D(pool_size=2))

m.add(Conv1D(16, 3))
m.add(Activation('relu'))
m.add(BatchNormalization())
m.add(MaxPooling1D(pool_size=2))

m.add(Conv1D(32, 3))
m.add(Activation('relu'))
m.add(BatchNormalization())
m.add(MaxPooling1D(pool_size=2))

# m.add(Conv1D(64, 3))
# m.add(Activation('relu'))
# m.add(BatchNormalization())
# m.add(MaxPooling1D(pool_size=2))

m.add(Flatten())
m.add(Dense(64))
m.add(Activation('relu'))
m.add(BatchNormalization())

m.add(Dense(64))
m.add(Activation('relu'))
m.add(BatchNormalization())

m.add(Dense(45))
m.add(Activation('relu'))
m.add(BatchNormalization())

m.add(Reshape((45, 1)))
m.add(Activation('linear'))

sgd = optimizers.SGD(lr=0.001)
m.compile(optimizer=sgd, loss='mean_squared_error')

p.m = m
p.save_m()

p.paras_fit['batch_size'] = 32
p.paras_fit['epochs'] = 500
p.fit()
