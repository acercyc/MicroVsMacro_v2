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


# ---------------------------------------------------------------------------- #
#                                   Read data                                  #
# ---------------------------------------------------------------------------- #
fPath = 'pics/'
wSize = 3


def readImgFromFolder(path):
    imgFile = glob.glob(path+'/*.png')
    imgFile.sort()
    imgs = [scipy.ndimage.imread(iImg)[:, :, 1] for iImg in imgFile]
    imgs = np.dstack(imgs)
    imgs = imgs / 255
    return imgs


def readImgFromFolders(path, iFilder=None):
    fPaths = glob.glob(path + '*')
    if iFilder is None:
        fPathsSelected = fPaths
    else:
        fPathsSelected = [fPaths[i] for i in iFilder]
    imgs = [readImgFromFolder(path) for path in fPathsSelected]
    imgs = np.dstack(imgs)
    imgs = np.rollaxis(imgs, 2, 0)
    return imgs


# def genXY(d):
#     nFrame = d.shape[2]
#     X = d[:, :, range(1, nFrame)]
#     Y = d[:, :, range(nFrame-1)]
#     return X, Y

def genXY(d, wSize):
    X, Y = timeSeries.timeSeriesBatchGen(d, wSize, nPredictTime=5)
    X = np.expand_dims(X, 4)
    Y = np.expand_dims(Y, 4)
    Y = Y[:, (0, 4), :, :, :]
    return X, Y


# ---------------------------------------------------------------------------- #
#                                Create pipeline                               #
# ---------------------------------------------------------------------------- #


p = Pipeline.Pipeline('m1')
p.id = 'CNN_try3_3P5F'
p.d_train = genXY(readImgFromFolders(fPath, range(60)), wSize)
p.d_test = genXY(readImgFromFolders(fPath, range(60, 80)), wSize)
p.d_valid = genXY(readImgFromFolders(fPath, range(80, 100)), wSize)


# ---------------------------------------------------------------------------- #
#                                     Model                                    #
# ---------------------------------------------------------------------------- #

m = Sequential()
m.add(Conv3D(16, (1, 3, 3), input_shape=(3, 108, 108, 1), data_format="channels_last"))
m.add(Activation('relu'))
m.add(BatchNormalization())
m.add(MaxPooling3D(pool_size=(1, 2, 2)))

m.add(Conv3D(32, (1, 3, 3)))
m.add(Activation('relu'))
m.add(BatchNormalization())
m.add(MaxPooling3D(pool_size=(1, 2, 2)))

m.add(Conv3D(64, (2, 3, 3)))
m.add(Activation('relu'))
m.add(BatchNormalization())
m.add(MaxPooling3D(pool_size=(1, 2, 2)))


m.add(Conv3D(128, (2, 3, 3)))
m.add(Activation('relu'))


m.add(Flatten())
m.add(Dense(1024))
m.add(Activation('relu'))

m.add(Dense(1024))
m.add(Activation('relu'))

m.add(Dense(23328))
m.add(Activation('relu'))

m.add(Reshape((2, 108, 108, 1)))
m.compile(optimizer='SGD', loss='mean_squared_error')
p.m = m
p.save()

p.paras_fit['batch_size'] = 32
p.paras_fit['epochs'] = 500
p.fit()
