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


pp = Pipeline.load_pipeline('m1_test/test_20170428_163048_pipeline')
