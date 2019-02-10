import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import optimizers
from keras import backend as K
import copy
import multiprocessing as mp
import time
from numpy import genfromtxt


x = genfromtxt("E:\\Work\\GANN\\Data\\mydata.csv", delimiter=',', skip_header=1, usecols=(1, 2, 3, 4))
y = genfromtxt("E:\\Work\\GANN\\Data\\mydata.csv", delimiter=',', skip_header=1, usecols=(0,))

y_test = y[:40000]
x_test = x[:40000]

y_train = y[40000:50000]
x_train = x[40000:50000]

neurons = [4, 10, 20, 1]

model = Sequential()
model.add(Dense(neurons[1], activation='relu', use_bias=False, input_shape=(neurons[0],)))
model.add(Dense(neurons[2], activation='relu', use_bias=False))
model.add(Dense(neurons[3], activation='sigmoid', use_bias=False))
model.compile(loss='mean_squared_error', optimizer=SGD(), metrics=['accuracy'])

model.fit(x_test, y_test)
loss = model.evaluate(x_test, y_test)[0]
weights = model.get_weights()
SGD.get_updates(SGD(), loss=loss, params=weights)
