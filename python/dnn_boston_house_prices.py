import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import logging

# adding L2 regularization, to reduce the effect of weights that aren't 
# needed for the general problem, but might be for specific problems
from tensorflow.keras.regularizers import l2

# adding dropout, to reduce the possibility of co-adaptation of neurons,
# which will cause specialization
# this works by adding a layer between neurons, 
# blocking a set of connections to a neuron, effectively rendering useless
from tensorflow.keras.layers import Dropout

tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 500
BATCH_SIZE = 16

# Read and standardize data
boston_housing = keras.datasets.boston_housing
(raw_x_train, y_train), (raw_x_test, y_test) = boston_housing.load_data()

x_mean = np.mean(raw_x_train, axis=0)
x_stddev = np.std(raw_x_train, axis=0);
x_train = (raw_x_train - x_mean) / x_stddev
x_test = (raw_x_test - x_mean) / x_stddev

# Create and train model
model = Sequential()

# for a real DL
model.add(Dense(
    64, 
    activation='relu', 
    kernel_regularizer=l2(1.0), # regularizer added
    bias_regularizer=l2(0.1),   # bias regularizer is separate
    input_shape=[13]))
# adding dropout, at a rate of 20%
model.add(Dropout(0.2))

model.add(Dense(
    64, 
    activation='relu',
    kernel_regularizer=l2(0.1),
    bias_regularizer=l2(0.1))) # get thems two layers
model.add(Dropout(0.2))

model.add(Dense(
    1, 
    activation='linear',
    kernel_regularizer=l2(0.1),
    bias_regularizer=l2(0.1)))

# standard linear model
# model.add(Dense(1, activation='linear', input_shape=[13]))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

model.summary()

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)

# show first 4
predictions = model.predict(x_test)

for i in range(0,4):
    print('Predicion: ', predictions[i], ', true value: ', y_test[i])

