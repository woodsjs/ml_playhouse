import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import logging

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
model.add(Dense(64, activation='relu', input_shape=[13]))
model.add(Dense(64, activation='relu')) # get thems two layers

model.add(Dense(1, activation='linear'))

# standard linear model
# model.add(Dense(1, activation='linear', input_shape=[13]))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

model.summary()

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)

# show first 4
predictions = model.predict(x_test)

for i in range(0,4):
    print('Predicion: ', predictions[i], ', true value: ', y_test[i])

