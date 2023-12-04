import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import logging

tf.get_logger().setLevel(logging.ERROR)
tf.random.set_seed(7)

EPOCHS = 20
BATCH_SIZE = 1

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load.data()

#standardize the data
mean = np.mean(train_images)
stddev = np.std(train_images)

train_images = (train_images - mean) / stddev
test_images = (test_images - mean) / stddev

#One hot labels
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# Initialize the weights
initializer = keras.initializers.RandomUniform( minval=-0.1, maxval=0.1)

# Create the model
# 784 inputs, 2 dense with 25 and 10 neurons
# Tanh as activation for hidden
# SIgmoid for output layer
model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(25, activation='tanh',
            kernel_intializer=initializer,
            bias_intiializer='zeros'),
        keras.layers.Dense(10, activation='sigmoid',
            kernel_initializer=initializer,
            bias_initializer='zeros')])

# SGD, learning rate 0.01. Let everything else ride
# MSE as loss, accuracy reporting
opt = keras.optimizers.SGD(learning_rate=0.01)

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

# Train for 20 Epochs, watch it blow away our cpp code...
history = model.fit(train_images, train_labels,
        validation_data=(test_images, test_labels),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        verbose=2, shuffle=True)

