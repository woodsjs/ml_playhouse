import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout

import numpy as np
import matplotlib.pyplot as plt
import logging

tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 128
BATCH_SIZE = 32

cifar_dataset = keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifar_dataset.load_data()

# STandardize
mean = np.mean(train_images)
stddev = np.std(train_images)

train_images = (train_images - mean) / stddev
test_images = (test_images - mean) /stddev

print('mean ', mean)
print('stddev ', stddev)

# One hot the labels
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# two convolutional and one fully connected layer
model = Sequential()

# we can play here, adding and removing layers
# adjusting params, to get a good fit to the data.
model.add(Conv2D(64, (5,5), strides=(2,2),
    activation='relu', padding='same',
    input_shape=(32,32,3),
    kernel_initializer='he_normal',
    bias_initializer='zeros'))

model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), strides=(2,2),
    activation='relu', padding='same',
    kernel_initializer='he_normal',
    bias_initializer='zeros'))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax',
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros'))

model.compile(loss='categorical_crossentropy',
        optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(
        train_images, train_labels, validation_data = (test_images, test_labels),
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)
