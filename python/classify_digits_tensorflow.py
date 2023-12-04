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




