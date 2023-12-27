import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import logging

tf.get_logger().setLevel(logging.ERROR)

cifar_dataset = keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifar_dataset.load_data()

print('Category: ', train_labels[100])
plt.figure(figsize=(1,1))
plt.imshow(train_images[100])

plt.show()
