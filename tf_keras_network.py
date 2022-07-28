import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import logging

tf.get_logger().setLevel(logging.ERROR)
tf.random.set_seed(7)

EPOCHS = 20
BATCH_SIZE = 64

#load dataset from MNIST
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

mean = np.mean(train_images)
stddev = np.std(train_images)

# normalize
train_images = (train_images - mean) / stddev
test_images = (test_images - mean) / stddev

# one-hot encode labels
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# object to initialize weights
#initializer = keras.initializers.RandomUniform(minval = -0.1, maxval = 0.1)

# Create a sequential model. 784 inputs. Two dense layers
# tanh as activation function for hidden layer.
# Sigmoid as activation for output layer
#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=(28, 28)),
#    keras.layers.Dense(25, activation = 'tanh',
#                       kernel_initializer=initializer,
#                       bias_initializer = 'zeros'),
#    keras.layers.Dense(10, activation = 'sigmoid',
#                            kernel_initializer = initializer,
#                            bias_initializer = 'zeros')
#])                          


# SGD with rate of 0.01, randomize order, update weights after EACH example (batch_size=1)
#opt = keras.optimizers.SGD(learning_rate = 0.01)

#model.compile(loss = 'mean_squared_error', optimizer = opt, metrics = ['accuracy'])

initializer = keras.initializers.HeNormal(seed=25)
output_intializer = keras.initializers.GlorotUniform(seed=42)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(25, activation = 'relu',
                       kernel_initializer=initializer,
                       bias_initializer = 'zeros'),
    keras.layers.Dense(10, activation = 'softmax',
                            kernel_initializer = output_intializer,
                            bias_initializer = 'zeros')
])  

# SGD with rate of 0.01, randomize order, update weights after EACH example (batch_size=1)
opt = keras.optimizers.Adam(learning_rate = 0.01)

model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

# train for EPOCHS epocs (std is 20 here)
history = model.fit(train_images, train_labels,
                        validation_data = (test_images, test_labels),
                        epochs = EPOCHS,
                        batch_size = BATCH_SIZE,
                        verbose = 2,
                        shuffle = True)