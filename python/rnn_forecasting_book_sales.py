import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN

import logging

tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 100
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.8
MIN = 12
FILE_NAME = '../data/book_store_sales.csv'

def readfile(file_name):
    file = open(file_name, 'r', encoding='utf-8')
    
    # skip the first line, it's our hearders.
    next(file)

    data = []

    for line in (file):
        values = line.split(',')
        data.append(float(values[1]))

    file.close()

    return np.array(data, dtype=np.float32)

def plotDataset(dataset):
    x = range(len(dataset))

    plt.plot(x, dataset, 'r-', label='book sales')
    plt.title('Book store sales change over time')
    plt.axis([0, 339, 0.0, 3000.0])
    plt.xlabel('Months')
    plt.ylabel('Sales (millions $)')

    plt.legend()
    plt.show()

def naivePrediction(test_sales):
    test_output = test_sales[MIN:]
    naive_prediction = test_sales[MIN-1:-1]

    x = range(len(test_output))

    plt.plot(x, test_output, 'g-', label='test_output')
    plt.plot(x, naive_prediction, 'm-', label='naive_prediction')

    plt.title('Book store sales naive prediction')

    plt.axis([0, len(test_output), 0.0, 3000.0])
    plt.xlabel('months')
    plt.ylabel('Monthly book store sales')

    plt.legend()
    plt.show()

def naiveStandardizedPrediction(test_sales_std):
    test_output = test_sales_std[MIN:]
    naive_prediction = test_sales_std[MIN-1:-1]

    mean_squared_error = np.mean(np.square(naive_prediction - test_output))
    mean_abs_error = np.mean(np.abs(naive_prediction - test_output))

    print('naive test mse: ', mean_squared_error)
    print('naive test mean abs: ', mean_abs_error)


sales = readfile(FILE_NAME)

months = len(sales)
split = int(months * TRAIN_TEST_SPLIT)

train_sales = sales[0:split]
test_sales = sales[split:]

plotDataset(sales)
naivePrediction(test_sales)

# standardize it
# be sure to use only the training set
mean = np.mean(train_sales)
stddev = np.std(train_sales)
train_sales_std = (train_sales - mean) / stddev
test_sales_std = (test_sales - mean) / stddev

train_months = len(train_sales)
train_X = np.zeros((train_months-MIN, train_months-1,1))
train_y = np.zeros((train_months-MIN, 1))

for i in range(0, train_months-MIN):
    train_X[i, -(i+MIN):, 0] = train_sales_std[0:i+MIN]
    train_y[i, 0] = train_sales_std[i+MIN]

# and now test
mean = np.mean(test_sales)
stddev = np.std(test_sales)
test_sales_std = (test_sales - mean) / stddev
test_sales_std = (test_sales - mean) / stddev

test_months = len(test_sales)
test_X = np.zeros((test_months-MIN, test_months-1,1))
test_y = np.zeros((test_months-MIN, 1))

for i in range(0, test_months-MIN):
    test_X[i, -(i+MIN):, 0] = test_sales_std[0:i+MIN]
    test_y[i, 0] = test_sales_std[i+MIN]

# model
model = Sequential()
model.add(SimpleRNN(128, activation='relu',
            input_shape=(None, 1)))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam',
            metrics=['mean_absolute_error'])

model.summary()

history = model.fit(train_X, train_y,
        validation_data = (test_X, test_y), epochs=EPOCHS,
        batch_size=BATCH_SIZE, verbose=2,
        shuffle=True)

naiveStandardizedPrediction(test_sales_std)
