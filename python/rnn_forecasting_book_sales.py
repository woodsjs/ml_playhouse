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

sales = readfile(FILE_NAME)

months = len(sales)
split = int(months * TRAIN_TEST_SPLIT)

train_sales = sales[0:split]
test_sales = sales[split:]

plotDataset(sales)
naivePrediction(test_sales)
