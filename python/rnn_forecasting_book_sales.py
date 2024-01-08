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

sales = readfile(FILE_NAME)

months = len(sales)
split = int(months * TRAIN_TEST_SPLIT)

train_sales = sales[0:split]
test_sales = sales[split:]

plotDataset(sales)
