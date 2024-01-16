import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 32
BATCH_SIZE = 256
INPUT_FILE_NAME = '../data/frankenstein.txt'
WINDOW_LENGTH = 40
WINDOW_STEP = 3
BEAM_SIZE = 8
NUM_LETTERS = 11

file = open(INPUT_FILE_NAME, 'r', encoding= 'utf-8-sig')
text = file.read()
file.close()

# clean data 
text = text.lower()
text.replace('\n', ' ')
text.replace('  ', ' ')

# encoding
unique_chars = list(set(text))
char_to_index = dict((ch, index) for index, ch in enumerate(unique_chars))
index_to_char = dict((index, ch) for index, ch in enumerate(unique_chars))
encoding_width = len(char_to_index)

# create training examples from text
fragments = []
targets = []

for i in range(0, len(text) - WINDOW_LENGTH, WINDOW_STEP):
    fragments.append(text[i: i + WINDOW_LENGTH])
    targets.append(text[i + WINDOW_LENGTH])

# convert to one hot
X = np.zeros((len(fragments), WINDOW_LENGTH, encoding_width))
y = np.zeros((len(fragments), encoding_width))

for i, fragment in enumerate(fragments):
    for j, char in enumerate(fragment):
        X[i, j, char_to_index[char]] = 1

    target_char = targets[i]

    y[i, char_to_index[target_char]] = 1

#model
model = Sequential()

# return sequences pushes our raw values out to the next layer
# needed because we're using two LSTM layers
model.add(LSTM(128, return_sequences=True,
    dropout=0.2, recurrent_dropout=0.2,
    input_shape=(None, encoding_width)))

model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(encoding_width, activation='softmax'))

model.compile(loss='categorical_crossentropy',
        optimizer='adam')

model.summary()

history = model.fit(X, y, validation_split=0.05,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2,
        shuffle=True)

#Create single beam as a tuple
# probability (log), string, one hot string
# the feels are that this code can be made much more efficient
letters = 'the body '

one_hots = []

# one hot the input
for i, char in enumerate(letters):
    x = np.zeros(encoding_width)
    x[char_to_index[char]] = 1
    one_hots.append(x)

# log is our probability
beams = [(np.log(1.0), letters, one_hots)]

# predict into the future
for i in range(NUM_LETTERS):
    minibatch_list = []

    # for each entry in beams, which is one right now...pull the one hots into the minibatch list
    for triple in beams:
        minibatch_list.append(triple[2])

    # convert the list into a numpy array 
    minibatch = np.array(minibatch_list)

    # gives one softmax prediction per beam
    # this prediction is one probability per char in the alphabet
    y_predict = model.predict(minibatch, verbose=0)

    new_beams = []

    # pull out each prediction, and create a new beam for it
    # we keep adding to new_beams as we go along, and keep predicting new BEAM_SIZE beams for
    # the beams we have. At the end we will disguard most of them.
    for j, softmax_vec in enumerate(y_predict):

        triple = beams[j]

        # how many beams do we want?
        for k in range(BEAM_SIZE):

            # our output is N probabilities, pull the largest probability
            char_index = np.argmax(softmax_vec)

            # get our current log, add that to the log of the prediction of the largest value
            # this helps to avoid underflow, as the probabilities are small
            new_prob = triple[0] + np.log(softmax_vec[char_index])

            # get our letters, add the letter that is our largest probability
            new_letters = triple[1] + index_to_char[char_index]

            # one hot our x value
            x = np.zeros(encoding_width)
            x[char_index] = 1

            # take our current one hots, and append our predicted as a one hot 
            new_one_hots = triple[2].copy()
            new_one_hots.append(x)

            # put all this data into a new beam
            new_beams.append((new_prob, new_letters, new_one_hots))

            # remove that prediction from our softmax_vector so it isn't the largest anymore
            softmax_vec[char_index] = 0

    # prune tree to BEAM_SIZE
    new_beams.sort(key=lambda tup: tup[0], reverse=True)
    beams = new_beams[0:BEAM_SIZE]

for item in beams:
    print(item[1])


