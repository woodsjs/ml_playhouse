import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence

import tensorflow as tf

import logging

tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 32
BATCH_SIZE = 256
INPUT_FILE_NAME = '../data/frankenstein.txt'
WINDOW_LENGTH = 40
WINDOW_STEP = 3
PREDICT_LENGTH = 3
MAX_WORDS = 10000
EMBEDDING_WIDTH = 100

file = open(INPUT_FILE_NAME, 'r', encoding='utf-8-sig')
text = file.read()
file.close()

text = text_to_word_sequence(text)

fragments = []
targets = []

for i in range(0, len(text) - WINDOW_LENGTH, WINDOW_STEP):
    fragments.append(text[i: i + WINDOW_LENGTH])
    targets.append(text[i + WINDOW_LENGTH])

# we'll only tokenize 10000 words. If a word isn't found, 'UNK' is used
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='UNK')
# assign indexes to words
tokenizer.fit_on_texts(text)

# convert each fragment into indicies
fragments_indexed = tokenizer.texts_to_sequences(fragments)
targets_indexed = tokenizer.texts_to_sequences(targets)

X = np.array(fragments_indexed, dtype=np.int64)
y = np.zeros((len(targets_indexed), MAX_WORDS))

for i, target_index in enumerate(targets_indexed):
    y[i, target_index] = 1

training_model = Sequential()
training_model.add(Embedding(
    output_dim=EMBEDDING_WIDTH,
    input_dim=MAX_WORDS,
    mask_zero=True,
    input_length=None))
training_model.add(LSTM(128, 
    return_sequences=True,
    dropout=0.2,
    recurrent_dropout=0.2))
training_model.add(LSTM(128, 
    dropout=0.2,
    recurrent_dropout=0.2))
training_model.add(Dense(128, activation='relu'))
training_model.add(Dense(MAX_WORDS, activation='softmax'))

training_model.compile(loss='categorical_crossentropy', optimizer='adam')

training_model.summary()

history = training_model.fit(X, y, validation_split=0.05,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2,
        shuffle=True)
 
