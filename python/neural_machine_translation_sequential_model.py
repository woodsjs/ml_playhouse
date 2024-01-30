import numpy as np
import random

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.models import Model 

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf

import logging

tf.get_logger().setLevel(logging.ERROR)

# Constants
EPOCHS = 20
BATCH_SIZE = 128
MAX_WORDS = 10000
READ_LINES = 60000

# model params
LAYER_SIZE = 256
EMBEDDING_WIDTH = 128
TEST_PERCENT = 0.2
SAMPLE_SIZE = 20

# 
OOV_WORD = 'UNK'
PAD_INDEX = 0
OOV_INDEX = 1
START_INDEX = MAX_WORDS - 2
STOP_INDEX = MAX_WORDS - 1
MAX_LENGTH = 60
SRC_DEST_FILE_NAME = '../data/fra.txt'

# read the file and parse
# return a tuple of source and destination word sequences
def read_file_combined(file_name, max_len):
    file = open(file_name, 'r', encoding='utf-8')

    src_word_sequences = []
    dest_word_sequences = []

    for i, line in enumerate(file):
        if i == READ_LINES:
            break

        pair = line.split('\t')
        word_sequence = text_to_word_sequence(pair[1])
        src_word_sequence = word_sequence[0:max_len]
        src_word_sequences.append(src_word_sequence)
        
        # work on the destination lang
        word_sequence = text_to_word_sequence(pair[0])
        dest_word_sequence = word_sequence[0:max_len]
        dest_word_sequences.append(dest_word_sequence)

    file.close()
    return src_word_sequences, dest_word_sequences

# tokenize and untokenize
# returns tuple of tokenizer object, and token sequences
def tokenize(sequences):
    # MAX_WORDS-2 used to reserve two indicies for start and stop
    tokenizer = Tokenizer(num_words=MAX_WORDS-2, oov_token=OOV_WORD)
    tokenizer.fit_on_texts(sequences)

    token_sequences = tokenizer.texts_to_sequences(sequences)

    return tokenizer, token_sequences

def tokens_to_words(tokenizer, seq):
    word_seq = []

    for index in seq:
        if index == PAD_INDEX:
            word_seq.append('PAD')
        elif index == OOV_INDEX:
            word_seq.append(OOV_WORD)
        elif index == START_INDEX:
            word_seq.append('START')
        elif index == STOP_INDEX:
            word_seq.append('STOP')
        else:
            word_seq.append(tokenizer.sequences_to_texts([[index]])[0])

    print(word_seq)

src_seq, dest_seq = read_file_combined(SRC_DEST_FILE_NAME, MAX_LENGTH)
src_tokenizer, src_token_seq = tokenize(src_seq)
dest_tokenizer, dest_token_seq = tokenize(dest_seq)

# prep training data
dest_target_token_seq = [x + [STOP_INDEX] for x in dest_token_seq]
dest_input_token_seq = [[START_INDEX] + x for x in dest_target_token_seq]

src_input_data = pad_sequences(src_token_seq)
dest_input_data = pad_sequences(dest_input_token_seq, padding='post')

dest_target_data = pad_sequences(dest_target_token_seq, padding='post', maxlen = len(dest_input_data[0]))

# splits
rows = len(src_input_data[:,0])
all_indicies = list(range(rows))
test_rows = int(rows * TEST_PERCENT)
test_indicies = random.sample(all_indicies, test_rows)
train_indicies = [x for x in all_indicies if x not in test_indicies]

train_src_input_data = src_input_data[train_indicies]
train_dest_input_data = dest_input_data[train_indicies]
train_dest_target_data = dest_target_data[train_indicies]

test_src_input_data = src_input_data[test_indicies]
test_dest_input_data = dest_input_data[test_indicies]
test_dest_target_data = dest_target_data[test_indicies]

# sample test set
test_indicies = list(range(test_rows))
sample_indicies = random.sample(test_indicies, SAMPLE_SIZE)
sample_input_data = test_src_input_data[sample_indicies]
sample_target_data = test_dest_target_data[sample_indicies]

# encoding model
# input is the source language
enc_embedding_input = Input(shape=(None,))

enc_embedding_layer = Embedding(output_dim=EMBEDDING_WIDTH,
        input_dim=MAX_WORDS, mask_zero=True)
enc_layer1 = LSTM(LAYER_SIZE,
        return_state=True,
        return_sequences=True)
enc_layer2 = LSTM(LAYER_SIZE,
        return_state=True)

# connecting the encoding layers
enc_embedding_layer_outputs = enc_embedding_layer(enc_embedding_input)
enc_layer1_outputs, enc_layer1_state_h, enc_layer1_state_c = enc_layer1(enc_embedding_layer_outputs)
_, enc_layer2_state_h, enc_layer2_state_c =  enc_layer2(enc_layer1_outputs)

# Build
enc_model = Model(enc_embedding_input,
    [ enc_layer1_state_h, enc_layer1_state_c,
        enc_layer2_state_h, enc_layer2_state_c])

enc_model.summary()

# decoder model
dec_layer1_state_input_h = Input(shape=(LAYER_SIZE,))
dec_layer1_state_input_c = Input(shape=(LAYER_SIZE,))
dec_layer2_state_input_h = Input(shape=(LAYER_SIZE,))
dec_layer2_state_input_c = Input(shape=(LAYER_SIZE,))

dec_embedding_input = Input(shape=(None,))

# layers
dec_embedding_layer = Embedding(output_dim=EMBEDDING_WIDTH,
        input_dim=MAX_WORDS,
        mask_zero=True)

dec_layer1 = LSTM(LAYER_SIZE, return_state=True,
        return_sequences=True)
dec_layer2 = LSTM(LAYER_SIZE,
        return_state=True,
        return_sequences=True)
dec_layer3 = Dense(MAX_WORDS, activation='softmax')

#connect the decoder layers
dec_embedding_layer_outputs = dec_embedding_layer(dec_embedding_input)
dec_layer1_outputs, dec_layer1_state_h, dec_layer1_state_c = dec_layer1(dec_embedding_layer_outputs,
        initial_state=[dec_layer1_state_input_h,
            dec_layer1_state_input_c])
dec_layer2_outputs, dec_layer2_state_h, dec_layer2_state_c = dec_layer2(dec_layer1_outputs,
        initial_state=[dec_layer2_state_input_h,
            dec_layer2_state_input_c])
dec_layer3_outputs = dec_layer3(dec_layer2_outputs)

# build decoding model
dec_model = Model([dec_embedding_input,
    dec_layer1_state_input_h,
    dec_layer1_state_input_c,
    dec_layer2_state_input_h,
    dec_layer2_state_input_c],
    [dec_layer3_outputs, dec_layer1_state_h,
        dec_layer1_state_c, dec_layer2_state_h,
        dec_layer2_state_c])

dec_model.summary()
