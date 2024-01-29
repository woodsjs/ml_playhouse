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
