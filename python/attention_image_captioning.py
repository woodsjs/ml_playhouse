import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Attention
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Reshape

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence

from tensorflow.keras.applications import vgg19
from tensorflow.keras.applications.vgg19 import preprocess_input

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.utils import Sequence

from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle
import gzip
import logging

tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 20
BATCH_SIZE = 128
MAX_WORDS = 10000
READ_IMAGES = 90000
LAYER_SIZE = 256
EMBEDDING_WIDTH = 128
OOV_WORD = 'UNK'
PAD_INDEX = 0
OOV_INDEX = 1
START_INDEX = MAX_WORDS - 2
STOP_INDEX = MAX_WORDS - 1
MAX_LENGTH = 60
TRAINING_FILE_DIR = '../data/test_images/'
TEST_IMAGES = ['boat.jpg',
        'cat.jpg',
        'table.jpg',
        'bird.jpg']

# read the directory with captions
def read_training_file(file_name, max_len):
    pickle_file = gzip.open(file_name, 'rb')
    image_dict = pickle.load(pickle_file)
    pickle_file.close()
    image_paths = []
    dest_word_sequences = []

    for i, key in enumerate(image_dict):
        if i == READ_IMAGES:
            break

        image_item = image_dict[key]
        image_paths.append(image_item[0])
        caption = image_item[1]
        word_sequence = text_to_word_sequence(caption)
        dest_word_sequence = word_sequence[0:max_len]
        dest_word_sequences.append(dest_word_sequence)

    return image_paths, dest_word_sequences

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

image_paths, dest_seq = read_training_file(TRAINING_FILE_DIR + 'caption_file.pickle.gz', MAX_LENGTH)
dest_tokenizer, dest_token_seq = tokenize(dest_seq)


