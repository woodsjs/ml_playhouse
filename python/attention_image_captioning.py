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
TRAINING_FILE_DIR = 'tf_data/feature_vectors/'
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

# Class derived from sequence, to do batch processing
class ImageCaptionSequence(Sequence):

    def __init__(self, image_paths, dest_input_data, dest_target_data, batch_size):
        self.image_paths = image_paths
        self.dest_input_data = dest_input_data
        self.dest_target_data = dest_target_data
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.dest_input_data) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x0 = self.image_paths[
                idx * 
                self.batch_size:(idx + 1) * 
                self.batch_size]
        batch_x1 = self.dest_input_data[
                idx *
                self.batch_size:(idx + 1) *
                self.batch_size]

        batch_y = self.dest_target_data[
                idx *
                self.batch_size:(idx + 1) *
                self.batch_size]

        image_features = []

        for image_id in batch_x0:
            file_name = TRAINING_FILE_DIR + image_id + '.pickle.gzip'
            pickle_file = gzip.open(file_name, 'rb')
            feature_vector = pickle.load(pickle_file)
            pickle_file.close()
            image_features.append(feature_vector)
        
        return [np.array(image_features), np.array(batch_x1)], np.array(batch_y)

# prep training data
dest_target_token_seq = [x + [STOP_INDEX] for x in dest_token_seq]
dest_input_token_seq = [[START_INDEX] + x for x in dest_target_token_seq]

dest_input_data = pad_sequences(dest_input_token_seq, padding='post')

dest_target_data = pad_sequences(dest_target_token_seq, padding='post', maxlen = len(dest_input_data[0]))

image_sequence = ImageCaptionSequence(image_paths, dest_input_data, dest_target_data, BATCH_SIZE)

feature_vector_input = Input(shape=(14,14,512))

enc_mean_layer = GlobalAveragePooling2D()
enc_layer_h = Dense(LAYER_SIZE)
enc_layer_c = Dense(LAYER_SIZE)

#connect those layers
enc_mean_layer_output = enc_mean_layer(feature_vector_input)
enc_layer_h_outputs = enc_layer_h(enc_mean_layer_output)
enc_layer_c_outputs = enc_layer_c(enc_mean_layer_output)

enc_layer_outputs = [enc_layer_h_outputs, enc_layer_c_outputs]

# build
enc_model_top = Model(feature_vector_input, enc_layer_outputs)
enc_model_top.summary()

# crazy that, with how complex a decoder is using attention, 
# how little code we need to implement. Huzaah to library authors!

dec_feature_vector_input = Input(shape=(14,14,512))
dec_embedding_input = Input(shape=(None, ))
dec_layer1_state_input_h = Input(shape=(LAYER_SIZE,))
dec_layer1_state_input_c = Input(shape=(LAYER_SIZE,))

dec_reshape_layer = Reshape((196, 512),
        input_shape=(14,14,512,))
dec_attention_layer = Attention()
dec_query_layer = Dense(512)
dec_embedding_layer = Embedding(
        output_dim=EMBEDDING_WIDTH,
        input_dim=MAX_WORDS,
        mask_zero=False)

dec_layer1 = LSTM(LAYER_SIZE, return_state=True, return_sequences=True)
dec_concat_layer = Concatenate()
dec_layer2 = Dense(MAX_WORDS, activation='softmax')

# connect the decoder layers
dec_embedding_layer_outputs = dec_embedding_layer(dec_embedding_input)

dec_reshape_layer_outputs = dec_reshape_layer(dec_feature_vector_input)

dec_layer1_outputs, dec_layer1_state_h, dec_layer1_state_c = dec_layer1(
        dec_embedding_layer_outputs,
        initial_state=[dec_layer1_state_input_h,
            dec_layer1_state_input_c]
        )

dec_query_layer_outputs = dec_query_layer(dec_layer1_outputs)

dec_attention_layer_outputs = dec_attention_layer(
        [dec_query_layer_outputs, dec_reshape_layer_outputs])

dec_layer2_inputs = dec_concat_layer(
        [dec_layer1_outputs, dec_attention_layer_outputs])
dec_layer2_outputs = dec_layer2(dec_layer2_inputs)

# build
dec_model = Model([dec_feature_vector_input,
    dec_embedding_input,
    dec_layer1_state_input_h,
    dec_layer1_state_input_c],
    [dec_layer2_outputs,
        dec_layer1_state_h,
        dec_layer1_state_c])

dec_model.summary()

# build full model
train_feature_vector_input = Input(shape=(14,14,512))
train_dec_embedding_input = Input(shape=(None,))

intermediate_state = enc_model_top(train_feature_vector_input)

train_dec_output,_,_ = dec_model([train_feature_vector_input,
    train_dec_embedding_input] + intermediate_state)

training_model = Model([train_feature_vector_input,
    train_dec_embedding_input],
    [train_dec_output])

training_model.compile(loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

training_model.summary()

# full encoder model
conv_model = vgg19.VGG19(weights='imagenet')
conv_model_outputs = conv_model.get_layer('block5_conv4').output

intermediate_state = enc_model_top(conv_model_outputs)
inference_enc_model = Model([conv_model.input],
        intermediate_state + [conv_model_outputs])
inference_enc_model.summary()

# train this.
# we're using the image sequence object to give us data batch by batch
for i in range(EPOCHS):
    print('step: ', i)

    history = training_model.fit(image_sequence, epochs=1)

    for filename in TEST_IMAGES:
        image = load_img(TEST_FILE_DIR + filename)

        width = image.size[0]
        height = image.size[1]

        if height > width:
            image = load_img(TEST_FILE_DIR + filename,
                    target_size=(int(height/width*256), 256))
        else:
            image = load_img(
                    TEST_FILE_DIR + filename,
                    target_size=(256, int(width/height*256)))

        width = image.size[0]
        height = image.size[1]
        image_np = img_to_array(image)

        # crop to center
        h_start = int((height-224)/2)
        w_start = int((width-224)/2)
        image_np = image_np[h_start:h_start+224,
                w_start:w_start+224]

        # run image through encoder
        image_np = np.expand_dims(image_np, axis=0)
        x = preprocess_image(image_np)
        dec_layer1_state_h, dec_layer1_state_c, feature_vector = inference_enc_model.predict(x, verbose=0)

        # predict sentence word for word
        prev_word_index = START_INDEX
        produced_string = ''
        pred_seq = []

        for j in range(MAX_LENGTH):
            x = np.reshape(np.array(prev_word_index), (1, 1))
            preds, dec_layer1_state_h, dec_layer1_state_c = dec_model.predict(
                    [feature_vector, x, dec_layer1_state_h, dec_layer1_state_c], verbose = 0)
            prev_word_index = np.asarray(preds[0][0]).argmax()
            pred_seq.append(prev_word_index)

            if prev_word_index == STOP_INDEX:
                break

        tokens_to_words(dest_tokenizer, pred_seq)
        print('\n\n')
