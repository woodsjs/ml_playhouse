import json
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications import vgg19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

import pickle
import gzip
import logging

tf.get_logger().setLevel(logging.ERROR)

TRAINING_FILE_DIR = '../data/coco/'
OUTPUT_FILE_DIR = 'tf_data/feature_vectors/'

with open(TRAINING_FILE_DIR + 'captions_train2014.json') as json_file:
    data = json.load(json_file)

image_dict = {}

for image in data['images']:
    image_dict[image['id']] = [image['file_name']]

for anno in data['annotations']:
    image_dict[anno['image_id']].append(anno['caption'])

# get a pretrained model, and rip out the top layers.
model = VGG19(weights='imagenet')
model.summary()

model_new = Model(inputs=model.input,
        outputs=model.get_layer('block5_conv4').output)

model_new.summary()


# get them outputs
for i, key in enumerate(image_dict.keys()):
    if i % 1000 == 0:
        print('Progress: ' + str(i) + ' images processed')

    item = image_dict.get(key)
    filename = TRAINING_FILE_DIR + 'train2014/' + item[0]

    # get dimensions
    image = load_img(filename)
    width = image.size[0]
    height = image.size[1]

    # resize
    if height > width:
        image = load_img(filename, target_size=(int(height/width*256), 256))
    else:
        image = load_img(filename, target_size=(256, int(height/width*256)))


    width = image.size[0]
    height = image.size[1]
    image_np = image_to_array(image)

    # crop image, leaving inner bits
    h_start = int((height-224)/2)
    w_start = int((width-224)/2)

    image_np = image_np[h_start:h_start+224,
            w_start:w_start+224]

    # increase dimenstions
    image_np = np.expand_dims(image_np, axis=0)

    # call model and save tensor to disk
    X = preprocess_input(image_np)
    y = model_new.predict(X)

    save_filename = OUTPUT_FILE_DIR + item[0] + '.pickle.gzip'

    pickle_file = gzip.open(sae_filename, 'wb')
    pickle.dump(y[0], pickle_file)
    pickle_file.close()

# save dict with captions and filenames
save_filename = OUTPUT_FILE_DIR + 'caption_file.pickle.gz'

pickle_file = gzip.open(save_filename, 'wb')
pickle.dump(image_dict, pickle_file)
pickle_file.close()

