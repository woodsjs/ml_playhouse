import json
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications import vgg19
from tensorflow.keras.applications.vgg19 import preprocess_input
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
