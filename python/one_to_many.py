import json
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models imort Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.VGG19 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

import pickle
import gzip
import logging

tf.get_logger().setLevel(logging.ERROR)

TRAINING_FILE_DIR = '../data/coco/'
OUTPUT_FILE_DIR = 'tf_data/feature_vectors/'


