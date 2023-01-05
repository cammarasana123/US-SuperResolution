import numpy as np
import tensorflow as tf
import tensorlayer as tl
from wdsr import wdsr_custom
import pathlib

### Input images are the output of the up-sampling algorithm, e.g., cubic-convolution
PATH_test = pathlib.Path('path/to/images')
batch_size_test = 1

def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_bmp(image)
  input_image = tf.cast(input_image, tf.float32)
  return input_image

   
def load_image_train(image_file):
  input_image, real_image = load(image_file)
  return input_image, real_image
       
test_ds = tf.data.Dataset.list_files(str(PATH_test/'*.bmp'))
test_ds = test_ds.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)

G = wdsr_custom(my_parameters)
G.load_weights('path/to/model.h5')


for test_input in test_ds.take(lengthTest):
    prediction = G(test_input)
