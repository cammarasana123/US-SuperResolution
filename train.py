#! /usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from wdsr import wdsr_custom
from config import config
import pathlib

###====================== DATA-SET====================================###
#The data set preparation requires down-sample images and up-sampled images through an interpolation algorithm (e.g., cubic convolution).
#In this example, we apply montage to LR and SR images and load them together; then, we split input and target images

###====================== HYPER-PARAMETERS ===========================###
#Initialize training parameters

PATH_train = pathlib.Path('path/to/train')

def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_bmp(image)
# split image into input and target
  return input_image, real_image
   
def load_image_train(image_file):
  input_image, real_image = load(image_file)
  return input_image, real_image


def train():
    train_ds = tf.data.Dataset.list_files(str(PATH_train/'*.bmp'))
    train_ds = train_ds.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    G = wdsr_custom(my_parameters)
# Initialize optimisation parameters

    ## initialize learning (G)
    for epoch in range(n_epoch_init):
        log_LOSS = 0
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):            
            with tf.GradientTape() as tape:
                fake_hr_patchs = G(lr_patchs)
                FP = fake_hr_patchs[:,1::2,:,:]
                HP = hr_patchs[:,1::2,:,:]
                log_loss = my_error_function(FP,HP)
            grad = tape.gradient(mse_loss, G.trainable_weights)
            g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))
            LOG_LOSS = LOG_LOSS + log_loss

train()
