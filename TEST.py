import numpy as np
import tensorflow as tf
import tensorlayer as tl
from SR_MODEL import get_G


# LOAD LOW-RESOLUTION  NOISY IMAGE
# IN THIS EXAMPLE, WE CONSIDER 128 X 128 X 1 image affected by Gaussian noise with mu = 0, and sigma = 0.01

lr_noisy_image = #LOAD IMAGE TENSOR
#----------------------------------------------------------------

# EVENTUALLY PROCESS INPUT IMAGE, ACCORDING TO THE DIMENSION OF THE TENSOR
lr_noisy_image = tf.expand_dims(lr_noisy_image,2)
lr_noisy_image = tf.expand_dims(lr_noisy_image,0)

G = get_G((batch_size, 128, 128, 1))
G.load_weights('./savedModel/my_model.h5')
G.eval()   
 
hr_image = G(lr_noisy_image,training=True)
