import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from adam import Adam
from siren import Siren
from util import *

# Getting Test Card F
img = get_image("Testcard_F.jpg") / 255
img = img.astype(np.float32)
# Preparing Poisson Model
sobel_siren = Siren(
    num_inputs=2, num_outputs=3, num_hidden_layers=5, hidden_layer_width=350
)

rainbow = get_image_resize("rainbow.jpg") / 255
rainbow = rainbow.astype(np.float32)

rainbow_sobel_stack = get_sobel(tf.convert_to_tensor(rainbow)[None, :])[-1]
img_sobel_stack = get_sobel(tf.convert_to_tensor(img)[None, :])[-1]


#to make rainbow more prominent scale gradients by 3 the gradients
combined_sobel = 3 * rainbow_sobel_stack + img_sobel_stack

iterations = 5000
optimizer = Adam(1e-4)

model_output = sobel_siren.train_model(
    train_on=combined_sobel,
    get_loss=get_loss_poisson,
    iterations=iterations,
    shape=[273, 365, -1],
    optimizer=optimizer,
    frame_folder="poisson",
)

img_tensor = tf.reshape(model_output, shape=[273, 365, -1])
plt.imshow(img_tensor)
plt.show()
tf_save_img(img_tensor, "artifacts/poisson_blend.jpg")
