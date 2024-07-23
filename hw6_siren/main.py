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


# Training Image Fitting Model
siren_model = Siren(
    num_inputs=2, num_outputs=3, num_hidden_layers=5, hidden_layer_width=350
)

iterations = 1500
optimizer = Adam(1e-4)
model_output = siren_model.train_model(
    train_on=img,
    get_loss=get_loss_imagefit,
    iterations=iterations,
    shape=[273, 365, -1],
    optimizer=optimizer,
    frame_folder="image_fit",
)

img_tensor = tf.reshape(model_output, shape=[273, 365, -1])

plt.imshow(img_tensor)
plt.show()

tf_save_img(img_tensor, "artifacts/imagefit.jpg")


## upscaling image x2 (730x546) by interpolating points

x_vals_double, y_vals_double = tf.meshgrid(
    tf.linspace(-1, 1, 365 * 2), tf.linspace(-1, 1, 273 * 2)
)

cord_vals_double = tf.stack(
    [tf.reshape(y_vals_double, -1), tf.reshape(x_vals_double, -1)], axis=1
)

double_model_output = siren_model(tf.cast(cord_vals_double, dtype=tf.float32))
img_tensor_upscaled = tf.reshape(double_model_output, shape=[2 * 273, 2 * 365, -1])

plt.imshow(img_tensor_upscaled)
plt.show()

tf_save_img(img_tensor_upscaled, "artifacts/imagefit_upscaled.jpg")


