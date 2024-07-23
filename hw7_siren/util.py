import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import trange

from adam import Adam


def get_image(image_path):
    img = np.asarray(Image.open(image_path))
    return img


def get_loss_imagefit(siren_model, cord_vals, image_true, shape):
    model_output = siren_model(tf.cast(cord_vals, dtype=tf.float32))
    return tf.reduce_mean((model_output - image_true) ** 2), model_output


def get_image_resize(image_path):
    img = Image.open(image_path)
    img = img.resize((365, 273), Image.Resampling.LANCZOS)
    return np.asarray(img)


def get_sobel(img):
    sobel_y = tf.image.sobel_edges(img)[0, :, :, :, 0]
    sobel_x = tf.image.sobel_edges(img)[0, :, :, :, 1]
    sobel_stack = tf.stack([sobel_x, sobel_y], axis=-1)
    return sobel_x, sobel_y, sobel_stack


def get_loss_poisson(siren_model, cord_vals, sobel_true, shape):
    model_output = siren_model(tf.cast(cord_vals, dtype=tf.float32))
    model_output_reshape = tf.reshape(model_output, shape=shape)[None, :]

    return (
        #assumning rgb (3 channel images)
        tf.reduce_mean((tf.reshape(get_sobel(model_output_reshape)[-1], shape=[shape[0] * shape[1], 6])- sobel_true)** 2), 
        model_output,
    )


def tf_save_img(img_tensor, filename):
    img = img_tensor.numpy() * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save(filename)
