
import tensorflow as tf
import numpy as np

#Utility functions 

def normalize(features):

    moments = tf.nn.moments(features, axes=[1, 2])
    mean = moments[0]
    stdev = moments[1]
    mean = np.repeat(mean[:, np.newaxis, np.newaxis, :], 32, axis=1)
    mean = np.repeat(mean, 32, axis=2)
    stdev = np.repeat(stdev[:, np.newaxis, np.newaxis, :], 32, axis=1)
    stdev = np.repeat(stdev, 32, axis=2)

    features = (features - mean) / np.sqrt(stdev)
    return features

def get_loss(labels, logits):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
    )

def random_crop(image):
    image = tf.image.random_flip_left_right(image)
    cropped_image = tf.image.random_crop(image, size=(32, 32, 3))
    return cropped_image

def augment(image_data):
    pad_width = [(0, 0), (4, 4), (4, 4), (0, 0)]
    image_data = np.pad(image_data, pad_width, constant_values=0)
    image_data = tf.map_fn(random_crop, image_data, dtype=tf.float32)
    return image_data
