import numpy as np
import tensorflow as tf


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


import numpy as np
import tensorflow as tf


def pos_encoding(length, d_model):
    pos_enc = np.zeros((length, d_model))
    for pos in range(length):
        for i in range(d_model // 2):
            angle = pos / np.power(10000, 2 * i / d_model)
            pos_enc[pos, 2 * i] = np.sin(angle)
            pos_enc[pos, 2 * i + 1] = np.cos(angle)

    pos_enc = tf.cast(pos_enc, dtype=tf.float32)  # Convert to a TensorFlow tensor
    return pos_enc


def pos_encoding(length, dim_model):
    pos_enc = np.zeros((length, dim_model))
    position = np.arange(0, length)[:, np.newaxis]
    div = np.power(10000, 2 * np.arange(0, dim_model // 2) / dim_model)

    pos_enc[:, 0::2] = np.sin(position / div[0::2])
    pos_enc[:, 1::2] = np.cos(position / div[1::2])

    return tf.cast(pos_enc, dtype=tf.float32)


def colin(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return P


breakpoint()
