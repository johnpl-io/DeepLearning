import numpy as np
import tensorflow as tf


def xavier_init(shape, in_dim, out_dim):
    # Computes the xavier initialization values for a weight matrix
    xavier_lim = tf.sqrt(6.0) / tf.sqrt(tf.cast(in_dim, tf.float32))
    weight_vals = tf.random.uniform(
        shape=shape, minval=-xavier_lim, maxval=xavier_lim, seed=22
    )
    return weight_vals


def he_init(shape, in_dim):
    # Computes the He initialization values for a weight matrix
    stddev = tf.sqrt(2.0 / tf.cast(in_dim, tf.float32))
    weight_vals = tf.random.normal(shape=shape, mean=0, stddev=stddev, seed=22)
    return weight_vals


class Conv2d(tf.Module):
    def __init__(
        self,
        input_depth,
        filters,
        kernel_size,
        strides=(1, 1),
        use_bias=True,
        padding="VALID",
    ):
        self.w = tf.Variable(
            he_init(
                shape=[kernel_size[0], kernel_size[1], input_depth, filters],
                in_dim=input_depth,
            ),
            trainable=True,
            name="Conv/w",
        )

        self.strides = strides
        self.output_shape = filters
        self.bias = use_bias
        if self.bias:
            self.b = tf.Variable(
                tf.zeros(
                    shape=[1, filters],
                ),
                trainable=True,
                name="Linear/b",
            )
        self.padding = padding

    def __call__(self, x):
        f = tf.nn.conv2d(x, filters=self.w, strides=self.strides, padding=self.padding)
        if self.bias:
            f = f + self.b
        return f
