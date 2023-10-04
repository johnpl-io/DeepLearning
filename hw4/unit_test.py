import math

import numpy as np
import pytest
import tensorflow as tf

from AdamW import AdamW
from Classifier import Classifier
from Conv2d import Conv2d
from Dense import DenseLayer
from dropout import DropLayer

rng = tf.random.get_global_generator()
rng.reset_from_seed(2384230948)


@pytest.mark.parametrize(
    "batch_size, input_depth, strides, kernel_size, output_depth, x_dim, y_dim",
    [(12, 32, (1, 1), (3, 4), 64, 30, 30), (10, 1, (2, 5), (3, 3), 32, 32, 10)],
)
def test_conv2d(
    batch_size, input_depth, strides, kernel_size, output_depth, x_dim, y_dim
):
    conv2dobj = Conv2d(
        input_depth=input_depth,
        strides=strides,
        filters=output_depth,
        kernel_size=kernel_size,
    )
    x = rng.normal(shape=[batch_size, x_dim, y_dim, input_depth])

    z = conv2dobj(x)

    z_dim_x = math.ceil((x_dim - kernel_size[0] + 1) / strides[0])

    z_dim_y = math.ceil((y_dim - kernel_size[1] + 1) / strides[1])

    tf.assert_equal((batch_size, z_dim_x, z_dim_y, output_depth), z.shape)


@pytest.mark.parametrize(
    "batch_size, input_depth, strides, kernel_size, output_depth, x_dim, y_dim",
    [(6, 11, (2, 2), (4, 4), 60, 31, 31), (12, 12, (3, 5), (5, 5), 32, 32, 10)],
)
def test_dim(batch_size, input_depth, strides, kernel_size, output_depth, x_dim, y_dim):
    conv2dobj = Conv2d(
        input_depth=input_depth,
        strides=strides,
        filters=output_depth,
        kernel_size=kernel_size,
    )
    x = rng.normal(shape=[batch_size, x_dim, y_dim, input_depth])
    with tf.GradientTape() as tape:
        z = conv2dobj(x)
        loss = tf.math.reduce_mean(z**2)
    grads = tape.gradient(loss, conv2dobj.trainable_variables)
    for grad, var in zip(grads, conv2dobj.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)


def test_classifier():
    input_shape = (1, 52, 52, 2)
    num_classes = 15
    conv_cnn = Classifier(
        input_shape=input_shape,
        layer_depths=(32, 63, 64),
        layer_kernel_sizes=[(3, 3), (3, 3), (3, 3)],
        num_classes=num_classes,
    )
    x = rng.normal(shape=input_shape)
    z = conv_cnn(x)
    tf.debugging.assert_equal((1, num_classes), z.shape)


def test_additivity():
    M = 15
    DenseLayer_model = DenseLayer(M, num_outputs=10)
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    a = rng.normal(shape=[1, 15])
    b = rng.normal(shape=[1, 15])

    tf.debugging.assert_near(
        DenseLayer_model(a + b), DenseLayer_model(a) + DenseLayer_model(b), summarize=2
    )


@pytest.mark.parametrize("num_inputs, num_outputs", [(2, 1), (3, 2), (4, 3)])
def test_dimensionality(num_inputs, num_outputs):
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    linear_model = DenseLayer(num_inputs=num_inputs, num_outputs=num_outputs)
    a = rng.normal(shape=[10, num_inputs])
    z = linear_model(a)

    tf.assert_equal(tf.shape(z)[-1], num_outputs)


def test_dropout():
    shape = shape = [15, 20, 15, 20, 15, 23]
    rate = 0.5
    z = DropLayer(rate)

    a = rng.normal(shape)

    tf.assert_equal(a.shape, shape)


def test_AdamW():
    optomizer = AdamW()
    input_shape = (1, 52, 52, 2)
    num_classes = 15
    conv_cnn = Classifier(
        input_shape=input_shape,
        layer_depths=(32, 63, 64),
        layer_kernel_sizes=[(3, 3), (3, 3), (3, 3)],
        num_classes=num_classes,
    )
    x = rng.normal(shape=input_shape)
    with tf.GradientTape() as tape:
        loss = tf.math.reduce_mean(conv_cnn(x) ** 2)
    grads = tape.gradient(loss, conv_cnn.trainable_variables)

    optomizer.apply_gradients(grads, conv_cnn.trainable_variables)

    assert conv_cnn.trainable_variables
