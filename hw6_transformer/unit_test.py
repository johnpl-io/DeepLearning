import math

import numpy as np
import pytest
import tensorflow as tf

from AdamW import AdamW
from multi_head_attention import MultiHeadAttention
from transformer_block import TransformerBlock
from decoder import TransformerDecoder

rng = tf.random.get_global_generator()
rng.reset_from_seed(2384230948)


# testing gradient with mha and mask
def test_grad_mha():
    x = tf.random.normal(shape=[4, 10, 32])
    head = MultiHeadAttention(32, 16, 10, True)

    with tf.GradientTape() as tape:
        z = head(x)
        loss = tf.math.reduce_mean(z**2)
    grads = tape.gradient(loss, head.trainable_variables)
    for grad, var in zip(grads, head.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)


def test_grad_mha_no_mask():
    x = tf.random.normal(shape=[4, 10, 32])
    head = MultiHeadAttention(32, 16, 10, False)

    with tf.GradientTape() as tape:
        z = head(x)
        loss = tf.math.reduce_mean(z**2)
    grads = tape.gradient(loss, head.trainable_variables)
    for grad, var in zip(grads, head.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)


# making sureshapes are consistent after mha
def test_shape_mha():
    x = tf.random.normal(shape=[4, 10, 16])
    head = MultiHeadAttention(16, 8, 12, True)
    z = head(x)
    tf.assert_equal(z.shape, x.shape)


# making sure shapes are consistent after transformer block
def test_shape_transformer_block():
    x = tf.random.uniform(shape=[4, 10, 32])
    transformer = TransformerBlock(32, 16, 10, 0.1, mask=True)
    z = transformer(x)
    tf.assert_equal(z.shape, x.shape)


# testing causal mask with jacobian explained more in the readME
def test_causual_mask():
    x = tf.random.normal(shape=[1, 10, 5])
    head = MultiHeadAttention(5, 16, 10, True)
    head2 = MultiHeadAttention(5, 16, 10, False)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        z = head(x)
        z2 = head2(x)
    jacobian_with_mask = tape.batch_jacobian(z, x)
    jacobian_no_mask = tape.batch_jacobian(z2, x)
    tf.debugging.assert_near(
        jacobian_with_mask[0, 0, 1:5, 1:], 0.0
    )  # testing partial derivatives with respect to future values only
    tf.debugging.assert_none_equal(jacobian_no_mask[0, 0, 1:5, 1:], 0.0)
