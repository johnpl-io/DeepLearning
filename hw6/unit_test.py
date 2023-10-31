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


def test_shape_mha():
    x = tf.random.normal(shape=[4, 10, 16])
    head = MultiHeadAttention(16, 8, 12, True)
    z = head(x)
    tf.assert_equal(z.shape, x.shape)


def test_shape_transformer_block():
    x = tf.random.uniform(shape=[4, 10, 32])
    transformer = TransformerBlock(32, 16, 10, 0.1, mask=True)
    z = transformer(x)
    tf.assert_equal(z.shape, x.shape)
'''
def test_shape_transformer_block():
    x = tf.convert_to_tensor([[0, 2, 3, 4, 5, 2, 3, 6, 7, 8, 9]])

    model = TransformerDecoder(
    dim_model=512, heads=8, blocks=5, is_train=[False], vocab_size=10)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        z = model(x)
        grad = tape.gradient(z, {"x": x})
    breakpoint()
'''
'''    
def test_grad_mha():
    x = tf.random.normal(shape=[4, 10, 32])
    head = MultiHeadAttention(32, 16, 10, False)
    with tf.GradientTape() as tape:
        z = head(x)
        loss = tf.math.reduce_mean(z**2)
    grads = tape.gradient(loss, head.trainable_variables)
    for grad, var in zip(grads, head.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)
'''
  

def test_causual_mask():
    x = tf.random.normal(shape=[1, 10, 5])
    head = MultiHeadAttention(5, 16, 10, True)
    head2 = MultiHeadAttention(5, 16, 10, False)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        z = head(x)
        z2 = head2(x)
    grad = tape.batch_jacobian(z, x)
    grad2 = tape.batch_jacobian(z2, x)
    breakpoint()


grad[0][0][4][1:]
