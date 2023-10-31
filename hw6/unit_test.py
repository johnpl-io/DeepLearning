import math

import numpy as np
import pytest
import tensorflow as tf
from multi_head_attention import MultiHeadAttention
from AdamW import AdamW
from transformer_block import TransformerBlock
rng = tf.random.get_global_generator()
rng.reset_from_seed(2384230948)
def test_causual_mask():

    x = tf.random.normal(shape = [4, 10, 32])
    head = MultiHeadAttention(32, 16, 10, True)

 
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        z = head(x)
        loss = tf.math.reduce_mean(z**2)
    grads = tape.gradient(loss, head.trainable_variables)
    breakpoint()
    for grad, var in zip(grads, head.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)


def test_grad_mha():

    x = tf.random.normal(shape = [4, 10, 32])
    head = MultiHeadAttention(32, 16, 10, True)

    
    with tf.GradientTape() as tape:
        z = head(x)
        loss = tf.math.reduce_mean(z**2)
    grads = tape.gradient(loss, head.trainable_variables)
    for grad, var in zip(grads, head.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)


def test_grad_mha():

    x = tf.random.normal(shape = [4, 10, 32])
    head = MultiHeadAttention(32, 16, 10, False)
    with tf.GradientTape() as tape:
        z = head(x)
        loss = tf.math.reduce_mean(z**2)
    grads = tape.gradient(loss, head.trainable_variables)
    for grad, var in zip(grads, head.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

def test_grad_transformer_block():

    x = tf.random.uniform(shape = [4, 10, 32])
    transformer = TransformerBlock(32, 16, 10, 0.1, mask=True)
    with tf.GradientTape() as tape:
        z = transformer(x)
        loss = tf.math.reduce_mean(z**2)
    grads = tape.gradient(loss, transformer.trainable_variables)
    for grad, var in zip(grads, transformer.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
   
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)



