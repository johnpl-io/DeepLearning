import tensorflow as tf
import pytest
from ResidualBlock import ResidualBlock

@pytest.mark.parametrize(
    "shape",
    [([5,32,32,32]), ([5,28,28,64]),  ],
)
def test_shortcut(shape):

    '''testing if shortcut connections work'''
    Block = ResidualBlock(kernels = [(3, 3), (3, 3)], depths=[16, 64], input_depth=shape[-1], groups=[8, 8], shortcut=True)
    x = tf.random.uniform(shape)
    z = Block(x)
    tf.assert_equal(z.shape[-1],64)



@pytest.mark.parametrize(
    "shape",
    [([5,32,32,32]), ([10,25,25,64]),  ],
)
def test_noshortcut(shape):
    Block = ResidualBlock(kernels = [(3, 3), (3, 3)], depths=[16, shape[-1]], input_depth=shape[-1], groups=[8, 8], shortcut=False)
    x = tf.random.uniform(shape)
    z = Block(x)
    tf.assert_equal(x.shape, z.shape)

@pytest.mark.parametrize(
    "shape",
    [([5,32,32,32]), ([10,25,25,64]),  ],
)

def test_grad(shape):
    x = tf.random.uniform(shape)
    Block = ResidualBlock(kernels = [(3, 3), (3, 3)], depths=[16, shape[-1]], input_depth=shape[-1], groups=[8, 8])
    with tf.GradientTape() as tape:
        z = Block(x)
        loss = tf.math.reduce_mean(z**2)
    grads = tape.gradient(loss, Block.trainable_variables)
    for grad, var in zip(grads, Block.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)