import tensorflow as tf
import pytest
from GroupNorm import GroupNorm
rng = tf.random.get_global_generator()
rng.reset_from_seed(2384230948)
@pytest.mark.parametrize(
    "shape, C, G",
    [([3, 32, 32, 10], 10, 5), ([3, 16, 16, 25], 25, 5)],
)
def test_sizeprop(shape, C, G):
    """This function ensures that groupnorm preserves size
    and that the values after groupnorm is performed is normalized"""
    x = rng.uniform(shape=shape)
    Group = GroupNorm(C=C, G=G)
    z = Group(x)
    tf.debugging.assert_near(tf.math.reduce_mean(z), 0., atol=0.01)
    tf.debugging.assert_near(tf.math.reduce_std(z), 1., atol=0.01)
    tf.assert_equal(x.shape, z.shape)
@pytest.mark.parametrize(
    "shape, C, G",
    [([3, 32, 32, 6], 6, 2), ([3, 16, 16, 25], 25, 1)],
)

def test_grad(shape, C, G):
    x = rng.normal(shape=shape)
    Group = GroupNorm(C=C, G=G)
    with tf.GradientTape() as tape:
        z = Group(x)
        loss = tf.math.reduce_mean(z**2)
    grads = tape.gradient(loss, Group.trainable_variables)
    for grad, var in zip(grads, Group.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)
