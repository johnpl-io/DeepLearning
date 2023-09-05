import tensorflow as tf
import numpy as np

from linear import Linear

from basisexpansion import BasisExpansion

def testofSizesCombined(X = 10, N = 5):
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)


    Guass = BasisExpansion(M = 5)
    a = rng.normal(shape=[X, N])
    linear = Linear(M = 5)
    phis = Guass(a)

    linear = Linear(M = 5)

    result = linear(phis)
 
    tf.assert_equal(tf.shape(phis),[X, N] )
    tf.assert_equal(tf.shape(result),[X, 1] )

def test_training_combined():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    num_inputs = 12
    num_outputs = 1
    Guass = BasisExpansion(num_inputs)
    linear = Linear(M = num_inputs, num_outputs = num_outputs)
    a = rng.normal(shape=[1, num_inputs])
    with tf.GradientTape() as tape:
        z = linear(Guass(a))
        loss = tf.math.reduce_mean(z**2)
    grads = tape.gradient(loss, linear.trainable_variables)
    for grad, var in zip(grads, linear.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)    

