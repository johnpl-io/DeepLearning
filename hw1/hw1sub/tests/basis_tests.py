#Test Guassian 
import pytest



from basisexpansion import BasisExpansion
import tensorflow as tf
import numpy as np


def testNonLinear():
    M = 10
    Guass = BasisExpansion(M)
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    a = rng.normal(shape=[10, 1])  
    b = rng.normal(shape=[10, 1]) 


    tf.debugging.assert_none_equal(Guass(a+b), Guass(a) + Guass(b))

@pytest.mark.parametrize("numOfBasis, batch_size", [(5, 100), (2, 50), (50, 3)])  
def testofGuass(numOfBasis, batch_size):
    phi = BasisExpansion(numOfBasis)
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    a = rng.normal(shape=[batch_size, 1])
    tf.debugging.assert_near(tf.exp(-((a - phi.mu) ** 2) / (phi.sigma) ** 2), phi(a))

def testofStat():
    #checking that the mean of a single guassian is the largest value
    phi = BasisExpansion(1)
    a = tf.linspace(-10, 10, 10000)
    a = tf.reshape(a, shape= [10000, 1])
    a = tf.cast(a, tf.float32)
    phi_values = phi(a).numpy()
    max_index = np.argmax(phi_values)
    max_a = a[max_index]
    a = phi.mu[0] 
    b = max_a
   
    tf.debugging.assert_near(a, b, atol=0.001)




def testofTrainingGuass():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    batch_size = 10
    numOfBasis = 12

    Guass = BasisExpansion(10)
    a = rng.normal(shape=[1, batch_size])
    with tf.GradientTape() as tape:
        z = Guass(a)
        loss = tf.math.reduce_mean(z**2)
    grads = tape.gradient(loss, Guass.trainable_variables)
    for grad, var in zip(grads, Guass.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    

