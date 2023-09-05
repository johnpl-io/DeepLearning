# /bin/env python3.8

import pytest



from linear import Linear
from basisexpansion import BasisExpansion
import tensorflow as tf
import numpy as np

#Linear Tests
def test_additivity():
    M = 10
    linear_model = Linear(M, num_outputs=10)
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    a = rng.normal(shape=[1, 10])  
    b = rng.normal(shape=[1, 10]) 


    tf.debugging.assert_near(linear_model(a + b), linear_model(a) + linear_model(b), summarize=2)

def test_homogeneity():

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    M = 15
    linear = Linear(M)
    num_test_cases = 20
    a = rng.normal(shape=[1, M])
    b = rng.normal(shape=[num_test_cases, 1])
    print(linear(a * b))
    tf.debugging.assert_near(linear(a * b), linear(a) * b, summarize=2)



@pytest.mark.parametrize("numOfBasis, num_outputs", [(2, 1), (3, 2), (4, 3)])  
def test_dimensionality(numOfBasis, num_outputs):
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    linear_model = Linear(M=numOfBasis, num_outputs=num_outputs)
    a = rng.normal(shape=[1, numOfBasis])
    z = linear_model(a)
    tf.assert_equal(tf.shape(z)[-1], num_outputs)

@pytest.mark.parametrize("bias", [True, False])
def test_trainable(bias):
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    num_inputs = 10
    num_outputs = 1
    linear = Linear(M = num_inputs, num_outputs = num_outputs, bias=bias)
    a = rng.normal(shape=[1, num_inputs])
    with tf.GradientTape() as tape:
        z = linear(a)
        loss = tf.math.reduce_mean(z**2)
    grads = tape.gradient(loss, linear.trainable_variables)
    for grad, var in zip(grads, linear.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)    
    if bias:
        assert len(grads) == 2
    else:
        assert len(grads) == 1
def test_bias():

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    linear_with_bias = Linear(M = 1, num_outputs=1, bias=True)
    assert hasattr(linear_with_bias, "b")

    linear_with_bias = Linear(M = 1, num_outputs=1, bias=False)
    assert not hasattr(linear_with_bias, "b")
#Test Guassian 
@pytest.mark.parametrize("numOfBasis, batch_size", [(5, 100), (2, 50), (50, 3)])  
def testofGuass(numOfBasis, batch_size):
    phi = BasisExpansion(numOfBasis)
    rng = tf.random.get_global_generator()
    a = rng.normal(shape=[batch_size, 1])
    tf.debugging.assert_near(tf.exp(-((a - phi.mu) ** 2) / (phi.sigma) ** 2), phi(a))

def testofStat():
    phi = BasisExpansion(1)
    import matplotlib.pyplot as plt
    a = tf.linspace(-10, 10, 10000)
    a = tf.reshape(a, shape= [10000, 1])
    a = tf.cast(a, tf.float32)
    plt.plot(a, phi(a))
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

def testofGradCombined():
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    batch_size = 15
    numOfBasis = 5

    Guass = BasisExpansion(numOfBasis)
    a = rng.normal(shape=[1, batch_size])
    linear = line(M = numofBasis)
    with tf.GradientTape() as tape:
        z = Guass(a)
        loss = tf.math.reduce_mean(z**2)
    grads = tape.gradient(loss, Guass.trainable_variables)
    for grad, var in zip(grads, Guass.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)    
    

#TestCombined
