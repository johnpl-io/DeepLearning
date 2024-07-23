import tensorflow as tf
import math
import numpy as np

'''Prof. Curro Linear Class'''

class Linear(tf.Module):
    def __init__(self, M, num_outputs = 1, bias=True):
        rng = tf.random.get_global_generator()
    
        self.M = M
        
        self.w = tf.Variable(rng.normal(shape=[self.M, num_outputs], ), 
                             trainable=True)

        self.bias = bias

        if self.bias:
            self.b = tf.Variable( tf.zeros(shape=[1, 1], ), 
                                  trainable=True)

    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b

        return z







