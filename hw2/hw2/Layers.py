from linear import Linear
import tensorflow as tf


class Layer(tf.Module):
    def __init__(self, num_inputs, num_outputs, activation=tf.identity):
        self.linear = Linear(num_inputs, num_outputs)
        self.activation = activation

    def __call__(self, x):
        z = self.linear(x)
        return self.activation(z)
