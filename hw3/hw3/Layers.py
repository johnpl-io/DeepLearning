from hw3.dense import Linear
import tensorflow as tf


class DenseLayer(tf.Module):
    def __init__(self, num_inputs, num_outputs, activation=tf.nn.relu):
        self.linear = Linear(num_inputs, num_outputs)
        self.activation = activation

    def __call__(self, x):
        z = self.linear(x)
        return self.activation(z)
