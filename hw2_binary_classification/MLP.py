import tensorflow as tf
import numpy as np
from Layers import Layer


class MLP(tf.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.identity,
        output_activation=tf.identity,
    ):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layer_width = hidden_layer_width
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.layers = [
            Layer(self.num_inputs, hidden_layer_width, self.hidden_activation)
        ]  # first hidden layer
        for i in range(num_hidden_layers):  # rest of the hidden layers
            self.layers.append(
                Layer(
                    self.hidden_layer_width,
                    self.hidden_layer_width,
                    self.hidden_activation,
                )
            )
        self.layers.append(
            Layer(self.hidden_layer_width, self.num_outputs, self.output_activation)
        )  # output layer

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
