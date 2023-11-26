import tensorflow as tf

# implementation of Siren network from https://arxiv.org/pdf/2006.09661.pdf
from sinelayer import *

from dense import DenseLayer

class Siren(tf.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, hidden_layer_width):
        # first layer has slightly different initialization scheme
        self.layers = [
            SineLayer(
                num_inputs, hidden_layer_width, siren_initializer=siren_init_first
            )
        ]

        for i in range(num_hidden_layers):
            self.layers.append(
                SineLayer(
                    num_inputs=hidden_layer_width,
                    num_outputs=hidden_layer_width,
                    siren_initializer=siren_init_layer,
                )
            )

      #  last layer
        self.layers.append(DenseLayer(num_inputs=hidden_layer_width, num_outputs=num_outputs, activation=tf.nn.relu))


      #  self.layers.append(SineLayer(num_inputs=hidden_layer_width, num_outputs=num_outputs, siren_initializer=siren_init_layer))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x



siren_model = Siren(num_inputs=2, num_outputs=3, num_hidden_layers=5, hidden_layer_width=256)