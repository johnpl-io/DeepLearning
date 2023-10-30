import tensorflow as tf
from Dense import DenseLayer
class FeedForwardNetwork(tf.Module):
    def __init__(self, d_model):  #conforming to the paper, there is a *4 increase in size for a single hidden layer of the FFN
        self.layers = [
        DenseLayer(num_inputs=d_model, num_outputs=d_model*4, activation=tf.nn.relu),
        DenseLayer(num_inputs=d_model*4, num_outputs=d_model) #no relu at the end
        ]


    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x