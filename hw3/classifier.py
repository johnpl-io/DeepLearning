import tensorflow as tf
import numpy as np
from Conv2d import Conv2d
from dense import DenseLayer
import math
from dropout import DropLayer
class Classifier(tf.Module):
    def __init__(
        self,
        input_shape: int,
        input_depth: int,
        layer_depths: list[int],
        layer_kernel_sizes: list,
        num_classes: int,
        strides=(1,1),
    ):
        self.conv_layers = [
            Conv2d(
                input_shape=input_depth,
                kernel_size=layer_kernel_sizes[0],
                filters=layer_depths[0],
                strides=strides,
                activation=tf.nn.relu,
            )
        ]
        for i in range(1, len(layer_depths)):
            self.conv_layers.append(
                Conv2d(
                    input_shape=self.conv_layers[i - 1].output_shape,
                    kernel_size=layer_kernel_sizes[i],
                    filters=layer_depths[i],
                    strides=strides,
                    activation=tf.nn.relu,
                )
            )
            self.conv_layers.append(DropLayer(0.2))

        out_width = 0
        self.batch_size = input_shape[0]
        i_width = input_shape[1]
        for x in range(0, len(layer_kernel_sizes)):
            out_width = math.ceil((i_width - layer_kernel_sizes[x][0] + 1)/strides[0])
            i_width = out_width
        self.dense_layer = [
            DenseLayer(
                num_inputs=(out_width**2) * layer_depths[-1],
                num_outputs=num_classes,
               
            )
        ]

    def __call__(self, x):
        input_shape = x.shape[0]
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = tf.reshape(x, [input_shape, -1])
        for dense_layer in self.dense_layer:
            x = dense_layer(x)
        return x

'''
z = Classifier(
    input_shape = (4, 28, 28, 1),
    input_depth=1,
    layer_depths=[6, 2, 3],
    layer_kernel_sizes=[(2, 2), (2, 2), (2, 2)],
    num_classes=10,
)
x = np.zeros((4, 28, 28, 1))
s = z(x)

'''