import tensorflow as tf
import numpy as np
from Conv2d import Conv2d
from Dense import DenseLayer
import math
from dropout import DropLayer


class Classifier(tf.Module):
    def __init__(
        self,
        input_shape: int,
        layer_depths: list[int],
        layer_kernel_sizes: list,
        num_classes: int,
        strides=[(1, 1)],
        dropout_rate=0.2,
    ):
        if len(strides) == 1 and len(strides) < len(layer_depths):
            strides = strides * len(layer_depths)
        self.conv_layers = [
            Conv2d(
                input_depth=input_shape[-1],
                kernel_size=layer_kernel_sizes[0],
                filters=layer_depths[0],
                strides=strides[0],
                activation=tf.nn.relu,
            )
        ]
        for i in range(1, len(layer_depths)):
            self.conv_layers.append(
                Conv2d(
                    input_depth=self.conv_layers[i - 1].output_shape,
                    kernel_size=layer_kernel_sizes[i],
                    filters=layer_depths[i],
                    strides=strides[i],
                    activation=tf.nn.relu,
                )
            )
            self.conv_layers.append(DropLayer(dropout_rate))
        self.is_train = True
        out_width = 0
        self.batch_size = input_shape[0]
        i_width = input_shape[1]
        for x in range(0, len(layer_kernel_sizes)):
            out_width = math.ceil(
                (i_width - layer_kernel_sizes[x][0] + 1) / strides[x][0]
            )
            i_width = out_width

        out_height = 0
        i_height = input_shape[2]
        for x in range(0, len(layer_kernel_sizes)):
            out_height = math.ceil(
                (i_height - layer_kernel_sizes[x][1] + 1) / strides[x][1]
            )
            i_height = out_height

        self.dense_layer = [
            DenseLayer(
                num_inputs=(out_width * out_height) * layer_depths[-1],
                num_outputs=num_classes,
            )
        ]

    def __call__(self, x):
        input_shape = x.shape[0]
        for conv_layer in self.conv_layers:
            if self.is_train == True:
                x = conv_layer(x)
            elif type(conv_layer) == Conv2d:
                x = conv_layer(x)
        x = tf.reshape(x, [input_shape, -1])
        for dense_layer in self.dense_layer:
            x = dense_layer(x)
        return x
