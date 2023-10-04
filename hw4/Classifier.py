import math

import numpy as np
import tensorflow as tf

from AveragePool import AveragePool2d
from Conv2d import Conv2d
from Dense import DenseLayer
from dropout import DropLayer
from GroupNorm import GroupNorm
from MaxPool2d import MaxPool2d
from ResidualBlock import ResidualBlock


class Classifier(tf.Module):
    def __init__(
        self,
        input_shape: int,
        layer_depths: list[int],
        layer_kernel_sizes: list,
        num_classes: int,
        res_depths,
        out_layer,
        strides=[(1, 1)],
    ):
        self.isTrain = True
        if len(strides) == 1 and len(strides) < len(layer_depths):
            strides = strides * len(layer_depths)
        self.conv1_layers = [
            Conv2d(
                input_depth=input_shape,
                kernel_size=layer_kernel_sizes[0],
                filters=layer_depths[0],
                strides=(1, 1),
            ),
            GroupNorm(layer_depths[0], 8),
            tf.nn.relu,
            MaxPool2d(ksize=2, strides=2),
        ]

        self.conv2_layers = [
            Conv2d(
                input_depth=layer_depths[0],
                kernel_size=layer_kernel_sizes[1],
                filters=layer_depths[1],
                strides=(1, 1),
            ),
            GroupNorm(layer_depths[1], 8),
            tf.nn.relu,
            MaxPool2d(ksize=2, strides=2),
        ]

        self.res1_layer = [
            ResidualBlock(
                kernels=[(3, 3), (3, 3)],
                depths=res_depths[0],
                input_depth=layer_depths[1],
                groups=[8, 8],
            )
        ]

        for i in range(1, len(res_depths) - 1):
            shortcut = False
            if i == 0:
                input_depth = layer_depths[1]
            else:
                input_depth = self.res1_layer[i - 1].layers[-2].output_shape
            if input_depth != res_depths[i][1]:
                shortcut = True
            self.res1_layer.append(
                ResidualBlock(
                    kernels=[(3, 3), (3, 3)],
                    depths=res_depths[i],
                    input_depth=input_depth,
                    groups=[8, 8],
                    shortcut=shortcut,
                )
            )

        self.res1_layer.append(MaxPool2d(ksize=4, strides=4))
        self.dense_layer = [DropLayer(0.4), DenseLayer(out_layer, num_classes)]

    def __call__(self, x):
        input_shape = x.shape[0]
        for layer in self.conv1_layers:
            x = layer(x)

        for layer in self.conv2_layers:
            x = layer(x)

        for layer in self.res1_layer:
            x = layer(x)
        x = tf.reshape(x, [input_shape, -1])
        for dense_layer in self.dense_layer:
            if self.isTrain == True:
                x = dense_layer(x)
            elif type(dense_layer) != DropLayer:
                x = dense_layer(x)

        return x
