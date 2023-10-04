import tensorflow as tf

from Conv2d import Conv2d
from GroupNorm import GroupNorm


# inspired by tensorflow source code
class ResidualBlock(tf.Module):
    def __init__(
        self,
        kernels,
        input_depth,
        depths,
        groups,
        activation=tf.nn.relu,
        shortcut=False,
    ):
        self.use_shortcut = shortcut
        if shortcut:
            self.shortcut = Conv2d(input_depth, depths[-1], (1, 1), padding="SAME")
            self.norm1 = GroupNorm(depths[-1], 1)
        self.layers = [
            activation,
            Conv2d(input_depth, depths[0], kernels[0], strides=(1, 1), padding="SAME"),
            GroupNorm(depths[0], 8),
        
        ]
    
        self.layers.append(activation)
        self.layers.append(
                Conv2d(
                    input_depth=depths[0],
                    kernel_size=kernels[1],
                    filters=depths[1],
                    strides=(1, 1),
                    padding="SAME",
                )
            )
        self.layers.append(GroupNorm(depths[1], 8))


        self.activation = activation

    def __call__(self, x):
        shortcut = x
        if self.use_shortcut:
            shortcut = self.shortcut(x)
            shortcut = self.norm1(shortcut)
        out = x
        for i in range(0, len(self.layers) - 1):
            out = self.layers[i](out)
        add = shortcut + self.layers[-1](out)
        out = add
        return self.activation(out)


"""
Block = ResidualBlock(kernels = [(3, 3), (3, 3)], depths=[32, 32], input_depth=32, groups=[8, 8], shortcut=True)
x = tf.random.uniform([5,32,32,32])
Block(x)
"""
