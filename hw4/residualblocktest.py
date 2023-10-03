import tensorflow as tf
import pytest
from ResidualBlock import ResidualBlock

def Block():
    Block = ResidualBlock(kernels = [(3, 3), (3, 3)], depths=[32, 32], input_depth=32, groups=[8, 8, 8], shortcut=True)
    x = tf.random.uniform([5,32,32,32])
    Block(x)
Block()