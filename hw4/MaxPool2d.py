import tensorflow as tf


class MaxPool2d(tf.Module):
    def __init__(self, ksize=(4, 4), strides=None, padding="VALID"):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def __call__(self, x):
        return tf.nn.max_pool2d(x, self.ksize, self.strides, self.padding)


s = MaxPool2d(ksize=2, strides=2)
x = tf.random.uniform([5, 32, 32, 32])
f = s(x)
