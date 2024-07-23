import tensorflow as tf


class AveragePool2d(tf.Module):
    def __init__(self, ksize=(2, 2), strides=None, padding="VALID"):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def __call__(self, x):
        return tf.nn.avg_pool2d(x, self.ksize, self.strides, self.padding)
