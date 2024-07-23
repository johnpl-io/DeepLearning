import tensorflow as tf


class GroupNorm(tf.Module):
    def __init__(self, C, G, eps=1e-5):
        self.gamma = tf.Variable(
            tf.ones(shape=[1, 1, 1, C]), trainable=True, name="gamma"
        )
        self.beta = tf.Variable(
            tf.zeros(shape=[1, 1, 1, C]), trainable=True, name="beta"
        )
        self.groups = G
        self.eps = eps

    def __call__(self, x):
        N, H, W, C = x.shape
        x = tf.reshape(x, [N, H, W, self.groups, C // self.groups])

        mean, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)

        x = (x - mean) / tf.sqrt(var + self.eps)

        x = tf.reshape(x, [N, H, W, C])
        return x * self.gamma + self.beta
