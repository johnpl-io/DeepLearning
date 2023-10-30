import tensorflow as tf

class LayerNorm(tf.Module):
    def __init__(self, shape, eps=1e-5):
        self.eps = eps
        self.gamma = tf.Variable(
            tf.ones(shape), trainable=True, name="gamma"
        )
        self.beta = tf.Variable(
            tf.zeros(shape), trainable=True, name="beta"
        )
    

    def __call__(self, x):
        mean, var = tf.nn.moments(x, [1], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.eps)  
        return x * self.gamma + self.beta

module = LayerNorm(shape=100)
x = tf.random.normal(shape=[32, 100])

y = module(x)


