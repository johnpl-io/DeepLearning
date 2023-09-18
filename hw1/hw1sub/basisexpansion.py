import tensorflow as tf
class BasisExpansion(tf.Module):
    def __init__(self, M, bias=True):
        rng = tf.random.get_global_generator()

        self.M = M
        
        self.mu = tf.Variable(rng.normal(shape=[1, self.M],), 
                              trainable=True)

        self.sigma = tf.Variable(tf.ones(shape=[1, self.M], name="sigma"), 
                                  trainable=True )

    def __call__(self, x):
        phi = tf.exp(-((x - self.mu) ** 2) / (self.sigma) ** 2)
        return phi

        return z