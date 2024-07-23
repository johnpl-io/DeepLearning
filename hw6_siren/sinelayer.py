import tensorflow as tf


def siren_init_first(shape, omega_0):
    in_dim, out_dim = shape
    weight_vals = tf.random.uniform(shape=shape, minval=-1 / in_dim, maxval=1 / out_dim)
    return weight_vals


def siren_init_layer(shape, omega_0):
    in_dim, out_dim = shape
    weight_vals = tf.random.uniform(
        shape=shape,
        minval=-tf.math.sqrt(6 / in_dim) / omega_0,
        maxval=tf.math.sqrt(6 / in_dim) / omega_0,
    )
    return weight_vals



class SineLayer(tf.Module):
    def __init__(
        self, num_inputs, num_outputs, siren_initializer, bias=True, omega_0=30
    ):
        """Imeplementation of sine layer as denoted in the paper. To speed up training omega_0
        is used in the same fashion as mentioned in Appendix 1.5, leveraging omega_0 for all layers of the Siren
        network
        """

        self.omega_0 = omega_0
        
        self.w = tf.Variable(
            siren_initializer(shape=[num_inputs, num_outputs], omega_0=self.omega_0),
            trainable=True,
            name="Linear/w",
        )

        self.bias = bias

        if self.bias:
            self.b = tf.Variable(
                tf.zeros(
                    shape=[1, num_outputs],
                ),
                trainable=True,
                name="Linear/b",
            )

    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b

        return tf.math.sin(self.omega_0 * z)
