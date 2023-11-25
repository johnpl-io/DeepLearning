import tensorflow as tf


def he_init(shape):
    # Computes the He initialization values for a weight matrix
    in_dim, out_dim = shape
    stddev = tf.sqrt(2.0 / tf.cast(in_dim, tf.float32))
    weight_vals = tf.random.normal(shape=shape, mean=0, stddev=stddev, seed=22)
    return weight_vals


class DenseLayer(tf.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        bias=True,
        activation=tf.identity,
        initializer=he_init,
    ):
        self.w = tf.Variable(
            initializer(shape=[num_inputs, num_outputs]),
            trainable=True,
            name="Linear/w",
        )
        self.activation = activation
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

        return self.activation(z)
