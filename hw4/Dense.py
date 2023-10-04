import tensorflow as tf


# Same as previous assignment and Prof Curro implementation
class DenseLayer(tf.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        bias=True,
        activation=tf.identity,
        initializer=tf.zeros,
    ):
        rng = tf.random.get_global_generator()

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
