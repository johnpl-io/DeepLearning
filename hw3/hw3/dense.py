import tensorflow as tf



#taken from tensorflow docs to initialize weights in an efficient manner
#it did not seem to help as much as I thought 
def xavier_init(shape):
    # Computes the xavier initialization values for a weight matrix
    in_dim, out_dim = shape
    xavier_lim = tf.sqrt(6.0) / tf.sqrt(tf.cast(in_dim + out_dim, tf.float32))
    weight_vals = tf.random.uniform(
        shape=(in_dim, out_dim), minval=-xavier_lim, maxval=xavier_lim, seed=22
    )
    return weight_vals

#Same as previous assignment and Prof Curro implementation
class DenseLayer(tf.Module):
    def __init__(self, num_inputs, num_outputs, bias=True, activation = tf.identity):
        rng = tf.random.get_global_generator()

        self.w = tf.Variable(
            xavier_init(shape=[num_inputs, num_outputs]),
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
