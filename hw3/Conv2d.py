import tensorflow as tf
import numpy as np
def xavier_init(shape, in_dim, out_dim):
    # Computes the xavier initialization values for a weight matrix
    xavier_lim = tf.sqrt(6.0) / tf.sqrt(tf.cast(in_dim + out_dim, tf.float32))
    weight_vals = tf.random.uniform(
        shape=shape, minval=-xavier_lim, maxval=xavier_lim, seed=22
    )
    return weight_vals
class Conv2d(tf.Module):
    def __init__(self, input_shape, filters, kernel_size, strides = (1, 1), activation = tf.identity):
        self.w = tf.Variable(
            xavier_init(shape=[kernel_size[0], kernel_size[1], input_shape,  filters], in_dim = 1, out_dim = filters),
            trainable=True,
            name="Conv/w",
        )

        self.activation = activation
        self.strides = strides
        self.output_shape = self.w.shape[-1]


        
    def __call__(self, x):
        f = tf.nn.conv2d(x, filters = self.w, strides = self.strides, padding = 'VALID')
        return self.activation(f)


conv2dobj = Conv2d(1, 5, kernel_size=(2,2))
x = np.zeros((4,28,28,1))

z = conv2dobj(x)

    #filters how many channels
    #kernel size size of kernel