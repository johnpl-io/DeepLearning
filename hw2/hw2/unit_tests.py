import pytest
import tensorflow as tf
from MLP import MLP
@pytest.mark.parametrize("inputs, outputs, num_width, num_hidden_layer", [(1, 2, 4, 5), (2, 8,  6, 9), (8, 4, 100, 3)])
def test_in_out(inputs, outputs, num_width, num_hidden_layer):
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    x = rng.normal(shape=[6, inputs])
    
    f = MLP(num_inputs=inputs, num_outputs=outputs, hidden_activation=tf.nn.relu, num_hidden_layers=num_hidden_layer, hidden_layer_width=num_width)
    z = f(x)

    tf.debugging.assert_equal(outputs, z.shape[1])



@pytest.mark.parametrize("inputs, outputs, num_width, num_hidden_layer", [(5, 4, 3, 2), (10, 20,  5, 6), (4, 4, 21, 3)])
def test_in_range(inputs, outputs, num_width, num_hidden_layer):
    f = MLP(num_inputs=inputs, num_outputs=outputs, hidden_activation=tf.nn.relu, num_hidden_layers=num_hidden_layer, hidden_layer_width=num_width, output_activation=tf.nn.sigmoid)
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    x = rng.normal(shape=[10, inputs])
    z = f(x)
    tf.debugging.assert_positive(z)

@pytest.mark.parametrize("inputs, outputs, num_width, num_hidden_layer", [(5, 7, 31, 23), (12, 2,  1, 1), (1, 1, 1, 0)]) #no hidden layers on the last
def test_layers(inputs, outputs, num_width, num_hidden_layer):
    f = MLP(num_inputs=inputs, num_outputs=outputs, hidden_activation=tf.nn.relu, num_hidden_layers=num_hidden_layer, hidden_layer_width=num_width, output_activation=tf.nn.sigmoid)
    #subtracting output and input layer
    test_layers = f.layers[1:-1]
    for layer in test_layers:
        tf.debugging.assert_equal(layer.linear.w.shape, [num_width, num_width])
    tf.debugging.assert_equal(len(f.layers) - 2, num_hidden_layer)
