import pytest
import tensorflow as tf
from classifier import Classifier
from Conv2d import Conv2d
import numpy as np
rng = tf.random.get_global_generator()
rng.reset_from_seed(2384230948)
@pytest.mark.parametrize("input_depth, kernel_size, output_depth", [(32, (3, 3), 64), (1, (2, 2), 32)])
def test_conv2d(input_depth, kernel_size, output_depth):
    conv2dobj = Conv2d(input_shape = input_depth, filters = output_depth, kernel_size=(3,3))
    x = np.zeros((4,28,28,input_depth))
    
    z = conv2dobj(x)
