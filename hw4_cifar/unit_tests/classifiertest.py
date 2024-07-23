import numpy as np
import pytest
import tensorflow as tf

from Classifier import Classifier
from util import *


def test_size():
    x = tf.random.uniform(shape=[64, 32, 32, 3])
    classifier = Classifier(
        3,
        [64, 128],
        [(3, 3), (3, 3)],
        out_layer=512,
        num_classes=100,
        res_depths=[[128, 128], [256, 256], [256, 512], [512, 512]],
    )
    z = classifier(x)
    """making sure batch size is preserved and output classes are correct"""
    tf.assert_equal(z.shape, (64, 100))
