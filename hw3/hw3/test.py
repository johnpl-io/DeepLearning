import tensorflow as tf
import numpy as np
# Example logits (predicted scores) and labels (ground truth)
logits = tf.constant([[10, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 10, 0, 0, 0, 0, 0, 0, 0]] , dtype=tf.float32)
labels = tf.constant([0,2], dtype=tf.int64)

# Calculate the softmax cross-entropy loss
#loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

def get_accuracy(est_output, correct_label=0):
    est_output = tf.math.argmax(est_output, axis=1)
    num_correct = tf.equal(est_output, correct_label)
    accuracy = tf.reduce_mean(tf.cast(num_correct, tf.float32)) * 100.0
    return accuracy

# Print the loss
x = get_accuracy(logits, labels)
breakpoint()