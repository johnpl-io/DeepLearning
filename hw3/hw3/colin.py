"""
ECE472 Assignment 1
Author: Colin Hwang

"""

import numpy as np
import tensorflow as tf
import argparse
from sklearn.model_selection import KFold
from tqdm import trange

def read_idx(filename):
    with open(filename, 'rb') as f:
        # Read the magic number
        magic_num = f.read(4)
        # Extract the number of dimensions (4th byte)
        dims = magic_num[3]
        # Read in sizes of each dim
        shape = tuple(int.from_bytes(f.read(4)) for i in range(dims))
        # Return reshaped data
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def load_mnist_data(validation_fraction=0.1):
    # Normalize images by dividing by 255
    train_images_all = read_idx('train-images-idx3-ubyte') / 255.0 
    train_labels_all = read_idx('train-labels-idx1-ubyte')
    test_images = read_idx('t10k-images-idx3-ubyte') / 255.0
    test_labels = read_idx('t10k-labels-idx1-ubyte')

    # Add a channel dimension for tf.nn.conv2d (1 channel for grayscale)
    train_images_all = np.expand_dims(train_images_all, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)
    
    # Create a validation set from the training set
    num_samples = len(train_images_all)
    num_val_samples = int(validation_fraction * num_samples)
    num_train_samples = num_samples - num_val_samples
    
    train_images = train_images_all[:num_train_samples]
    train_labels = train_labels_all[:num_train_samples]
    val_images = train_images_all[num_train_samples:num_samples]
    val_labels = train_labels_all[num_train_samples:num_samples]

    return train_images, train_labels, val_images, val_labels, test_images, test_labels

# Linear Module
class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs):
        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))
        self.w = tf.Variable(
            rng.normal(shape=[num_inputs, num_outputs], stddev=stddev),
            trainable=True,
            name="Linear/w"
        )
        self.b = tf.Variable(
            tf.zeros(shape=[1, num_outputs]), 
            trainable=True, 
            name="Linear/b"
        )

    def __call__(self, x):
        y_hat = x @ self.w + self.b
        return y_hat

# Convolution module that wraps tf.nn.conv2d
class Conv2d(tf.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        rng = tf.random.get_global_generator()
        stddev = tf.math.sqrt(2 / (input_channels + output_channels))
        self.w = tf.Variable(
            rng.normal(shape=[kernel_size[0], kernel_size[1], input_channels, output_channels], stddev=stddev),
            trainable=True,
            name="Conv/w"
        )
        self.b = tf.Variable(
            tf.zeros(shape=[output_channels]),
            trainable=True,
            name="Conv/b"
        )
    def __call__(self, x):
        x = tf.cast(x, tf.float32)
        # Use a stride of 1 for height and width
        return tf.nn.conv2d(x, self.w, strides=[1, 1], padding='SAME') + self.b

# Classifier module
class Classifier(tf.Module):
    def __init__(self, input_depth, layer_depths, layer_kernel_sizes, num_classes, dropout_rate=0.5):      
        self.layers = []
        for depth, kernel_size in zip(layer_depths, layer_kernel_sizes):
            self.layers.append(Conv2d(input_depth, depth, kernel_size))
            self.layers.append(tf.nn.relu)
            self.layers.append(tf.nn.dropout)
            input_depth = depth
        flattened = 28 * 28 * input_depth
        self.linear = Linear(flattened, num_classes)
        self.dropout_rate = dropout_rate
    
    def __call__(self, x, training=True):
        for layer in self.layers:
            # only apply dropout during training
            if layer == tf.nn.dropout:
                x = layer(x, rate=self.dropout_rate) if training else x
            else:
                x = layer(x)
        x = tf.reshape(x, [x.shape[0], -1])
        return self.linear(x)

# L2 penalty function
def l2_penalty(variables, lambda_val):
    l2_loss = 0
    for v in variables:
        l2_loss += tf.reduce_sum(v**2)
    return lambda_val * l2_loss

# Same update function as example code
def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)
    
if __name__ == "__main__":
    # Command-line argument for whether to evaluate on test set
    parser = argparse.ArgumentParser(description='MNIST Classifier')
    parser.add_argument('--test_set', type=bool, default=False, help='Whether or not to evaluate on the test set.')
    args = parser.parse_args()

    # Initialize rng
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    # Load data
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_mnist_data()

    # Define parameters for training
    batch_size = 128
    num_iters = 5
    decay_rate = 0.9999
    refresh_rate = 10
    lambda_val = 0.0001
    step_size = 0.1

    # Initalize classifier with 2 layers with 64 nodes and 3x3 kernels
    model = Classifier(input_depth=1, layer_depths=[64, 64], layer_kernel_sizes=[(3, 3), (3, 3)], num_classes=10, dropout_rate = 0.5)

    # Training Loop
    bar = trange(num_iters)
    for i in bar: 
        # Randomly select batches of samples
        batch_indices = rng.uniform(shape=[batch_size], maxval=len(train_images), dtype=tf.int32)
        
        with tf.GradientTape() as tape:
            x_batch = tf.gather(train_images, batch_indices)
            y_batch = tf.gather(train_labels, batch_indices)
            logits = model(x_batch)
            l2 = l2_penalty(model.trainable_variables, lambda_val)
            ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y_batch, tf.int32), logits=logits))
            loss = ce_loss + l2

        grads = tape.gradient(loss, model.trainable_variables)
        grad_update(step_size, model.trainable_variables, grads)

        # Decrease stepsize as training progresses
        step_size *= decay_rate

        # tqdm progress bar
        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, step_size => {step_size:0.4f}"
            )
            bar.refresh()

    # Evaluation on validation set
    eval_logits = model(val_images)
    predicted_labels = tf.argmax(eval_logits, axis=1)
    correct_predictions = tf.equal(predicted_labels, val_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    print(f"Validation Accuracy: {accuracy.numpy() * 100:.2f}%")

    # Final evaluation on test set after cross-validation
    if args.test_set:
        test_logits = model(test_images)
        predicted_labels = tf.argmax(test_logits, axis=1)
        correct_predictions = tf.equal(predicted_labels, test_labels)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        print(f"Final Test Accuracy: {accuracy.numpy() * 100:.2f}%")
