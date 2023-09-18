import numpy as np
from numpy.random import Generator, PCG64
import tensorflow as tf
import matplotlib.pyplot as plt
import yaml
import argparse
from classifier import Classifier
from pathlib import Path
from tqdm import trange
import idx2numpy
from Adam import Adam
import os as os


parser = argparse.ArgumentParser(
    prog="Spiral MLP",
    description="Fits a linear model to some data, given a config",
)

parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
args = parser.parse_args()

config = yaml.safe_load(args.config.read_text())
epochs = config["learning"]["num_iters"]
step_size = config["learning"]["step_size"]
decay_rate = config["learning"]["decay_rate"]
batch_size = config["learning"]["batch_size"]
refresh_rate = config["display"]["refresh_rate"]
lambda_val = config["learning"]["lambda_val"]

def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)


def get_l2_regularization(lambda_val, weights):
    l2 = tf.reduce_sum([tf.reduce_sum(w**2) for w in weights if w.name != 'Linear/b:0'])
    return lambda_val * l2

def get_accuracy(est_output, correct_label):
    est_output = tf.math.argmax(est_output, axis=1)
    num_correct = tf.equal(est_output, correct_label)
    accuracy = tf.reduce_mean(tf.cast(num_correct, tf.float32)) * 100.0
    return accuracy

def get_loss(labels, logits):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels, logits))

def get_l2_regularization(lambda_val, weights):
    l2 = tf.reduce_sum([tf.reduce_sum(w**2) for w in weights if w.name != 'Linear/b:0'])
    return lambda_val * l2



                 
labels_file = 'train-labels-idx1-ubyte'
images_file = 'train-images-idx3-ubyte'

labels_data = idx2numpy.convert_from_file(labels_file)
images_data = idx2numpy.convert_from_file(images_file)
images_data = images_data / 255.0
images_data = images_data.astype(np.float32)
labels_data = labels_data.astype(np.int64)
# Display number of instances:
n_images, im_size = images_data.shape[:2]
n_labels = labels_data.shape[0]
print(f"There is {n_images} images.")
print(f"There is {n_labels} labels.")
print(f"The images size is {im_size}.")
import matplotlib.pyplot as plt
images_data = images_data.reshape(n_images, im_size, im_size, 1)

#spliting test and training data 
train_images_data = images_data[:50000]
train_labels_data = labels_data[:50000]

val_images_data = images_data[50000:]
val_labels_data = labels_data[50000:]
conv_cnn = Classifier(
    input_shape = (batch_size, im_size, im_size, 1),
    input_depth=1,
    layer_depths=[32, 64, 64],
    layer_kernel_sizes=[(3, 3), (3, 3), (3,3)],
    num_classes=10,
)


bar = trange(5000)

rng = tf.random.get_global_generator()
rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

optomizer = Adam()
acc = 0
for i in bar:
    batch_indices = rng.uniform(shape=[batch_size], maxval = 50000, dtype=tf.int32)
    with tf.GradientTape() as tape:
        train_images_batch = tf.gather(train_images_data, batch_indices)
        train_labels_batch = tf.gather(train_labels_data, batch_indices)
        est_labels = conv_cnn(train_images_batch)
        loss = get_loss(labels=train_labels_batch, logits = est_labels)
        l2 = get_l2_regularization(0.001, conv_cnn.trainable_variables)
        cost = loss + l2

    
        
    grads = tape.gradient(cost, conv_cnn.trainable_variables)
    #step_size *= decay_rate
    optomizer.apply_gradients(grads, conv_cnn.trainable_variables)
    #grad_update(step_size, conv_cnn.trainable_variables, grads)
    
    if(i% 100 == 99):
        batch_indices_val = rng.uniform(shape=[1000], maxval = 10000, dtype=tf.int32)
        acc = get_accuracy(conv_cnn(tf.gather(val_images_data, batch_indices_val)), tf.gather(val_labels_data, batch_indices_val))
    if i % refresh_rate == (refresh_rate - 1):
        bar.set_description(
            f"Step {i}; Cost => {cost.numpy():0.4f}, acc => {acc}"
        )
        bar.refresh()
        


print(get_accuracy(conv_cnn(val_images_data[0:1000]), val_labels_data[0:1000]))
#print(get_accuracy(conv_cnn(val_images_data[5000:]), val_labels_data[5000:]))
# Display an element:
#element = 10
#plt.imshow(images_data[element], cmap=plt.cm.binary)
#plt.title(labels_data[element])
#plt.show()