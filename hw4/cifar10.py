import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from tqdm import trange

from AdamW import AdamW
from classifier import Classifier
from util import *
parser = argparse.ArgumentParser(
    prog="MNIST classifier",
    description="Classifieres MNIST",
)


def load_preprocess(cifar10_dataset_folder_path, batch_ids=None):
    features_list = []
    labels_list = []
    for id in batch_ids:
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, id)
        # preprocess and augment
        features_list.append(features)
        labels_list.append(labels)

    return np.array(features_list), np.array(labels_list)


def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(
        cifar10_dataset_folder_path + "/data_batch_" + str(batch_id), mode="rb"
    ) as file:
        # note the encoding type is 'latin1'
        import pickle

        batch = pickle.load(file, encoding="latin1")

    features = (
        batch["data"].reshape((len(batch["data"]), 3, 32, 32)).transpose(0, 2, 3, 1)
    )
    labels = batch["labels"]

    return features, labels


def get_accuracy(est_output, correct_label):
    resnet.isTrain = False
    est_output = tf.math.argmax(est_output, axis=1)
    num_correct = tf.equal(est_output, correct_label)
    accuracy = tf.reduce_mean(tf.cast(num_correct, tf.float32)) * 100.0
    resnet.isTrain = True
    return accuracy


parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
args = parser.parse_args()
config = yaml.safe_load(args.config.read_text())
refresh_rate = config["display"]["refresh_rate"]
features, labels = load_preprocess("cifar-10-batches-py", [1, 2, 3, 4, 5])

features = features.reshape(50000, 32, 32, 3)


labels = labels.reshape(50000)

features = features / 255.0

moments = tf.nn.moments(features, axes=[1, 2])
mean = moments[0]
stdev = moments[1]
mean = np.repeat(mean[:, np.newaxis, np.newaxis, :], 32, axis=1)
mean = np.repeat(mean, 32, axis=2)
stdev = np.repeat(stdev[:, np.newaxis, np.newaxis, :], 32, axis=1)
stdev = np.repeat(stdev, 32, axis=2)

features = (features - mean) / np.sqrt(stdev)


features = features.astype(np.float32)

labels = labels.astype(np.int64)

train_features_data = features[:40000]

padding_size = 4


train_labels_data = labels[:40000]
pad_width = [(0, 0), (4, 4), (4, 4), (0, 0)]
#train_features_data = np.pad(train_features_data, pad_width, constant_values=0)

#train_features_data = tf.map_fn(random_crop, train_features_data, dtype=tf.float32)

val_features_data = features[40000:]
val_labels_data = labels[40000:]
bar = trange(4000)

rng = tf.random.get_global_generator()
rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)


resnet = Classifier((64, 32, 32, 3), [64, 128], [(3,3), (3, 3)],num_classes=10, res_depths=[[128, 128] , [256,512], [512, 512]])

acc = 0
optimizer = AdamW(learning_rate=0.001, weight_decay=1e-5)
for i in bar:
    with tf.GradientTape() as tape:
        batch_indices = rng.uniform(shape=[512], maxval=40000, dtype=tf.int32)
        train_images_batch = tf.gather(features, batch_indices)

        train_labels_batch = tf.gather(labels, batch_indices)

        est_labels = resnet(train_images_batch)

        cost = get_loss(labels=train_labels_batch, logits=est_labels)
        grads = tape.gradient(cost, resnet.trainable_variables)
        optimizer.apply_gradients(grads, resnet.trainable_variables)

    if i % 100 == 99:
        batch_indices_val = rng.uniform(shape=[256], maxval=10000, dtype=tf.int32)
        acc = get_accuracy(
            resnet(tf.gather(val_features_data, batch_indices_val)),
            tf.gather(val_labels_data, batch_indices_val),
        )


    if i % refresh_rate == (refresh_rate - 1):
        bar.set_description(f"Step {i}; Cost => {cost.numpy():0.4f}, acc => {acc}")
        bar.refresh()
