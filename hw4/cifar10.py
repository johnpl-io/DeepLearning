import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from tqdm import trange
import pickle
from AdamW import AdamW
from Classifier import Classifier
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
        features_list.append(features)
        labels_list.append(labels)

    return np.array(features_list), np.array(labels_list)


def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(
        cifar10_dataset_folder_path + "/data_batch_" + str(batch_id), mode="rb"
    ) as file:
        batch = pickle.load(file, encoding="latin1")

    features = (
        batch["data"].reshape((len(batch["data"]), 3, 32, 32)).transpose(0, 2, 3, 1)
    )
    labels = batch["labels"]

    return features, labels


def load_cfar10_test():
    with open("cifar-10-batches-py/test_batch", mode="rb") as file:
        batch = pickle.load(file, encoding="latin1")

    features = (
        batch["data"].reshape((len(batch["data"]), 3, 32, 32)).transpose(0, 2, 3, 1)
    )
    labels = batch["labels"]

    return features, labels


def get_accuracy(correct_label, est_output):
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
features = normalize(features)

features = features.astype(np.float32)
labels = labels.astype(np.int64)

train_features_data = features[:40000]
train_labels_data = labels[:40000]


#train_features_data = pad(train_features_data)

test_features, test_labels = load_cfar10_test()
test_features = test_features / 255.0
test_features = normalize(test_features)

images_test_data = test_features.astype(np.float32)
labels_test_data = np.array(test_labels).astype(np.int64)

val_features_data = features[40000:]
val_labels_data = labels[40000:]
bar = trange(2000)

rng = tf.random.get_global_generator()
rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)


resnet = Classifier(
    3,
    [64, 128],
    [(3, 3), (3, 3)],
    out_layer=512,
    num_classes=10,
    res_depths=[[128, 128], [256, 512], [512, 512]],
    
)


optimizer = AdamW(learning_rate=0.01, weight_decay=1e-5)
train_model(resnet, optimizer, train_features_data, train_labels_data, val_features_data, val_labels_data, 1000, get_accuracy, get_loss, rng)
get_testacc(resnet, images_test_data, labels_test_data, get_accuracy)
