import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import top_k_accuracy_score

from AdamW import AdamW
from Classifier import Classifier
from util import *


def unpickle(file):
    with open(file, "rb") as fo:
        myDict = pickle.load(fo, encoding="latin1")
    return myDict


def preprocess(file_path, N):
    dict = unpickle(file_path)
    features = dict["data"].reshape(N, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(dict["fine_labels"])
    features = features / 255.0
    features = normalize(features)
    features = features.astype(np.float32)
    labels = labels.astype(np.int64)
    features = features / 255.0
    features = normalize(features)
    features = features.astype(np.float32)
    labels = labels.astype(np.int64)
    return features, labels


def get_accuracy(y_true, y_score):
    resnet.isTrain = False
    acc = top_k_accuracy_score(y_true, y_score, k=5, labels=list(range(0, 100))) * 100
    resnet.isTrain = True
    return acc


features, labels = preprocess("cifar-100-python/train", 50000)
features_test, labels_test = preprocess("cifar-100-python/test", 10000)


train_labels_data = labels[:40000]
train_features_data = features[:40000]
train_features_data = pad(train_features_data)
val_features_data = features[40000:]
val_labels_data = labels[40000:]
rng = tf.random.get_global_generator()
rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

optimizer = AdamW(learning_rate=0.01, weight_decay=1e-3)
resnet = Classifier(
    3,
    [64, 128],
    [(3, 3), (3, 3)],
    out_layer=512,
    num_classes=100,
    res_depths=[[128, 128], [128, 128], [256, 512], [512, 512]],
)

train_model(
    resnet,
    optimizer,
    train_features_data,
    train_labels_data,
    val_features_data,
    val_labels_data,
    1500,
    get_accuracy,
    get_loss,
    rng,
)


get_testacc(resnet, features_test, labels_test, get_accuracy)
