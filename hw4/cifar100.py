import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from tqdm import trange
from sklearn.metrics import top_k_accuracy_score
from AdamW import AdamW
from classifier import Classifier
import pickle
from util import *
def unpickle(file):
    with open(file, 'rb') as fo:
        myDict = pickle.load(fo, encoding='latin1')
    return myDict

def preprocess(file_path, N): 
    dict = unpickle(file_path)
    features = dict['data'].reshape(N, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(dict['fine_labels'])
    features = features / 255.0 
    features = normalize(features)
    features = features.astype(np.float32)
    labels = labels.astype(np.int64)
    features = features / 255.0 
    features = normalize(features)
    features = features.astype(np.float32)
    labels = labels.astype(np.int64)
    return features, labels

def get_accuracy(y_score, y_true):
    return top_k_accuracy_score(y_true, y_score, k=5, labels=list(range(0,100)))

features, labels = preprocess('cifar-100-python/train', 50000)
features_test, labels_test = preprocess('cifar-100-python/test', 10000)




refresh_rate = 10

train_labels_data = labels[:40000]
val_features_data = features[40000:]
val_features_data = tf.map_fn(random_crop, val_features_data, dtype=tf.float32)
val_features_data = features[40000:]
val_labels_data = labels[40000:]
bar = trange(1000)
rng = tf.random.get_global_generator()
rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

optomizer = AdamW(learning_rate=0.001,weight_decay=0.01)
resnet = Classifier((64, 32, 32, 3), [64, 128], [(3,3), (3, 3)],num_classes=100, res_depths=[[128, 128] , [256,512], [512, 512]])
acc = 0

for i in bar:
    with tf.GradientTape() as tape:
        batch_indices = rng.uniform(shape=[512], maxval=40000, dtype=tf.int32)
        train_images_batch = tf.gather(features, batch_indices)

        train_labels_batch = tf.gather(labels, batch_indices)

        est_labels = resnet(train_images_batch)

        cost = get_loss(labels=train_labels_batch, logits=est_labels)
        grads = tape.gradient(cost, resnet.trainable_variables)
        optomizer.apply_gradients(grads, resnet.trainable_variables)

    if i % 100 == 99:
        batch_indices_val = rng.uniform(shape=[256], maxval=10000, dtype=tf.int32)
        y_score = resnet(tf.gather(val_features_data, batch_indices_val))
        y_true = tf.gather(val_labels_data, batch_indices_val)
      
        acc = get_accuracy(y_true, y_score)
    if i % refresh_rate == (refresh_rate - 1):
        bar.set_description(f"Step {i}; Cost => {cost.numpy():0.4f}, acc => {acc * 100:.2f}")
        bar.refresh()

        

individual_accuracies = []


chunk_size = 1000
for i in range(0, len(features_test), chunk_size):
    chunk_images = features_test[i : i + chunk_size]
    chunk_labels = labels_test[i : i + chunk_size]

    chunk_predictions = resnet(chunk_images)
    accuracy = get_accuracy(chunk_labels,chunk_predictions)

    individual_accuracies.append(accuracy)

mean_accuracy = np.mean(individual_accuracies)
print("accuracy", mean_accuracy*100)
