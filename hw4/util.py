import tensorflow as tf
import numpy as np
from tqdm import trange

# Utility functions


def normalize(features):
    moments = tf.nn.moments(features, axes=[1, 2])
    mean = moments[0]
    stdev = moments[1]
    mean = np.repeat(mean[:, np.newaxis, np.newaxis, :], 32, axis=1)
    mean = np.repeat(mean, 32, axis=2)
    stdev = np.repeat(stdev[:, np.newaxis, np.newaxis, :], 32, axis=1)
    stdev = np.repeat(stdev, 32, axis=2)
    features = (features - mean) / np.sqrt(stdev)
    return features


def get_loss(labels, logits):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
    )


def random_crop(image):
    image = tf.image.random_flip_left_right(image)
    cropped_image = tf.image.random_crop(image, size=(32, 32, 3))
    return cropped_image

def pad(image_data):
    pad_width = [(0, 0), (4, 4), (4, 4), (0, 0)]
    return np.pad(image_data, pad_width, constant_values=0)

def augment(image_data):
    image_data = pad(image_data)
    image_data = tf.map_fn(random_crop, image_data, dtype=tf.float32)
    return image_data


def train_model(
    resnet,
    optimizer,
    train_data,
    train_label,
    val_data,
    val_label,
    iterations,
    get_acc,
    get_loss,
    rng,
):
    acc = 0
    bar = trange(iterations)
    for i in bar:
        with tf.GradientTape() as tape:
            batch_indices = rng.uniform(shape=[512], maxval=40000, dtype=tf.int32)
            train_images_batch = tf.gather(train_data, batch_indices)
            train_labels_batch = tf.gather(train_label, batch_indices)
            train_images_batch =  tf.map_fn(random_crop, train_images_batch, dtype=tf.float32)
            est_labels = resnet(train_images_batch)

            cost = get_loss(labels=train_labels_batch, logits=est_labels)
            grads = tape.gradient(cost, resnet.trainable_variables)
            optimizer.apply_gradients(grads, resnet.trainable_variables)

        if i % 100 == 99:
            batch_indices_val = rng.uniform(shape=[256], maxval=10000, dtype=tf.int32)
            y_score = resnet(tf.gather(val_data, batch_indices_val))
            y_true = tf.gather(val_label, batch_indices_val)

            acc = get_acc(y_true, y_score)
        if i % 10 == (10 - 1):
            bar.set_description(f"Step {i}; Cost => {cost.numpy():0.4f}, acc => {acc}")
            bar.refresh()


def get_testacc(resnet, features_test, labels_test, get_acc):
    individual_accuracies = []
    chunk_size = 1000
    for i in range(0, len(features_test), chunk_size):
        chunk_images = features_test[i : i + chunk_size]
        chunk_labels = labels_test[i : i + chunk_size]

        chunk_predictions = resnet(chunk_images)
        accuracy = get_acc(chunk_labels, chunk_predictions)

        individual_accuracies.append(accuracy)

        mean_accuracy = np.mean(individual_accuracies)
    print(f"\nAccuracy: {mean_accuracy}%")
