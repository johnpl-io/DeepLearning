import tensorflow as tf
from tqdm import trange

from dense import DenseLayer
from Dropout import DropLayer


class Sequential(tf.Module):
    def __init__(self, layers):
        self.isTrain = True
        self.layers = layers

    def train_model(
        self,
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
                batch_indices = rng.uniform(shape=[1024], maxval=108000, dtype=tf.int32)
                train_images_batch = tf.gather(train_data, batch_indices)
                train_labels_batch = tf.gather(train_label, batch_indices)
                est_labels = self(train_images_batch)

                cost = get_loss(labels=train_labels_batch, logits=est_labels)
                grads = tape.gradient(cost, self.trainable_variables)
                optimizer.apply_gradients(grads, self.trainable_variables)

            if i % 100 == 99:
                batch_indices_val = rng.uniform(
                    shape=[1024], maxval=12000, dtype=tf.int32
                )
                y_score = self(tf.gather(val_data, batch_indices_val))
                y_true = tf.gather(val_label, batch_indices_val)

                acc = get_acc(y_true, y_score)
            if i % 10 == (10 - 1):
                bar.set_description(
                    f"Step {i}; Cost => {cost.numpy():0.4f}, acc => {acc}"
                )
                bar.refresh()

    def __call__(self, x):
        for layer in self.layers:
            if self.isTrain == True:
                x = layer(x)
            elif type(layer) != (DropLayer):
                x = layer(x)
        return x

