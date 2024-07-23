import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from AdamW import AdamW
from Dense import DenseLayer
from Dropout import DropLayer
from sequential import Sequential

dataset = load_dataset("ag_news")


# pre-trained model loading
model = SentenceTransformer("all-mpnet-base-v2")

data_text = model.encode(dataset["train"]["text"])
data_label = np.array(dataset["train"]["label"])


val_text = data_text[108000:]
train_text = data_text[:108000]

val_label = data_label[108000:]
train_label = data_label[:108000]


def get_loss(labels, logits):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
    )


def get_accuracy(correct_label, est_output):
    model_mlp.isTrain = False
    est_output = tf.math.argmax(est_output, axis=1)
    num_correct = tf.equal(est_output, correct_label)
    accuracy = tf.reduce_mean(tf.cast(num_correct, tf.float32)) * 100.0
    model_mlp.isTrain = True
    return accuracy


model_mlp = Sequential(
    [
        DenseLayer(768, 256, activation=tf.nn.relu),
        DropLayer(0.45),
        DenseLayer(256, 512, activation=tf.nn.relu),
        DropLayer(0.45),
        DenseLayer(512, 4, initializer=tf.zeros),
    ]
)


optimizer_custom = AdamW(learning_rate=0.01, weight_decay=0.02)
rng = tf.random.get_global_generator()
rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
model_mlp.train_model(
    optimizer_custom,
    train_text,
    train_label,
    val_text,
    val_label,
    1200,
    get_accuracy,
    get_loss,
    rng,
)


model_keras = tf.keras.Sequential(
    layers=[
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.45, seed=3489024),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.45, seed=3489024),
        tf.keras.layers.Dense(4, activation="softmax"),
    ]
)


loss = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.AdamW(learning_rate=0.01, weight_decay=0.02)
model_keras.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
model_keras.fit(
    data_text, data_label, validation_split=0.10, epochs=10, batch_size=1024
)

label_test = np.array(dataset["test"]["label"])
data_test = model.encode(dataset["test"]["text"])


model_keras.evaluate(data_test, label_test)
print("non-keras accuracy: ", get_accuracy(y_test, model_mlp(data_test)))
