import numpy as np
from numpy.random import Generator, PCG64
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
import yaml
import argparse
from MLP import MLP
from pathlib import Path
from tqdm import trange

parser = argparse.ArgumentParser(
    prog="Spiral MLP",
    description="Fits a linear model to some data, given a config",
)

parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
args = parser.parse_args()

config = yaml.safe_load(args.config.read_text())
num_of_samples = config["spiral_data"]["num_samples"]
noise_val = config["spiral_data"]["noise_stdev"]
epochs = config["learning"]["num_iters"]
step_size = config["learning"]["step_size"]
decay_rate = config["learning"]["decay_rate"]
batch_size = config["learning"]["batch_size"]
refresh_rate = config["display"]["refresh_rate"]
lambda_val = config["learning"]["lambda_val"]


def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)


def binary_cross_entropy_loss(y, y_hat):
    loss = -y * tf.math.log(y_hat + 1e-15) - (1 - y) * tf.math.log(1 - y_hat + 1e-15)
    return tf.math.reduce_mean(loss)


def get_l2_regularization(lambda_val, weights):
    l2 = tf.reduce_sum([tf.reduce_sum(w**2) for w in weights if w.name != 'Linear/b:0'])
    return lambda_val * l2


def get_batches(x1, y1, x2, y2, batch_indices):
    x1_batch = tf.gather(x1, batch_indices)
    y1_batch = tf.gather(y1, batch_indices)
    x2_batch = tf.gather(x2, batch_indices)
    y2_batch = tf.gather(y2, batch_indices)
    z = np.random.choice([0, 1], size=(batch_size, 1))
    xy_values = np.hstack(
        ((z * x1_batch) + (x2_batch) * (1 - z), (z * y1_batch) + (y2_batch) * (1 - z))
    )
    return xy_values, z.astype(np.float32)


rng = Generator(PCG64())

rad_vals = rng.uniform(np.pi / 4, 11, size=[num_of_samples, 1])
x1 = rad_vals * np.cos(rad_vals) + rng.normal(size=[num_of_samples, 1], scale=noise_val)
y1 = -rad_vals * np.sin(rad_vals) + rng.normal(size=[num_of_samples, 1], scale=noise_val)

x2 = -rad_vals * np.cos(rad_vals) + rng.normal(size=[num_of_samples, 1], scale=noise_val)
y2 = rad_vals * np.sin(rad_vals) + rng.normal(size=[num_of_samples, 1], scale=noise_val)

mlp_model = MLP(
    num_inputs=2,
    num_outputs=1,
    num_hidden_layers=3,
    hidden_layer_width=100,
    hidden_activation=tf.nn.relu,
    output_activation=tf.nn.sigmoid,
)
bar = trange(epochs)
rng2 = tf.random.get_global_generator()
rng2.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)


for i in bar:
    batch_indices = rng2.uniform(shape=[batch_size], maxval=200, dtype=tf.int32)

    with tf.GradientTape() as tape:
        xy_batch, real_labels = get_batches(x1, y1, x2, y2, batch_indices)
        est_labels = mlp_model(xy_batch)
        loss = binary_cross_entropy_loss(real_labels, est_labels)
        l2 = get_l2_regularization(lambda_val, mlp_model.trainable_variables)
        cost = loss + l2

    grads = tape.gradient(cost, mlp_model.trainable_variables)

    grad_update(step_size, mlp_model.trainable_variables, grads)

    step_size *= decay_rate

    if i % refresh_rate == (refresh_rate - 1):
        bar.set_description(
            f"Step {i}; Cost => {cost.numpy():0.4f}, step_size => {step_size:0.4f}"
        )
        bar.refresh()

xx0, xx1 = np.meshgrid(np.linspace(-12, 12), np.linspace(-12, 12))

grid = np.vstack([xx0.ravel(), xx1.ravel()]).T

y_pred = np.reshape(mlp_model(grid), xx0.shape)
display = DecisionBoundaryDisplay(xx0=xx0, xx1=xx1, response=y_pred)
display.plot()


display.ax_.scatter(x1, y1, edgecolor="black")
display.ax_.scatter(x2, y2, edgecolor="black")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("MLP Decision boundary with L2 regularization")

plt.savefig('artifacts/plotL2.png')
plt.show()
