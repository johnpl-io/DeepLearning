
import argparse

from pathlib import Path

import math
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import yaml

import tensorflow as tf
import numpy as np

from linear import Linear

from basisexpansion import BasisExpansion
from tqdm import trange

def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)

parser = argparse.ArgumentParser(
    prog="Linear",
    description="Fits a linear model to some data, given a config",
)

parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
args = parser.parse_args()

config = yaml.safe_load(args.config.read_text())

rng = tf.random.get_global_generator()
rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)


noisy_x = rng.uniform(minval=0, maxval=1, shape=[50, 1])
real_x = np.linspace(0, 1, 50)
real_y = tf.math.sin(2 * math.pi * real_x)

real_y = real_y.numpy()
noisy_y = tf.math.sin(2 * math.pi * noisy_x) + rng.normal(
    shape=[50, 1], mean=0, stddev=0.1
)
x_values = np.linspace(0, 5, 100)
# y_values = line(x_values)
# Plot the results
num_samples = config["data"]["num_samples"]
epochs = config["learning"]["num_iters"]
step_size = config["learning"]["step_size"]
decay_rate = config["learning"]["decay_rate"]
batch_size = config["learning"]["batch_size"]
refresh_rate = config["display"]["refresh_rate"]
M = config["learning"]["no_basis_func"]
loss_arr = []
bar = trange(epochs)
phi = BasisExpansion(M = M)
linear = Linear(M = M)
for i in bar:
    batch_indices = rng.uniform(shape=[batch_size], maxval=num_samples, dtype=tf.int32)
    
    with tf.GradientTape() as tape:
        x_batch = tf.gather(noisy_x, batch_indices)
        y_batch = tf.gather(noisy_y, batch_indices)
        phis = phi(x_batch)
        y_hat = linear(phis)
        loss = tf.math.reduce_mean(1 / 2 * (y_batch - y_hat) ** 2)
        loss_arr.append(loss.numpy())
    
    grads = tape.gradient(
        loss, phi.trainable_variables + linear.trainable_variables
    )

    grad_update(
        step_size, phi.trainable_variables + linear.trainable_variables, grads
    )

    step_size *= decay_rate

    if i % refresh_rate == (refresh_rate - 1):
        bar.set_description(
            f"Step {i}; Loss => {loss.numpy():0.4f}, step_size => {step_size:0.4f}"
        )
        bar.refresh()

y_val_guass = []
for x in real_x:
    y_val_guass.append(linear(phi(x)).numpy()[0])
    #there is probably a cleaner way to do this 

fig, ax = plt.subplots()
ax.set_title("SGD estimation of Sine Wave")
ax.grid(True)
ax.plot(real_x, y_val_guass, label="Linear Combinations of Guassians")
ax.plot(real_x, real_y, label="Clean Sine Wave")
ax.scatter(noisy_x, noisy_y, label="Noisey Sine Wave", s=10)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True)
fig.savefig("regression.png")

fig2, ax2 = plt.subplots()
ax2.set_title("Guassian Basis Functions")
for i in range(7):
    basis_y = tf.exp(-((real_x - phi.mu[0, i]) ** 2) / (phi.sigma[0, i]) ** 2)
    ax2.plot(real_x, basis_y, label="Basis "  + str(i + 1))
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.legend()
fig3, ax3  = plt.subplots()
ax3.set_title("MSE vs Epochs")
ax3.plot(list(range(2000)), loss_arr)
ax3.set_xlabel("Epochs")
ax3.set_ylabel("MSE")



fig.savefig("estimation.pdf")
fig2.savefig("basis.pdf")
fig3.savefig("mse.pdf")