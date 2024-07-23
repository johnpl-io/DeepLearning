import tensorflow as tf

from dense import DenseLayer
# implementation of Siren network from https://arxiv.org/pdf/2006.09661.pdf
from sinelayer import *
from util import *


class Siren(tf.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, hidden_layer_width):
        # first layer has slightly different initialization scheme
        self.layers = [
            SineLayer(
                num_inputs, hidden_layer_width, siren_initializer=siren_init_first
            )
        ]

        for i in range(num_hidden_layers):
            self.layers.append(
                SineLayer(
                    num_inputs=hidden_layer_width,
                    num_outputs=hidden_layer_width,
                    siren_initializer=siren_init_layer,
                )
            )

        
        """last layer uses a clipped relu as points can only go from [0, 1]. 
            Doing so helped improved training time"""
        
        clipped_relu = lambda x: tf.clip_by_value(tf.nn.relu(x), 0, 1)
        self.layers.append(
            DenseLayer(
                num_inputs=hidden_layer_width,
                num_outputs=num_outputs,
                activation=clipped_relu,
            )
        )

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def train_model(
        self, train_on, get_loss, iterations, shape, optimizer=Adam, frame_folder=None
    ):
        
        """"arranging the points from -1 to 1 proved to be the most effective way to train the network.
        this is similar to what the Siren paper does"""
        
        bar = trange(iterations)
        x_vals, y_vals = tf.meshgrid(
            tf.linspace(-1, 1, shape[1]), tf.linspace(-1, 1, shape[0])
        )
        
        cord_vals = tf.stack([tf.reshape(y_vals, -1), tf.reshape(x_vals, -1)], axis=1)
        train_on = tf.convert_to_tensor(train_on, dtype=tf.float32)
        train_on_true = tf.reshape(train_on, shape=[shape[0] * shape[1], -1])

        frames = []

        for i in bar:
            with tf.GradientTape() as tape:
                loss_output = get_loss(self, cord_vals, train_on_true, shape)
                grads = tape.gradient(loss_output[0], self.trainable_variables)
                optimizer.apply_gradients(grads=grads, vars=self.trainable_variables)

            if i % 10 == (10 - 1):
                bar.set_description(f"Step {i}; loss => {loss_output[0].numpy():0.4f}")
                bar.refresh()

                if frame_folder:
                    plt.imshow(tf.reshape(loss_output[1], shape=[273, 365, -1]))
                    plt.axis("off")
                    plt.savefig(f"{frame_folder}/frame_{i}.png", bbox_inches="tight")
                    plt.close()
                    frames.append(Image.open(f"{frame_folder}/frame_{i}.png"))

        if frame_folder:
            frames[0].save(
                f"{frame_folder}.gif",
                save_all=True,
                append_images=frames,
                duration=200,
                loop=0,
            )
        return loss_output[1]