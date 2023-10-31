import numpy as np
import tensorflow as tf


class PositionalEmbedding(tf.Module):
    def __init__(self, vocab_size, dim_model):
        self.embedding = tf.Variable(
            tf.random.uniform(shape=[vocab_size, dim_model]),
            trainable=True,
            name="embed_matrix",
        )
        self.dim_model = dim_model

    def pos_encoding(self, length):
        pos_enc = np.zeros((length, self.dim_model))
        for pos in range(length):
            for i in range(self.dim_model // 2):
                angle = pos / np.power(10000, 2 * i / self.dim_model)
                pos_enc[pos, 2 * i] = np.sin(angle)
                pos_enc[pos, 2 * i + 1] = np.cos(angle)
        pos_enc = tf.cast(pos_enc, dtype=tf.float32)  # Convert to a TensorFlow tensor
        return pos_enc

    def __call__(self, x):
        length = int(tf.shape(x)[-1])
        x = tf.nn.embedding_lookup(self.embedding, tf.cast(x, tf.int32))
        return x * tf.math.sqrt(
            tf.cast(self.dim_model, tf.float32)
        ) + self.pos_encoding(length)


x = np.array([[0, 1, 2, 3]])
