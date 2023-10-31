import numpy as np
import tensorflow as tf

from Dense import DenseLayer
from positional_embedding import PositionalEmbedding
from transformer_block import TransformerBlock


class TransformerDecoder(tf.Module):
    def __init__(self, dim_model, heads, blocks, vocab_size, is_train):
        self.embed_layer = PositionalEmbedding(vocab_size, dim_model)
        self.blocks = [
            TransformerBlock(
                dim_model, dim_model // heads, heads, mask=True, is_train=is_train
            )
            for _ in range(blocks)
        ]

        self.fc = DenseLayer(dim_model, vocab_size)

    def __call__(self, x):
        x = self.embed_layer(x)
        for block in self.blocks:
            x = block(x)

        return self.fc(x)
