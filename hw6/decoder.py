import tensorflow as tf
from transformer_block import TransformerBlock

from Dense import DenseLayer
from positional_embedding import PositionalEmbedding
import numpy as np
class TransformerDecoder(tf.Module):
    def __init__(self, dim_model, heads, blocks, vocab_size, max_len):
        self.embed_layer = PositionalEmbedding(vocab_size, dim_model, max_len)
        self.blocks = [TransformerBlock(dim_model, dim_model//heads, heads, mask = True) for _ in range(blocks)]
        self.max_len = max_len

        self.fc = DenseLayer(dim_model, vocab_size)
    def generate(self, prompt, max_token_gen):
        
        for _ in range(max_token_gen):

            output = self(prompt[:, -self.max_len:])

            predicted_tokens = tf.math.argmax(output, axis = - 1)
            first_predicted_token = predicted_tokens[:,0]


            prompt = tf.concat([prompt, first_predicted_token], axis=1)


        return prompt






    def __call__(self, x):
        x = self.embed_layer(x)
        for block in self.blocks:
            x = block(x)

        return self.fc(x)


model = TransformerDecoder(64, 8,1, 5, 4)