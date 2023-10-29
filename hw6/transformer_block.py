
import tensorflow as tf
from multi_head_attention import MultiHeadAttention
from LayerNorm import LayerNorm
from feed_forward_network import FeedForwardNetwork
class TransformerBlock(tf.Module):
    def __init__(self, dim_model, dim_v, heads, dropout):
        self.mha = MultiHeadAttention(dim_model, dim_v, heads)
        self.layer_norm_0 = LayerNorm(dim_model)

        self.fnn = FeedForwardNetwork(dim_model)
        self.layer_norm_1 = LayerNorm(dim_model)


    def __call__(self, x):
        x = self.layer_norm_0(x + self.mha(x))
        x = self.layer_norm_1(x + self.fnn(x))

        return x




