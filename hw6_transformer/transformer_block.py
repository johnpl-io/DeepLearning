import tensorflow as tf

from Dense import DenseLayer
from Dropout import DropLayer
from feed_forward_network import FeedForwardNetwork
from layer_norm import LayerNorm
from multi_head_attention import MultiHeadAttention


class TransformerBlock(tf.Module):
    def __init__(
        self, dim_model, dim_v, heads, dropout=0.1, is_train=[True], mask=False
    ):
        self.mha = MultiHeadAttention(dim_model, dim_v, heads, mask)
        self.layer_norm_0 = LayerNorm(dim_model)

        self.fnn = FeedForwardNetwork(dim_model)
        self.layer_norm_1 = LayerNorm(dim_model)

        self.is_train = is_train
        self.dropout = DropLayer(0.1)

    def __call__(self, x):
        if self.is_train[0]:
            x = self.layer_norm_0(x + self.dropout(self.mha(x)))
            x = self.layer_norm_1(x + self.dropout(self.fnn(x)))
        else:
            x = self.layer_norm_0(x + self.mha(x))
            x = self.layer_norm_1(x + self.fnn(x))

        return x
