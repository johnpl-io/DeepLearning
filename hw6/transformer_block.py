
import tensorflow as tf
from multi_head_attention import MultiHeadAttention
from LayerNorm import LayerNorm
from feed_forward_network import FeedForwardNetwork
from Dropout import DropLayer
from Dense import DenseLayer
class TransformerBlock(tf.Module):
    def __init__(self, dim_model, dim_v, heads, dropout=0.1, isTrain=[True], mask =False):
        self.mha = MultiHeadAttention(dim_model, dim_v, heads, mask)
        self.layer_norm_0 = LayerNorm(dim_model)

        self.fnn = FeedForwardNetwork(dim_model)
        self.layer_norm_1 = LayerNorm(dim_model)
        self.isTrain = isTrain
        self.dropout = DropLayer(0.1)

    def __call__(self, x):
        if self.isTrain[0]:
            x = self.layer_norm_0(x + self.dropout(self.mha(x)))
            x = self.layer_norm_1(x + self.dropout(self.fnn(x)))
        else:
            x = self.layer_norm_0(x + self.mha(x))
            x = self.layer_norm_1(x + self.fnn(x))
            
        return x


