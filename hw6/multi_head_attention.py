import tensorflow as tf
from Dense import DenseLayer
import einops
''' Implementation of self attention for multiple heads as in 
https://arxiv.org/abs/1706.03762'''

class MultiHeadAttention(tf.Module):
    def __init__(self, dim_model, dim_v, heads):
        self.key = DenseLayer(dim_model, dim_v * heads, bias=False)
        self.query = DenseLayer(dim_model, dim_v * heads, bias=False)
        self.value = DenseLayer(dim_model, dim_v * heads, bias=False)
        self.weights = DenseLayer(heads * dim_v, dim_model, bias = False)
        self.heads = heads
        self.mask = False
        self.dim_k = dim_v #self attention
    def __call__(self, x):
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)
        
        query = einops.rearrange(Q, 'b e (d h) ->  b h e d', h = self.heads)
        value = einops.rearrange(K, 'b e (d h) ->  b h e d', h = self.heads)
        key = einops.rearrange(V, 'b e (d h) ->  b h e d', h = self.heads)
        mask = True
        q_dot_v = tf.einsum('b h e d,  b h j d -> b h e j', query, key) * self.dim_k ** -0.5
       
        if mask:
            inf_mask = tf.fill(q_dot_v.shape, float('-inf'))
            q_dot_v = q_dot_v + tf.linalg.band_part(inf_mask, 0, -1)

        out_softmax = tf.nn.softmax(q_dot_v)
       
        out_softmax_dot_k = tf.einsum('b h e j,  b h j d -> b h e d',   out_softmax , value)
        out_concat = einops.rearrange(out_softmax_dot_k, 'b h e d -> b e (h d)')

        return self.weights(out_concat)
        

      


head = MultiHeadAttention(32, 16, 8)
x = tf.random.normal(shape = [4, 10, 32])

y = head(x)
breakpoint()