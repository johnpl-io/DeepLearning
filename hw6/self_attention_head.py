import tensorflow as tf
from Dense import DenseLayer

''' Implementation of self attention head from 
https://arxiv.org/abs/1706.03762'''

class SelfAttentionHead(tf.Module):
    def __init__(self, n_embd, dim_model):
        self.key = DenseLayer(n_embd, dim_model, bias=False)
        self.query = DenseLayer(n_embd, dim_model, bias=False)
        self.value = DenseLayer(n_embd, dim_model, bias=False)
    
    def __call__(self, x):
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)
        
        test = tf.einsum('b i d, b j d -> b i j', Q, K)


        breakpoint()




head = SelfAttentionHead(32, 16)
x = tf.random.normal(shape = [4, 10, 32])

y = head(x)
