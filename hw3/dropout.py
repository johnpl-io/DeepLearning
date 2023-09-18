import tensorflow as tf

class DropLayer(tf.Module):
    def __init__(self, rate):
        self.rate = rate


    def __call__(self, x):
        return tf.nn.dropout(x, rate = self.rate, seed = 5840938542930)
    
test = DropLayer(0.5)
x = tf.ones([3,5])     
z = test(x)
