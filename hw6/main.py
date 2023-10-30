#!/usr/bin/env python
# coding: utf-8

# In[1]:


from decoder import TransformerDecoder
import numpy as np
import tensorflow as tf
from AdamW import AdamW


# In[2]:


#simple token cheme from https://www.tensorflow.org/text/tutorials/word2vec
def gen_vocab(corpus):
    vocab = {}
    vocab["<start>"] = 0
    vocab["<end>"] = 1
    i = 2
    tokens = corpus.lower().split()
    for token in tokens:
        if token  not in vocab:
            vocab[token] = i
        i+=1
    return vocab



# In[3]:


corpus = "hello how are you"

vocab = gen_vocab(corpus)

vocab
inverse_vocab = {index: token for token, index in vocab.items()}
inverse_vocab


# In[4]:


sentence = "<start> hello how are you <end>"
tokenize = lambda sentence : [vocab[word] for word in sentence.lower().split()]
detokenize = lambda tokenized_text : ' '.join([inverse_vocab[x] for x in tokenized_text])

tokenized_text = tokenize(sentence)
input_text = np.array(tokenized_text[:-1])[np.newaxis,:]
target_text = np.array(tokenized_text[1:])[np.newaxis,:]
target_text


# In[5]:


def get_loss(labels, logits):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
    )


# In[6]:


model = TransformerDecoder(dim_model=64, heads=8, blocks=1, vocab_size=len(vocab), max_len=5)

x = np.array(tokenized_text[0:2])[np.newaxis,:]


output = model(x)

predicted_tokens = tf.math.argmax(output, axis = - 1)

get_loss(target_text  ,model(input_text))


# In[7]:


from tqdm import trange
def train(model, optimizer, input_text, output_text, iterations, get_loss):
    bar = trange(iterations)
    for i in bar:
        with tf.GradientTape() as tape:
            est_labels = model(input_text)
            loss = get_loss(labels=target_text, logits=est_labels)

            grads = tape.gradient(loss, model.trainable_variables)

            grads = [tf.convert_to_tensor(grad) for grad in grads]
    
            optimizer.apply_gradients(grads, model.trainable_variables)



        if i%10 == (10 - 1):
            bar.set_description(f"Step {i}; Loss=> {loss.numpy():0.4f}")
            bar.refresh()
    








# In[8]:


optimizer = AdamW()
train(model, optimizer, input_text, target_text, 100, get_loss)


# In[60]:


x = np.array(tokenized_text)[np.newaxis,:]


#output = model(x)

#predicted_tokens = tf.math.argmax(output, axis = - 1)
#predicted_tokens
int(tf.shape(x)[-1])



# In[10]:


def generate(model, prompt, max_token_gen):
        
    for _ in range(max_token_gen):

        output = model(prompt[:, -5:])

        predicted_tokens = tf.math.argmax(output, axis = - 1)
        first_predicted_token = predicted_tokens[:,0]


        prompt = tf.concat([prompt, first_predicted_token[tf.newaxis, :]], axis=1)


    return prompt


# In[11]:


output = model(x)

predicted_tokens = tf.math.argmax(output, axis = - 1)
first_predicted_token = predicted_tokens[:,0]


first_predicted_token

predicted_tokens
int(tf.shape(x)[-1])

# generate()
