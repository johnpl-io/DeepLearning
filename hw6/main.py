import numpy as np
import tensorflow as tf
from tqdm import trange

from AdamW import AdamW
from decoder import TransformerDecoder


# simple token cheme from https://www.tensorflow.org/text/tutorials/word2vec
def gen_vocab(corpus):
    vocab = {}
    vocab["<start>"] = 0
    vocab["<end>"] = 1
    i = 2
    tokens = corpus.lower().split()
    for token in tokens:
        if token not in vocab:
            vocab[token] = i
            i += 1
    return vocab


corpus = "to be, or not to be, that is the question"


vocab = gen_vocab(corpus)
inverse_vocab = {index: token for token, index in vocab.items()}


sentence = "<start> " + corpus + " <end>"
tokenize = lambda sentence: [vocab[word] for word in sentence.lower().split()]
detokenize = lambda tokenized_text: " ".join([inverse_vocab[x] for x in tokenized_text])

tokenized_text = tokenize(sentence)
input_text = np.array(tokenized_text[:-1])[np.newaxis, :]
target_text = np.array(tokenized_text[1:])[np.newaxis, :]


def get_loss(labels, logits):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
    )


is_train = np.array([True])
model = TransformerDecoder(
    dim_model=512, heads=8, blocks=1, vocab_size=len(vocab), is_train=is_train
)


def train(model, optimizer, input_text, output_text, iterations, get_loss):
    bar = trange(iterations)
    for i in bar:
        with tf.GradientTape() as tape:
            est_labels = model(input_text)
            loss = get_loss(labels=target_text, logits=est_labels)

            grads = tape.gradient(loss, model.trainable_variables)
            grads = [tf.convert_to_tensor(grad) for grad in grads]

            optimizer.apply_gradients(grads, model.trainable_variables)

        if i % 10 == (10 - 1):
            bar.set_description(f"Step {i}; Loss=> {loss.numpy():0.4f}")
            bar.refresh()


optimizer = AdamW()
train(model, optimizer, input_text, target_text, 500, get_loss)


is_train[0] = False
x = np.array(tokenize("<start>"))[np.newaxis, :]

output = model(x)


while 1:
    predicted_tokens = tf.math.argmax(output, axis=-1)

    output = model(x)

    predicted_tokens = tf.math.argmax(output, axis=-1)
    first_predicted_token = predicted_tokens[:, -1]

    x = tf.concat([x, first_predicted_token[tf.newaxis, :]], axis=1)
    if int(first_predicted_token) == 1:
        break

print(detokenize(np.array(x[0])))
