import collections
import math
import tensorflow as tf
import numpy as np
import pdb

# credit: https://github.com/Kyubyong/transformer

'''
Self-Attention Neural Network for Sequence Labelling Tasks
'''

class Labeller(object):
    def __init__(self, config):
        # hyper-parameters / configurations
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.num_stacks = config['num_stacks'] # N=6 in the paper
        self.num_units = config['num_units']
        self.max_sentence_length = config['max_sentence_length']
        self.vocab_size = config['vocab_size']

    def build_netowrk(self):
        # placeholders
        self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
        self.sent_lengths = tf.placeholder(tf.int32, [None], name="sent_lengths")
        self.labels = tf.placeholder(tf.int32, [None, None], name="labels")

        # embeddings
        word_embedded = embedding(self.word_ids, self.vocab_size, self.num_units)

        # positional encoding
        positional_enc = positional_encoding([self.batch_size, self.max_sentence_length], self.num_units)

        # inputs into the neural network
        inputs = word_embedded + positional_enc

        if self.num_stacks > 1:
            for n in range(self.num_stacks - 1):
                scope_name = "self_attn" + str(n)
                inputs = self_attention_layer(inputs, 512, 8, 0.1, True, scope_name)

        scope_name = "self_attn" + str(self.num_stacks-1)
        self.outputs = self_attention_layer(inputs, 512, 8, 0.1, True, scope_name)


        print("build neural network model successful")



# -------------- Modules high-level --------------- #
def self_attention_layer(inputs, num_units=512, num_heads=8, dropout=0.0, is_training=True,
                         scope="self_attention_layer"):

    with tf.variable_scope(scope):
        attn = multihead_attention(query=inputs, key=inputs, value=inputs,
                            num_units=num_units, num_heads=num_heads, dropout=dropout,
                            is_training=is_training, causality=False,
                            scope="multihead_attention", reuse=None)

        ffn = feedforward(attn, num_units=2048, d_model=num_units, scope="feedforward", reuse=None)

    return ffn

# --------------- Modules low-level --------------- #
def add_residual(activations, inputs):
    return activations + inputs

def layer_norm(inputs, epsilon=1e-8, scope="layer_norm", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta

    return outputs

def embedding(inputs, vocab_size, num_units,
              # zero_pad=True,
              scale=True,
              scope="embedding", reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable("word_embeddings",
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        # if zero_pad:
        #     lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
        #                               lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)

    return outputs

def positional_encoding(inputs_shape, num_units, zero_pad=True, scale=True,
                        scope="positional_encoding", reuse=None):

    N, T = inputs_shape

    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units**0.5

        return outputs


def multihead_attention(query, key, value,
                        num_units=None, num_heads=8, dropout=0.0,
                        is_training=True, causality=False,
                        scope="multihead_attention", reuse=None):

    """Multihead Attention sub-layer
    Args:
        query: 3d Tensor
        key: 3d Tensor
        value: 3d Tensor ---- self-attention value = key
        num_units: d_model let's call it C
        num_heads: h
    """

    with tf.variable_scope(scope, reuse=reuse):
        if num_units == None:
            num_units = query.get_shape().as_list()[-1]

        # linear projections --- output dimension [N, timestep, C]
        Q = tf.layers.dense(query, num_units, activation=tf.nn.relu)
        K = tf.layers.dense(key, num_units, activation=tf.nn.relu)
        V = tf.layers.dense(value, num_units, activation=tf.nn.relu)

        # split and combine multihead --- output dimension [h*N, timestep,  C/h]
        _Q = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)
        _K = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)
        _V = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)

        # scoring - matrix multiplication & scaling
        scores = tf.matmul(_Q, tf.transpose(_K, perm=[0, 2, 1]))
        scores = scores / (math.sqrt(num_units/num_heads))

        # scoring - key masking
        key_masks = tf.sign(tf.reduce_sum(tf.abs(key), axis=-1))
        key_masks = tf.tile(key_masks, [num_heads, 1])
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(query)[1], 1])
        paddings = tf.ones_like(scores)*(-2**32+1)
        scores = tf.where(tf.equal(key_masks, 0), paddings, scores)

        # scoring - Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(scores[0, :, :])
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(scores)[0], 1, 1])

            paddings = tf.ones_like(masks)*(-2**32+1)
            scores = tf.where(tf.equal(masks, 0), paddings, scores)

        # scoring - softmax --- output dimension [h*N, timestep_Q, timestep_K]
        scores = tf.nn.softmax(scores)

        # Query Masking
        query_masks = tf.sign(tf.reduce_sum(tf.abs(query), axis=-1))
        query_masks = tf.tile(query_masks, [num_heads, 1])
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(key)[1]])
        scores = scores * query_masks # broadcasting. (N, T_q, C)

        # dropout
        scores = tf.layers.dropout(scores, rate=dropout, training=tf.convert_to_tensor(is_training))

        # outputs!!!
        outputs = tf.matmul(scores, _V) # ( h*N, timestep_q, C/h)

        # get the original shape back
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1) # (N, timestep_Q, C)

        # residual connection
        outputs = add_residual(outputs, query)

        # layer normalisation
        outputs = layer_norm(outputs)

    return outputs

def feedforward(inputs, num_units=2048, d_model=512, scope="feedforward", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        s = inputs.get_shape().as_list()
        x0 = tf.reshape(inputs, [s[0]*s[1], s[2]])

        # layer 1
        w1 = tf.Variable(tf.random_normal([d_model, num_units], stddev=0.01), name="w1")
        b1 = tf.Variable(tf.constant(0.0, shape=(1, num_units)), name="b1")
        a1 = tf.nn.relu(tf.matmul(x0,w1) + b1)

        # layer 2
        w2 = tf.Variable(tf.random_normal([num_units, d_model], stddev=0.01), name="w2")
        b2 = tf.Variable(tf.constant(0.0, shape=(1, d_model)), name="b2")
        z2 = tf.matmul(a1,w2) + b2

        # residual connection
        outputs = z2 + x0

        # get the original shape back
        outputs = tf.reshape(outputs, [s[0], s[1], s[2]])

        # layer normalisation
        outputs = layer_norm(outputs)

        return outputs
