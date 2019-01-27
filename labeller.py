import collections
import math
import tensorflow as tf
import numpy as np
import pdb

# credit: https://github.com/Kyubyong/transformer
# http://jalammar.github.io/illustrated-transformer/

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
        self.label_ids = tf.placeholder(tf.int32, [None, None], name="label_ids")
        self.sent_lengths = tf.placeholder(tf.int32, [None], name="sent_lengths")

        s = tf.shape(self.word_ids)

        # embeddings
        word_embedded = embedding(self.word_ids, self.vocab_size, self.num_units)

        # positional encoding
        positional_enc = positional_encoding([s[0], self.max_sentence_length], self.num_units)

        # inputs into the neural network
        inputs = word_embedded + positional_enc

        if self.num_stacks > 1:
            for n in range(self.num_stacks - 1):
                scope_name = "self_attn" + str(n)
                inputs = self_attention_layer(inputs, self.sent_lengths, 512, 8, 0.0, True, scope_name)

        scope_name = "self_attn" + str(self.num_stacks-1)
        self_attn_outputs = self_attention_layer(inputs, self.sent_lengths, 512, 8, 0.0, True, scope_name) # shape = [batch_size, max_sentence_length, num_units]


        # ---------- linear & softmax --------- #
        with tf.variable_scope("linear_projection", reuse=None):
            w1 = tf.get_variable("w1", shape=[512, 200], initializer=tf.glorot_normal_initializer())
            b1 = tf.get_variable("b1", shape=[1, 200], initializer=tf.zeros_initializer())
            w2 = tf.get_variable("w2", shape=[200, 2], initializer=tf.glorot_normal_initializer())
            b2 = tf.get_variable("b2", shape=[1, 2], initializer=tf.zeros_initializer())

        x0 = tf.reshape(self_attn_outputs, shape=[s[0]*self.max_sentence_length, 512])
        a1 = tf.nn.relu(tf.matmul(x0, w1) + b1)
        logits = tf.matmul(a1, w2) + b2

        self.logits = tf.reshape(logits, shape=[s[0], self.max_sentence_length, 2])


        self.probabilities = tf.nn.softmax(self.logits)
        self.predictions = tf.argmax(self.probabilities, 2)

        # ----------- cost function ----------- #
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label_ids)
        target_weights = tf.sequence_mask(lengths=self.sent_lengths,
                                            maxlen=self.max_sentence_length,
                                            dtype=tf.float32)

        self.crossent_masked = crossent * target_weights
        self.loss = tf.reduce_sum(self.crossent_masked) / self.batch_size


        # ----- Gradient and Optimisation ----- #
        trainable_params = tf.trainable_variables() # return a list of Variable objects
        gradients = tf.gradients(self.loss, trainable_params)
        # max_gradient_norm = 1.0 # set to value like 1.0, 5.0
        # clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

        # optimisation
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        # self.train_op = self.optimizer.apply_gradients(zip(clipped_gradients, trainable_params))
        self.train_op = self.optimizer.apply_gradients(zip(gradients, trainable_params))


        print("build neural network model successful")


# -------------- Modules high-level --------------- #
def self_attention_layer(inputs, input_lengths, num_units=512, num_heads=8,
                         dropout=0.0, is_training=True,
                         scope="self_attention_layer"):

    with tf.variable_scope(scope):
        attn = multihead_attention(inputs=inputs, input_lengths=input_lengths,
                            num_units=num_units, num_heads=num_heads,
                            dropout=dropout, is_training=is_training,
                            # causality=False,
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
        beta= tf.Variable(tf.zeros(params_shape), name="beta")
        gamma = tf.Variable(tf.ones(params_shape), name="gamma")
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


def multihead_attention(inputs, input_lengths, num_units=None, num_heads=8,
                        dropout=0.0, is_training=True,
                        # causality=False,
                        scope="multihead_attention", reuse=None):

    """Multihead Attention sub-layer
    Args:
        inputs: shape = [batch_size (N), max_time_steps (T), vector_size]
        input_lengths: shape = [batch_size] # e.g. len of each sentence in the batch
        num_units: d_model let's call it C
        num_heads: h
    """

    with tf.variable_scope(scope, reuse=reuse):
        if num_units == None:
            num_units = inputs.get_shape().as_list()[-1]

        # linear projections --- output dimension [N, timestep, C]
        # Q = tf.layers.dense(inputs, num_units, activation=tf.nn.relu, name="query")
        # K = tf.layers.dense(inputs, num_units, activation=tf.nn.relu, name="key")
        # V = tf.layers.dense(inputs, num_units, activation=tf.nn.relu, name="value")
        Q = tf.layers.dense(inputs, num_units, activation=None, use_bias=False, name="query")
        K = tf.layers.dense(inputs, num_units, activation=None, use_bias=False, name="key")
        V = tf.layers.dense(inputs, num_units, activation=None, use_bias=False, name="value")

        # split and combine multihead --- output dimension [h*N, timestep,  C/h]
        _Q = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)
        _K = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)
        _V = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)

        # scoring - matrix multiplication & scaling
        scores = tf.matmul(_Q, tf.transpose(_K, perm=[0, 2, 1]))
        scores = scores / (math.sqrt(num_units/num_heads)) # shape = (h*N, T, T)

        # scoring - key masking
        max_length = inputs.get_shape().as_list()[1]
        key_masks = tf.sequence_mask(lengths=input_lengths, maxlen=max_length, dtype=tf.bool) # (N, T)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(inputs)[1], 1]) # (h*N, T, T)

        paddings = tf.ones_like(scores)*(-2**32+1)
        scores = tf.where(key_masks, scores, paddings)

        # scoring - Causality = Future blinding
        # if causality:
        #     diag_vals = tf.ones_like(scores[0, :, :])
        #     tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        #     masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(scores)[0], 1, 1])
        #
        #     paddings = tf.ones_like(masks)*(-2**32+1)
        #     scores = tf.where(tf.equal(masks, 0), paddings, scores)

        # scoring - softmax --- output dimension [h*N, timestep_Q, timestep_K]
        scores = tf.nn.softmax(scores)

        # Query Masking
        # query_masks = tf.sequence_mask(lengths=input_lengths, maxlen=max_length, dtype=tf.float32) # (N, T)
        # query_masks = tf.tile(query_masks, [num_heads, 1])
        # query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(inputs)[1]])
        #
        # scores = scores * query_masks # broadcasting. (N, T, C)

        # dropout
        scores = tf.layers.dropout(scores, rate=dropout, training=tf.convert_to_tensor(is_training))

        # outputs!!!
        outputs = tf.matmul(scores, _V) # ( h*N, T, C/h)

        # get the original shape back
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1) # (N, T, C)

        # residual connection
        outputs = add_residual(outputs, inputs)

        # layer normalisation
        outputs = layer_norm(outputs)

    return outputs

def feedforward(inputs, num_units=2048, d_model=512, scope="feedforward", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # s = inputs.get_shape().as_list()
        # x0 = tf.reshape(inputs, [s[0]*s[1], s[2]])

        # layer 1
        # w1 = tf.Variable(tf.random_normal([d_model, num_units], stddev=0.01), name="w1")
        # b1 = tf.Variable(tf.constant(0.0, shape=(1, num_units)), name="b1")
        # a1 = tf.nn.relu(tf.matmul(x0,w1) + b1)

        a1 = tf.layers.dense(inputs, num_units, activation=tf.nn.relu, use_bias=True, name="block1")

        # layer 2
        # w2 = tf.Variable(tf.random_normal([num_units, d_model], stddev=0.01), name="w2")
        # b2 = tf.Variable(tf.constant(0.0, shape=(1, d_model)), name="b2")
        # z2 = tf.matmul(a1,w2) + b2
        z2 = tf.layers.dense(a1, d_model, activation=None, use_bias=True, name="block2")

        # residual connection
        # outputs = add_residual(z2, x0)
        outputs = add_residual(z2, inputs)

        # get the original shape back
        # outputs = tf.reshape(outputs, [s[0], s[1], s[2]])

        # layer normalisation
        outputs = layer_norm(outputs)

        return outputs
