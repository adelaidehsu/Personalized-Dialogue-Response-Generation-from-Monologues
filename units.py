import tensorflow as tf
import numpy as np
import data_utils
import collections

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

def encode(cell, embedding, encoder_inputs, seq_len=None, dtype=tf.float32):
    with variable_scope.variable_scope("embedding_rnn_seq2seq") as scope:
        scope.set_dtype(dtype) 
        encoder_inputs = [tf.cast(embedding_ops.embedding_lookup(embedding, i), tf.float32) for i in encoder_inputs]
        
        return tf.nn.static_rnn(
            cell,
            encoder_inputs,
            sequence_length=seq_len,
            dtype=dtype)


def decode(cell, init_state, embedding, decoder_inputs, feature_inputs, feature_proj, maxlen, feed_prev=False, loop_function=None, dtype=tf.float32):
    with variable_scope.variable_scope("embedding_rnn_decoder") as scope:
        outputs = []
        hiddens = []
        state = init_state
        
        feature_inputs = [tf.matmul(feature_inputs[0], feature_proj)]

        if not feed_prev:
            emb_inputs = [tf.cast(embedding_ops.embedding_lookup(embedding, i), tf.float32) for i in decoder_inputs]
            for i, emb_inp in enumerate(emb_inputs):
                if i >= maxlen:
                    break
                if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()

                emb_inp = tf.concat([emb_inp, feature_inputs[0]],1)
                output, state = cell(emb_inp, state)
                outputs.append(output)
                hiddens.append(state)
            return outputs, hiddens, state
        else:
            samples = []
            i = 0
            emb_inp = tf.cast(embedding_ops.embedding_lookup(embedding, decoder_inputs[0]), tf.float32)
            prev = None
            tmp = None

            index = 0
            while(True):
                index = index + 1
                if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()
                emb_inp = tf.concat([emb_inp, feature_inputs[0]],1) # concat_feature

                output, state = cell(emb_inp, state)
                outputs.append(output)
                hiddens.append(state)
                prev = output
                with tf.variable_scope('loop', reuse=True):
                    if prev is not None:
                        tmp = loop_function(prev)

                if tmp is not None:
                    if isinstance(tmp, list):
                        emb_inp, prev_symbol = tmp
                        samples.append(prev_symbol)
                    else:
                        emb_inp = tmp
                i += 1
                if i >= maxlen:
                    break
            return outputs, samples, hiddens
