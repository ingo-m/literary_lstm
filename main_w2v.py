#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM fun main function using word2vec embedding.
"""


# import time
import random
import numpy as np
import tensorflow as tf

# from tensorflow.contrib import rnn
from tensorflow.keras import layers

# from tensorflow.nn.objRnn import DropoutWrapper
# from tf.contrib.rnn.DropoutWrapper


# -----------------------------------------------------------------------------
# *** Define parameters

# Path of input data file:
strPthIn = '/Users/john/1_PhD/GitLab/literary_lstm/log_w2v/word2vec_data.npz'

# Log directory:
strPthLog = '/Users/john/1_PhD/GitLab/literary_lstm/log_lstm'

# Learning rate:
varLrnRte = 0.001

# Number of training iterations:
varNumItr = 1

# Display steps (after x number of iterations):
varDspStp = 1

# Number of input words from which to predict next word:
varNumIn = 8

# Number of neurons in first hidden layer:
varNrn01 = 256

# Number of neurons in second hidden layer:
varNrn02 = 100

# -----------------------------------------------------------------------------
# *** Load data

# Load npz file:
objNpz = np.load(strPthIn)

# Enable object array:
objNpz.allow_pickle = True

# Coded text:
vecC = objNpz['vecC']

# Dictionary, with words as keys:
dicWdCnOdr = objNpz['dicWdCnOdr'][()]

# Reverse dictionary, with ordinal word count as keys:
dictRvrs = objNpz['dictRvrs'][()]

# Embedding matrix:
aryEmb = objNpz['aryEmbFnl']

# Tensorflow constant fo embedding matrix:
aryTfEmb = tf.constant(aryEmb, dtype=tf.float32)

# Transposed version of embedding matrix:
aryEmbT = aryEmb.T


# -----------------------------------------------------------------------------
# *** Preparations

# Vocabulary size (number of words):
varNumWrds = aryEmb.shape[0]

print(('Size of vocabulary (number of unique words): ' + str(varNumWrds)))

# Number of words in text (corpus):
varLenTxt = vecC.shape[0]

print(('Length of text: ' + str(varLenTxt)))

# Ratio number of words / text length:
varNumRto = np.around((float(varNumWrds) / float(varLenTxt)), decimals=3)

print(('Vocabulary / text ratio: ' + str(varNumRto)))

# Size of embedding vector:
varSzeEmb = aryEmb.shape[1]

# Total number of inputs (to input layer):
varNumInTtl = varNumIn * varSzeEmb

# Placeholder for inputs (to input layer):
aryWrdsIn = tf.placeholder(tf.float32, [1, varNumIn, varSzeEmb])

# Placeholder for output:
vecWrdsOut = tf.placeholder(tf.float32, [varSzeEmb, 1])


# -----------------------------------------------------------------------------
# *** RNN LSTM function

def fncRnn(aryWrdsIn, varNrn01, varNrn02):
    """Recurrent neural network with LSTM cell and dropout."""

    objRnn = layers.StackedRNNCells(
                                    [layers.LSTMCell(
                                                     varNrn01,
                                                     activation='tanh',
                                                     recurrent_activation='hard_sigmoid',
                                                     use_bias=True,
                                                     kernel_initializer='glorot_uniform',
                                                     recurrent_initializer='orthogonal',
                                                     bias_initializer='zeros',
                                                     unit_forget_bias=True,
                                                     kernel_regularizer=None,
                                                     recurrent_regularizer=None,
                                                     bias_regularizer=None,
                                                     kernel_constraint=None,
                                                     recurrent_constraint=None,
                                                     bias_constraint=None,
                                                     dropout=0.1,
                                                     recurrent_dropout=0.1,
                                                     implementation=1
                                                     ),
                                     layers.LSTMCell(
                                                     varNrn02,
                                                     activation='tanh',
                                                     recurrent_activation='hard_sigmoid',
                                                     use_bias=True,
                                                     kernel_initializer='glorot_uniform',
                                                     recurrent_initializer='orthogonal',
                                                     bias_initializer='zeros',
                                                     unit_forget_bias=True,
                                                     kernel_regularizer=None,
                                                     recurrent_regularizer=None,
                                                     bias_regularizer=None,
                                                     kernel_constraint=None,
                                                     recurrent_constraint=None,
                                                     bias_constraint=None,
                                                     dropout=0.1,
                                                     recurrent_dropout=0.1,
                                                     implementation=1
                                                     )]
                                    )

    # ?
    # lstOut, objState = rnn.static_rnn(objRnn, [aryWrdsIn], dtype=tf.float32)  # keras.layers.RNN(cell, unroll=True)

    # keras.layers.RNN(cell, unroll=True)

    # lstOut, objState = layers.RNN(objRnn, unroll=True)(aryWrdsIn)
    lstOut = layers.RNN(objRnn,
                        return_sequences=False,
                        return_state=False,
                        stateful=True,
                        unroll=True,
                        time_major=False
                        )(aryWrdsIn)

    # TODO: What is the output type/shape of keras.layers.RNN()?

    print('type(lstOut)')
    print(type(lstOut))

    # objState
    # (
    # LSTMStateTuple(c=<tf.Tensor 'rnn/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/Add_1:0' shape=(1, 256) dtype=float32>,
    # h=<tf.Tensor 'rnn/rnn/multi_rnn_cell/cell_0/dropout_1/mul:0' shape=(1, 256) dtype=float32>),
    # LSTMStateTuple(c=<tf.Tensor 'rnn/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/Add_1:0' shape=(1, 100) dtype=float32>,
    # h=<tf.Tensor 'rnn/rnn/multi_rnn_cell/cell_1/dropout_1/mul:0' shape=(1, 100) dtype=float32>)
    # )

    # Activation function (select second element in lstOut, corresponding to
    # the output layer):
    # aryOut = tf.add(tf.matmul(lstOut[0], aryWghts02), vecBias02)
    aryOut = lstOut[0]

    return aryOut


# -----------------------------------------------------------------------------
# ***

aryOut = fncRnn(aryWrdsIn, varNrn01, varNrn02)

# Cost function:
objCost = tf.sqrt(tf.reduce_sum(tf.squared_difference(aryOut, vecWrdsOut)))

# Optimiser:
objOpt = tf.train.RMSPropOptimizer(
                                   learning_rate=varLrnRte
                                   ).minimize(objCost)

# Predicted word (integer code):
objPrd = tf.to_int32(
    tf.argmin(tf.reduce_sum(tf.squared_difference(aryTfEmb,
    tf.broadcast_to(aryOut, [varNumWrds, varSzeEmb])), axis=1), axis=0)
    )
# objPrd = tf.reduce_sum(tf.squared_difference(aryTfEmb, tf.broadcast_to(aryOut, [varNumWrds, varSzeEmb])), axis=1)

# Variable initialiser:
objInit = tf.global_variables_initializer()

# To keep track of total accuracy & loss:
varAccTtl = 0
varLssTtl = 0


# -----------------------------------------------------------------------------
# *** Run optimisation

# Launch the graph
with tf.Session() as objSess:

    # Initializing the variables:
    objSess.run(objInit)

    # Loop through iterations:
    for idxItr in range(varNumItr):

        # Random number for status feedback:
        varRndm = random.randint(20, (varLenTxt - 20))

        # Loop through text (corpus). Index refers to target word (i.e. word
        # to be predicted).
        for idxWrd in range(varNumIn, varLenTxt):
            print(idxWrd)

            # Get integer codes of context word(s):
            vecCntxt = vecC[(idxWrd - varNumIn):idxWrd]

            # Get embedding vectors for words:
            aryCntxt = np.array(aryEmb[vecCntxt, :]).reshape((1, varNumIn, varSzeEmb))

            # Word to predict (target):
            varTrgt = vecC[idxWrd]

            # Get embedding vector for target word:
            vecTrgt = aryEmb[varTrgt, :].reshape((varSzeEmb, 1))

            # Run optimisation:
            _, varPred, varLoss = objSess.run([objOpt, objPrd, objCost],
                                              feed_dict={aryWrdsIn: aryCntxt,
                                                         vecWrdsOut: vecTrgt}
                                              )

            # Status feedback:
            if (idxItr % varDspStp == 0) and (idxWrd == varRndm):

                try:

                    print('---')

                    print(('Context: '
                           + str([dictRvrs[x] for x in vecC[(idxWrd - 15):idxWrd]])))

                    print(('Target: '
                           + dictRvrs[varTrgt]))

                    # Get prediction for current context word(s):
                    #vecTmp = objSess.run([objPrd],
                    #                     feed_dict={aryWrdsIn: aryCntxt}  #, vecWrdsOut: vecTrgt}
                    #                     )[0].flatten()

                    # Minimum squared deviation between prediciton and embedding
                    # vectors:
                    #vecTmp = np.sum(
                    #                np.square(
                    #                          np.subtract(
                    #                                      aryEmb,
                    #                                      vecTmp[None, :]
                    #                                      )
                    #                          ),
                    #                axis=1
                    #                )

                    # Look up predicted word in dictionary:
                    #strWrdPrd = dictRvrs[int(np.argmin(vecTmp))]
                    strWrdPrd = dictRvrs[varPred]

                    print(('Prediction: ' + strWrdPrd))

                    print(('Loss: ' + str(varLoss)))

                except:

                    pass
