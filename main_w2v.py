#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM fun main function using word2vec embedding.
"""


# import time
# import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

# from tensorflow.nn.objRnn import DropoutWrapper
# from tf.contrib.rnn.DropoutWrapper


# -----------------------------------------------------------------------------
# *** Define parameters

# Path of input data file:
strPthIn = '/home/john/PhD/GitLab/literary_lstm/tf_log/word2vec_data.npz'

# Log directory:
strPthLog = '/home/john/PhD/GitLab/literary_lstm/log_lstm'

# Learning rate:
varLrnRte = 0.001

# Number of training iterations:
varNumItr = 1000000

# Display steps (after x number of iterations):
varDspStp = 1000

# Number of input words from which to predict next word:
varNumIn = 1

# Number of neurons in first hidden layer:
varNrn01 = 256

# Number of neurons in second hidden layer:
varNrn02 = 100


# -----------------------------------------------------------------------------
# *** Load data

# Load npz file:
objNpz = np.load(strPthIn)

# Coded text:
lstC = objNpz['lstC']

# Dictionary, with words as keys:
dicWdCnOdr = objNpz['dicWdCnOdr'][()]

# Reverse dictionary, with ordinal word count as keys:
dictRvrs = objNpz['dictRvrs'][()]

# Embedding matrix:
aryEmbFnl = objNpz['aryEmbFnl']

# Transposed version of embedding matrix:
aryEmbFnlT = aryEmbFnl.T


# -----------------------------------------------------------------------------
# *** Preparations

# Vocabulary size (number of words):
varNumWrds = aryEmbFnl.shape[0]

print(('Size of vocabulary (number of unique words): ' + str(varNumWrds)))

# Number of words in text (corpus):
varLenTxt = len(lstC)

print(('Length of text: ' + str(varLenTxt)))

# Ratio number of words / text length:
varNumRto = np.around((float(varNumWrds) / float(varLenTxt)), decimals=3)

print(('Vocabulary / text ratio: ' + str(varNumRto)))

# Size of embedding vector:
varSzeEmb = aryEmbFnl.shape[1]

# Total number of inputs (to input layer):
varNumInTtl = varNumIn * varSzeEmb

# Placeholder for inputs (to input layer):
vecWrdsIn = tf.placeholder("float", [1, varNumInTtl])

# Placeholder for output:
vecWrdsOut = tf.placeholder("float", [varSzeEmb, 1])

# Weights of first & second hidden layers:
aryWghts01 = tf.Variable(tf.random_normal([varNumInTtl, varNrn01]))
aryWghts02 = tf.Variable(tf.random_normal([varNrn01, varNrn02]))

# Biases for first & second hidden layers:
vecBias01 = tf.Variable(tf.random_normal([varNrn01]))
vecBias02 = tf.Variable(tf.random_normal([varNrn02]))

# -----------------------------------------------------------------------------
# *** RNN LSTM function

def fncRnn(vecWrdsIn, aryWghts01, aryWghts02, vecBias01, vecBias02,
           varNrn01, varNrn02):
    """Recurrent neural network with LSTM cell and dropout."""

    objRnn = rnn.MultiRNNCell(
                              [rnn.DropoutWrapper(
                                                  rnn.BasicLSTMCell(varNrn01),
                                                  input_keep_prob=0.9,
                                                  output_keep_prob=0.9,
                                                  state_keep_prob=0.9,
                                                  ),
                               rnn.DropoutWrapper(
                                                  rnn.BasicLSTMCell(varNrn02),
                                                  input_keep_prob=0.9,
                                                  output_keep_prob=0.9,
                                                  state_keep_prob=0.9,
                                                  )]
                               )

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    # x = tf.split(x, n_input, 1)

    # ?
    lstOut, objState = rnn.static_rnn(objRnn, [vecWrdsIn], dtype=tf.float32)

    #print('type(lstOut)')
    #print(type(lstOut))
    #print('len(lstOut)')
    #print(len(lstOut))
    #print('lstOut')
    #print(lstOut)

    #print('type(objState)')
    #print(type(objState))
    #print('len(objState)')
    #print(len(objState))
    #print('objState')
    #print(objState)

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

aryOut = fncRnn(vecWrdsIn, aryWghts01, aryWghts02, vecBias01, vecBias02,
                varNrn01, varNrn02)

# Cost function:
objCost = tf.reduce_mean(tf.squared_difference(aryOut, vecWrdsOut))

# Optimiser:
objOpt = tf.train.RMSPropOptimizer(
                                   learning_rate=varLrnRte
                                    ).minimize(objCost)

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

        # Loop through text (corpus). Index refers to target word (i.e. word
        # to be predicted).
        for idxWrd in range(varNumIn, varLenTxt):

            # Get integer codes of context word(s):
            lstCntxt = lstC[(idxWrd - varNumIn):idxWrd]

            # Get embedding vectors for words:
            aryCntxt = np.array(aryEmbFnl[lstCntxt, :])

            # Word to predict (target):
            varTrgt = lstC[idxWrd]

            # Get embedding vector for target word:
            vecTrgt = aryEmbFnl[varTrgt, :].reshape((varSzeEmb, 1))

            objSess.run(objOpt,
                        feed_dict={vecWrdsIn: aryCntxt,
                                   vecWrdsOut: vecTrgt}
                        )

# -----------------------------------------------------------------------------
# ***

assert False

varLssTtl += loss
varAccTtl += acc
if (idxItr+1) % varDspStp == 0:
    print("Iter= " + str(idxItr+1) + ", Average Loss= " + \
          "{:.6f}".format(varLssTtl/varDspStp) + ", Average Accuracy= " + \
          "{:.2f}%".format(100*varAccTtl/varDspStp))
    varAccTtl = 0
    varLssTtl = 0
    symbols_in = [dictRvrs[lstC[i]] for i in range(offset, offset + varNumIn)]
    symbols_out = dictRvrs[lstC[offset + varNumIn]]

    #symbols_out_pred = dictRvrs[int(tf.argmax(onehot_pred, 1).eval())]
    # Sum of squares
    vecTmp = np.sum(np.square(np.subtract(aryEmbFnlT, onehot_pred)), axis=0)
    symbols_out_pred = dictRvrs[int(np.argmin(vecTmp))]

    print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
idxItr += 1
offset += (varNumIn+1)

print("Optimization Finished!")
print("Run on command line.")
# print("\ttensorboard --logdir=%s" % (logs_path))
# print("Point your web browser to: http://localhost:6006/")
while True:
    prompt = "%s words: " % varNumIn
    sentence = input(prompt)
    sentence = sentence.strip()
    words = sentence.split(' ')
    if len(words) != varNumIn:
        continue
    try:
        symbols_in_keys = [dicWdCnOdr[str(words[i])] for i in range(len(words))]
        for i in range(32):
            keys = np.reshape(np.array(symbols_in_keys), [-1, varNumIn, 1])
            onehot_pred = objSess.run(pred, feed_dict={vecWrdsIn: keys})

            onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())

            sentence = "%s %s" % (sentence,dictRvrs[onehot_pred_index])
            symbols_in_keys = symbols_in_keys[1:]
            symbols_in_keys.append(onehot_pred_index)
        print(sentence)
    except:
        print("Word not in dictionary")
