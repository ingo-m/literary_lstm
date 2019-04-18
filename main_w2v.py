#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM fun main function using word2vec embedding.
"""


# import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


#from tensorflow.nn.rnn_cell import DropoutWrapper
#from tf.contrib.rnn.DropoutWrapper


# -----------------------------------------------------------------------------
# *** Define parameters

# Path of input data file:
strPthIn = '/home/john/PhD/GitLab/literary_lstm/tf_log/word2vec_data.npz'

# Log directory:
strPthLog = '/home/john/PhD/GitLab/literary_lstm/log_lstm'

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


aryEmbFnlT = aryEmbFnl.T

# -----------------------------------------------------------------------------
### Parameters

# Learning rate:
varLrnRte = 0.001

# Number of training iterations:
varNumItr = 10000000

# Display steps (after x number of iterations):
varDspStp = 1000

# Number of input words from which to predict next word: (?)
varNumIn = 5

# Number of hidden units in RNN:
varNumNrn = 512

# Vocabulary size (number of words):
varNumWrds = aryEmbFnl.shape[0]

# Size of embedding vector:
varSzeEmb = aryEmbFnl.shape[1]

# tf Graph input
vecWrdsIn = tf.placeholder("float", [None, (varNumIn * varSzeEmb), 1])
vecWrdsOut = tf.placeholder("float", [None, varNumWrds])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([varNumNrn, varNumWrds]))
}
biases = {
    'out': tf.Variable(tf.random_normal([varNumWrds]))
}

def RNN(vecWrdsIn, weights, biases):

    # reshape to [1, varNumIn]
    vecWrdsIn = tf.reshape(vecWrdsIn, [-1, varNumIn])

    # Generate a varNumIn-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    vecWrdsIn = tf.split(vecWrdsIn, varNumIn, 1)

    rnn_cell = rnn.MultiRNNCell([rnn.DropoutWrapper(
                                                    rnn.BasicLSTMCell(varNumNrn),
                                                    input_keep_prob=0.7,
                                                    output_keep_prob=0.7,
                                                    state_keep_prob=0.7,
                                                    ),
                                 rnn.DropoutWrapper(
                                                    rnn.BasicLSTMCell(varNumNrn),
                                                    input_keep_prob=0.7,
                                                    output_keep_prob=0.7,
                                                    state_keep_prob=0.7,
                                                    )
                                 ])

    # 2-layer LSTM, each layer has varNumNrn units.
    # Average Accuracy= 95.20% at 50k iter
    # rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(varNumNrn),
    #                              rnn.BasicLSTMCell(varNumNrn)])

    # 1-layer LSTM with varNumNrn units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(varNumNrn)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, vecWrdsIn, dtype=tf.float32)

    # there are varNumIn outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(vecWrdsIn, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(
                      tf.nn.softmax_cross_entropy_with_logits(logits=pred,
                                                              labels=vecWrdsOut)
                      )

optimizer = tf.train.RMSPropOptimizer(
                                      learning_rate=varLrnRte
                                      ).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(vecWrdsOut,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

training_data = lstC

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0,varNumIn+1)
    end_offset = varNumIn + 1
    acc_total = 0
    loss_total = 0

    while step < varNumItr:
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(training_data)-end_offset):
            offset = random.randint(0, varNumIn+1)

        # symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+varNumIn) ]
        # symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, varNumIn, 1])

        # Get integer codes of next chunk of words:
        lstTmpWrds = [lstC[x] for x in range(offset, offset+varNumIn)]
        # Get embedding vectors for words:
        aryTmpWrds = np.array(aryEmbFnl[lstTmpWrds, :]).reshape((-1, (varNumIn * varSzeEmb), 1))

        #if step < 10:
        #    print(aryTmpWrds.shape)

        # symbols_out_onehot = np.zeros([varNumWrds], dtype=float)
        # symbols_out_onehot[dictionary[str(training_data[offset+varNumIn])]] = 1.0
        # symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

        # Word to predict:
        # lstC[(offset + varNumIn)]
        symbols_out_onehot = np.zeros([varNumWrds], dtype=float)
        symbols_out_onehot[lstC[(offset + varNumIn)]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={vecWrdsIn: aryTmpWrds, vecWrdsOut: symbols_out_onehot})

        # if step < 10:
        #     print(onehot_pred.shape)

        loss_total += loss
        acc_total += acc
        if (step+1) % varDspStp == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/varDspStp) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/varDspStp))
            acc_total = 0
            loss_total = 0
            symbols_in = [dictRvrs[training_data[i]] for i in range(offset, offset + varNumIn)]
            symbols_out = dictRvrs[training_data[offset + varNumIn]]

            #symbols_out_pred = dictRvrs[int(tf.argmax(onehot_pred, 1).eval())]
            # Sum of squares 
            vecTmp = np.sum(np.square(np.subtract(aryEmbFnlT, onehot_pred)), axis=0)
            symbols_out_pred = dictRvrs[int(np.argmin(vecTmp))]

            print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
        step += 1
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
                onehot_pred = session.run(pred, feed_dict={vecWrdsIn: keys})

                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())

                sentence = "%s %s" % (sentence,dictRvrs[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
        except:
            print("Word not in dictionary")
