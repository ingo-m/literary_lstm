#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM fun main function using word2vec embedding.
"""

import os
import random
import datetime
import numpy as np
import tensorflow as tf

# from tensorflow.contrib import rnn
# from tensorflow.keras import layers

# from tensorflow.nn.objRnn import DropoutWrapper
# from tf.contrib.rnn.DropoutWrapper


# -----------------------------------------------------------------------------
# *** Define parameters

# Path of input data file:
strPthIn = '/home/john/PhD/GitLab/literary_lstm/log_w2v/word2vec_data.npz'

# Log directory:
strPthLog = '/home/john/PhD/GitLab/literary_lstm/log_lstm'

# Learning rate:
varLrnRte = 0.001

# Number of training iterations:
varNumItr = 1000

# Display steps (after x number of iterations):
varDspStp = 1

# Number of input words from which to predict next word:
varNumIn = 1

# Number of neurons in first hidden layer:
varNrn01 = 300

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
# aryEmbT = aryEmb.T


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

# Number of neurons in second hidden layer:
varNrn02 = varSzeEmb

# Total number of inputs (to input layer):
varNumInTtl = varNumIn * varSzeEmb

# Placeholder for inputs (to input layer):
aryWrdsIn = tf.keras.Input(shape=(varNumIn, varSzeEmb),
                           batch_size=1,
                           dtype=tf.float32)

# Placeholder for output:
vecWrdsOut = tf.placeholder(tf.float32, [1, varSzeEmb])


# -----------------------------------------------------------------------------
# *** Logging

## Get date string as default session name:
#strDate = str(datetime.datetime.now())
#lstD = strDate[0:10].split('-')
#lstT = strDate[11:19].split(':')
#strDate = (lstD[0] + lstD[1] + lstD[2] + '_' + lstT[0] + lstT[1] + lstT[2])
#
## Log directory:
#strPthLog = os.path.join(strPthLog, strDate)
#
#objCallback = tf.keras.callbacks.TensorBoard(log_dir=strPthLog)


# -----------------------------------------------------------------------------
# ***

lstClls = [tf.keras.layers.LSTMCell(varNrn01,
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
           tf.keras.layers.LSTMCell(
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

# tf.keras.layers.StackedRNNCells ?
aryOut = tf.keras.layers.RNN(lstClls)(aryWrdsIn)

objMdl = tf.keras.Model(inputs=aryWrdsIn, outputs=aryOut)

# Cost function:
# objCost = tf.sqrt(tf.reduce_sum(tf.squared_difference(aryOut, vecWrdsOut)))
#objCost = tf.losses.mean_squared_error(
#    vecWrdsOut,
#    aryOut) #,
#    reduction=tf.losses.Reduction.SUM
#    )
objCost = tf.losses.mean_squared_error


# Optimiser:
# objOpt = tf.train.RMSPropOptimizer(learning_rate=varLrnRte).minimize(objCost)
objOpt = tf.keras.optimizers.RMSprop(lr=varLrnRte)


objMdl.compile(optimizer=objOpt,
               loss=objCost,
               metrics=['accuracy'])




# Loop through iterations:
for idxItr in range(varNumItr):

    # Random number for status feedback:
    varRndm = random.randint(20, (varLenTxt - 20))

    # Loop through text (corpus). Index refers to target word (i.e. word
    # to be predicted).
    for idxWrd in range(varNumIn, varLenTxt):

        # Get integer codes of context word(s):
        vecCntxt = vecC[(idxWrd - varNumIn):idxWrd]

        # Get embedding vectors for words:
        aryCntxt = np.array(aryEmb[vecCntxt, :]).reshape((1, varNumIn, varSzeEmb))

        # Word to predict (target):
        varTrgt = vecC[idxWrd]

        # Get embedding vector for target word:
        vecTrgt = aryEmb[varTrgt, :].reshape((1, varSzeEmb))

        # Run optimisation:
        #objMdl.fit(x=aryCntxt,  # run on entire dataset
        #           y=vecTrgt,
        #           verbose=0,
        #           epochs=1)
        #           # callbacks=[objCallback])
        varLoss01 = objMdl.train_on_batch(aryCntxt,  # run on single batch
                                          y=vecTrgt)



        # Status feedback:
        #if (idxItr % varDspStp == 0) and (idxWrd == varRndm):
        if (idxWrd % 10000 == 0):

            try:

                print('---')

                # Get prediction for current context word(s):
                vecTmp = objMdl.predict_on_batch(aryCntxt)

                # Current loss:
                varLoss02 = np.sum(
                                   np.square(
                                             np.subtract(
                                                         vecTrgt,
                                                         vecTmp
                                                         )
                                             )
                                   )

                print(('Loss auto:   ' + str(varLoss01)))
                print(('Loss manual: ' + str(varLoss02)))

                print(('Context: '
                       + str([dictRvrs[x] for x in vecC[(idxWrd - 15):idxWrd]])))

                print(('Target: '
                       + dictRvrs[varTrgt]))

                # Minimum squared deviation between prediciton and embedding
                # vectors:
                vecTmp = np.sum(
                                np.square(
                                          np.subtract(
                                                      aryEmb,
                                                      vecTmp[None, :]
                                                      )
                                          ),
                                axis=1
                                )

                # Look up predicted word in dictionary:
                strWrdPrd = dictRvrs[int(np.argmin(vecTmp))]
                #strWrdPrd = dictRvrs[varPred]

                print(('Prediction: ' + strWrdPrd))



            except:

                pass


#objMdl.evaluate(x_test, y_test)
