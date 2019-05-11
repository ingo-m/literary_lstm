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
from utilities import read_text


# -----------------------------------------------------------------------------
# *** Define parameters

# Path of input data file:
strPthIn = '/home/john/PhD/GitLab/literary_lstm/log_w2v/word2vec_data_200.npz'

# Log directory:
strPthLog = '/home/john/PhD/GitLab/literary_lstm/log_lstm'

# Path of sample text to base new predictions on (when generating new text):
strPthBse = '/home/john/Dropbox/Ernest_Hemingway/redacted/new_base.txt'

# Learning rate:
varLrnRte = 0.001

# Number of training iterations:
varNumItr = 1000

# Display steps (after x number of iterations):
varDspStp = 1

# Number of input words from which to predict next word:
varNumIn = 10

# Number of neurons in first hidden layer:
varNrn01 = 200

# Length of new text to generate:
varLenNewTxt = 100


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
# varNumInTtl = varNumIn * varSzeEmb

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

# Note that this cell is not optimized for performance on GPU.
# Please use tf.keras.layers.CuDNNLSTM for better performance on GPU.
aryOut01 = tf.keras.layers.LSTM(varNrn01,
                                #input_shape=(varNumIn, varSzeEmb),
                                #batch_size=1,
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
                                activity_regularizer=None,
                                kernel_constraint=None,
                                recurrent_constraint=None,
                                bias_constraint=None,
                                dropout=0.1,
                                recurrent_dropout=0.1,
                                implementation=1,
                                return_sequences=False,  # ?
                                return_state=False,
                                go_backwards=False,
                                stateful=True,
                                unroll=False,
                                name='LSTMlayer01'
                                )(aryWrdsIn)

#aryOut02 = tf.keras.layers.LSTM(varNrn02,
#                                #input_shape=(varNumIn, varNrn01),
#                                #batch_size=1,
#                                activation='tanh',
#                                recurrent_activation='hard_sigmoid',
#                                use_bias=True,
#                                kernel_initializer='glorot_uniform',
#                                recurrent_initializer='orthogonal',
#                                bias_initializer='zeros',
#                                unit_forget_bias=True,
#                                kernel_regularizer=None,
#                                recurrent_regularizer=None,
#                                bias_regularizer=None,
#                                activity_regularizer=None,
#                                kernel_constraint=None,
#                                recurrent_constraint=None,
#                                bias_constraint=None,
#                                dropout=0.1,
#                                recurrent_dropout=0.1,
#                                implementation=1,
#                                return_sequences=False,  # ?
#                                return_state=False,
#                                go_backwards=False,
#                                stateful=True,
#                                unroll=False,
#                                name='LSTMlayer02'
#                                )(aryOut01)

# objMdl.add(tf.keras.layers.Dense(1))

# inputs=model1_inputs, outputs = model1_outputs, name='model1
#objMdl = tf.keras.models.Sequential([objLyr01, objLyr02])
objMdl = tf.keras.models.Model(inputs=aryWrdsIn, outputs=aryOut01)

objMdl.summary()

objMdl.compile(optimizer=tf.keras.optimizers.RMSprop(lr=varLrnRte),
               loss=tf.losses.mean_squared_error,
               metrics=['accuracy'])

# Loop through iterations:
for idxItr in range(varNumItr):

    # Random number for status feedback:
    varRndm = random.randint(20, (varLenTxt - 20))

    # List for word indices to loop over, in randomised order:
    lstRnd = list(range(varNumIn, varLenTxt))
    random.shuffle(lstRnd)

    # Loop through text (corpus). Index refers to target word (i.e. word
    # to be predicted).
    for idxWrd in lstRnd:

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
        if (idxItr % varDspStp == 0) and (idxWrd == varRndm):

            try:

                print('---')

                # Get prediction for current context word(s):
                vecWrd = objMdl.predict_on_batch(aryCntxt)

                # Current loss:
                varLoss02 = np.sum(
                                   np.square(
                                             np.subtract(
                                                         vecTrgt,
                                                         vecWrd
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
                vecDiff = np.sum(
                                 np.square(
                                           np.subtract(
                                                       aryEmb,
                                                       vecWrd[None, :]
                                                       )
                                           ),
                                 axis=1
                                 )

                # Look up predicted word in dictionary:
                strWrdPrd = dictRvrs[int(np.argmin(vecDiff))]
                #strWrdPrd = dictRvrs[varPred]

                print(('Prediction: ' + strWrdPrd))

                # ** Generate new text

                # Vector for next text (coded):
                vecNew = np.zeros(varLenNewTxt, dtype=np.int32)

                # Use last context word as starting point for new text:
                # vecWrd = aryCntxt  # TODO: only works with input size one

                # Generate new text:
                for idxNew in range(varLenNewTxt):

                    # Get prediction for current word:
                    vecWrd = objMdl.predict_on_batch(aryCntxt)  # .reshape(1, varNumIn, varSzeEmb))  # TODO: only works with input size one?

                    # Minimum squared deviation between prediciton and embedding
                    # vectors:
                    vecDiff = np.sum(
                                     np.square(
                                               np.subtract(
                                                           aryEmb,
                                                           vecWrd[None, :]
                                                           )
                                               ),
                                     axis=1
                                     )

                    # Get code of closest word vector:
                    varTmp = int(np.argmin(vecDiff))

                    # Save code of predicted word:
                    vecNew[idxNew] = varTmp

                    # Update context (leave out first - i.e. oldest - word in
                    # context, and append newly predicted word):
                    aryCntxt = np.concatenate((aryCntxt[:, 1:, :],
                                               vecWrd.reshape(1, 1, varSzeEmb)
                                               ), axis=1)

                # Decode newly generated words:
                lstNew = [dictRvrs[x] for x in vecNew]

                # List to string:
                strNew = ' '.join(lstNew)
                print('New text:')
                print(strNew)

            except:

                pass

#objMdl.evaluate(x_test, y_test)


# -----------------------------------------------------------------------------
# *** Generate new text

if False:

    # TODO: Does running data through model with model.predict_on_batch actually
    # change the state of the LSTM?

    # Load text to base new predictions on:
    lstBase = read_text(strPthBse)

    # Base text to code:
    varLenBse = len(lstBase)
    vecBase = np.zeros(varLenBse, dtype=np.int32)

    # Loop through base text:
    for idxWrd in range(varLenBse):

        # Try to look up words in dictionary. If there is an unkown words, replace
        # with unknown token.
        try:
            varTmp = dicWdCnOdr[lstBase[idxWrd].lower()]
        except KeyError:
            varTmp = 0
        vecBase[idxWrd] = varTmp


    # Get embedding vectors for words:
    aryBase = np.array(aryEmb[vecBase, :])

    for idxWrd in range(varLenBse):

        # Get prediction for current word:
        vecWrd = objMdl.predict_on_batch(aryBase[idxWrd, :].reshape(1, 1, varSzeEmb))  # TODO: only works with input size one

    # Vector for new text (coded):
    vecNew = np.zeros(varLenNewTxt, dtype=np.int32)

    # Generate new text:
    for idxNew in range(varLenNewTxt):

        # Get prediction for current word:
        vecWrd = objMdl.predict_on_batch(vecWrd.reshape(1, 1, varSzeEmb))  # TODO: only works with input size one

        # Minimum squared deviation between prediciton and embedding
        # vectors:
        vecDiff = np.sum(
                         np.square(
                                   np.subtract(
                                               aryEmb,
                                               vecWrd[None, :]
                                               )
                                   ),
                         axis=1
                         )

        # Get code of closest word vector:
        varTmp = int(np.argmin(vecDiff))

        # Save code of predicted word:
        vecNew[idxNew] = varTmp

    # Decode newly generated words:
    lstNew = [dictRvrs[x] for x in vecNew]

    # List to string:
    strBase = ' '.join(lstBase)
    strNew = ' '.join(lstNew)

    print('---')
    print('Base text:')
    print(strBase)
    print('---')
    print('New text:')
    print(strNew)
    print('---')
