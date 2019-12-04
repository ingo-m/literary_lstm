#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LSTM function using word2vec embedding."""


import os
import random
import datetime
import threading
import queue
import numpy as np
import tensorflow as tf
import time


# -----------------------------------------------------------------------------
# *** Define parameters

# Path of input data file (containing text and word2vec embedding):
strPthIn = '/home/john/Dropbox/Harry_Potter/embedding/word2vec_data_all_books_e300_w5000.npz'

# Path of npz file containing previously trained model's weights to load (if
# None, new model is created):
strPthMdl = '/home/john/Dropbox/Harry_Potter/lstm/512/20191202_191428/lstm_data.npz'

# Log directory (parent directory, new session directory will be created):
strPthLog = '/home/john/Downloads'

# Learning rate:
varLrnRte = 0.000001

# Number of training iterations over the input text:
varNumItr = 4.0

# Display steps (after x number of optimisation steps):
varDspStp = 100000

# Reset internal model states after x number of optimisation steps:
varResStp = 1000

# Length of new text to generate:
varLenNewTxt = 1000

# Batch size:
varSzeBtch = 64

# Input dropout:
varInDrp = 0.5

# Recurrent state dropout:
varStDrp = 0.5

# Number of words from which to sample next word (n most likely words) when
# generating new text. This parameter has no effect during training, but during
# validation.
varNumWrdSmp = 40  # 128k

# Exponent to apply over likelihoods of predictions. Higher value biases the
# selection towards words predicted with high likelihood, but leads to
# repetitive sequences of frequent words. Lower value biases selection towards
# less frequent words and breaks repetitive sequences, but leads to incoherent
# sequences without gramatical structure or semantic meaning.
varTemp = 1.0  # 1.4


try:
    # Prepare import from google drive, if on colab:
    from google.colab import drive
    # Mount google drive:
    drive.mount('drive')
except ModuleNotFoundError:
    pass

# Load npz file:
objNpz = np.load(strPthIn)

# Enable object array:
objNpz.allow_pickle = True

# Coded text:
vecC = objNpz['vecC']

# Only train on part of text (retain copy of full text for weights):
vecFullC = np.copy(vecC)
# vecC = vecC[15:15020]

# Dictionary, with words as keys:
dicWdCnOdr = objNpz['dicWdCnOdr'][()]

# Reverse dictionary, with ordinal word count as keys:
dictRvrs = objNpz['dictRvrs'][()]

# Embedding matrix:
aryEmb = objNpz['aryEmbFnl']

# Scale embedding matrix:
varAbsMax = np.max(np.absolute(aryEmb.flatten()))
varAbsMax = varAbsMax / 0.5
aryEmb = np.divide(aryEmb, varAbsMax)

# Tensorflow constant fo embedding matrix:
aryTfEmb = tf.constant(aryEmb, dtype=tf.float32)


# -----------------------------------------------------------------------------
# *** Preparations

# Create tf session:
objSess = tf.Session()

# Tell keras about tf session:
tf.keras.backend.set_session(objSess)

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

# Number of optimisation steps:
varNumOpt = int(np.floor(float(varLenTxt * varNumItr)))

# Get information on model architecture from npz file:
objNpz = np.load(strPthMdl)
objNpz.allow_pickle = True

# Number of neurons per layer (LSTM layers, plus two dense layers):
lstNumNrn = objNpz['lstNumNrn']

# Indices of weights to asssign to layer:
lstLoadW = list(range(len(lstNumNrn)))

# Which layers are trainable?
lstLyrTrn = [False] * len(lstNumNrn)


# -----------------------------------------------------------------------------
# *** Prepare queues

# Batches are prepared in a queue-feeding-function that runs in a separate
# thread.

# Queue capacity:
varCapQ = 32

# Queue for training batches of context words:
objQ01 = tf.FIFOQueue(capacity=varCapQ,
                      dtypes=[tf.float32],
                      shapes=[(varSzeBtch, 1, varSzeEmb)])

# Queue for training batches of target words:
objQ02 = tf.FIFOQueue(capacity=varCapQ,
                      dtypes=[tf.float32],
                      shapes=[(varSzeBtch, varSzeEmb)])

# Queue for training weights (weights are applied to outputs, therefore the
# shape is independent of number of input words to make prediction; the model
# predicts one word at a time):
objQ03 = tf.FIFOQueue(capacity=varCapQ,
                      dtypes=[tf.float32],
                      shapes=[(varSzeBtch)])

# Method for getting queue size:
objSzeQ = objQ01.size()

# Placeholder that is used to put batch on computational graph:
objPlcHld01 = tf.placeholder(tf.float32,
                             shape=[varSzeBtch, 1, varSzeEmb])
objPlcHld02 = tf.placeholder(tf.float32,
                             shape=[varSzeBtch, varSzeEmb])
objPlcHld03 = tf.placeholder(tf.float32,
                             shape=[varSzeBtch])

# The enqueue operation that puts data on the graph.
objEnQ01 = objQ01.enqueue([objPlcHld01])
objEnQ02 = objQ02.enqueue([objPlcHld02])
objEnQ03 = objQ03.enqueue([objPlcHld03])

# Number of threads that will be created per queue:
varNumThrd = 1

# The queue runner (places the enqueue operation on the queue?).
objRunQ01 = tf.train.QueueRunner(objQ01, [objEnQ01] * varNumThrd)
tf.train.add_queue_runner(objRunQ01)
objRunQ02 = tf.train.QueueRunner(objQ02, [objEnQ02] * varNumThrd)
tf.train.add_queue_runner(objRunQ02)
objRunQ03 = tf.train.QueueRunner(objQ03, [objEnQ03] * varNumThrd)
tf.train.add_queue_runner(objRunQ03)

# The tensor object that is retrieved from the queue. Functions like
# placeholders for the data in the queue when defining the graph.
# Training context placebolder (input):
objTrnCtxt = tf.keras.Input(shape=(varSzeBtch, 1, varSzeEmb),
                            batch_size=varSzeBtch,
                            tensor=objQ01.dequeue(),
                            dtype=tf.float32)
# Training target placeholder:
objTrgt = objQ02.dequeue()
# Testing context placeholder:
objTstCtxt = tf.keras.Input(shape=(1, varSzeEmb),
                            batch_size=1,
                            dtype=tf.float32)
objWght = objQ03.dequeue()


# -----------------------------------------------------------------------------
# *** Build the network

# Regularisation:
# objRegL1 = tf.keras.regularizers.l1(l=0.001)
objRegL2 = None  # tf.keras.regularizers.l2(l=0.0001)

# Stateful model:
lgcState = True

# Number of LSTM layers (not including the two final dense layers):
varNumLstm = len(lstNumNrn) - 2

# Lists used to assign output of one layer as input of next layer (for training
# and validation model, respectively).
lstIn = [objTrnCtxt]
lstInT = [objTstCtxt]

# List for `return_sequences` flag of LSTM (needs to be 'True' for all but last
# layer).
lstRtrnSq = [True] * varNumLstm
lstRtrnSq[-1] = False

sigmoid = tf.keras.activations.sigmoid
relu = tf.keras.activations.relu
tanh = tf.keras.activations.tanh

# The actual LSTM layers.
# Note that this cell is not optimized for performance on GPU.
# Please use tf.keras.layers.CuDNNLSTM for better performance on GPU.
for idxLry in range(varNumLstm):

    objInTmp = tf.keras.layers.LSTM(lstNumNrn[idxLry],
                                    activation=tanh,
                                    recurrent_activation='hard_sigmoid',
                                    dropout=varInDrp,
                                    recurrent_dropout=varStDrp,
                                    kernel_regularizer=objRegL2,
                                    return_sequences=lstRtrnSq[idxLry],
                                    return_state=False,
                                    go_backwards=False,
                                    stateful=lgcState,
                                    unroll=False,
                                    trainable=lstLyrTrn[idxLry],
                                    name=('LstmLayer' + str(idxLry))
                                    )(lstIn[idxLry])
    lstIn.append(objInTmp)

# Dense feedforward layer:
aryDense01 = tf.keras.layers.Dense(lstNumNrn[-2],
                                   activation=tanh,
                                   kernel_regularizer=objRegL2,
                                   trainable=lstLyrTrn[-2],
                                   name='DenseFf01'
                                   )(lstIn[-1])
aryDense02 = tf.keras.layers.Dense(lstNumNrn[-1],
                                   activation=tanh,
                                   kernel_regularizer=objRegL2,
                                   trainable=lstLyrTrn[-1],
                                   name='DenseFf02'
                                   )(aryDense01)

# Initialise the model:
objMdl = tf.keras.models.Model(inputs=[objTrnCtxt], outputs=aryDense02)

# An almost idential version of the model used for testing, without dropout
# and possibly different input size (fixed batch size of one).
for idxLry in range(varNumLstm):

    objInTmp = tf.keras.layers.LSTM(lstNumNrn[idxLry],
                                    activation=tanh,
                                    recurrent_activation='hard_sigmoid',
                                    dropout=0.0,
                                    recurrent_dropout=0.0,
                                    kernel_regularizer=objRegL2,
                                    return_sequences=lstRtrnSq[idxLry],
                                    return_state=False,
                                    go_backwards=False,
                                    stateful=lgcState,
                                    unroll=False,
                                    trainable=False,
                                    name=('TestLstmLayer' + str(idxLry))
                                    )(lstInT[idxLry])
    lstInT.append(objInTmp)

# Dense feedforward layer:
aryDenseT1 = tf.keras.layers.Dense(lstNumNrn[-2],
                                   activation=tanh,
                                   kernel_regularizer=objRegL2,
                                   trainable=False,
                                   name='TestingDenseFf01'
                                   )(lstInT[-1])
aryDenseT2 = tf.keras.layers.Dense(lstNumNrn[-1],
                                   activation=tanh,
                                   kernel_regularizer=objRegL2,
                                   trainable=False,
                                   name='TestingDenseFf02'
                                   )(aryDenseT1)

# Initialise the model:
objTstMdl = tf.keras.models.Model(inputs=objTstCtxt, outputs=aryDenseT2)
init = tf.global_variables_initializer()
objSess.run(init)

# Load pre-trained weights from disk?
if strPthMdl is None:
    print('Building new model.')
else:
    print('Loading pre-trained model weights from disk.')

    # Get weights from npz file:
    objNpz = np.load(strPthMdl)
    objNpz.allow_pickle = True
    lstWghts = list(objNpz['lstWghts'])

    # Number of layers:
    varNumLyr = len(objMdl.layers)
    # Counter (necessary because layers without weights, such as input layer,
    # are skipped):
    varCntLyr = 0
    # Loop through layers:
    for idxLry in range(varNumLyr):
        # Skip layers without weights (e.g. input layer):
        if len(objMdl.layers[idxLry].get_weights()) != 0:
            # Load weights for this layer?
            if lstLoadW[varCntLyr] is not None:
                print(('---Assigning pre-trained weights to layer: '
                       + str(varCntLyr)
                       + ', from list item '
                       + str(lstLoadW[varCntLyr])))
                # Assign weights to model:
                objMdl.layers[idxLry].set_weights(lstWghts[lstLoadW[varCntLyr]])
            # Increment counter:
            varCntLyr += 1

# Print model summary:
print('Training model:')
objMdl.summary()
print('Testing model:')
objTstMdl.summary()

# Define the optimiser and loss function:
objMdl.compile(optimizer=tf.keras.optimizers.Adam(lr=varLrnRte),
               loss=tf.keras.losses.mean_squared_error)


# -----------------------------------------------------------------------------
# *** Training

# Length of context to use to initialise the state of the prediction
# model:
varLenCntx = 100

varTmpWrd = 350487

# Avoid beginning of text (not enough preceding context words):
if varTmpWrd > varLenCntx:

    # Get model weights from training model:
    lstWghts = objMdl.get_weights()

    # Copy weights from training model to test model:
    objTstMdl.set_weights(lstWghts)

    # Reset model states:
    objTstMdl.reset_states()

    # Loop through context window:
    for idxCntx in range(1, varLenCntx):
        # Get integer code of context word (the '- 1' is so as not
        # to predict twice on the word right before the target
        # word, see below):
        varCtxt = vecC[(varTmpWrd - 1 - varLenCntx + idxCntx)]
        # Get embedding vectors for context word(s):
        aryCtxt = np.array(aryEmb[varCtxt, :]
                           ).reshape(1, 1, varSzeEmb)
        # Predict on current context word:
        vecWrd = objTstMdl.predict_on_batch(aryCtxt)

    # Get integer code of context word:
    varTstCtxt = vecC[(varTmpWrd - 1)]

    # Get embedding vector for context word:
    aryTstCtxt = np.array(aryEmb[varTstCtxt, :]
                          ).reshape(1, 1, varSzeEmb)

    # Word to predict (target):
    varTrgt = vecC[varTmpWrd]

    # Get embedding vector for target word:
    vecTstTrgt = aryEmb[varTrgt, :].reshape(1, varSzeEmb)

    # Get test prediction for current context word(s):
    vecWrd = objTstMdl.predict_on_batch(aryTstCtxt)

    # Current loss:
    varLoss02 = np.sum(
                       np.square(
                                 np.subtract(
                                             vecTstTrgt,
                                             vecWrd
                                             )
                                 )
                       )

    print(('Loss manual: ' + str(varLoss02)))

    # Context:
    lstCtxt = [dictRvrs[x] for x in vecC[(varTmpWrd - 15):varTmpWrd]]
    strCtxt = ' '.join(lstCtxt)

    print(('Context: ' + strCtxt))

    print(('Target: ' + dictRvrs[vecC[varTmpWrd]]))

    # Minimum squared deviation between prediciton and embedding
    # vectors:
    vecDiff = np.sum(
                     np.square(
                               np.subtract(
                                           aryEmb,
                                           vecWrd[0, :]
                                           )
                               ),
                     axis=1
                     )

    # Get code of closest word vector:
    varTmp = int(np.argmin(vecDiff))

    # Look up predicted word in dictionary:
    strWrdPrd = dictRvrs[varTmp]

    print(('Prediction: ' + strWrdPrd))

    # ** Generate new text

    # Vector for next text (coded):
    vecNew = np.zeros(varLenNewTxt, dtype=np.int32)

    # Generate new text:
    for idxNew in range(varLenNewTxt):

        # Update context (leave out first - i.e. oldest - word in
        # context, and append newly predicted word):
        aryTstCtxt = vecWrd.reshape(1, 1, varSzeEmb)

        # Get test prediction for current context word(s):
        vecWrd = objTstMdl.predict_on_batch(aryTstCtxt)

        # Minimum squared deviation between prediciton and embedding
        # vectors. We skip the first row of the embedding matrix
        # (corresponding to the unknown-token).
        vecDiff = np.sum(
                         np.square(
                                   np.subtract(
                                               aryEmb[1:, :],
                                               vecWrd[0, :]
                                               )
                                   ),
                         axis=1
                         )

        # Percentile corresponding to number of words to sample:
        varPrcntSmp = float(varNumWrdSmp) / float(varNumWrds) * 100.0

        # Construct vector with probability distrubition of next word.
        # The  difference between the prediction and all words in the
        # vocabulary is the starting point.
        vecProbDist = vecDiff.astype('float64')

        # The next word will be selected from a subset of words (the n
        # words that are clostest to the predicted word vector). Create
        # a boolean vector for words to include in sampling.
        varPrcnt = np.percentile(vecProbDist, varPrcntSmp)
        # Words in vocabulary to exclude from sampling:
        vecLgcExc = np.greater(vecProbDist, varPrcnt)
        # Words in vocabulary to include in sampling:
        vecLgcInc = np.logical_not(vecLgcExc)

        # Invert the difference between prediction and word vectors, so
        # high values correspond to words that are more likely, given
        # the prediction.
        vecProbDist = np.divide(1.0, vecProbDist)

        # Normalise the range of the distribution:
        varMin = np.min(vecProbDist[vecLgcInc])
        vecProbDist = np.subtract(vecProbDist, varMin)
        vecProbDist[vecLgcExc] = 0.0
        vecProbDist = np.divide(vecProbDist, np.max(vecProbDist))

        # Apply exponent, to bias selection towards more or less likely
        # words.
        vecProbDist = np.power(vecProbDist, varTemp)

        # Turn the vector into a probability distribution (so that the
        # sum of all elements is one).
        vecProbDist = np.divide(vecProbDist, np.sum(vecProbDist))

        # Sample next word from probability distribution - code of next
        # word.
        varTmp = int(np.argmax(np.random.multinomial(1, vecProbDist)))

        # Add one, because we skipped the first row of the embedding
        # matrix when calculating the minimum squared deviation between
        # prediciton and embedding vectors (to avoid the unknown-token).
        varTmp = varTmp + 1

        # Save code of predicted word:
        vecNew[idxNew] = varTmp

        # Replace predicted embedding vector with embedding vector of
        # closest word:
        vecWrd = aryEmb[varTmp, :]

    # Decode newly generated words:
    lstNew = [dictRvrs[x] for x in vecNew]

    # List to string:
    strNew = ' '.join(lstNew)
    print('New text:')
    print(strNew)

# Reset model states:
print('Resetting model states.')
objMdl.reset_states()
objTstMdl.reset_states()
