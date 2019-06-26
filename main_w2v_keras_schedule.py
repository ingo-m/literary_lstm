#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LSTM function using word2vec embedding."""

import os
import random
import datetime
import threading
import numpy as np
import tensorflow as tf
from utilities import read_text


# -----------------------------------------------------------------------------
# *** Define parameters

# Path of input data file (containing text and word2vec embedding):
strPthIn = '/home/john/Dropbox/Harry_Potter/embedding/word2vec_data_all_books_e300_w5000.npz'

# Path of previously trained model (parent directory containing training and
# test models; if None, new model is created):
strPthMdl = None

# Log directory (parent directory, new session directory will be created):
strPthLog = '/home/john/Dropbox/Harry_Potter/lstm'

# Path of sample text to base new predictions on (when generating new text):
strPthBse = 'new_base.txt'

# Learning rate:
varLrnRte = 0.001

# Number of training iterations over the input text:
varNumItr = 1000000

# Display steps (after x number of optimisation steps):
varDspStp = 10000

# Number of input words from which to predict next word:
varNumIn = 1

# Number of neurons in first hidden layer:
varNrn01 = 300

# Number of neurons in second hidden layer:
varNrn02 = 300

# Length of new text to generate:
varLenNewTxt = 100

# Batch size:
varSzeBtch = 250

# Input dropout:
varInDrp = 0.3

# Recurrent state dropout:
varStDrp = 0.2


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

# Scale embedding matrix (this is only for a more convenience range of loss
# values during visualisation):
aryEmb = np.multiply(aryEmb, 100.0)

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
# varNumOpt = int(np.floor(float(varLenTxt * varNumItr) / float(varSzeBtch)))

# Total number of inputs (to input layer):
# varNumInTtl = varNumIn * varSzeEmb

# Placeholder for inputs (to input layer):
#aryWrdsIn = tf.keras.Input(shape=(varNumIn, varSzeEmb),
#                           batch_size=1,
#                           dtype=tf.float32)

# Placeholder for output:
#vecWrdsOut = tf.placeholder(tf.float32, [1, varSzeEmb])

# Index for testing word:
varTst = 0

# Global variable for control flow:
varLgcGlb = True


# -----------------------------------------------------------------------------
# *** Prepare queues

# Batches are prepared in a queue-feeding-function that runs in a separate
# thread.

# Queue capacity:
varCapQ = 5

# Queue for training batches of context words:
objQ01 = tf.FIFOQueue(capacity=varCapQ,
                      dtypes=[tf.float32],
                      shapes=[(varSzeBtch, varNumIn, varSzeEmb)])

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
                             shape=[varSzeBtch, varNumIn, varSzeEmb])
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
objTrnCtxt = tf.keras.Input(shape=(varSzeBtch, varNumIn, varSzeEmb),
                            batch_size=varSzeBtch,
                            tensor=objQ01.dequeue(),
                            dtype=tf.float32)
# Training target placeholder:
objTrgt = objQ02.dequeue()
# Testing context placeholder:
objTstCtxt = tf.keras.Input(shape=(varNumIn, varSzeEmb),
                            batch_size=1,
                            dtype=tf.float32)
objWght = objQ03.dequeue()


# -----------------------------------------------------------------------------
# *** Build the network

# Load pre-trained model, or create new one:
if strPthMdl is None:

    print('Building new model.')

    # Regularisation:
    # objRegL2 = tf.keras.regularizers.l2(l=0.005)
    objRegL2 = None

    # The actual LSTM layers.
    # Note that this cell is not optimized for performance on GPU.
    # Please use tf.keras.layers.CuDNNLSTM for better performance on GPU.
    # objMdl.add(tf.keras.layers.LSTM(varNrn01,
    aryOut01 = tf.keras.layers.LSTM(varNrn01,
                                    # input_shape=(varNumIn, varSzeEmb),
                                    # batch_size=varSzeBtch,
                                    activation='tanh',
                                    recurrent_activation='hard_sigmoid',
                                    use_bias=True,
                                    kernel_initializer='glorot_uniform',
                                    recurrent_initializer='orthogonal',
                                    bias_initializer='zeros',
                                    unit_forget_bias=True,
                                    kernel_regularizer=objRegL2,
                                    recurrent_regularizer=objRegL2,
                                    bias_regularizer=objRegL2,
                                    activity_regularizer=objRegL2,
                                    kernel_constraint=None,
                                    recurrent_constraint=None,
                                    bias_constraint=None,
                                    dropout=varInDrp,
                                    recurrent_dropout=varStDrp,
                                    implementation=1,
                                    return_sequences=True,
                                    return_state=False,
                                    go_backwards=False,
                                    stateful=False,
                                    unroll=False,
                                    name='LSTMlayer01'
                                    )(objTrnCtxt)

    # Second LSTM layer:
    # objMdl.add(tf.keras.layers.LSTM(varNrn02,
    aryOut02 = tf.keras.layers.LSTM(varNrn02,
                                    # input_shape=(varNumIn, varNrn01),
                                    # batch_size=varSzeBtch,
                                    activation='tanh',
                                    recurrent_activation='hard_sigmoid',
                                    use_bias=True,
                                    kernel_initializer='glorot_uniform',
                                    recurrent_initializer='orthogonal',
                                    bias_initializer='zeros',
                                    unit_forget_bias=True,
                                    kernel_regularizer=objRegL2,
                                    recurrent_regularizer=objRegL2,
                                    bias_regularizer=objRegL2,
                                    activity_regularizer=objRegL2,
                                    kernel_constraint=None,
                                    recurrent_constraint=None,
                                    bias_constraint=None,
                                    dropout=varInDrp,
                                    recurrent_dropout=varStDrp,
                                    implementation=1,
                                    return_sequences=False,
                                    return_state=False,
                                    go_backwards=False,
                                    stateful=False,
                                    unroll=False,
                                    name='LSTMlayer02'
                                    )(aryOut01)

    # Dense feedforward layer:
    # activity_regularizer=tf.keras.layers.ActivityRegularization(l2=0.1)
    aryOut03 = tf.keras.layers.Dense(varSzeEmb,
                                     # batch_size=varSzeBtch,
                                     activation=tf.keras.activations.tanh,
                                     name='Dense_FF_output'
                                     )(aryOut02)

    # Initialise the model:
    objMdl = tf.keras.models.Model(inputs=objTrnCtxt, outputs=aryOut03)

    # An almost idential version of the model used for testing, with different
    # input size (only one batch).
    aryOut04 = tf.keras.layers.LSTM(varNrn01,
                                    # input_shape=(varNumIn, varSzeEmb),
                                    # batch_size=1,
                                    activation='tanh',
                                    recurrent_activation='hard_sigmoid',
                                    use_bias=True,
                                    kernel_initializer='glorot_uniform',
                                    recurrent_initializer='orthogonal',
                                    bias_initializer='zeros',
                                    unit_forget_bias=True,
                                    kernel_regularizer=objRegL2,
                                    recurrent_regularizer=objRegL2,
                                    bias_regularizer=objRegL2,
                                    activity_regularizer=objRegL2,
                                    kernel_constraint=None,
                                    recurrent_constraint=None,
                                    bias_constraint=None,
                                    dropout=varInDrp,
                                    recurrent_dropout=varStDrp,
                                    implementation=1,
                                    return_sequences=True,
                                    return_state=False,
                                    go_backwards=False,
                                    stateful=True,
                                    unroll=False,
                                    name='Test_LSTMlayer01'
                                    )(objTstCtxt)

    # Second LSTM layer:
    # objMdl.add(tf.keras.layers.LSTM(varNrn02,
    aryOut05 = tf.keras.layers.LSTM(varNrn02,
                                    # input_shape=(varNumIn, varNrn01),
                                    # batch_size=1,
                                    activation='tanh',
                                    recurrent_activation='hard_sigmoid',
                                    use_bias=True,
                                    kernel_initializer='glorot_uniform',
                                    recurrent_initializer='orthogonal',
                                    bias_initializer='zeros',
                                    unit_forget_bias=True,
                                    kernel_regularizer=objRegL2,
                                    recurrent_regularizer=objRegL2,
                                    bias_regularizer=objRegL2,
                                    activity_regularizer=objRegL2,
                                    kernel_constraint=None,
                                    recurrent_constraint=None,
                                    bias_constraint=None,
                                    dropout=varInDrp,
                                    recurrent_dropout=varStDrp,
                                    implementation=1,
                                    return_sequences=False,
                                    return_state=False,
                                    go_backwards=False,
                                    stateful=True,
                                    unroll=False,
                                    name='Test_LSTMlayer02'
                                    )(aryOut04)

    # Dense feedforward layer:
    # activity_regularizer=tf.keras.layers.ActivityRegularization(l2=0.1)
    aryOut06 = tf.keras.layers.Dense(varSzeEmb,
                                     activation=tf.keras.activations.tanh
                                     )(aryOut05)

    # Initialise the model:
    objTstMdl = tf.keras.models.Model(inputs=objTstCtxt, outputs=aryOut06)

else:
    print('Loading pre-trained model from disk.')

    # Load pre-trained model from disk:
    objMdl = tf.keras.models.load_model(os.path.join(strPthMdl,
                                                     'lstm_training_model'))
    objTstMdl = tf.keras.models.load_model(os.path.join(strPthMdl,
                                                        'lstm_test_model'))

# Print model summary:
print('Training model:')
objMdl.summary()
print('Testing model:')
objTstMdl.summary()

# Define the optimiser and loss function:
objMdl.compile(optimizer=tf.keras.optimizers.RMSprop(lr=varLrnRte),
               loss=tf.keras.losses.mean_squared_error)


# -----------------------------------------------------------------------------
# *** Logging

# Get date string as default session name:
strDate = str(datetime.datetime.now())
lstD = strDate[0:10].split('-')
lstT = strDate[11:19].split(':')
strDate = (lstD[0] + lstD[1] + lstD[2] + '_' + lstT[0] + lstT[1] + lstT[2])

# Log directory:
strPthLogSes = os.path.join(strPthLog, strDate)

# Create session subdirectory:
if not os.path.exists(strPthLogSes):
    os.makedirs(strPthLogSes)

# Tf 2.0 / keras implementation does not work with train_on_batch.
# Create object for tensorboard visualisations:
# objCallback = tf.keras.callbacks.TensorBoard(log_dir=strPthLogSes,
#                                              write_graph=True,
#                                              write_images=True,
#                                              update_freq='batch')
# objCallback.set_model(objMdl)

# Old (tf 1.13) summary implementation for tensorboard:
objLogWrt = tf.summary.FileWriter(strPthLogSes)


# -----------------------------------------------------------------------------
# *** Queues

# Batches are prepared in a queue-feeding-function that runs in a separate
# thread.

def training_queue():
    """Place training data on queue."""

    # Target word counter (index of current target word in corpus):
    global varTst

    # Global logical variable for ending training:
    global varLgcGlb

    # Batch index:
    idx01 = 0

    # Optimisation step index:
    idx02 = 0

    # Array for new batch of context words:
    aryCntxt = np.zeros((varSzeBtch, varNumIn, varSzeEmb), dtype=np.float32)

    # Array for new batch of target words:
    aryTrgt = np.zeros((varSzeBtch, varSzeEmb), dtype=np.float32)

    # Array for new batch of sample weights:
    # aryWght = np.zeros((varSzeBtch, varNumIn), dtype=np.float32)
    aryWght = np.zeros((varSzeBtch), dtype=np.float32)

    # Sample weighting.
    # In order to reduce the impact of very frequent words (e.g. 'the'), sample
    # weights can be applied during training. There is one sample weight per
    # output. (Thus, the size of the sample weight vector depends only on batch
    # size, and not on the number of context words used to make a prediction,
    # because the model predicts one word at a time.) The following
    # implementation relies on the word dictionary, in which the order of words
    # corresponds to their relative frequency (i.e. the order is from the most
    # frequent word to the least frequent word). Alternatively, the actual
    # number of occurences could be used.

    # Minimum weight to use (for most frequent word):
    varWghtMin = 0.002

    # Maximum weight to use (for least frequent word):
    varWghtMax = 2.0

    # Vector with word count in corpus (returns vector with unique values,
    # which  is identical to word codes, and corresponding word counts):
    _, vecCnt = np.unique(vecC, return_counts=True)

    # Minimum number of occurences:
    vecCntMin = np.min(vecCnt)

    # Weights that are inversely proportional to number of occurences:
    vecWght = np.divide(vecCntMin, vecCnt)

    # Scale weights to respective range:
    vecWght = np.multiply(vecWght, (varWghtMax - varWghtMin))
    vecWght = np.add(varWghtMin, vecWght)

    # Exponent (slope of weighting function, higher value gives higher relative
    # weight to infrequent words):
    # varPow = 3.0

    # Weight vector:
    # vecWght = np.linspace(1.0,
    #                       0.0,
    #                       num=varNumWrds)
    # vecWght = np.power(vecWght, varPow)
    # vecWght = np.multiply(vecWght, (varWghtMax - varWghtMin))
    # vecWght = np.subtract(varWghtMax, vecWght)

    # Training schedule
    
    # In order to support convergence, first train on small sample of text,
    # and increase training text gradually.

    # Number of words to start training on:
    varNumTrnWrds = 1077

    # Initial segment of corpus to train on:
    vecTrainC = vecC[0:varNumTrnWrds]

    # Start incrementing training text length after this iterations:
    varIncAftr = 1000000

    # Loop through iterations:
    for idxFeed in range(varNumItr):

        # Increase training text size:
        if (idxFeed > varIncAftr):

            # Make sure the end of the text has not been reached yet:
            if (varLenTxt > varNumTrnWrds):
                vecTrainC = vecC[0:varNumTrnWrds]
                varNumTrnWrds += 1
 
        # Length of current training text:
        varLenTrn = vecTrainC.shape[0]
    
        # Loop through text (corpus). Index refers to target word (i.e. word
        # to be predicted).
        for idxWrd in range(varNumIn, varLenTrn):

            # Index of current target word to global variable (needed for
            # validation/status feedback):
            varTst = idxWrd
            
            # Get integer codes of context word(s):
            vecCntxt = vecTrainC[(idxWrd - varNumIn):idxWrd]

            # Get embedding vectors for context word(s):
            aryCntxt[idx01, :, :] = np.array(aryEmb[vecCntxt, :])

            # Word to predict (target):
            varTrgt = vecTrainC[idxWrd]

            # Get embedding vector for target word:
            aryTrgt[idx01, :] = aryEmb[varTrgt, :]

            # Get sample weight for current target word:
            aryWght[idx01] = vecWght[varTrgt]

            # Increment index:
            idx01 += 1

            # Check whether end of batch has been reached:
            if idx01 == int(varSzeBtch):

                # TODO

                aryTmp01 = aryCntxt
                dicIn01 = {objPlcHld01: aryTmp01}
                aryTmp02 = aryTrgt
                dicIn02 = {objPlcHld02: aryTmp02}
                aryTmp03 = aryWght
                dicIn03 = {objPlcHld03: aryTmp03}

                # Batch is complete, push to the queue:
                objSess.run(objEnQ01, feed_dict=dicIn01)
                objSess.run(objEnQ02, feed_dict=dicIn02)
                objSess.run(objEnQ03, feed_dict=dicIn03)

                # Array for new batch of context words:
                aryCntxt = np.zeros((varSzeBtch, varNumIn, varSzeEmb),
                                    dtype=np.float32)

                # Array for new batch of target words:
                aryTrgt = np.zeros((varSzeBtch, varSzeEmb),
                                   dtype=np.float32)

                # Array for new batch of sample weights:
                aryWght = np.zeros((varSzeBtch), dtype=np.float32)

                # Reset index:
                idx01 = 0

                # Count optimisation steps:
                idx02 += 1

                # Exit threat when last optimisation step has been reached:
                if idx02 == varNumOpt:

                    # Signal ending of training:
                    varLgcGlb = False
                    
                    break


# -----------------------------------------------------------------------------
# *** Fill queue

# Buffer size (number of samples to put on queue before starting
# execution of graph):
varBuff = 3

# Define & run extra thread with graph that places data on queue:
objThrd = threading.Thread(target=training_queue)
objThrd.setDaemon(True)
objThrd.start()

# Stay in this while loop until the specified number of samples
# (varBuffer) have been placed on the queue).
varTmpSzeQ = 0
while varTmpSzeQ < varBuff:
    varTmpSzeQ = objSess.run(objSzeQ)


# -----------------------------------------------------------------------------
# *** Embedding lookup


def emb_lookup(aryEmb, dictRvrs, aryWrdVec):
    """
    Vector to word lookup based on embedding matrix.

    Parameters
    ----------
    aryEmb : np.array
        Embedding matrix (dimensions aryEmb[words, dimensions]).
    dictRvrs : dict
        Reverse dictionary (where keys are word order).
    aryWrdVec : np.array
        Array of word vectors for which to look up words (dimensions
        aryWrdVec[words, dimensions]).

    Returns
    -------
    lstWrds : list
        List of strings with words closest to input vectors.
    aryDev : np.array
        Deviationbetween input word vectors and closest word vector in
        embedding matrix (i.e. deviation between looked up word and input
        vector).
    """
    # Number of input words to look up:
    varNumLook = aryWrdVec.shape[0]

    # Dimensionality of embedding:
    varNumDim = aryWrdVec.shape[1]

    # List of looked up words:
    lstWrds = [None] * varNumLook

    # Array for deviations:
    vecDev = np.zeros((varNumLook, varNumDim), dtype=np.float32)

    # Loop through words to look up:
    for idxLk in range(varNumLook):

        # Minimum squared deviation between prediciton and embedding
        # vectors:
        vecDiff = np.sum(
                         np.square(
                                   np.subtract(
                                               aryEmb,
                                               aryWrdVec[idxLk, :]
                                               )
                                   ),
                         axis=1
                         )

        # Get code of closest word vector:
        varTmp = int(np.argmin(vecDiff))

        # Look up predicted word in dictionary:
        lstWrds[idxLk] = dictRvrs[varTmp]

        # Deviation between input word vector and looked up word:
        vecDev[idxLk] = np.min(vecDiff)

    return lstWrds, vecDev


# -----------------------------------------------------------------------------
# *** Training

# Loop through optimisation steps (one batch per optimisation step):
for idxOpt in range(varNumOpt):

    # Give status feedback or train next batch.

    # Status feedback:
    if (idxOpt % varDspStp == 0):

        # ** Give status feedback

        print('---')

        print(('Optimisation step: '
               + str(idxOpt)
               + ' out of '
               + str(varNumOpt)))


        print(('                   '
               + str(np.around((float(idxOpt) / float(varNumOpt) * 100))
                     ).split('.')[0]
               + '%'))

        # Avoid beginning of text (not enough preceding context words):
        # if varTst > varNumIn:
        if varTst > 50:

            # Copy weights from training model to test model:
            objTstMdl.set_weights(objMdl.get_weights())

            # +++
            varLenStt = 50
            # Get integer codes of context word(s) for new state:
            vecSttCtxt = vecC[(varTst - varLenStt):varTst]
            # Get embedding vectors for context word(s):
            arySttCtxt = np.array(aryEmb[vecSttCtxt, :]
                                  ).reshape(1, varLenStt, varSzeEmb)
            # TODO only works with varNumIn = 1
            for idxStt in range(varLenStt):
                _ = objTstMdl.predict_on_batch(arySttCtxt[:, idxStt, :].reshape(1, 1, varSzeEmb))
            # +++

            # Get integer codes of context word(s):
            vecTstCtxt = vecC[(varTst - varNumIn):varTst]

            # Get embedding vectors for context word(s):
            aryTstCtxt = np.array(aryEmb[vecTstCtxt, :]
                                  ).reshape(1, varNumIn, varSzeEmb)

            # Word to predict (target):
            varTrgt = vecC[varTst]

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

            print(('Loss auto:   ' + str(varLoss01)))
            print(('Loss manual: ' + str(varLoss02)))

            # Context:
            lstCtxt = [dictRvrs[x] for x in vecC[(varTst - 15):varTst]]
            strCtxt = ' '.join(lstCtxt)

            print(('Context: ' + strCtxt))

            print(('Target: ' + dictRvrs[vecC[varTst]]))

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

                # Place testing context word(s) on queue:
                # testing_queue(aryTstCtxt)

                # Update context (leave out first - i.e. oldest - word in
                # context, and append newly predicted word):
                if varNumIn > 1:
                    aryTstCtxt = np.concatenate((aryTstCtxt[:, 1:, :],
                                                 vecWrd.reshape(1, 1, varSzeEmb)
                                                 ), axis=1)
                else:
                    aryTstCtxt = vecWrd.reshape(1, 1, varSzeEmb)

                # Get test prediction for current context word(s):
                vecWrd = objTstMdl.predict_on_batch(aryTstCtxt)

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

                # Save code of predicted word:
                vecNew[idxNew] = varTmp

            # Decode newly generated words:
            lstNew = [dictRvrs[x] for x in vecNew]

            # List to string:
            strNew = ' '.join(lstNew)
            print('New text:')
            print(strNew)

            # Reset model states:
            # objMdl.reset_states()
            # objTstMdl.reset_states()

    else:

        # Run optimisation:
        # objMdl.fit(x=objTrnCtxt,
        #            y=objTrgt,
        #            epochs=10,
        #            shuffle=False,
        #            steps_per_epoch=1,
        #            verbose=0)
        #          callbacks=[objCallback])
        varLoss01 = objMdl.train_on_batch(objTrnCtxt,  # run on single batch
                                          y=objTrgt,
                                          sample_weight=objWght)

        if (idxOpt % 50 == 0):
            # Use old (tf 1.13) implementation for writing loss to tensorboard
            # summary.
            objSmry = tf.Summary(value=[tf.Summary.Value(tag="loss",
                simple_value=varLoss01), ])
            objLogWrt.add_summary(objSmry, global_step=idxOpt)

        #objMdl.reset_states()

        # Increment test word index:
        # varTst = varTst + varSzeBtch
        # if varTst >= varLenTxt:
        #     varTst = varTst - varLenTxt

# objMdl.evaluate(x_test, y_test)

# Get model weights:
lstWghts = objMdl.get_weights()

# print('len(lstWghts)')
# print(len(lstWghts))

# Save model weights and training parameters to disk:
np.savez(os.path.join(strPthLogSes, 'lstm_data.npz'),
         varLrnRte=varLrnRte,
         varNumItr=varNumItr,
         varNumIn=varNumIn,
         varNrn01=varNrn01,
         varSzeEmb=varSzeEmb,
         varSzeBtch=varSzeBtch,
         varInDrp=varInDrp,
         varStDrp=varStDrp,
         lstWghts=lstWghts,
         )

# Save model to disk:
tf.keras.models.save_model(objMdl,
                           os.path.join(strPthLogSes, 'lstm_training_model'))
tf.keras.models.save_model(objTstMdl,
                           os.path.join(strPthLogSes, 'lstm_test_model'))


# -----------------------------------------------------------------------------
# *** Validation - generate new text

## TODO: Does running data through model with model.predict_on_batch actually
## change the state of the LSTM?
#
## Load text to base new predictions on:
#lstBase = read_text(strPthBse)
#
## Base text to code:
#varLenBse = len(lstBase)
#vecBase = np.zeros(varLenBse, dtype=np.int32)
#
## Loop through base text:
#for idxWrd in range(varLenBse):
#
#    # Try to look up words in dictionary. If there is an unkown words, replace
#    # with unknown token.
#    try:
#        varTmp = dicWdCnOdr[lstBase[idxWrd].lower()]
#    except KeyError:
#        varTmp = 0
#    vecBase[idxWrd] = varTmp
#
#
## Get embedding vectors for words:
#aryBase = np.array(aryEmb[vecBase, :])
#
#for idxWrd in range(varLenBse):
#
#    # Get prediction for current word:
#    vecWrd = objMdl.predict_on_batch(aryBase[idxWrd, :].reshape(1, 1, varSzeEmb))  # TODO: only works with input size one
#
## Vector for new text (coded):
#vecNew = np.zeros(varLenNewTxt, dtype=np.int32)
#
## Generate new text:
#for idxNew in range(varLenNewTxt):
#
#    # Get prediction for current word:
#    vecWrd = objMdl.predict_on_batch(vecWrd.reshape(1, 1, varSzeEmb))  # TODO: only works with input size one
#
#    # Minimum squared deviation between prediciton and embedding
#    # vectors:
#    vecDiff = np.sum(
#                     np.square(
#                               np.subtract(
#                                           aryEmb,
#                                           vecWrd[None, :]
#                                           )
#                               ),
#                     axis=1
#                     )
#
#    # Get code of closest word vector:
#    varTmp = int(np.argmin(vecDiff))
#
#    # Save code of predicted word:
#    vecNew[idxNew] = varTmp
#
## Decode newly generated words:
#lstNew = [dictRvrs[x] for x in vecNew]
#
## List to string:
#strBase = ' '.join(lstBase)
#strNew = ' '.join(lstNew)
#
#print('---')
#print('Base text:')
#print(strBase)
#print('---')
#print('New text:')
#print(strNew)
#print('---')
