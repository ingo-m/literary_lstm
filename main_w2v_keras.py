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

# Path of input data file:
strPthIn = '/home/john/Dropbox/Ernest_Hemingway/redacted/word2vec_data_100.npz'

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
varNumIn = 1

# Number of neurons in first hidden layer:
varNrn01 = 100

# Length of new text to generate:
varLenNewTxt = 100

# Batch size:
varSzeBtch = 5000

# Input dropout:
varInDrp = 0.2

# Recurrent state dropout:
varStDrp = 0.0


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

# Number of neurons in second hidden layer:
varNrn02 = varSzeEmb

# Total number of inputs (to input layer):
# varNumInTtl = varNumIn * varSzeEmb

# Placeholder for inputs (to input layer):
#aryWrdsIn = tf.keras.Input(shape=(varNumIn, varSzeEmb),
#                           batch_size=1,
#                           dtype=tf.float32)

# Placeholder for output:
#vecWrdsOut = tf.placeholder(tf.float32, [1, varSzeEmb])


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

## Queue for testing batch, context words (one batch only):
#objQ03 = tf.FIFOQueue(capacity=1,
#                      dtypes=[tf.float32],
#                      shapes=[(1, varNumIn, varSzeEmb)])

# Method for getting queue size:
objSzeQ = objQ01.size()

# Placeholder that is used to put batch on computational graph:
objPlcHld01 = tf.placeholder(tf.float32,
                             shape=[varSzeBtch, varNumIn, varSzeEmb])
objPlcHld02 = tf.placeholder(tf.float32,
                             shape=[varSzeBtch, varSzeEmb])
#objPlcHld03 = tf.placeholder(tf.float32,
#                             shape=[1, varNumIn, varSzeEmb])

# The enqueue operation that puts data on the graph.
objEnQ01 = objQ01.enqueue([objPlcHld01])
objEnQ02 = objQ02.enqueue([objPlcHld02])
#objEnQ03 = objQ03.enqueue([objPlcHld03])

# Number of threads that will be created per queue:
varNumThrd = 1

# The queue runner (places the enqueue operation on the queue?).
objRunQ01 = tf.train.QueueRunner(objQ01, [objEnQ01] * varNumThrd)
tf.train.add_queue_runner(objRunQ01)
objRunQ02 = tf.train.QueueRunner(objQ02, [objEnQ02] * varNumThrd)
tf.train.add_queue_runner(objRunQ02)
#objRunQ03 = tf.train.QueueRunner(objQ03, [objEnQ03] * 1)
#tf.train.add_queue_runner(objRunQ03)

# The tensor object that is retrieved from the queue. Functions like
# placeholders for the data in the queue when defining the graph.
# Training context placebolder (input):
objTrnCtxt = tf.keras.Input(shape=(varSzeBtch, varNumIn, varSzeEmb),
                            tensor=objQ01.dequeue(),
                            dtype=tf.float32)
# Training target placeholder:
objTrgt = objQ02.dequeue()
# Testing context placeholder:
objTstCtxt = tf.keras.Input(shape=(varNumIn, varSzeEmb),
                            batch_size=1,
                            dtype=tf.float32)


# -----------------------------------------------------------------------------
# *** Build the network

# Initialise the model:
# objMdl = tf.keras.models.Sequential()

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
                                kernel_regularizer=None,
                                recurrent_regularizer=None,
                                bias_regularizer=None,
                                activity_regularizer=None,
                                kernel_constraint=None,
                                recurrent_constraint=None,
                                bias_constraint=None,
                                dropout=varInDrp,
                                recurrent_dropout=varStDrp,
                                implementation=1,
                                return_sequences=True,  # ?
                                return_state=False,
                                go_backwards=False,
                                stateful=True,
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
                                kernel_regularizer=None,
                                recurrent_regularizer=None,
                                bias_regularizer=None,
                                activity_regularizer=None,
                                kernel_constraint=None,
                                recurrent_constraint=None,
                                bias_constraint=None,
                                dropout=varInDrp,
                                recurrent_dropout=varStDrp,
                                implementation=1,
                                return_sequences=False,  # ?
                                return_state=False,
                                go_backwards=False,
                                stateful=True,
                                unroll=False,
                                name='LSTMlayer02'
                                )(aryOut01)

# Dense feedforward layer:
# objMdl.add(tf.keras.layers.Dense(1))

# Initialise the model:
objMdl = tf.keras.models.Model(inputs=objTrnCtxt, outputs=aryOut02)

# Inidialise copy of the model used for testing, with different input size
# (only one batch).

aryOut03 = tf.keras.layers.LSTM(varNrn01,
                                # input_shape=(varNumIn, varSzeEmb),
                                # batch_size=1,
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
                                dropout=varInDrp,
                                recurrent_dropout=varStDrp,
                                implementation=1,
                                return_sequences=True,  # ?
                                return_state=False,
                                go_backwards=False,
                                stateful=True,
                                unroll=False,
                                name='Test_LSTMlayer01'
                                )(objTstCtxt)

# Second LSTM layer:
# objMdl.add(tf.keras.layers.LSTM(varNrn02,
aryOut04 = tf.keras.layers.LSTM(varNrn02,
                                # input_shape=(varNumIn, varNrn01),
                                # batch_size=1,
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
                                dropout=varInDrp,
                                recurrent_dropout=varStDrp,
                                implementation=1,
                                return_sequences=False,  # ?
                                return_state=False,
                                go_backwards=False,
                                stateful=True,
                                unroll=False,
                                name='Test_LSTMlayer02'
                                )(aryOut03)

objTstMdl = tf.keras.models.Model(inputs=objTstCtxt, outputs=aryOut04)

# Print model summary:
print('Training model:')
objMdl.summary()
print('Testing model:')
objTstMdl.summary

# Define the optimiser and loss function:
objMdl.compile(optimizer=tf.keras.optimizers.RMSprop(lr=varLrnRte),
               loss=tf.losses.mean_squared_error,
               metrics=['accuracy'])


# -----------------------------------------------------------------------------
# *** Queues

# Batches are prepared in a queue-feeding-function that runs in a separate
# thread.

def training_queue():
    """Place training data on queue."""

    # Batch index:
    idx01 = 0

    # Array for new batch of context words:
    aryCntxt = np.zeros((varSzeBtch, varNumIn, varSzeEmb), dtype=np.float32)

    # Array for new batch of target words:
    aryTrgt = np.zeros((varSzeBtch, varSzeEmb), dtype=np.float32)

    # varNumBtch = int(np.ceil(float(varLenTxt) / float(varSzeBtch)))

    # Loop through iterations:
    for idxItr in range(varNumItr):

        # Loop through text (corpus). Index refers to target word (i.e. word
        # to be predicted).
        for idxWrd in range(varNumIn, varLenTxt):

            # Get integer codes of context word(s):
            vecCntxt = vecC[(idxWrd - varNumIn):idxWrd]

            # Get embedding vectors for context word(s):
            aryCntxt[idx01, :, :] = np.array(aryEmb[vecCntxt, :])

            # Word to predict (target):
            varTrgt = vecC[idxWrd]

            # Get embedding vector for target word:
            aryTrgt[idx01, :] = aryEmb[varTrgt, :]

            # Increment index:
            idx01 += 1

            # Check whether end of batch has been reached:
            if idx01 == int(varSzeBtch):

                # TODO

                aryTmp01 = aryCntxt
                dicIn01 = {objPlcHld01: aryTmp01}
                aryTmp02 = aryTrgt
                dicIn02 = {objPlcHld02: aryTmp02}

                # Batch is complete, push to the queue:
                objSess.run(objEnQ01, feed_dict=dicIn01)
                objSess.run(objEnQ02, feed_dict=dicIn02)

                # Array for new batch of context words:
                aryCntxt = np.zeros((varSzeBtch, varNumIn, varSzeEmb),
                                    dtype=np.float32)

                # Array for new batch of target words:
                aryTrgt = np.zeros((varSzeBtch, varSzeEmb),
                                   dtype=np.float32)

                # Reset index:
                idx01 = 0


#def testing_queue(aryTstCntxt):
#    """Place testing data on queue."""
#    #
#    aryTmp03 = aryTstCntxt
#    dicIn03 = {objPlcHld03: aryTmp03}
#
#    # Batch is complete, push to the queue:
#    objSess.run(objEnQ03, feed_dict=dicIn03)


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
# *** Training

# Loop through iterations:
for idxItr in range(varNumItr):

    # Random number for status feedback:
    varRndm = random.randint(20, (varLenTxt - 20))

    # List for word indices to loop over, in randomised order:
    # lstRnd = list(range(varNumIn, varLenTxt))
    # random.shuffle(lstRnd)

    # Loop through text (corpus). Index refers to target word (i.e. word
    # to be predicted).
    for idxWrd in range(varNumIn, varLenTxt):

        # Give status feedback or train next batch.

        # Status feedback:
        if (idxItr % varDspStp == 0) and (idxWrd == varRndm):

            # ** Give status feedback

            print('---')

            print(('Iteration: ' + str(idxItr)))

            # Copy weights from training model to test model:
            objTstMdl.set_weights(objMdl.get_weights())

            # Get integer codes of context word(s):
            vecTstCtxt = vecC[(idxWrd - varNumIn):idxWrd]

            # Get embedding vectors for context word(s):
            aryTstCtxt = np.array(aryEmb[vecTstCtxt, :]
                                  ).reshape(1, varNumIn, varSzeEmb)

            # Word to predict (target):
            varTrgt = vecC[idxWrd]

            # Get embedding vector for target word:
            aryTstTrgt = aryEmb[varTrgt, :].reshape(1, varSzeEmb)

            # Get test prediction for current context word(s):
            vecWrd = objTstMdl.predict_on_batch(aryTstCtxt)

            print('vecWrd.shape')
            print(vecWrd.shape)

            # Current loss:
            varLoss02 = np.sum(
                               np.square(
                                         np.subtract(
                                                     aryTstTrgt,
                                                     vecWrd
                                                     )
                                         )
                               )

            # print(('Loss auto:   ' + str(varLoss01)))
            print(('Loss manual: ' + str(varLoss02)))

            print(('Context: '
                   + str([dictRvrs[x] for x in vecC[(idxWrd - 15):idxWrd]])))

            print(('Target: '
                   + dictRvrs[vecC[idxWrd]]))

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

            # Generate new text:
            for idxNew in range(varLenNewTxt):

                # Place testing context word(s) on queue:
                # testing_queue(aryTstCtxt)

                # Get test prediction for current context word(s):
                vecWrd = objTstMdl.predict_on_batch(aryTstCtxt)

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
                aryTstCtxt = np.concatenate((aryTstCtxt[:, 1:, :],
                                             vecWrd.reshape(1, 1, varSzeEmb)
                                             ), axis=1)

            # Decode newly generated words:
            lstNew = [dictRvrs[x] for x in vecNew]

            # List to string:
            strNew = ' '.join(lstNew)
            print('New text:')
            print(strNew)

        else:

            # Run optimisation:
            #objMdl.fit(x=aryCntxt,  # run on entire dataset
            #           y=vecTrgt,
            #           verbose=0,
            #           epochs=1)
            #           # callbacks=[objCallback])
            varLoss01 = objMdl.train_on_batch(objTrnCtxt,  # run on single batch
                                              y=objTrgt)


# objMdl.evaluate(x_test, y_test)


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
