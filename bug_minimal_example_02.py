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

# Learning rate:
varLrnRte = 0.0001

# Number of optimisation steps:
varNumOpt = 10000

# Display steps (after x number of optimisation steps):
varDspStp = 100

# Number of input words from which to predict next word:
varNumIn = 1

# Number of neurons in first hidden layer:
varNrn01 = 10

# Length of new text to generate:
varLenNewTxt = 100

# Batch size:
varSzeBtch = 1

# Input dropout:
# varInDrp = 0.3

# Recurrent state dropout:
# varStDrp = 0.3


# -----------------------------------------------------------------------------
# *** Preparations

# Create tf session:
objSess = tf.Session()

# Tell keras about tf session:
tf.keras.backend.set_session(objSess)

# Sequence to learn:
vecS = np.array([-1.0, 1.0] * 100, dtype=np.float32)

# Length of sequence:
varLenS = vecS.shape[0]

objTrnCtxtA = tf.keras.Input(shape=(varNumIn, 1),
                             batch_size=varSzeBtch,
                             # tensor=objQ01.dequeue(),
                             dtype=tf.float32)

objTstCtxt = tf.keras.Input(shape=(varNumIn, 1),
                            batch_size=varSzeBtch,
                            # tensor=objQ01.dequeue(),
                            dtype=tf.float32)

# Dummy  sample weights:
vecSmpWgt = np.ones(varSzeBtch, dtype=np.float32)


# -----------------------------------------------------------------------------
# *** Build the network

# Adjust model's statefullness according to batch size:
if varSzeBtch == 1:
    print('Stateful training model.')
    lgcState = True
else:
    print('Stateless training model.')
    lgcState = False

# Regularisation:
objRegL2 = None

# The actual LSTM layers.
# aryOut01 = tf.keras.layers.CuDNNLSTM(varNrn01,
aryOut01 = tf.keras.layers.LSTM(varNrn01,
                                # activation='tanh',
                                # recurrent_activation='hard_sigmoid',
                                kernel_regularizer=objRegL2,
                                recurrent_regularizer=objRegL2,
                                bias_regularizer=objRegL2,
                                activity_regularizer=objRegL2,
                                return_sequences=False,
                                return_state=False,
                                go_backwards=False,
                                stateful=lgcState,
                                name='LSTM01'
                                )(objTrnCtxtA)
# aryOut01D = tf.keras.layers.Dropout(varInDrp)(aryOut01)

# Dense feedforward layer:
aryOut02 = tf.keras.layers.Dense(1,
                                 activation=tf.keras.activations.tanh,
                                 name='Dense_FF'
                                 )(aryOut01)

# Initialise the model:
objMdl = tf.keras.models.Model(inputs=objTrnCtxtA,
                               outputs=aryOut02)  # [aryOut06, aryOut06])

# An almost idential version of the model used for testing, without dropout
# and possibly different input size (fixed batch size of one).
aryOut01T = tf.keras.layers.LSTM(varNrn01,
                                # activation='tanh',
                                # recurrent_activation='hard_sigmoid',
                                kernel_regularizer=objRegL2,
                                recurrent_regularizer=objRegL2,
                                bias_regularizer=objRegL2,
                                activity_regularizer=objRegL2,
                                return_sequences=False,
                                return_state=False,
                                go_backwards=False,
                                stateful=True,
                                name='TestLSTM01'
                                )(objTstCtxt)
# aryOut01D = tf.keras.layers.Dropout(varInDrp)(aryOut01)

# Dense feedforward layer:
aryOut02T = tf.keras.layers.Dense(1,
                                  activation=tf.keras.activations.tanh,
                                  name='TestDense_FF'
                                  )(aryOut01T)

# Initialise the model:
objTstMdl = tf.keras.models.Model(inputs=objTstCtxt, outputs=aryOut02T)

# Print model summary:
print('Training model:')
objMdl.summary()
print('Testing model:')
objTstMdl.summary()

#def prediction_loss(objTrgt, aryOut06):
#    return tf.reduce_mean(tf.math.squared_difference(objTrgt, aryOut06))

def repetition_loss(objTrnCtxtB, aryOut06):
    return tf.math.log(tf.math.add(tf.math.divide(1.0, tf.reduce_mean(tf.math.squared_difference(objTrnCtxtA, aryOut02))), 1.0))

# Define the optimiser and loss function:
objMdl.compile(optimizer=tf.keras.optimizers.Adam(lr=varLrnRte),  # Or use RMSprop?
               loss=repetition_loss)  # [prediction_loss, repetition_loss])
# loss_weights=[1.0, 0.7]


# -----------------------------------------------------------------------------
# *** Training

print('--> Beginning of training.')

# Sample index:
idxSmp = 0

# Loop through optimisation steps (one batch per optimisation step):
for idxOpt in range(varNumOpt):

    aryIn = vecS[idxSmp].reshape(varSzeBtch, varNumIn, 1)
    aryOut = vecS[(idxSmp + 1)].reshape(varSzeBtch, varNumIn, 1)
    
    varLoss = objMdl.train_on_batch(aryIn,  # run on single batch
                                    y=aryOut,
                                    sample_weight=vecSmpWgt)

    if varLenS > (idxSmp + 1):
        idxSmp += 1
    else:
        idxSmp = 0


    # Give status feedback:
    if (idxOpt % varDspStp == 0):

        print('---')

        print(('Optimisation step: '
               + str(idxOpt)
               + ' out of '
               + str(varNumOpt)))

        print(('                   '
               + str(np.around((float(idxOpt) / float(varNumOpt) * 100))
                     ).split('.')[0]
               + '%'))

        # Length of context to use to initialise the state of the prediction
        # model:
        varLenCntx = 5

        # Avoid beginning of text (not enough preceding context words):
        if idxSmp > varLenCntx:

            # Copy weights from training model to test model:
            objTstMdl.set_weights(objMdl.get_weights())

            # If the training model is stateless, initialise state of the
            # (statefull) prediction model with context. This assumes that
            # the model can be stateful during prediction (which is should be
            # according to the documentation).
            if not lgcState:
                objTstMdl.reset_states()
                # Loop through context window:
                for idxCntx in range(1, varLenCntx):
                    # Get integer code of context word (the '- 1' is so as not
                    # to predict twice on the word right before the target
                    # word, see below):
                    varCtxt = vecS[(idxSmp - 1 - varLenCntx + idxCntx)]
                    # Get embedding vectors for context word(s):
                    aryCtxt = np.array(vecS[varCtxt]
                                       ).reshape(1, varNumIn, 1)
                    # Predict on current context word:
                    vecWrd = objTstMdl.predict_on_batch(aryCtxt)

            # Word to predict (target):
            varTrgt = vecS[(idxSmp + 1)]

            # Get embedding vectors for context word(s):
            aryTstCtxt = np.array(vecS[(idxSmp)]
                                  ).reshape(1, varNumIn, 1)

            # Get test prediction for current context word(s):
            vecWrd = objTstMdl.predict_on_batch(aryTstCtxt)

            #objSmry = objSess.run(objMrgSmry,
            #                      feed_dict={objPlcPredWrd: vecWrd.flatten()})
            #objLogWrt.add_summary(objSmry, global_step=idxOpt)

            print(('Loss auto:   ' + str(varLoss)))

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
        print('Resetting model states.')
        objMdl.reset_states()
        objTstMdl.reset_states()

print('--> End of training.')
