#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal example re custom loss function bug.

Update: I initially assumed that something using a custom loss function did not
        work in 1.14.0 (but in 1.13.1). However, this suspicion was wrong; the
        below model does converge in both tf versions.
"""


import numpy as np
import tensorflow as tf


# -----------------------------------------------------------------------------
# *** Define parameters

# Learning rate:
varLrnRte = 0.01

# Number of optimisation steps:
varNumOpt = 10000

# Display steps (after x number of optimisation steps):
varDspStp = 1000

# Number of neurons in first hidden layer:
varNrn01 = 1

# Length of new text to generate:
varLenNew = 100

# Batch size:
varSzeBtch = 1


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

objTrnCtxtA = tf.keras.Input(shape=(1, 1),
                             batch_size=varSzeBtch,
                             # tensor=objQ01.dequeue(),
                             dtype=tf.float32)
objTrnCtxtB = tf.keras.Input(shape=(varSzeBtch, 1),
                             batch_size=varSzeBtch,
                             # tensor=objQ04.dequeue(),
                             dtype=tf.float32)

objTstCtxt = tf.keras.Input(shape=(1, 1),
                            batch_size=1,
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

# The actual LSTM layers.
aryOut01 = tf.keras.layers.LSTM(varNrn01,
                                return_sequences=True,
                                return_state=False,
                                go_backwards=False,
                                stateful=lgcState,
                                name='LSTM01'
                                )(objTrnCtxtA)

aryOut02 = tf.keras.layers.LSTM(varNrn01,
                                return_sequences=False,
                                return_state=False,
                                go_backwards=False,
                                stateful=lgcState,
                                name='LSTM02'
                                )(aryOut01)

# Dense feedforward layer:
aryOut03 = tf.keras.layers.Dense(1,
                                 activation=tf.keras.activations.tanh,
                                 name='Dense_FF'
                                 )(aryOut02)

# Initialise the model:
objMdl = tf.keras.models.Model(inputs=objTrnCtxtA,
                               outputs=[aryOut03, aryOut03])

# An almost idential version of the model used for testing, without dropout
# and possibly different input size (fixed batch size of one).
# The actual LSTM layers.
aryOut01T = tf.keras.layers.LSTM(varNrn01,
                                 return_sequences=True,
                                 return_state=False,
                                 go_backwards=False,
                                 stateful=True,
                                 name='Test_LSTM01'
                                 )(objTstCtxt)

aryOut02T = tf.keras.layers.LSTM(varNrn01,
                                 return_sequences=False,
                                 return_state=False,
                                 go_backwards=False,
                                 stateful=True,
                                 name='Test_LSTM02'
                                 )(aryOut01T)

# Dense feedforward layer:
aryOut03T = tf.keras.layers.Dense(1,
                                  activation=tf.keras.activations.tanh,
                                  name='Test_Dense_FF'
                                  )(aryOut02T)

# Initialise the model:
objTstMdl = tf.keras.models.Model(inputs=objTstCtxt, outputs=aryOut03T)

# Print model summary:
print('Training model:')
objMdl.summary()
print('Testing model:')
objTstMdl.summary()

def prediction_loss(objTrgt, aryOut03):
    return tf.reduce_mean(tf.math.squared_difference(objTrgt, aryOut03))

def repetition_loss(objTrnCtxtB, aryOut03):
    return tf.math.abs(tf.math.add(objTrnCtxtB, aryOut03))

# Define the optimiser and loss function:
objMdl.compile(optimizer=tf.keras.optimizers.Adam(lr=varLrnRte),
               loss=[prediction_loss, repetition_loss])  # [prediction_loss, repetition_loss])
# loss_weights=[1.0, 0.7]


# -----------------------------------------------------------------------------
# *** Training

print('--> Beginning of training.')

# Sample index (refering to target):
idxSmp = 1

varLenS = 117

vecWght = np.ones(varSzeBtch, dtype=np.float32)

# Loop through optimisation steps:
for idxOpt in range(varNumOpt):

    # Create batch:
    aryIn = np.zeros((varSzeBtch, 1, 1), dtype=np.float32)
    aryOut = np.zeros((varSzeBtch, 1, 1), dtype=np.float32)
    for idxBtch in range(varSzeBtch):

        # print((idxSmp - 1))
        aryIn[idxBtch, 0, 0] = vecS[(idxSmp - 1)]
        aryOut[idxBtch, 0, 0] = vecS[idxSmp]

        # Increment / reset sample coutner:
        idxSmp += 1
        if idxSmp >= varLenS:
            idxSmp = 1

    varLoss = objMdl.train_on_batch(aryIn,
                                    y=[aryOut.reshape(varSzeBtch, 1),
                                       aryIn.reshape(varSzeBtch, 1)],
                                    sample_weight=[vecWght, vecWght])

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
                    # Get context (the '- 1' is so as not to predict twice on
                    # the sample right before the target  word, see below):
                    varCtxt = vecS[(idxSmp - 1 - varLenCntx + idxCntx)]
                    # Get embedding vectors for context word(s):
                    aryCtxt = np.array(varCtxt).reshape(1, 1, 1)
                    # Predict on current context word:
                    vecPrd = objTstMdl.predict_on_batch(aryCtxt)

            # Context for validation prediction:
            varCtxt = vecS[(idxSmp - 1)]
            aryTstCtxt = np.array(varCtxt).reshape(1, 1, 1)

            # Get test prediction for current context:
            vecPrd = objTstMdl.predict_on_batch(aryTstCtxt)

            #objSmry = objSess.run(objMrgSmry,
            #                      feed_dict={objPlcPredWrd: vecWrd.flatten()})
            #objLogWrt.add_summary(objSmry, global_step=idxOpt)

            print(('Loss auto:   ' + str(varLoss)))

            # Context:
            lstCtxt = [str(x) for x in vecS[(idxSmp - 15):idxSmp]]
            strCtxt = ' '.join(lstCtxt)
            print(('Context: ' + strCtxt))

            # Sample to predict (target):
            varTrgt = vecS[idxSmp]
            print(('Target: ' + str(varTrgt)))

            print(('Prediction: ' + str(vecPrd[0][0])))

            # ** Generate new text

            vecNew = np.zeros(varLenNew, dtype=np.float32)

            # Generate new text:
            for idxNew in range(varLenNew):

                # Update context (leave out first - i.e. oldest - word in
                # context, and append newly predicted word):
                aryTstCtxt = vecPrd.reshape(1, 1, 1)

                # Get test prediction for current context word(s):
                vecPrd = objTstMdl.predict_on_batch(aryTstCtxt)

                # Save code of predicted word:
                vecNew[idxNew] = vecPrd[0][0]

            # Newly generated numbers to list:
            lstNew = [str(x) for x in vecNew]

            # List to string:
            strNew = ' '.join(lstNew)
            print('New text:')
            print(strNew)

        # Reset model states:
        print('Resetting model states.')
        objMdl.reset_states()
        objTstMdl.reset_states()

print('--> End of training.')
