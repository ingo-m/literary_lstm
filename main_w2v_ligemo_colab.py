# --- Cell ---

# Check which hardware is available:
!nvidia-smi

# --- Cell ---

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Literature generating model.

At the core of the model there is a memory matrix. A single weight vector is
used to control reading, erasing, and writing to/from the memory matrix.
Separate recurrent layers control the content of the erase and write vectors.
The text is encoded by means of word-level embedding (word2vec).
"""


import os
import random
import datetime
import threading
import queue
import time
import numpy as np
import tensorflow as tf
# from memory_module import MeLa


# -----------------------------------------------------------------------------
# *** Define parameters

# Path of input data file (containing text and word2vec embedding):
strPthIn = 'drive/My Drive/word2vec_data_all_books_e300_w5000.npz'

# Path of npz file containing previously trained model's weights to load (if
# None, new model is created):
strPthMdl = 'drive/My Drive/ligemo_log/20191127_092038/ligemo_data.npz'

# Log directory (parent directory, new session directory will be created):
strPthLog = 'drive/My Drive/ligemo_log'

# Number of neurons:
varNrn01 = 512

# Number of memory locations:
varNumMem = 1024

# Size of memory locations:
varSzeMem = 512

# Learning rate:
varLrnRte = 0.0000001

# Number of training iterations over the input text:
varNumItr = 1.3  # K80: 1.3, P100: 2.3

# Display steps (after x number of optimisation steps):
varDspStp = 10000

# Reset internal model states after x number of optimisation steps:
varResStp = 1000

# Length of new text to generate:
varLenNewTxt = 200

# Batch size:
varSzeBtch = 64

# Input dropout:
varInDrp = 0.5

# Recurrent state dropout:
varStDrp = 0.5

# Memory dropbout:
varMemDrp = 0.4

# Number of words from which to sample next word (n most likely words) when
# generating new text. This parameter has no effect during training, but during
# validation.
varNumWrdSmp = 100

# Exponent to apply over likelihoods of predictions. Higher value biases the
# selection towards words predicted with high likelihood, but leads to
# repetitive sequences of frequent words. Lower value biases selection towards
# less frequent words and breaks repetitive sequences, but leads to incoherent
# sequences without gramatical structure or semantic meaning.
varTemp = 1.5


# -----------------------------------------------------------------------------
# *** Use GPU if available:

try:
    from tensorflow.python.client import device_lib
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    print(('--> Using device: ' + gpus[0].name))
    lgcGpu = True
except AttributeError:
    lgcGpu = False


# -----------------------------------------------------------------------------
# *** Load data

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

# --- Cell ---

# Shortcut for activation functions:
tanh = tf.keras.activations.tanh
sigmoid = tf.keras.activations.sigmoid
softmax = tf.keras.activations.softmax
relu = tf.keras.activations.relu


class MeLa(tf.keras.layers.Layer):
    """
    Memory module for literature generating model.

    At the core of the module there is a memory matrix. A single weight vector
    is used to control reading, erasing, and writing to/from the memory matrix.
    Separate recurrent layers control the content of the erase and write
    vectors. The text (input) is encoded by means of word-level embedding
    (word2vec).
    """

    def __init__(self,  # noqa
                 batch_size=None,
                 emb_size=None,
                 units_01=None,
                 mem_locations=None,
                 mem_size=None,
                 drop_in=None,
                 drop_state=None,
                 drop_mem=None,
                 name='LiGeMo',
                 **kwargs):
        super(MeLa, self).__init__(name=name, **kwargs)

        # ** Model parameters **
        self.lgm_batch_size = batch_size
        self.lgm_mem_locations = mem_locations
        self.lgm_mem_size = mem_size

        # ** Layers **

        # Input feedforward module:
        self.drop1 = tf.keras.layers.Dropout(drop_in)
        self.d1 = tf.keras.layers.Dense(units_01,
                                        input_shape=(batch_size,
                                                     1,
                                                     emb_size),
                                        activation=tanh,
                                        name='dense_in')

        # Layer controlling weight vector:
        self.mem_weights = tf.keras.layers.Dense(mem_locations,
                                                 activation=softmax,
                                                 name='memory_weights')

        # Layer controlling content of write vector:
        self.write = tf.keras.layers.Dense(mem_size,
                                           activation=softmax,
                                           name='memory_write')

        # Layer controlling content of erase vector:
        self.erase = tf.keras.layers.Dense(mem_size,
                                           activation=softmax,
                                           name='memory_erase')

        # Output feedforward module:
        self.drop2 = tf.keras.layers.Dropout(drop_in)
        self.d2 = tf.keras.layers.Dense(emb_size,
                                        activation=tanh,
                                        name='dense_out')

        # ** Dropouts **

        # Dropout layers for state arrays and memory:
        self.drop_memory = tf.keras.layers.Dropout(drop_mem)
        self.drop_state_weights = tf.keras.layers.Dropout(drop_state)
        self.drop_state_erase = tf.keras.layers.Dropout(drop_state)
        self.drop_state_write = tf.keras.layers.Dropout(drop_state)

        # ** Recurrent states **

        # The three layers (controlling the weights, erase, and write vectors)
        # have a recurrent state vector.

        # State of weights vector:
        vec_rand_01 = tf.random.normal((batch_size, mem_locations),
                                       mean=0.0,
                                       stddev=0.001,
                                       dtype=tf.float32)
        self.state_weights = tf.Variable(initial_value=vec_rand_01,
                                         trainable=False,
                                         dtype=tf.float32,
                                         name='state_weights')

        # State of erase vector:
        vec_rand_02 = tf.random.normal((batch_size, mem_size),
                                       mean=0.0,
                                       stddev=0.001,
                                       dtype=tf.float32)
        self.state_erase = tf.Variable(initial_value=vec_rand_02,
                                       trainable=False,
                                       dtype=tf.float32,
                                       name='state_erase')

        # State of write vector:
        vec_rand_03 = tf.random.normal((batch_size, mem_size),
                                       mean=0.0,
                                       stddev=0.001,
                                       dtype=tf.float32)
        self.state_write = tf.Variable(initial_value=vec_rand_03,
                                       trainable=False,
                                       dtype=tf.float32,
                                       name='state_write')

        # ** Memory **

        # Random values for initial state of memory vector:
        vec_rand_04 = tf.random.normal((batch_size, mem_locations, mem_size),
                                       mean=0.0,
                                       stddev=0.001,
                                       dtype=tf.float32)

        # The actual memory matrix:
        self.memory = tf.Variable(initial_value=vec_rand_04,
                                  trainable=False,
                                  dtype=tf.float32,
                                  name='memory_matrix')

        # Matrix of ones (needed for memory update):
        self.ones = tf.ones((batch_size, mem_locations, mem_size),
                            dtype=tf.float32,
                            name='ones_matrix')

        # Math ops:
        # self.add = tf.keras.layers.Add()
        # self.sub = tf.keras.layers.Subtract()
        # self.mult = tf.keras.layers.Multiply()
        self.conc = tf.keras.layers.Concatenate(axis=1)

        # Reshape op for output (batch size gets added automatically as first
        # dimension):
        self.reshape = tf.keras.layers.Reshape((1, emb_size))


    def call(self, inputs):  # noqa

        # Activate of first feedforward module:
        f1 = self.drop1(inputs)
        f1 = self.d1(f1)

        # Activate recurrent layer that controls weights vector:
        conc_01 = self.conc([f1[:, 0, :],
                            self.state_weights,   # batch_size * mem_locations
                            self.state_erase,     # batch_size * mem_size
                            self.state_write])    # batch_size * mem_size
        weight_vec = self.mem_weights(conc_01)

        # Activate recurrent layer that controls write vector:
        conc_02 = self.conc([f1[:, 0, :],
                            self.state_weights,   # batch_size * mem_locations
                            self.state_erase,     # batch_size * mem_size
                            self.state_write])    # batch_size * mem_size
        write_vec = self.write(conc_02)

        # Activate recurrent layer that controls erase vector:
        conc_03 = self.conc([f1[:, 0, :],
                            self.state_weights,   # batch_size * mem_locations
                            self.state_erase,     # batch_size * mem_size
                            self.state_write])    # batch_size * mem_size
        erase_vec = self.erase(conc_03)

        # Calculate read vector:
        read_vec = tf.linalg.matvec(self.memory,  # batch_size * mem_locations * mem_size
                                    weight_vec,   # batch_size * mem_locations
                                    transpose_a=True)

        # ** Calculate new memory matrix **

        # Multiply weight vector with erase vector:
        erase_mat = tf.linalg.matmul(tf.reshape(weight_vec,
                                                (self.lgm_batch_size,
                                                 self.lgm_mem_locations,
                                                 1)),
                                     tf.reshape(erase_vec,
                                                (self.lgm_batch_size,
                                                 1,
                                                 self.lgm_mem_size)),
                                     transpose_a=False,
                                     transpose_b=False)
        # Subtract resulting erase matrix from ones:
        erase_mat = tf.math.subtract(self.ones,
                                     erase_mat)

        # Multiply previous memory matrix with erase matrix (element-wise):
        new_memory = tf.math.multiply(self.memory,
                                      erase_mat)

        # Multiply weight vector with write vector:
        write_mat = tf.linalg.matmul(tf.reshape(weight_vec,
                                                (self.lgm_batch_size,
                                                 self.lgm_mem_locations,
                                                 1)),
                                     tf.reshape(write_vec,
                                                (self.lgm_batch_size,
                                                 1,
                                                 self.lgm_mem_size)),
                                     transpose_a=False,
                                     transpose_b=False)

        # Add write matrix to new memory matrix:
        new_memory = tf.math.add(new_memory,
                                 write_mat)

        # Apply dropout:
        new_memory = self.drop_memory(new_memory)
        weight_vec = self.drop_state_weights(weight_vec)
        erase_vec = self.drop_state_erase(erase_vec)
        write_vec = self.drop_state_write(write_vec)

        # Update memory and states:
        self.memory = new_memory
        self.state_weights = weight_vec
        self.state_erase = erase_vec
        self.state_write = write_vec

        # Concatenate output of first feedforward module and memory readout:
        mem_and_states = self.conc([read_vec,
                                    weight_vec,
                                    erase_vec,
                                    write_vec])

        # Activation of second feedforward module:
        f2 = self.drop2(mem_and_states)
        f2 = self.d2(f2)

        return self.reshape(f2)

    def erase_memory(self, batch_size=None, mem_locations=None, mem_size=None):
        """Re-initialise memory and recurrent states."""
        # Reset state of weights vector:
        vec_rand_01 = tf.random.normal((batch_size, mem_locations),
                                       mean=0.0,
                                       stddev=0.001,
                                       dtype=tf.float32)
        self.state_weights = vec_rand_01

        # Reset state of erase vector:
        vec_rand_02 = tf.random.normal((batch_size, mem_size),
                                       mean=0.0,
                                       stddev=0.001,
                                       dtype=tf.float32)
        self.state_erase = vec_rand_02

        # Reset state of write vector:
        vec_rand_03 = tf.random.normal((batch_size, mem_size),
                                       mean=0.0,
                                       stddev=0.001,
                                       dtype=tf.float32)
        self.state_write = vec_rand_03

        # Reset state of memory matrix:
        vec_rand_04 = tf.random.normal((batch_size, mem_locations, mem_size),
                                       mean=0.0,
                                       stddev=0.001,
                                       dtype=tf.float32)
        self.memory = vec_rand_04

# --- Cell ---

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

# The actual network:
aryMemMod01 = MeLa(batch_size=varSzeBtch,
                   emb_size=varSzeEmb,
                   units_01=varNrn01,
                   mem_locations=varNumMem,
                   mem_size=varSzeMem,
                   drop_in=varInDrp,
                   drop_state=varStDrp,
                   drop_mem=varMemDrp,
                   name='MeLa01')(objTrnCtxt)
aryMemMod02 = MeLa(batch_size=varSzeBtch,
                   emb_size=varSzeEmb,
                   units_01=varNrn01,
                   mem_locations=varNumMem,
                   mem_size=varSzeMem,
                   drop_in=varInDrp,
                   drop_state=varStDrp,
                   drop_mem=varMemDrp,
                   name='MeLa02')(aryMemMod01)

# Initialise the model:
objMdl = tf.keras.models.Model(inputs=[objTrnCtxt], outputs=aryMemMod02)

# An almost idential version of the model used for testing, without dropout
# and possibly different input size (fixed batch size of one).
aryMemModT1 = MeLa(batch_size=1,
                   emb_size=varSzeEmb,
                   units_01=varNrn01,
                   mem_locations=varNumMem,
                   mem_size=varSzeMem,
                   drop_in=0.0,
                   drop_state=0.0,
                   drop_mem=0.0,
                   name='TeMeLa01')(objTstCtxt)
aryMemModT2 = MeLa(batch_size=1,
                   emb_size=varSzeEmb,
                   units_01=varNrn01,
                   mem_locations=varNumMem,
                   mem_size=varSzeMem,
                   drop_in=0.0,
                   drop_state=0.0,
                   drop_mem=0.0,
                   name='TeMeLa02')(aryMemModT1)

# Initialise the model:
objTstMdl = tf.keras.models.Model(inputs=objTstCtxt, outputs=aryMemModT2)
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

    # Assign pre-trained weights to model:
    objMdl.set_weights(lstWghts)

# Print model summary:
print('Training model:')
objMdl.summary()
print('Testing model:')
objTstMdl.summary()

# Define the optimiser and loss function:
objMdl.compile(optimizer=tf.keras.optimizers.Adam(lr=varLrnRte),
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

# Old (tf 1.13) summary implementation for tensorboard:
objLogWrt = tf.summary.FileWriter(strPthLogSes,
                                  graph=objSess.graph)
objMrgSmry = tf.summary.merge_all()


# -----------------------------------------------------------------------------
# *** Queues

# Create FIFO queue for target word index (to get current target word index
# from batch-creating queue for validation):
objIdxQ = queue.Queue(maxsize=varCapQ)

# Batches are prepared in a queue-feeding-function that runs in a separate
# thread.


def training_queue():
    """Place training data on queue."""
    # Initial index of last sample in batch:
    varLast = float(varLenTxt) - (float(varLenTxt) / float(varSzeBtch))
    varLast = int(np.floor(varLast))

    # Word index; refers to position of target word (i.e. word to be predicted)
    # in the corpus.
    # varIdxWrd = 1
    vecIdxWrd = np.array(random.sample(range(varLenTxt), varSzeBtch),
                         dtype=np.int64)

    # Array for new batch of sample weights:
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
    varWghtMin = 0.1

    # Maximum weight to use (for least frequent word):
    varWghtMax = 1.0

    # Vector with word count in corpus (returns vector with unique values,
    # which  is identical to word codes, and corresponding word counts):
    _, vecCnt = np.unique(vecFullC, return_counts=True)

    # Minimum number of occurences:
    vecCntMin = np.min(vecCnt)

    # Weights that are inversely proportional to number of occurences:
    vecWght = np.divide(vecCntMin, vecCnt)

    # Scale weights to respective range:
    vecWght = np.multiply(vecWght, (varWghtMax - varWghtMin))
    vecWght = np.add(varWghtMin, vecWght)
    vecWght = vecWght.astype(np.float32)

    # Loop through optimisation steps (one batch per optimisation step):
    for idxOpt in range(varNumOpt):

        # Get integer codes of context word (the word preceeding the target
        # word):
        vecCntxt = vecC[np.subtract(vecIdxWrd, 1)]

        # Get embedding vector for context words (the words preceeding the
        # target words):
        aryCntxt = aryEmb[vecCntxt, :].reshape(varSzeBtch, 1, varSzeEmb)

        # Words to predict (targets):
        vecTrgt = vecC[vecIdxWrd]

        # Get embedding vector for target words:
        aryTrgt = aryEmb[vecTrgt, :]

        # Get sample weights for target words:
        aryWght = vecWght[vecTrgt]

        # Increment word indicies for next batch:
        vecIdxWrd = np.add(vecIdxWrd, 1)
        # Reset word index to beginning of text if end has been reached:
        vecGrtr = np.greater_equal(vecIdxWrd, varLenTxt)
        vecIdxWrd[vecGrtr] = 1

        # Chose one of the target word indices at random (for testing):
        varIdxWrd = random.choice(vecIdxWrd)
        # Put index of next target word on the queue (target word after current
        # batch, because index has already been incremented):
        objIdxQ.put(varIdxWrd)

        dicIn01 = {objPlcHld01: aryCntxt}
        dicIn02 = {objPlcHld02: aryTrgt}
        dicIn03 = {objPlcHld03: aryWght}

        # Batch is complete, push to the queue:
        objSess.run(objEnQ01, feed_dict=dicIn01)
        objSess.run(objEnQ02, feed_dict=dicIn02)
        objSess.run(objEnQ03, feed_dict=dicIn03)

        # Array for new batch of context words:
        aryCntxt = np.zeros((varSzeBtch, 1, varSzeEmb),
                            dtype=np.float32)

        # Array for new batch of target words:
        aryTrgt = np.zeros((varSzeBtch, varSzeEmb),
                           dtype=np.float32)

        # Array for new batch of sample weights:
        aryWght = np.zeros((varSzeBtch), dtype=np.float32)

    print('--> End of feeding thread.')


def gpu_status():
    """Print GPU status information."""
    while True:
        # Print nvidia GPU status information:
        !nvidia-smi
        # Sleep some time before next status message:
        time.sleep(600)


# -----------------------------------------------------------------------------
# *** Fill queue

# Buffer size (number of samples to put on queue before starting
# execution of graph):
varBuff = (varCapQ - 1)

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
# Additional thread for GPU status information:
if lgcGpu:
    objThrdGpuStt = threading.Thread(target=gpu_status)
    objThrdGpuStt.setDaemon(True)
    objThrdGpuStt.start()


# -----------------------------------------------------------------------------
# *** Training

print('--> Beginning of training.')

# Loop through optimisation steps (one batch per optimisation step):
for idxOpt in range(varNumOpt):

    # Run optimisation on single batch:
    varLoss01 = objMdl.train_on_batch(objTrnCtxt,
                                      y=objTrgt,
                                      sample_weight=objWght)

    # Take target word index from queue:
    varTmpWrd = objIdxQ.get()

    # Update tensorboard information:
    if (idxOpt % 100 == 0):
        # Use old (tf 1.13) implementation for writing loss to tensorboard
        # summary.
        objSmry = tf.Summary(value=[tf.Summary.Value(tag="loss",
                                                     simple_value=varLoss01),
                                    ])
        objLogWrt.add_summary(objSmry, global_step=idxOpt)

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
        varLenCntx = 100

        # Avoid beginning of text (not enough preceding context words):
        if varTmpWrd > varLenCntx:

            # Get model weights from training model:
            lstWghts = objMdl.get_weights()

            # Copy weights from training model to test model:
            objTstMdl.set_weights(lstWghts)

            # Reset model states:
            objTstMdl.reset_states()

            # The memory vector of the custom memory layer needs to be reset
            # manually:
            objTstMdl.get_layer(name='TeMeLa01').erase_memory(
                batch_size=1, mem_locations=varNumMem, mem_size=varSzeMem)
            objTstMdl.get_layer(name='TeMeLa02').erase_memory(
                batch_size=1, mem_locations=varNumMem, mem_size=varSzeMem)

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

            print(('Loss auto:   ' + str(varLoss01)))
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

    # Reset internal model states:
    if (idxOpt % varResStp == 0):

        objMdl.reset_states()

        # The memory vector of the custom memory layer needs to be reset
        # manually:
        objMdl.get_layer(name='MeLa01').erase_memory(
            batch_size=varSzeBtch, mem_locations=varNumMem, mem_size=varSzeMem)
        objMdl.get_layer(name='MeLa02').erase_memory(
            batch_size=varSzeBtch, mem_locations=varNumMem, mem_size=varSzeMem)

print('--> End of training.')

# Get model weights:
lstWghts = objMdl.get_weights()

# Save model weights and training parameters to disk:
np.savez(os.path.join(strPthLogSes, 'ligemo_data.npz'),
         strPthIn=strPthIn,
         strPthMdl=strPthMdl,
         strPthLog=strPthLog,
         varLrnRte=varLrnRte,
         varNumItr=varNumItr,
         varResStp=varResStp,
         varNrn01=varNrn01,
         varSzeEmb=varSzeEmb,
         varNumMem=varNumMem,
         varSzeMem=varSzeMem,
         varSzeBtch=varSzeBtch,
         varInDrp=varInDrp,
         varStDrp=varStDrp,
         varMemDrp=varMemDrp,
         lstWghts=lstWghts)
