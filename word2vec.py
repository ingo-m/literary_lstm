# =============================================================================
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# See
# https://www.tensorflow.org/tutorials/representation/word2vec

"""Basic word2vec example."""

# cd '/Users/john/PhD/GitLab/literary_lstm'

import os
import numpy as np
import math
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from utilities import read_text
from utilities import build_dataset
from utilities import generate_batch_n

# from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.tensorboard.plugins import projector

# ------------------------------------------------------------------------------
# *** Define parameters

# Path of input text files:
strPthIn = '/Users/john/Dropbox/Ernest_Hemingway/redacted/compilation.txt'

# Tensorflow log directory:
strTfLog = '/Users/john/1_PhD/GitLab/literary_lstm/log_w2v'

# Batch size:
varBatSze = 500

# Size of context window, i.e. how many words to consider to the left and to
# the right of each target word.
varConWin = 10

# Dimension of the embedding vector. (Number of neurons in hidden layer?)
varSzeEmb = 300

# Number of negative examples to sample.
varNumNgtv = 300

# Random set of words to evaluate similarity on:
varSzeEval = 10

# Vocabulary size (number of words; rare words are replaced with 'unknown'
# code if the vocabulary size is exceeded by the number of words in the
# text).
varVocSze = 15000

# Number of training iterations:
varNumIt = 100001

# ------------------------------------------------------------------------------
# *** Preparations

print('-Word2vec')

print('---Preparations.')

# Read text from file:
lstTxt = read_text(strPthIn)

# Make all words lower case:
lstTxt = [x.lower() for x in lstTxt]

# Create the directory for TensorBoard variables if there is not.
if not os.path.exists(strTfLog):
    os.makedirs(strTfLog)

# ------------------------------------------------------------------------------
# *** Code text

# Build coded dataset from text:
vecC, lstWrdCnt, dicWdCnOdr, dictRvrs = build_dataset(lstTxt, varVocSze)

# Delete non-coded text (reduce memory usage):
del lstTxt

# Generate example batch:
#vecWrds, aryCntxt = generate_batch_n(vecC,
#                                     50,
#                                     varBatSze=50,
#                                     varConWin=5.0,
#                                     varTrnk=10)

# ------------------------------------------------------------------------------
# *** Build skip-gram model

print('---Initialising skip-gram model.')

# We pick a random validation set to sample nearest neighbors. Here we limit
# the validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.

# Only pick dev samples in the head of the distribution:
varSzeEvalWin = int(np.around((varVocSze * 0.1)))
vecEvalSmple = np.random.choice(varSzeEvalWin, varSzeEval, replace=False)

graph = tf.Graph()

with graph.as_default():

    # Input data.
    with tf.name_scope('inputs'):

        # Placeholder for batch of words (coded as integers) for which to
        # predict the context.
        vecTfWrds = tf.placeholder(tf.int32, shape=[varBatSze])

        # Placeholder for context words to predict (coded as integers):
        aryTfCntxt = tf.placeholder(tf.int32, shape=[varBatSze, 1])

        # Evaluation dataset (vector of words coded as integer). Used for
        # displaying model accuracy.
        vecTfEvalSmple = tf.constant(vecEvalSmple, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):

        # Look up embeddings for inputs.
        with tf.name_scope('embeddings'):

            # Embedding matrix, size: vocabulary * number of neurons in hidden
            # layer (?).
            aryTfEmbd = tf.Variable(tf.random_uniform([varVocSze, varSzeEmb],
                                                      -1.0,
                                                      1.0))

            # Embedding lookup, size: batch size (number of words) * number of
            # neurons in hidden layer (?)
            aryTfEbd = tf.nn.embedding_lookup(aryTfEmbd, vecTfWrds)

    # Construct the variables for the NCE loss
    with tf.name_scope('weights'):

        # Weights of neurons in hidden layer (number of words times number of
        # neurons).
        aryWghts = tf.Variable(
            tf.truncated_normal([varVocSze, varSzeEmb],
            stddev=(1.0 / math.sqrt(varSzeEmb)))
            )

    with tf.name_scope('biases'):

        # Biases of neurons in hidden layer (size: number of words ???)
        aryBiases = tf.Variable(tf.zeros([varVocSze]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # Explanation of the meaning of NCE loss:
    #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

    with tf.name_scope('loss'):

        # Objective function to reduce (?)
        varLoss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=aryWghts,
                biases=aryBiases,
                labels=aryTfCntxt,
                inputs=aryTfEbd,
                num_sampled=varNumNgtv,
                num_classes=varVocSze
                )
            )

    # Add the loss value as a scalar to summary.
    tf.summary.scalar('loss', varLoss)

    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
        objOpt = tf.train.GradientDescentOptimizer(1.0).minimize(varLoss)

    # Compute the cosine similarity between minibatch examples and all
    # embeddings.
    aryNorm = tf.sqrt(tf.reduce_sum(tf.square(aryTfEmbd), 1, keepdims=True))
    # "normalized_embeddings", size: number of words * number of neurons in
    # hidden layer (?)
    aryNormEmb = aryTfEmbd / aryNorm

    # Get embeddings of validation dataset (?), size: number of words in
    # validation dataset * number of neurons in hidden layer.
    aryEvalEmbd = tf.nn.embedding_lookup(aryNormEmb,
                                         vecTfEvalSmple)


    # "similarity", size: number of words in validation set * number of words
    # in vocabulary.
    arySim = tf.matmul(aryEvalEmbd, aryNormEmb, transpose_b=True)

    # Merge all summaries.
    objMrgSmry = tf.summary.merge_all()



    # Add variable initializer.
    objInit = tf.global_variables_initializer()

    # Create a saver.
    objSaver = tf.train.Saver()

print('---Training skip-gram model.')

with tf.Session(graph=graph) as objSess:

    # Open a writer to write summaries.
    objWrtr = tf.summary.FileWriter(strTfLog, objSess.graph)

    # We must initialize all variables before we use them.
    objInit.run()
    print('Initialized')

    # Average loss, will be updated after certain number of steps.
    varAvrgLoss = 0

    # Initialise index (for looping through corpus), do not start at zero
    # to avoid sampling at limits of corpus (words without sufficient context):
    varTrnk = 10
    varIdx = varTrnk

    # Number of words in text:
    varLenTxt = vecC.shape[0]

    # Upper limit for sampling from corpus:
    varSmpLimUp = ((varLenTxt - varTrnk) - varBatSze)

    for idxItr in range(varNumIt):

        vecWrds, aryCntxt = generate_batch_n(vecC,
                                             varIdx,
                                             varBatSze=varBatSze,
                                             varConWin=varConWin,
                                             varTrnk=varTrnk)

        dicFeed = {vecTfWrds: vecWrds, aryTfCntxt: aryCntxt}

        # Define metadata variable.
        objMetadata = tf.RunMetadata()

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        # Also, evaluate the merged op to get all summaries from the returned
        # "summary" variable. Feed metadata variable to session for visualizing
        # the graph in TensorBoard.
        _, objSmry, varTmpLoss = objSess.run([objOpt, objMrgSmry, varLoss],
                                              feed_dict=dicFeed,
                                              run_metadata=objMetadata)
        varAvrgLoss += varTmpLoss

        # Add returned summaries to writer in each step.
        objWrtr.add_summary(objSmry, idxItr)

        # Add metadata to visualize the graph for the last run.
        if idxItr == (varNumIt - 1):
            objWrtr.add_run_metadata(objMetadata, 'step%d' % idxItr)

        if idxItr % 10000 == 0:
            if idxItr > 0:
                varAvrgLoss /= 10000

            # The average loss is an estimate of the loss over the last 2000
            # batches.
            print('Average loss at step ', idxItr, ': ', varAvrgLoss)
            varAvrgLoss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if idxItr % 50000 == 0:

            arySimTmp = arySim.eval()

            # Loop through words in validation set:
            for idxEval in range(varSzeEval):

                strEvalWrd = dictRvrs[vecEvalSmple[idxEval]]
                varNnn = 8  # number of nearest neighbors
                vecNearest = (-arySimTmp[idxEval, :]).argsort()[1:varNnn + 1]
                strLogMsg = 'Nearest to %s:' % strEvalWrd

                # Loop through nearest neighbouring words:
                for idxNghb in range(varNnn):

                    # Close word:
                    strWrdCls = dictRvrs[vecNearest[idxNghb]]
                    strLogMsg = '%s %s,' % (strLogMsg, strWrdCls)

                print(strLogMsg)

        # Increment index, reset if end of corpus has been reached:
        varIdx += varBatSze
        if varSmpLimUp <= varIdx:
            varIdx = varTrnk

    # "Final embeddings":
    aryEmbFnl = aryNormEmb.eval()

    # Write corresponding labels for the embeddings.
    with open(strTfLog + '/metadata.tsv', 'w') as f:
        for i in range(varVocSze):
            f.write(dictRvrs[i] + '\n')

    # Save the model for checkpoints.
    objSaver.save(objSess, os.path.join(strTfLog, 'model.ckpt'))

    # Create a configuration for visualizing embeddings with the labels in
    # TensorBoard.
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = aryTfEmbd.name
    embedding_conf.metadata_path = os.path.join(strTfLog, 'metadata.tsv')
    projector.visualize_embeddings(objWrtr, config)

objWrtr.close()

# Save text, dictionary, and embeddings to disk:
np.savez(os.path.join(strTfLog, 'word2vec_data.npz'),
         vecC=vecC,  # Coded text
         dicWdCnOdr=dicWdCnOdr,  # Dictionary, keys=words
         dictRvrs=dictRvrs,  # Reservse dictionary, keys=ordinal-word-count
         aryEmbFnl=aryEmbFnl  # Embedding matrix
         )

# ------------------------------------------------------------------------------
# *** Visualise embeddings

print('---Visualising embeddings.')

# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, aryCntxt, filename):
    assert low_dim_embs.shape[0] >= len(aryCntxt), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(aryCntxt):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')

    plt.savefig(filename)

tsne = TSNE(perplexity=30,
            n_components=2,
            init='pca',
            n_iter=5000,
            method='exact')
plot_only = 500
low_dim_embs = tsne.fit_transform(aryEmbFnl[:plot_only, :])
aryCntxt = [dictRvrs[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, aryCntxt, os.path.join(strTfLog,
                                                      'tsne.png'))

print('-Done.')

## All functionality is run after tf.app.run() (b/122547914). This could be split
## up but the methods are laid sequentially with their usage for clarity.
#def main(unused_argv):
#    # Give a folder path as an argument with '--log_dir' to save
#    # TensorBoard summaries. Default is a log folder in current directory.
#    # current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
#
#    parser = argparse.ArgumentParser()
#    parser.add_argument(
#        '--strTfLog',
#        type=str,
#        default=os.path.join(strTfLog, 'log'),
#        help='The log directory for TensorBoard summaries.')
#    flags, unused_flags = parser.parse_known_args()
#    word2vec_basic(flags.strTfLog)
#
#if __name__ == '__main__':
#    tf.app.run()
