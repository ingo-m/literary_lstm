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

# cd '/Users/john/1_PhD/GitLab/literary_lstm'

import os
# import collections
# import random
import numpy as np
import tensorflow as tf
from utilities import read_text
from utilities import build_dataset
from utilities import generate_batch


# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#import argparse
import math
#import sys
#from tempfile import gettempdir
#from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.tensorboard.plugins import projector


# Path of input text files:
# strPthIn = '/Users/john/Dropbox/Thomas_Mann/Thomas_Mann_1909_Buddenbrooks.txt'
strPthIn = '/Users/john/Dropbox/Thomas_Mann/Thomas_Mann_1909_Buddenbrooks_excerpt.txt'

# Tensorflow log directory:
strTfLog = '/Users/john/1_PhD/GitLab/literary_lstm/tf_log'

# Read text from file:
lstTxt = read_text(strPthIn)


#def word2vec_basic(strTfLog):
#    """Example of building, training and visualizing a word2vec model."""

# Create the directory for TensorBoard variables if there is not.
if not os.path.exists(strTfLog):
    os.makedirs(strTfLog)

# Step 2: Build the dictionary and replace rare words with UNK token.

# Vocabulary size (number of words; rare words are replaced with 'unknown'
# code if the vocabulary size is exceeded by the number of words in the
# text).
varVocSze = 140 # 20000  # 50000

# Build coded dataset from text:
lstC, lstWrdCnt, dicWdCnOdr, dictRvrs = build_dataset(lstTxt, varVocSze)

# Delete non-coded text (reduce memory usage):
del lstTxt

# print('Most common words (+UNK)', lstWrdCnt[:5])
# print('Sample data', lstC[:10], [dictRvrs[i] for i in lstC[:10]])

# Batch size: (?)
vecBatSze = 4

# How many times to reuse an input to generate a label.
# ???
varNumSkp = 1

# Size of context window, i.e. how many words to consider to the left and to
# the right of each target word.
varConWin = 3

# Global index. ?
glbVarIdx = 0

# Generate batch
# ???
vecWrds, aryCntxt, glbVarIdx = generate_batch(lstC,
                                              glbVarIdx,
                                              vecBatSze=vecBatSze,
                                              varNumSkp=varNumSkp,
                                              varConWin=varConWin)

# ???
# print("vecWrds: " + str([dictRvrs[x] for x in list(vecWrds)]))
# print("aryCntxt: " + str([dictRvrs[x] for x in list(aryCntxt[:, 0])]))

# for i in range(4):
#     print(vecWrds[i], dictRvrs[vecWrds[i]], '->', aryCntxt[i, 0],
#           dictRvrs[aryCntxt[i, 0]])

# Step 4: Build and train a skip-gram model.

# Dimension of the embedding vector. (Number of neurons in hidden layer?)
varSzeEmb = 24 # 128

# Number of negative examples to sample.
varNumNgtv = 12 # 64

# We pick a random validation set to sample nearest neighbors. Here we limit
# the validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.

# Random set of words to evaluate similarity on:
varSzeEval = 16

# Only pick dev samples in the head of the distribution:
varSzeEvalWin = 100
vecEvalSmple = np.random.choice(varSzeEvalWin, varSzeEval, replace=False)

graph = tf.Graph()

with graph.as_default():

    # Input data.
    with tf.name_scope('inputs'):

        # Placeholder for batch of words (coded as integers) for which to
        # predict the context.
        vecTfWrds = tf.placeholder(tf.int32, shape=[vecBatSze])

        # Placeholder for context words to predict (coded as integers):
        aryTfCntxt = tf.placeholder(tf.int32, shape=[vecBatSze, 1])

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
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=aryWghts,
                biases=aryBiases,
                labels=aryTfCntxt,
                inputs=aryTfEbd,
                num_sampled=varNumNgtv,
                num_classes=varVocSze
                )
            )

type(loss)
loss.shape

    # Add the loss value as a scalar to summary.
    tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all
    # embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(aryTfEmbd), 1, keepdims=True))
    normalized_embeddings = aryTfEmbd / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                              vecTfEvalSmple)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    # Merge all summaries.
    merged = tf.summary.merge_all()

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()

    # Step 5: Begin training.
    num_steps = 100001

with tf.Session(graph=graph) as session:

    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(strTfLog, session.graph)

    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels, glbVarIdx = generate_batch(lstC,
                                                               glbVarIdx,
                                                               vecBatSze=vecBatSze,
                                                               varNumSkp=varNumSkp,
                                                               varConWin=varConWin)
        feed_dict = {vecTfWrds: batch_inputs, aryTfCntxt: batch_labels}

        # Define metadata variable.
        run_metadata = tf.RunMetadata()

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        # Also, evaluate the merged op to get all summaries from the returned
        # "summary" variable. Feed metadata variable to session for visualizing
        # the graph in TensorBoard.
        _, summary, loss_val = session.run([optimizer, merged, loss],
                                           feed_dict=feed_dict,
                                           run_metadata=run_metadata)
        average_loss += loss_val

        # Add returned summaries to writer in each step.
        writer.add_summary(summary, step)
        # Add metadata to visualize the graph for the last run.
        if step == (num_steps - 1):
            writer.add_run_metadata(run_metadata, 'step%d' % step)

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
        # The average loss is an estimate of the loss over the last 2000
        # batches.
        print('Average loss at step ', step, ': ', average_loss)
        average_loss = 0

      # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(varSzeEval):
                valid_word = dictRvrs[vecEvalSmple[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = dictRvrs[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()

    # Write corresponding labels for the embeddings.
    with open(strTfLog + '/metadata.tsv', 'w') as f:
        for i in xrange(varVocSze):
            f.write(dictRvrs[i] + '\n')

    # Save the model for checkpoints.
    saver.save(session, os.path.join(strTfLog, 'model.ckpt'))

    # Create a configuration for visualizing embeddings with the labels in
    # TensorBoard.
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = os.path.join(strTfLog, 'metadata.tsv')
    projector.visualize_embeddings(writer, config)

    writer.close()

    # Step 6: Visualize the embeddings.

    # pylint: disable=missing-docstring
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

    try:
        # pylint: disable=g-import-not-at-top
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        tsne = TSNE(
            perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        aryCntxt = [dictRvrs[i] for i in xrange(plot_only)]
        plot_with_aryLbl(low_dim_embs, aryCntxt, os.path.join(gettempdir(),
                                                            'tsne.png'))

    except ImportError as ex:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')
        print(ex)


# All functionality is run after tf.app.run() (b/122547914). This could be split
# up but the methods are laid sequentially with their usage for clarity.
def main(unused_argv):
    # Give a folder path as an argument with '--strTfLog' to save
    # TensorBoard summaries. Default is a log folder in current directory.
    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--strTfLog',
        type=str,
        default=os.path.join(current_path, 'log'),
        help='The log directory for TensorBoard summaries.')
    flags, unused_flags = parser.parse_known_args()
    word2vec_basic(flags.strTfLog)

if __name__ == '__main__':
    tf.app.run()
