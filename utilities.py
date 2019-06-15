#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM fun helper functions.
"""


import re
import collections
import random
import numpy as np


def read_text(strPth):
    """
    Read text from file.

    Parameters
    ----------
    strPth : str
         Path of text file.

    Returns
    -------
    lstTxt : list
        List with text, separated into individual words and punctuation maks.
    """
    # List of permitted characters (all other characters will be removed):
    strChr = ('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
              + '.,;!?-ÄäÖöÜüèêßà()')

    # Open text file:
    objTxt = open(strPth, 'r')

    # Read complete text into list:
    strTxt = objTxt.read()

    # List of unique characters in text:
    lstUnq = list(set(strTxt))

    # Identify characters to be removed:
    for strTmp in lstUnq:
        if not(strTmp in strChr):
            # Replace character with blank space:
            # print(strTmp)
            strTxt = strTxt.replace(strTmp, " ")

    # Custom replacements in order facilitate convergence:
    strTxt = strTxt.replace('...', ' ellipsis')
    strTxt = strTxt.replace(',', '')

    # Split text into words and punctuation marks:
    lstTxt = re.findall(r"[\w']+|[-.,;!?()]", strTxt)

    # Close text file:
    objTxt.close()

    return lstTxt


def build_dataset(lstTxt, varVocSze=50000):
    """
    Build dataset from text.

    Parameters
    ----------
    lstTxt : list
        Input text (corpus).
    varVocSze : int
        Vocabulary size (number of words; rare words are replaced with 'unknown'
        code if the vocabulary size is exceeded by the number of words in the
        text).

    Returns
    -------
    vecC : np.array
        Coded version of original text (corpus), where words are coded as
        integers. The integer code of a word is its ordinal occurence
        number (i.e. the 50th most common word has the code 50).
    lstWrdCnt : list
        List of tuples with words and corresponding count of occurences in
        text (word, count).
    dicWdCnOdr : dict
        Dictionary for words (as keys) and ordinal word count (values);
        i.e. words in order of number of occurences.
    dictRvrs : dict
        Reverse dictionary (where keys are word order).
    """
    # List of tuples with words and corresponding count of occurences in
    # text (word, count):
    lstWrdCnt = [('UNK', -1)]
    # count = [['UNK', -1]]

    # Append (word, count) tuples to list:
    lstWrdCnt.extend(collections.Counter(lstTxt).most_common(
        varVocSze - 1))

    # Dictionary for words (as keys) and ordinal word count (values); i.e.
    # words in order of number of occurences.
    dicWdCnOdr = {}
    # dictionary = {}

    for strWrd, _ in lstWrdCnt:
        # Current word gets assigned current length of dictionary. Since
        # words in list are in order of number of occurences, this results
        # in an ordinal word index.
        dicWdCnOdr[strWrd] = len(dicWdCnOdr)

    # Coded text (words are coded as integers, and the code of a word
    # is its ordinal occurence number).
    lstC = []

    # Counter for 'unknown' words (words not in the vocabulary of most
    # common words):
    varCntUnk = 0

    # Translate original text (lstTxt) into code (lstC), i.e. replace words
    # with their integer codes.
    for strWrd in lstTxt:

        # Code of current word:
        varTmpC = dicWdCnOdr.get(strWrd, 0)

        # Count words that are not in vocabulary (dicWdCnOdr['UNK']):
        if varTmpC == 0:
            varCntUnk += 1

        # Append code to code-version of text:
        lstC.append(varTmpC)

    # Update word count of 'unknown' code:
    lstWrdCnt[0] = (lstWrdCnt[0][0], varCntUnk)

    # Create reverse dictionary (where keys are word order):
    dictRvrs = dict(zip(dicWdCnOdr.values(), dicWdCnOdr.keys()))

    # List to vector:
    vecC = np.array(lstC)

    return vecC, lstWrdCnt, dicWdCnOdr, dictRvrs


def generate_batch_n(vecC, varIdx, varBatSze=5, varConWin=5.0, varTrnk=10):
    """
    Generate training batch for skip-gram model.

    Inputs
    ------
    vecC : np.array
        Coded version of original text (corpus), where words are coded as
        integers. The integer code of a word is its ordinal occurence
        number (i.e. the 50th most common word has the code 50).
    varIdx: int
        Position (index) within corpus where to start sampling.
    varBatSze : int
        Batch size (number of words).
    varConWin : float
        Standard deviation of the context window (i.e. number of words to
        consider to the left and to the right of the target word). In order to
        more frequently sample nearby words, we base the selection of context
        words on a Gaussian distribution.
    varTrnk : int
        Cutoff for truncation of context word window. Context words are sampled
        from a Gaussian distrubution around the sample word, so that nearby
        words are sampled with higher probability. However, the probability
        distribution needs to be truncated so as not to run over the limits of
        the corpus.

    Returns
    -------
    vecWrds : np.array
        Batch of (integer) codes of input words (whose context to predict).
    aryCntxt : np.array
        Vector (batch_size * 1) of context words to predict.

    Notes
    -----
    The distance between sample word and context words are drawn from a normal
    probability distribution, in order to sample close words with higher
    probability.

    TODO:
    Sample common words (with repsect to occurence in corpus) more often.
    """
    # Vector with indices of context words, relative to sample word (e.g. -1
    # would refer to word before sample word, +1 to the word after the sample
    # word. We start with a Gaussian distribution (float), which will
    # subsequently be rounded to integer indices. Negative values are rounded
    # to floor, positive values to ceiling, so there will be no zeros (zero =
    # index of sample word).
    vecRndn = np.multiply(np.random.randn(varBatSze), varConWin)
    # Which indices are negative?
    vecLgcNgt = np.less(vecRndn, 0.0)
    vecLgcPst = np.logical_not(vecLgcNgt)
    # Negative indices are rounded to the floor:
    vecRndn[vecLgcNgt] = np.floor(vecRndn[vecLgcNgt])
    # Positive indices are rounded to ceiling:
    vecRndn[vecLgcPst] = np.ceil(vecRndn[vecLgcPst])
    # Cast indices to integer:
    vecRndn = vecRndn.astype(np.int64)

    # Truncate context words outside of cutoff limit (i.e. too far from sample
    # word) - positive limit:
    vecLgcTrc = np.logical_and(
                               np.greater(vecRndn, int(varTrnk)),
                               vecLgcPst
                               )
    vecRndn[vecLgcTrc] = int(varTrnk)

    # Truncate context words outside of cutoff limit (i.e. too far from sample
    # word) - negative limit:
    vecLgcTrc = np.logical_and(
                               np.less(vecRndn, int(-varTrnk)),
                               vecLgcNgt
                               )
    vecRndn[vecLgcTrc] = int(-varTrnk)


    # Indices of sample words in corpus (linear):
    vecIdxWrds = np.arange(varIdx, (varIdx + varBatSze))

    # Indices of context words in corpus:
    vecIdxCtx = np.add(vecIdxWrds, vecRndn)

    # Look up words in corpus.

    # Batch of (integer) codes of input words (whose context to predict).
    vecWrds = vecC[vecIdxWrds]

    # Batch of (integer) codes of context words (to predict):
    aryCntxt = vecC[vecIdxCtx].reshape(varBatSze, 1)

    return vecWrds, aryCntxt
