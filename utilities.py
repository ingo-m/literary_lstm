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
    strTxt = strTxt.replace('...', 'punktpunktpunkt')
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
    lstC : list
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

    return lstC, lstWrdCnt, dicWdCnOdr, dictRvrs


def generate_batch(lstC, glbVarIdx, vecBatSze=8, varNumSkp=2, varConWin=1):
    """
    Generate training batch for skip-gram model.

    Inputs
    ------
    lstC : list
        Coded version of original text (corpus), where words are coded as
        integers. The integer code of a word is its ordinal occurence
        number (i.e. the 50th most common word has the code 50).
    glbVarIdx : int
        ?
    vecBatSze : int
        ?
    varNumSkp : int
        ?
    varConWin : int
        ?

    Returns
    -------
    vecBat : np.array
        ?
    aryLbl : np.array
        ?
    glbVarIdx : int
        ?

    Notes
    -----
    ???
    """

    # global glbVarIdx
    assert vecBatSze % varNumSkp == 0
    assert varNumSkp <= 2 * varConWin

    # batch = np.ndarray(shape=(vecBatSze), dtype=np.int32)
    vecBat = np.zeros(vecBatSze, dtype=np.int32)

    # labels = np.ndarray(shape=(vecBatSze, 1), dtype=np.int32)
    aryLbl = np.zeros((vecBatSze, 1), dtype=np.int32)

    # span = 2 * varConWin + 1  # [ varConWin target varConWin ]
    varSpan = 2 * varConWin + 1

    # Buffer. ?
    objBuf = collections.deque(maxlen=varSpan)  # pylint: disable=redefined-builtin

    if glbVarIdx + varSpan > len(lstC):
        glbVarIdx = 0

    # Get coded words (integers) from corpus and put them on buffer. (?)
    objBuf.extend(lstC[glbVarIdx:glbVarIdx + varSpan])

    # Increment index:
    glbVarIdx += varSpan

    # Loop through ?
    for idx01 in range(vecBatSze // varNumSkp):

        # print("---")
        # print("idx01 = " + str(idx01))

        context_words = [w for w in range(varSpan) if w != varConWin]

        words_to_use = random.sample(context_words, varNumSkp)

        #print("context_words: " + str([dictRvrs[x] for x in context_words]))
        #print("words_to_use: " + str([dictRvrs[x] for x in words_to_use]))

        for idx02, context_word in enumerate(words_to_use):
            #print("idx02 = " + str(idx02))
            #print("context_word: " + dictRvrs[context_word])

            vecBat[idx01 * varNumSkp + idx02] = objBuf[varConWin]

            aryLbl[idx01 * varNumSkp + idx02, 0] = objBuf[context_word]

            # print("vecBat: " + dictRvrs[vecBat[idx01 * varNumSkp + idx02]])
            # print("aryLbl: " + dictRvrs[[idx01 * varNumSkp + idx02, 0][0]])

            if glbVarIdx == len(lstC):
                objBuf.extend(lstC[0:varSpan])
                glbVarIdx = varSpan
                print("!!!")

            else:
                # print("glbVarIdx = " + dictRvrs[lstC[glbVarIdx]])
                objBuf.append(lstC[glbVarIdx])
                glbVarIdx += 1

            # Backtrack a little bit to avoid skipping words in the end of a
            # batch
            glbVarIdx = (glbVarIdx + len(lstC) - varSpan) % len(lstC)

    return vecBat, aryLbl, glbVarIdx
