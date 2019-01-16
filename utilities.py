#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM fun helper functions.
"""

import re

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
