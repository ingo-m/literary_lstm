
# -*- coding: utf-8 -*-

import numpy as np

from plot import plot

# -----------------------------------------------------------------------------
# *** Define parameters

# List of session IDs:
lstSess = ['20191019_103037']

# Path of npz file containing previously trained model's weights to load (if
# None, new model is created):
strPthMdl = '/home/john/Dropbox/Harry_Potter/lstm/{}/lstm_data.npz'

# Output path for plots:
strPthOut = '/home/john/Dropbox/Harry_Potter/plots/'


# -----------------------------------------------------------------------------
# *** Loop through sessions

for strSess in lstSess:

    # -------------------------------------------------------------------------
    # *** Preparations
    
    # Get weights from npz file:
    objNpz = np.load(strPthMdl.format(strSess))
    objNpz.allow_pickle = True
    lstNpz = list(objNpz['lstWghts'])
    
    # The npz file contains 2D arrays (weights) and 1D arrays (biases). We put
    # them into separate lists:
    lstWghts = []
    lstBias = []
    for lstTmp in lstNpz:
        # 2D weights:
        if lstTmp.ndim == 2:
            lstWghts.append(lstTmp)
        # 1D biases:
        elif lstTmp.ndim == 1:
            lstBias.append(lstTmp)
    
    
    # -----------------------------------------------------------------------------
    # *** Plot weight matrices
            
    for idxLry in range(len(lstWghts)):

        # Plot output path:
        strOutTmp = (strPthOut + strSess + '_layer_' + str(idxLry)
                     + '_weights.png')

        # Plot title:
        strTtlTmp = ('Weights layer ' + str(idxLry))

        plot(lstWghts[idxLry],
             strTtlTmp,
             ' ',
             ' ',
             strOutTmp,
             tpleLimX=None,
             tpleLimY=None,
             varMin=None,
             varMax=None,
             varDpi=200.0)

