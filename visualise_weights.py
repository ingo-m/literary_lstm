# -*- coding: utf-8 -*-
"""Visualise weights and biases of language model."""

import numpy as np
from plot import plot
import seaborn as sns


# -----------------------------------------------------------------------------
# *** Define parameters

# List of session IDs:
lstSess = ['20191030_110148']

# Path of npz file containing previously trained model's weights to load (if
# None, new model is created):
strPthMdl = '/home/john/Dropbox/Harry_Potter/lstm/{}/lstm_data.npz'

# Output path for plots:
strPthOut = '/home/john/Dropbox/Harry_Potter/plots/'

# Only plot data within following percentile range (i.e. exclude outliers):
tplPrcntl = (0.5, 99.5)


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

    # Number of layers with weights:
    varNumLry = len(lstWghts)


    # -------------------------------------------------------------------------
    # *** Plot weight matrices

    for idxLry in range(varNumLry):

        # Plot output path:
        strOutTmp = (strPthOut
                     + 'layer_'
                     + str(idxLry)
                     + '_weights_session_'
                     + strSess
                     + '.png')

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
             varDpi=300.0,
             tplPrcntl=tplPrcntl)


    # -------------------------------------------------------------------------
    # *** Plot distribution of weights

    # Flatten weight arrays:
    lstWghts = [x.flatten() for x in lstWghts]

    for idxLry in range(varNumLry):

        # Exclude datapoints outside of percentile range:
        tplLim = np.percentile(lstWghts[idxLry], tplPrcntl)
        vecLgc = np.logical_and(np.greater(lstWghts[idxLry], tplLim[0]),
                                np.less(lstWghts[idxLry], tplLim[1]))
        lstWghts[idxLry] = lstWghts[idxLry][vecLgc]

    # Colour palette:
    objClr = sns.cubehelix_palette(varNumLry, rot=-.5, dark=.3)

    # Show each distribution with both violins and points
    objPlt = sns.violinplot(data=lstWghts,
                            palette=objClr,
                            scale='area',
                            inner=None)

    # Plot output path:
    strOutTmp = (strPthOut
                 + 'weights_distribution_session_'
                 + strSess
                 + '.png')

    # Save figure to disk:
    objFig = objPlt.get_figure()
    objFig.savefig(strOutTmp)
    objFig.clf()


    # -------------------------------------------------------------------------
    # *** Plot distribution of biases

    # Number of layers with bias vector:
    varNumLryB = len(lstBias)

    for idxLry in range(varNumLryB):

        # Exclude datapoints outside of percentile range:
        tplLim = np.percentile(lstBias[idxLry], tplPrcntl)
        vecLgc = np.logical_and(np.greater(lstBias[idxLry], tplLim[0]),
                                np.less(lstBias[idxLry], tplLim[1]))
        lstBias[idxLry] = lstBias[idxLry][vecLgc]

    # Colour palette:
    objClr = sns.cubehelix_palette(varNumLry, rot=-.5, dark=.3)

    # Show each distribution with both violins and points
    objPlt = sns.violinplot(data=lstBias,
                            palette=objClr,
                            scale='area',
                            inner=None)

    # Plot output path:
    strOutTmp = (strPthOut
                 + 'biases_distribution_session_'
                 + strSess
                 + '.png')

    # Save figure to disk:
    objFig = objPlt.get_figure()
    objFig.savefig(strOutTmp)
    objFig.clf()

    del(objNpz, lstNpz, lstWghts, lstBias)
# -----------------------------------------------------------------------------
