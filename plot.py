# -*- coding: utf-8 -*-
"""Plot 2D weight matrices."""

# Copyright (C) 2019 Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def plot(aryWghts, strTtl, strXlabel, strYlabel, strPathOut, tpleLimX=None,
         tpleLimY=None, varMin=None, varMax=None, varDpi=200.0,
         tplPrcntl=None):
    """Plot 2D weight matrix."""
    # Font type:
    strFont = 'Liberation Sans'

    # Font colour:
    vecFontClr = np.array([17.0/255.0, 85.0/255.0, 124.0/255.0])

    # Colour scale minimum:
    if varMin is None:
        varMin = np.percentile(aryWghts, tplPrcntl[0])
        varMin = (np.floor(varMin * 10.0) / 10.0)
        # varMin = (np.floor(varMin * 0.1) / 0.1)
        # varMin = np.floor(varMin)

    # Colour scale maximum:
    if varMax is None:
        varMax = np.percentile(aryWghts, tplPrcntl[1])
        varMax = (np.ceil(varMax * 10.0) / 10.0)
        # varMax = (np.ceil(varMax * 0.1) / 0.1)
        # varMax = np.ceil(varMax)

    # Saveguard to avoid division by zero in case of no negative values:
    # if np.less_equal(0.0, varMin):
    #     varMin = -1.0

    # Same scale for negative and positive colour bar:
    if np.greater(np.absolute(varMin), varMax):
        varMax = np.absolute(varMin)
    else:
        varMin = np.multiply(-1.0, np.absolute(varMax))

    # Maximum width or height (whichever is larger) of the figure, in inches:
    varFigSze = 8.0

    varSzeX = (float(aryWghts.shape[0])
               / float(np.max(aryWghts.shape))
               * varFigSze)
    varSzeY = (float(aryWghts.shape[1])
               / float(np.max(aryWghts.shape))
               * varFigSze)
    tplFigSze = (varSzeX, varSzeY)

    # Create main figure:
    fig01 = plt.figure(figsize=tplFigSze,
                       dpi=varDpi,
                       facecolor=([1.0, 1.0, 1.0]),
                       edgecolor=([1.0, 1.0, 1.0]))

    # Big subplot in the background for common axes labels:
    axsCmn = fig01.add_subplot(111)

    # Turn off axis lines and ticks of the big subplot:
    axsCmn.spines['top'].set_color('none')
    axsCmn.spines['bottom'].set_color('none')
    axsCmn.spines['left'].set_color('none')
    axsCmn.spines['right'].set_color('none')
    axsCmn.tick_params(labelcolor='w',
                       top=False,
                       bottom=False,
                       left=False,
                       right=False)

    # Set and adjust common axes labels:
    axsCmn.set_xlabel(strXlabel,
                      alpha=1.0,
                      fontname=strFont,
                      fontweight='normal',
                      fontsize=7.0,
                      color=vecFontClr,
                      position=(0.5, 0.0))
    axsCmn.set_ylabel(strYlabel,
                      alpha=1.0,
                      fontname=strFont,
                      fontweight='normal',
                      fontsize=7.0,
                      color=vecFontClr,
                      position=(0.0, 0.5))
    axsCmn.set_title(strTtl,
                     alpha=1.0,
                     fontname=strFont,
                     fontweight='bold',
                     fontsize=10.0,
                     color=vecFontClr,
                     position=(0.5, 1.1))

    # Create colour-bar axis:
    axsTmp = fig01.add_subplot(111)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Number of colour increments:
    varNumClr = 20

    # Colour values for the first colormap (used for negative values):
    aryClr01 = plt.cm.PuBu(np.linspace(0.1, 1.0, varNumClr))

    # Invert the first colour map:
    aryClr01 = np.flipud(np.array(aryClr01, ndmin=2))

    # Colour values for the second colormap (used for positive values):
    aryClr02 = plt.cm.OrRd(np.linspace(0.1, 1.0, varNumClr))

    # Combine negative and positive colour arrays:
    aryClr03 = np.vstack((aryClr01, aryClr02))

    # Create new custom colormap, combining two default colormaps:
    objCustClrMp = colors.LinearSegmentedColormap.from_list('custClrMp',
                                                            aryClr03)

    # Lookup vector for negative colour range:
    vecClrRngNeg = np.linspace(varMin, 0.0, num=varNumClr)

    # Lookup vector for positive colour range:
    vecClrRngPos = np.linspace(0.0, varMax, num=varNumClr)

    # Stack lookup vectors:
    vecClrRng = np.hstack((vecClrRngNeg, vecClrRngPos))

    # 'Normalize' object, needed to use custom colour maps and lookup table
    # with matplotlib:
    objClrNorm = colors.BoundaryNorm(vecClrRng, objCustClrMp.N)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Array needs to be transposed for image coordinate conventions.
    aryWghts = aryWghts.T

    # Create plot:
    pltTmpCorr = plt.imshow(aryWghts,
                            interpolation='none',  # 'bicubic',
                            origin='lower',
                            norm=objClrNorm,
                            cmap=objCustClrMp,
                            aspect='equal')

    if not (tpleLimX is None):

        # Position of x labels:
        vecPosLblX = np.linspace(0.0,
                                 (aryWghts.shape[1] - 1),
                                 num=int(tpleLimX[2]),
                                 endpoint=True)
        vecPosLblX = np.around(vecPosLblX, decimals=2)
        lstPosLblX = vecPosLblX.tolist()

        # Set position of x labels:
        axsTmp.set_xticks(lstPosLblX)

        # Labels for x axis:
        vecLblX = np.linspace(tpleLimX[0],
                              tpleLimX[1],
                              num=int(tpleLimX[2]),
                              endpoint=True).tolist()
        vecLblX = np.around(vecLblX, decimals=2)
        lstLblX = vecLblX.tolist()

        # Set axis labels:
        axsTmp.set_xticklabels(lstLblX)

    if not (tpleLimY is None):

        # Position of y labels:
        vecPosLblY = np.linspace(0.0,
                                 (aryWghts.shape[0] - 1),
                                 num=int(tpleLimY[2]),
                                 endpoint=True)
        vecPosLblY = np.around(vecPosLblY, decimals=2)
        lstPosLblY = vecPosLblY.tolist()

        # Set position of y labels:
        axsTmp.set_yticks(lstPosLblY)

        # Labels for y axis:
        vecLblY = np.linspace(tpleLimY[0],
                              tpleLimY[1],
                              num=int(tpleLimY[2]),
                              endpoint=True).tolist()
        vecLblY = np.around(vecLblY, decimals=2)
        lstLblY = vecLblY.tolist()

        # Set axis labels:
        axsTmp.set_yticklabels(lstLblY)

    # Turn of ticks:
    axsTmp.tick_params(labelcolor=([0.0, 0.0, 0.0]),
                       top=False,
                       bottom=False,
                       left=False,
                       right=False)

    # We create invisible axes for the colour bar slightly to the right of the
    # position of the last data-axes. First, retrieve position of last
    # data-axes:
    objBbox = axsTmp.get_position()
    # We slightly adjust the x-position of the colour-bar axis, by shifting
    # them to the right:
    vecClrAxsPos = np.array([(objBbox.x0 * 7.5),
                             objBbox.y0,
                             objBbox.width,
                             objBbox.height])
    # Create colour-bar axis:
    axsClr = fig01.add_axes(vecClrAxsPos,
                            frameon=False)

    # Add colour bar:
    pltClrbr = fig01.colorbar(pltTmpCorr,
                              ax=axsClr,
                              fraction=1.0,
                              shrink=1.0)

    # The values to be labeled on the colour bar:
    # vecClrLblsPos01 = np.arange(varMin, 0.0, 10)
    # vecClrLblsPos02 = np.arange(0.0, varMax, 100)
    vecClrLblsPos01 = np.linspace(varMin, 0.0, num=3)
    vecClrLblsPos02 = np.linspace(0.0, varMax, num=3)
    vecClrLblsPos = np.hstack((vecClrLblsPos01, vecClrLblsPos02))

    # The labels (strings):
    # vecClrLblsStr = map(str, vecClrLblsPos)
    vecClrLblsStr = [str(x) for x in vecClrLblsPos]

    # Set labels on coloubar:
    pltClrbr.set_ticks(vecClrLblsPos)
    pltClrbr.set_ticklabels(vecClrLblsStr)
    # Set font size of colour bar ticks, and remove the 'spines' on the right
    # side:
    pltClrbr.ax.tick_params(labelsize=8.0,
                            tick2On=False)

    # Make colour-bar axis invisible:
    axsClr.axis('off')

    # Save figure:
    fig01.savefig(strPathOut,
                  dpi=varDpi,
                  facecolor='w',
                  edgecolor='w',
                  orientation='landscape',
                  bbox_inches='tight',
                  pad_inches=0.2,
                  transparent=False)

    # Close figure:
    plt.close(fig01)
