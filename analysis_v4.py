# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
import sys
import os
import openseespy.opensees as op
import opsvis as opsv
import matplotlib.pyplot as plt
import math
import numpy as np
import vfo.vfo as vfo
import time as tt
from matplotlib.animation import FuncAnimation
import opstool as ops
#import opstool.vis.plotly as opsvis
from units_v4 import get_units, CreateEqDof, CollectMaxNodes
from getMat_v4 import *
from getSection2_v4 import *

from config_inputs_v4 import *
def ModalAnalysis(Nmodes, ExictationDir, model):
    vfo.createODB(model=model, loadcase="ModalAnalysis", Nmodes=Nmodes )
    # omega2 = op.eigen('fullGenLapack', Nmodes)
    # omega2 = op.eigen(Nmodes)
    modes  = list(range(1, Nmodes))
    # op.modalProperties('-print')
    op.printA('-file', 'modalAnalysis.txt')
    modes = list(range(1, Nmodes + 1))
    
    DOFs = [1, 2]       # DOFs per node (1-based)
    Nmodes = len(modes)
    Nnodes = len(op.getNodeTags())
    
    # Initialize Phi: rows = nodes, columns = modes*DOFs
    Phi = np.zeros((Nnodes, Nmodes * len(DOFs)))
    
    for i_node, nodeid in enumerate(op.getNodeTags()):
        col_idx = 0
        for mode in modes:
            for dof in DOFs:
                Phi[i_node, col_idx] = op.nodeEigenvector(nodeid, mode, dof)
                col_idx += 1
    
    # Save CSV pattern of mode1dof1 mode1dof1 mode2dof1 mode2dof2...... mode_n_dof1 mode_n_dof2 
    np.savetxt("Phi_8Mode1-2dofs_dated12092025-1.csv", Phi, delimiter=",")
   
    fig_wi_he = 100., 100. 
    # for Nmode in  modes:
    #     opsv.plot_mode_shape(Nmode, endDispFlag=0, 
    #                           fmt_defo={'color': 'blue', 'linestyle': 'solid', 'linewidth': 1.2, 'marker': '', 'markersize': 1}, 
    #                           fmt_undefo={'color': 'red', 'linestyle': (0, (1, 1)), 'linewidth': 1, 'marker': '', 'markersize': 1}, 
    #                           fig_wi_he=fig_wi_he)
    #     plt.savefig(Model_dir+str(Nmode)+'ModeShape.png')
    return Phi
# fig = opst.vis.plotly.plot_model(show_local_axes=True)
# fig.show()
