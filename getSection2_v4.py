# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 18:41:56 2023

@author: abahugu
"""
import openseespy.opensees as op
import opsvis as opsv
import matplotlib.pyplot as plt
import math
import numpy as np
from units_v4 import get_units
from getMat_v4 import * 
import os as os
#op.wipe()
m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi = get_units()
Ubig = 1.e10 			# a really large number
Usmall = 1.0/Ubig 		# a really small number
dpi=500
# m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi, lbf, pcf, psf, inch2, inch4, g, Ubig, Usmall= get_units2()

path = os.getcwd() #get the current directory path
section_dir = os.path.join(path, 'SectionFigs/')
try:
    os.makedirs(section_dir)
except:
    print ('skipping')
#%%

def getZerolengthSection (DSec, coverSec, numBarsSec, barAreaSec, SecTag, IDconcCore, IDconcCover, IDreinf):
    # MATERIAL parameters -------------------------------------------------------------------
    # IDconcCore  = 1 				# material ID tag -- confined core concrete
    # IDconcCover = 2 				# material ID tag -- unconfined cover concrete
    # IDreinf     = 3 				# material ID tag -- reinforcement

    # Generate a circular reinforced concrete section
    # with one layer of steel evenly distributed around the perimeter and a confined core.
    # confined core.
    #
    # Notes
    #    The center of the reinforcing bars are placed at the inner radius
    #    The core concrete ends at the inner radius (same as reinforcing bars)
    #    The reinforcing bars are all the same size
    #    The center of the section is at (0,0) in the local axis system
    #    Zero degrees is along section y-axis
    
    # path = os.getcwd() #get the current directory path
    # section_dir = os.path.join(path, 'SectionFigs\\')
    
    # if not os.path.isdir(section_dir):
    #     os.makedirs(section_dir)
    
    # Circular Column
    ri = 0.0        		# inner radius of the section, only for hollow sections
    ro = DSec/2         	# overall (outer) radius of the section
    nfCoreR = 20     		# number of radial divisions in the core (number of "rings")
    nfCoreT = 20     		# number of theta divisions in the core (number of "wedges")
    nfCoverR = 5   		# number of radial divisions in the cover
    nfCoverT = 20	       	# number of theta divisions in the cover
    rc = ro-coverSec        # Core radius
    theta = 360.0/numBarsSec	# Determine angle increment between bars
    
    # Define the fiber section
    # print('=======================================================')
    print('Zero Length Element Fiber Section : SecTag', SecTag)
    fib_sec_11 = [['section', 'Fiber', SecTag , '-GJ', Ubig],
                 ['patch', 'circ', IDconcCore, nfCoreT, nfCoreR, 0, 0, ri, rc, 0, 360], # Define the core patch
                 ['patch', 'circ', IDconcCover, nfCoverT, nfCoverR, 0, 0, rc, ro, 0, 360],
                 ['layer', 'circ', IDreinf, numBarsSec, barAreaSec, 0, 0, rc, theta, 360]
                 ]
    matcolor = ['darkgrey', 'lightgrey', 'gold', 'w', 'w', 'w']
    # opsv.plot_fiber_section(fib_sec_1, matcolor=matcolor)
    # plt.axis('equal')
    # plt.grid(True,linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    
    # plt.savefig(section_dir+'ColumnSection.png', dpi=dpi)
    # plt.show()
    
    #     Apply the section
    opsv.fib_sec_list_to_cmds(fib_sec_11)
    
    
    
    return SecTag


def getColumnSection2D (DSec, coverSec, numBarsSec, barAreaSec, SecTag, IDconcCore, IDconcCover, IDreinf, spacing):
    # MATERIAL parameters -------------------------------------------------------------------
    # IDconcCore  = 1                 # material ID tag -- confined core concrete
    # IDconcCover = 2                 # material ID tag -- unconfined cover concrete
    # IDreinf     = 3                 # material ID tag -- reinforcement

    # Generate a circular reinforced concrete section
    # with one layer of steel evenly distributed around the perimeter and a confined core.
    
    # Circular Column
    ri = 0.0                # inner radius of the section, only for hollow sections
    ro = DSec/2             # overall (outer) radius of the section
    nfCoreR = 20             # number of radial divisions in the core (number of "rings")
    nfCoreT = 20             # number of theta divisions in the core (number of "wedges")
    nfCoverR = 5           # number of radial divisions in the cover
    nfCoverT = 20               # number of theta divisions in the cover
    rc = ro-coverSec        # Core radius
    theta = 360.0/numBarsSec    # Determine angle increment between bars
    
    # Define the fiber section
    # print('=======================================================')
    print('Column Fiber Section : SecTag', SecTag)
    fib_sec_1 = [['section', 'Fiber', SecTag , '-GJ', Ubig],
                 
                 # --- CENTER COLUMN --- (y = 0, z = 0)
                 ['patch', 'circ', IDconcCore, nfCoreT, nfCoreR, 0, 0, ri, rc, 0, 360], 
                 ['patch', 'circ', IDconcCover, nfCoverT, nfCoverR, 0, 0, rc, ro, 0, 360],
                 ['layer', 'circ', IDreinf, numBarsSec, barAreaSec, 0, 0, rc, theta, 360],
                 
                 # --- LEFT COLUMN --- (y = 0, z = -spacing)
                 ['patch', 'circ', IDconcCore, nfCoreT, nfCoreR, 0, -spacing, ri, rc, 0, 360], 
                 ['patch', 'circ', IDconcCover, nfCoverT, nfCoverR, 0, -spacing, rc, ro, 0, 360],
                 ['layer', 'circ', IDreinf, numBarsSec, barAreaSec, 0, -spacing, rc, theta, 360],
                 
                 # --- RIGHT COLUMN --- (y = 0, z = spacing)
                 ['patch', 'circ', IDconcCore, nfCoreT, nfCoreR, 0, spacing, ri, rc, 0, 360], 
                 ['patch', 'circ', IDconcCover, nfCoverT, nfCoverR, 0, spacing, rc, ro, 0, 360],
                 ['layer', 'circ', IDreinf, numBarsSec, barAreaSec, 0, spacing, rc, theta, 360]
                 ]
                 
    matcolor = ['darkgrey', 'lightgrey', 'gold', 'w', 'w', 'w']
    opsv.plot_fiber_section(fib_sec_1, matcolor=matcolor)
    plt.axis('equal')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    
    # plt.savefig(section_dir+'ColumnSection.png', dpi=dpi)
    plt.show()
    
    #     Apply the section
    opsv.fib_sec_list_to_cmds(fib_sec_1)


def getColumnSection (DSec, coverSec, numBarsSec, barAreaSec, SecTag, IDconcCore, IDconcCover, IDreinf):
    # MATERIAL parameters -------------------------------------------------------------------
    # IDconcCore  = 1 				# material ID tag -- confined core concrete
    # IDconcCover = 2 				# material ID tag -- unconfined cover concrete
    # IDreinf     = 3 				# material ID tag -- reinforcement

    # Generate a circular reinforced concrete section
    # with one layer of steel evenly distributed around the perimeter and a confined core.
    # confined core.
    #
    # Notes
    #    The center of the reinforcing bars are placed at the inner radius
    #    The core concrete ends at the inner radius (same as reinforcing bars)
    #    The reinforcing bars are all the same size
    #    The center of the section is at (0,0) in the local axis system
    #    Zero degrees is along section y-axis
    
    # path = os.getcwd() #get the current directory path
    # section_dir = os.path.join(path, 'SectionFigs\\')
    
    # if not os.path.isdir(section_dir):
    #     os.makedirs(section_dir)
    
    # Circular Column
    ri = 0.0        		# inner radius of the section, only for hollow sections
    ro = DSec/2         	# overall (outer) radius of the section
    nfCoreR = 20     		# number of radial divisions in the core (number of "rings")
    nfCoreT = 20     		# number of theta divisions in the core (number of "wedges")
    nfCoverR = 5   		# number of radial divisions in the cover
    nfCoverT = 20	       	# number of theta divisions in the cover
    rc = ro-coverSec        # Core radius
    theta = 360.0/numBarsSec	# Determine angle increment between bars
    
    # Define the fiber section
    # print('=======================================================')
    print('Column Fiber Section : SecTag', SecTag)
    fib_sec_1 = [['section', 'Fiber', SecTag , '-GJ', Ubig],
                 ['patch', 'circ', IDconcCore, nfCoreT, nfCoreR, 0, 0, ri, rc, 0, 360], # Define the core patch
                 ['patch', 'circ', IDconcCover, nfCoverT, nfCoverR, 0, 0, rc, ro, 0, 360],
                 ['layer', 'circ', IDreinf, numBarsSec, barAreaSec, 0, 0, rc, theta, 360]
                 ]
    matcolor = ['darkgrey', 'lightgrey', 'gold', 'w', 'w', 'w']
    # opsv.plot_fiber_section(fib_sec_1, matcolor=matcolor)
    # plt.axis('equal')
    # plt.grid(True,linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    
    # plt.savefig(section_dir+'ColumnSection.png', dpi=dpi)
    # plt.show()
    
    #     Apply the section
    opsv.fib_sec_list_to_cmds(fib_sec_1)
    
    
    
    return SecTag

#%%
def getBeamSection(WSec,  DSec, coverSec, barAreaSec, SecTag, IDconcCore, IDconcCover, IDreinf):
    
    path = os.getcwd() #get the current directory path
    section_dir = os.path.join(path, 'SectionFigs/')
    
    
    # MATERIAL parameters -------------------------------------------------------------------
    IDconcCore  = 1 				# material ID tag -- confined core concrete
    IDconcCover = 2 				# material ID tag -- unconfined cover concrete
    IDreinf     = 3 				# material ID tag -- reinforcement

    # some variables derived from the parameters
    y1 = DSec/2.0   # neutral axis point in local y axis
    z1 = WSec/2.0   # neutral axis point in local z axis
    barAreaSec1 = 0.31*inch**2
#     # Define the fiber section
    # print('=======================================================')
    print('Interior Bent Cap Beam Fiber Section : SecTag', SecTag)
    fib_sec_2 = [['section', 'Fiber', SecTag , '-GJ', 1.0e6],
                  ['patch','rect', IDconcCore, 20, 20, coverSec-y1, coverSec-z1, y1-coverSec, z1-coverSec], # Define the core patch
                  ['patch','rect', IDconcCover, 20, 2, -y1, z1-coverSec, y1, z1],
                  ['patch','rect', IDconcCover, 20, 2, -y1, -z1, y1, coverSec-z1],
                  ['patch','rect', IDconcCover,  2, 20, -y1, coverSec-z1, coverSec-y1, z1-coverSec],
                  ['patch','rect', IDconcCover,  2, 20, y1-coverSec, coverSec-z1, y1, z1-coverSec],
                  ['layer','straight', IDreinf, 9, barAreaSec, y1-coverSec, z1-coverSec, y1-coverSec, coverSec-z1],
                  ['layer','straight', IDreinf, 2, barAreaSec1, (y1-coverSec)*0.333, z1-coverSec, (y1-coverSec)*0.333, coverSec-z1],
                  ['layer','straight', IDreinf, 2, barAreaSec1, (y1-coverSec)*0.666, z1-coverSec, (y1-coverSec)*0.666, coverSec-z1],
                  ['layer','straight', IDreinf, 2, barAreaSec1, 0.0     , z1-coverSec, 0.0      , coverSec-z1],
                  ['layer','straight', IDreinf, 2, barAreaSec1, -(y1-coverSec)*0.333, z1-coverSec, -(y1-coverSec)*0.333, coverSec-z1],
                  ['layer','straight', IDreinf, 2, barAreaSec1, -(y1-coverSec)*0.666, z1-coverSec, -(y1-coverSec)*0.666, coverSec-z1],
                  ['layer','straight', IDreinf, 9, barAreaSec, coverSec-y1, z1-coverSec, coverSec-y1, coverSec-z1]
                  ]
    matcolor = ['darkgrey', 'lightgrey', 'r', 'w', 'w', 'w']
    # opsv.plot_fiber_section(fib_sec_2, matcolor=matcolor)
    # plt.axis('equal')
    # plt.grid(True,linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    # plt.savefig(section_dir+'CapBeamSection.png', dpi=dpi)
    # plt.show()
    
    #     Apply the section
    opsv.fib_sec_list_to_cmds(fib_sec_2)

     
    return SecTag

#%%
def getEndBeamSection(WSec,  DSec, coverSec, barAreaSec, SecTag, IDconcCore, IDconcCover, IDreinf):
    # MATERIAL parameters -------------------------------------------------------------------
    # material ID tag -- confined core concrete
    # material ID tag -- unconfined cover concrete
    # material ID tag -- reinforcement

    # some variables derived from the parameters
    y1 = DSec/2.0   # neutral axis point in local y axis
    z1 = WSec/2.0   # neutral axis point in local z axis
    barAreaSec1 = 0.31*inch**2
    
    path = os.getcwd() #get the current directory path
    section_dir = os.path.join(path, 'SectionFigs/')
    
    #if not os.path.isdir(results_dir):
        #os.makedirs(results_dir)
        
#     # Define the fiber section
    # print('=======================================================')
    print('EndBent Cap Beam Fiber Section : SecTag', SecTag)
    fib_sec_2 = [['section', 'Fiber', SecTag , '-GJ', 1.0e6],
                  ['patch','rect', IDconcCore, 20, 20, coverSec-y1, coverSec-z1, y1-coverSec, z1-coverSec], # Define the core patch
                  ['patch','rect', IDconcCover, 20, 2, -y1, z1-coverSec, y1, z1],
                  ['patch','rect', IDconcCover, 20, 2, -y1, -z1, y1, coverSec-z1],
                  ['patch','rect', IDconcCover,  2, 20, -y1, coverSec-z1, coverSec-y1, z1-coverSec],
                  ['patch','rect', IDconcCover,  2, 20, y1-coverSec, coverSec-z1, y1, z1-coverSec],
                  ['layer','straight', IDreinf, 9, barAreaSec, y1-coverSec, z1-coverSec, y1-coverSec, coverSec-z1],
                  ['layer','straight', IDreinf, 2, barAreaSec1, (y1-coverSec)*0.2, z1-coverSec, (y1-coverSec)*0.2, coverSec-z1],
                  ['layer','straight', IDreinf, 2, barAreaSec1, (y1-coverSec)*0.6, z1-coverSec, (y1-coverSec)*0.6, coverSec-z1],
                  #['layer','straight', IDreinf, 2, barAreaSec, (y1-coverSec)*0.75, z1-coverSec, (y1-coverSec)*0.75, coverSec-z1],
                  # ['layer','straight', IDreinf, 2, barAreaSec1, 0.0     , z1-coverSec, 0.0      , coverSec-z1],
                  ['layer','straight', IDreinf, 2, barAreaSec1, -(y1-coverSec)*0.2, z1-coverSec, -(y1-coverSec)*0.2, coverSec-z1],
                  ['layer','straight', IDreinf, 2, barAreaSec1, -(y1-coverSec)*0.6, z1-coverSec, -(y1-coverSec)*0.6, coverSec-z1],
                  #['layer','straight', IDreinf, 2, barAreaSec, -(y1-coverSec)*0.75, z1-coverSec, -(y1-coverSec)*0.75, coverSec-z1]
                  ['layer','straight', IDreinf, 9, barAreaSec, coverSec-y1, z1-coverSec, coverSec-y1, coverSec-z1],
                 ]
    matcolor = ['darkgrey', 'lightgrey', 'r', 'w', 'w', 'w']
    # opsv.plot_fiber_section(fib_sec_2, matcolor=matcolor)
    # plt.axis('equal')
    # plt.grid(True,linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    # plt.savefig(section_dir+'EndBentCapBeam.png', dpi=dpi)
    # plt.show()
    
    #     Apply the section
    opsv.fib_sec_list_to_cmds(fib_sec_2)

     
    return SecTag


#%%
def getShaftSection2D (DSec, coverSec, numBarsSec, barAreaSec, SecTag, IDconcCore, IDconcCover, IDreinf, IDreinf1, spacing):
     # Generate a circular reinforced concrete section
    # with one layer of steel evenly distributed around the perimeter and a confined core.
    # confined core.
    #		by:  Michael H. Scott, 2003
    # 
    #
    # Notes
    #    The center of the reinforcing bars are placed at the inner radius
    #    The core concrete ends at the inner radius (same as reinforcing bars)
    #    The reinforcing bars are all the same size
    #    The center of the section is at (0,0) in the local axis system
    #    Zero degrees is along section y-axis
    
    # path = os.getcwd() #get the current directory path
    # section_dir = os.path.join(path, 'SectionFigs\\')
    
    # Circular Column
    ri = 0.0        		# inner radius of the section, only for hollow sections
    ro = DSec/2         	# overall (outer) radius of the section
    nfCoreR = 20     		# number of radial divisions in the core (number of "rings")
    nfCoreT = 20     		# number of theta divisions in the core (number of "wedges")
    nfCoverR = 5    		# number of radial divisions in the cover
    nfCoverT = 20	       	# number of theta divisions in the cover
    rc = ro-coverSec        # Core radius
    pipeThickness = 0.5*inch
    theta = 360.0/numBarsSec	# Determine angle increment between bars
    
    
  
    # Define the fiber section
    # print('=======================================================')
    print('Column Fiber Section : SecTag', SecTag)
    fib_sec_3 = [['section', 'Fiber', SecTag , '-GJ', Ubig],
                 
                 # --- CENTER COLUMN --- (y = 0, z = 0)
                 
                 ['patch', 'circ', IDreinf1, nfCoverT, nfCoverR, 0, 0, ro, ro+pipeThickness, 0, 360], #define the steel casing of pipe
                 ['patch', 'circ', IDconcCover, nfCoverT, nfCoverR, 0, 0, rc, ro, 0, 360], #define the cover path
                 ['patch', 'circ', IDconcCore, nfCoreT, nfCoreR, 0, 0, ri, rc, 0, 360], # Define the core patch
                 ['layer', 'circ', IDreinf, numBarsSec, barAreaSec, 0, 0, rc, theta, 360],
                 
                
                 # --- LEFT COLUMN --- (y = 0, z = -spacing)
                 ['patch', 'circ', IDreinf1, nfCoverT, nfCoverR, 0, -spacing, ro, ro+pipeThickness, 0, 360], #define the steel casing of pipe
                 ['patch', 'circ', IDconcCover, nfCoverT, nfCoverR, 0, -spacing, rc, ro, 0, 360], #define the cover path
                 ['patch', 'circ', IDconcCore, nfCoreT, nfCoreR, 0, -spacing, ri, rc, 0, 360], # Define the core patch
                 ['layer', 'circ', IDreinf, numBarsSec, barAreaSec, 0, -spacing, rc, theta, 360],
                 
                 # --- RIGHT COLUMN --- (y = 0, z = spacing)
                 ['patch', 'circ', IDreinf1, nfCoverT, nfCoverR, 0, spacing, ro, ro+pipeThickness, 0, 360], #define the steel casing of pipe
                 ['patch', 'circ', IDconcCover, nfCoverT, nfCoverR, 0, spacing, rc, ro, 0, 360], #define the cover path
                 ['patch', 'circ', IDconcCore, nfCoreT, nfCoreR, 0, spacing, ri, rc, 0, 360], # Define the core patch
                 ['layer', 'circ', IDreinf, numBarsSec, barAreaSec, 0, spacing, rc, theta, 360]
                 ]
                 
    matcolor = ['darkgrey', 'lightgrey', 'gold', 'w', 'w', 'w']
    opsv.plot_fiber_section(fib_sec_3, matcolor=matcolor)
    plt.axis('equal')
    plt.grid(True,linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    
    # plt.savefig(section_dir+'ColumnSection.png', dpi=dpi)
    plt.show()
    
    #     Apply the section
    opsv.fib_sec_list_to_cmds(fib_sec_3)
    return SecTag
    
def getShaftSection (DSec, coverSec, numBarsSec, barAreaSec, SecTag, IDconcCore, IDconcCover, IDreinf, IDreinf1):
    
    # Generate a circular reinforced concrete section
    # with one layer of steel evenly distributed around the perimeter and a confined core.
    # confined core.
    #		by:  Michael H. Scott, 2003
    # 
    #
    # Notes
    #    The center of the reinforcing bars are placed at the inner radius
    #    The core concrete ends at the inner radius (same as reinforcing bars)
    #    The reinforcing bars are all the same size
    #    The center of the section is at (0,0) in the local axis system
    #    Zero degrees is along section y-axis
    
    # path = os.getcwd() #get the current directory path
    # section_dir = os.path.join(path, 'SectionFigs\\')
    
    # Circular Column
    ri = 0.0        		# inner radius of the section, only for hollow sections
    ro = DSec/2         	# overall (outer) radius of the section
    nfCoreR = 20     		# number of radial divisions in the core (number of "rings")
    nfCoreT = 20     		# number of theta divisions in the core (number of "wedges")
    nfCoverR = 5    		# number of radial divisions in the cover
    nfCoverT = 20	       	# number of theta divisions in the cover
    rc = ro-coverSec        # Core radius
    pipeThickness = 0.5*inch
    theta = 360.0/numBarsSec	# Determine angle increment between bars
    
    
    
    # Define the fiber section
    # print('=======================================================')
    print('Shaft Fiber Section: SecTag', SecTag)
    fib_sec_3 = [['section', 'Fiber', SecTag , '-GJ', Ubig],
                 ['patch', 'circ', IDreinf1, nfCoverT, nfCoverR, 0, 0, ro, ro+pipeThickness, 0, 360], #define the steel casing of pipe
                 ['patch', 'circ', IDconcCover, nfCoverT, nfCoverR, 0, 0, rc, ro, 0, 360], #define the cover path
                 ['patch', 'circ', IDconcCore, nfCoreT, nfCoreR, 0, 0, ri, rc, 0, 360], # Define the core patch
                 ['layer', 'circ', IDreinf, numBarsSec, barAreaSec, 0, 0, rc, theta, 360]
                 ]
    matcolor = ['darkgrey', 'lightgrey', 'gold', 'w', 'w', 'w']
    # opsv.plot_fiber_section(fib_sec_3, matcolor=matcolor)
    # plt.axis('equal')
    # plt.grid(True,linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    # plt.savefig(section_dir+'ShaftSection.png', dpi=dpi)
    # # plt.savefig('fibsec_wshape.png')
    # plt.show()
    
    #     Apply the section
    opsv.fib_sec_list_to_cmds(fib_sec_3)
    
    
    
    return SecTag
#%%

def getSocketSection (DSec, coverSec, numBarsSec, barAreaSec, SecTag, IDconcCore, IDconcCover, IDreinf):
    '''
    

    Parameters
    ----------
    DSec : TYPE
        DESCRIPTION. Diameter of the section
    coverSec : TYPE
        DESCRIPTION. cover to reinforcing steel
    numBarsSec : TYPE
        DESCRIPTION. Number of longitudinal steel bars
    barAreaSec : TYPE
        DESCRIPTION. Area of the steel bar
    SecTag : TYPE
        DESCRIPTION. Section tag given 
    IDconcCore : TYPE
        DESCRIPTION. Material tag for core concerete
    IDconcCover : TYPE
        DESCRIPTION. Material tag for cover concrete
    IDreinf : TYPE
        DESCRIPTION. Mtarial tag for steel bar

    Returns
    -------
    SecTag : TYPE
        DESCRIPTION.

    '''
    # Generate a circular reinforced concrete section
    # with one layer of steel evenly distributed around the perimeter and a confined core.
    # confined core.
    #
    # Notes
    #    The center of the reinforcing bars are placed at the inner radius
    #    The core concrete ends at the inner radius (same as reinforcing bars)
    #    The reinforcing bars are all the same size
    #    The center of the section is at (0,0) in the local axis system
    #    Zero degrees is along section y-axis
    # path = os.getcwd() #get the current directory path
    # section_dir = os.path.join(path, 'SectionFigs\\')
    
    # Circular Column
    ri = 0.0        		# inner radius of the section, only for hollow sections
    ro = DSec/2         	# overall (outer) radius of the section
    nfCoreR = 20     		# number of radial divisions in the core (number of "rings")
    nfCoreT = 20     		# number of theta divisions in the core (number of "wedges")
    nfCoverR = 5    		# number of radial divisions in the cover
    nfCoverT = 20	       	# number of theta divisions in the cover
    rc = ro-coverSec        # Core radius
    theta = 360.0/numBarsSec	# Determine angle increment between bars
    
    # Define the fiber section
    # print('=======================================================')
    print('Socket Fiber Section: SecTag', SecTag)
    fib_sec_4 = [['section', 'Fiber', SecTag , '-GJ', Ubig],
                 ['patch', 'circ', IDconcCore, nfCoreT, nfCoreR, 0, 0, ri, rc, 0, 360], # Define the core patch
                 ['patch', 'circ', IDconcCover, nfCoverT, nfCoverR, 0, 0, rc, ro, 0, 360],
                 ['layer', 'circ', IDreinf, numBarsSec, barAreaSec, 0, 0, rc, theta, 360]
                 ]
    matcolor = ['darkgrey', 'lightgrey', 'gold', 'w', 'w', 'w']
    # opsv.plot_fiber_section(fib_sec_4, matcolor=matcolor)
    # plt.axis('equal')
    # plt.grid(True,linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    # plt.savefig(section_dir+'Socket Section.png',dpi=dpi)
    # plt.show()
    
    
    #     Apply the section
    opsv.fib_sec_list_to_cmds(fib_sec_4)
    
    
    
    return SecTag

#%%
def HSection (SecTag, matTag, HsectionDepth, TopFlangeWidth, TopFlangeThickness, BottomFlangeWidth, BottomFlangeThickness, WebThickness):
    
    f1 = TopFlangeWidth    
    f2 = BottomFlangeWidth
    d   = HsectionDepth
    tw  = WebThickness
    tf1 = TopFlangeThickness
    tf2 = BottomFlangeThickness
    h   = HsectionDepth-(TopFlangeThickness+BottomFlangeThickness)

    # print('=======================================================')
    print('H Pile Fiber Section: SecTag', SecTag)
    fib_sec_5 = [['section', 'Fiber', SecTag, '-GJ', 1.0e6],
                 ['patch', 'quad', matTag, 12,  4,  -f1/2,  (d/2-tf1), f1/2,  (d/2-tf1),  f1/2,       d/2,       -f1/2,  d/2      ],
                 ['patch', 'quad', matTag,  4, 64,  -tw/2, (-d/2+tf2), tw/2,  (-d/2+tf2), tw/2,       (d/2-tf1), -tw/2, (d/2-tf1) ],
                 ['patch', 'quad', matTag, 12,  4,  -f2/2,  -d/2,      f2/2, -d/2,        f2/2,     (-d/2+tf2),  -f2/2, (-d/2+tf2)],
                 ]
    matcolor = ['lightgrey', 'grey', 'w', 'gold', 'w', 'w', 'w']
    # opsv.plot_fiber_section(fib_sec_5, matcolor=matcolor)
    # plt.axis('equal')
    # plt.grid(True,linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    # plt.savefig(section_dir+'H-Section.png')
    # plt.show()
    opsv.fib_sec_list_to_cmds(fib_sec_5)
    
    return SecTag
#%%
def HSection1 (SecTag, matTag, HsectionDepth, TopFlangeWidth, TopFlangeThickness, BottomFlangeWidth, BottomFlangeThickness, WebThickness):
    
    f1 = TopFlangeWidth    
    f2 = BottomFlangeWidth
    d   = HsectionDepth
    tw  = WebThickness
    tf1 = TopFlangeThickness
    tf2 = BottomFlangeThickness
    h   = HsectionDepth-(TopFlangeThickness+BottomFlangeThickness)

    # print('=======================================================')
    print('H Pile Fiber Section: SecTag', SecTag)
    fib_sec_5 = [['section', 'Fiber', SecTag, '-GJ', 1.0e6],
                 ['patch', 'quad', matTag, 12,  4,  -f1/2,  (d/2-tf1)+1.5*ft, f1/2,  (d/2-tf1)+1.5*ft,  f1/2,       d/2+1.5*ft,       -f1/2,  d/2+1.5*ft      ],
                 ['patch', 'quad', matTag,  4, 64,  -tw/2, (-d/2+tf2)+1.5*ft, tw/2,  (-d/2+tf2)+1.5*ft, tw/2,       (d/2-tf1)+1.5*ft, -tw/2, (d/2-tf1)+1.5*ft ],
                 ['patch', 'quad', matTag, 12,  4,  -f2/2,  -d/2+1.5*ft,      f2/2, -d/2+1.5*ft,        f2/2,     (-d/2+tf2)+1.5*ft,  -f2/2, (-d/2+tf2)+1.5*ft],
                 
                 ['patch', 'quad', matTag, 12,  4,  -f1/2,  (d/2-tf1)-1.5*ft, f1/2,  (d/2-tf1)-1.5*ft,  f1/2,       d/2-1.5*ft,       -f1/2,  d/2-1.5*ft      ],
                 ['patch', 'quad', matTag,  4, 64,  -tw/2, (-d/2+tf2)-1.5*ft, tw/2,  (-d/2+tf2)-1.5*ft, tw/2,       (d/2-tf1)-1.5*ft, -tw/2, (d/2-tf1)-1.5*ft ],
                 ['patch', 'quad', matTag, 12,  4,  -f2/2,  -d/2-1.5*ft,      f2/2, -d/2-1.5*ft,        f2/2,     (-d/2+tf2)-1.5*ft,  -f2/2, (-d/2+tf2)-1.5*ft],
                 ]
    matcolor = ['lightgrey', 'grey', 'w', 'gold', 'w', 'w', 'w']
    # opsv.plot_fiber_section(fib_sec_5, matcolor=matcolor)
    # plt.axis('equal')
    # plt.grid(True,linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    # plt.savefig(section_dir+'H-Section1.png')
    # plt.show()
    opsv.fib_sec_list_to_cmds(fib_sec_5)
    
    return SecTag

#%%
def HSection2 (SecTag, matTag, HsectionDepth, TopFlangeWidth, TopFlangeThickness, BottomFlangeWidth, BottomFlangeThickness, WebThickness):
    
    f1 = TopFlangeWidth    
    f2 = BottomFlangeWidth
    d   = HsectionDepth
    tw  = WebThickness
    tf1 = TopFlangeThickness
    tf2 = BottomFlangeThickness
    h   = HsectionDepth-(TopFlangeThickness+BottomFlangeThickness)
    s = 2*ft
    # print('=======================================================')
    print('H Pile Fiber Section: SecTag', SecTag)
    fib_sec_5 = [['section', 'Fiber', SecTag, '-GJ', 1.0e6],
                 ['patch', 'quad', matTag, 12,  4,  -f1/2,  (d/2-tf1)+s, f1/2,  (d/2-tf1)+s,  f1/2,       d/2+s,       -f1/2,  d/2+s      ],
                 ['patch', 'quad', matTag,  4, 64,  -tw/2, (-d/2+tf2)+s, tw/2,  (-d/2+tf2)+s, tw/2,       (d/2-tf1)+s, -tw/2, (d/2-tf1)+s ],
                 ['patch', 'quad', matTag, 12,  4,  -f2/2,  -d/2+s,      f2/2, -d/2+s,        f2/2,     (-d/2+tf2)+s,  -f2/2, (-d/2+tf2)+s],
                 
                 ['patch', 'quad', matTag, 12,  4,  -f1/2,  (d/2-tf1)-s, f1/2,  (d/2-tf1)-s,  f1/2,       d/2-s,       -f1/2,  d/2-s      ],
                 ['patch', 'quad', matTag,  4, 64,  -tw/2, (-d/2+tf2)-s, tw/2,  (-d/2+tf2)-s, tw/2,       (d/2-tf1)-s, -tw/2, (d/2-tf1)-s ],
                 ['patch', 'quad', matTag, 12,  4,  -f2/2,  -d/2-s,      f2/2, -d/2-s,        f2/2,     (-d/2+tf2)-s,  -f2/2, (-d/2+tf2)-s],
                 ]
    matcolor = ['lightgrey', 'grey', 'w', 'gold', 'w', 'w', 'w']
    # opsv.plot_fiber_section(fib_sec_5, matcolor=matcolor)
    # plt.axis('equal')
    # plt.grid(True,linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    # plt.savefig(section_dir+'H-Section2.png')
    # plt.show()
    opsv.fib_sec_list_to_cmds(fib_sec_5)
    
    return SecTag

#%%

def DeckSection (SecTag, matTag):
    
    

    NumGirder = 6
    OneSide   = NumGirder/2 
    AcualNum = OneSide-1
    spacing = 8.6666
    # print('=======================================================')
    print('Deck Section: SecTag', SecTag)
    fib_sec_6 = [['section', 'Fiber', SecTag, '-GJ', 1.0e6],
                 
                 
                    ['patch', 'quad', matTag,  7, 3, -0.6875*ft, -24.125*ft,  -0.6875*ft,  24.125*ft,  0.*ft,      24.125*ft,     0.*ft,      -24.125*ft],
                    
                                           
                    ['patch', 'quad', matTag,  2, 2, -0.8958*ft, -5*ft,       -0.8958*ft, -3.6667*ft, -0.6875*ft, -3.6667*ft, -0.6875*ft, -5.*ft],
                    ['patch', 'quad', matTag,  2, 2, -1.4791*ft, -5*ft,       -1.4791*ft, -3.6667*ft, -0.8958*ft, -3.6667*ft, -0.8958*ft, -5.*ft  ],
                    ['patch', 'quad', matTag,  2, 2, -1.8541*ft, -4.625*ft,   -1.8541*ft, -4.0417*ft, -1.4791*ft, -3.6667*ft, -1.4791*ft, -5.*ft ],
                    ['patch', 'quad', matTag,  2, 2, -3.4375*ft, -4.625*ft,   -3.4375*ft, -4.0417*ft, -1.8541*ft, -4.0417*ft, -1.8541*ft, -4.625*ft],
                    ['patch', 'quad', matTag,  2, 2, -4.0635*ft, -5.25*ft,    -4.0625*ft, -3.4167*ft, -3.4375*ft, -4.0417*ft, -3.4375*ft, -4.625*ft],
                    ['patch', 'quad', matTag,  2, 2, -4.6458*ft, -5.25*ft,    -4.6458*ft, -3.4167*ft, -4.0625*ft, -3.4167*ft, -4.0635*ft, -5.25*ft],
                      
                    ['patch', 'quad', matTag,  2, 2, -0.8958*ft, (-5-spacing)*ft,       -0.8958*ft, (-3.6667-spacing)*ft, -0.6875*ft, (-3.6667-spacing)*ft, -0.6875*ft, (-5.-spacing)*ft ],
                    ['patch', 'quad', matTag,  2, 2, -1.4791*ft, (-5-spacing)*ft,       -1.4791*ft, (-3.6667-spacing)*ft, -0.8958*ft, (-3.6667-spacing)*ft, -0.8958*ft, (-5.-spacing)*ft ],
                    ['patch', 'quad', matTag,  2, 2, -1.8541*ft, (-4.625-spacing)*ft,   -1.8541*ft, (-4.0417-spacing)*ft, -1.4791*ft, (-3.6667-spacing)*ft, -1.4791*ft, (-5.-spacing)*ft],
                    ['patch', 'quad', matTag,  2, 2, -3.4375*ft, (-4.625-spacing)*ft,   -3.4375*ft, (-4.0417-spacing)*ft, -1.8541*ft, (-4.0417-spacing)*ft, -1.8541*ft, (-4.625-spacing)*ft],
                    ['patch', 'quad', matTag,  2, 2, -4.0635*ft, (-5.25-spacing)*ft,    -4.0625*ft, (-3.4167-spacing)*ft, -3.4375*ft, (-4.0417-spacing)*ft, -3.4375*ft, (-4.625-spacing)*ft],
                    ['patch', 'quad', matTag,  2, 2, -4.6458*ft, (-5.25-spacing)*ft,    -4.6458*ft, (-3.4167-spacing)*ft, -4.0625*ft, (-3.4167-spacing)*ft, -4.0635*ft, (-5.25-spacing)*ft],
              
                    ['patch', 'quad', matTag,  2, 2, -0.8958*ft, (-5-2*spacing)*ft,       -0.8958*ft, (-3.6667-2*spacing)*ft, -0.6875*ft, (-3.6667-2*spacing)*ft, -0.6875*ft, (-5.-2*spacing)*ft ],
                    ['patch', 'quad', matTag,  2, 2, -1.4791*ft, (-5-2*spacing)*ft,       -1.4791*ft, (-3.6667-2*spacing)*ft, -0.8958*ft, (-3.6667-2*spacing)*ft, -0.8958*ft, (-5.-2*spacing)*ft ],
                    ['patch', 'quad', matTag,  2, 2, -1.8541*ft, (-4.625-2*spacing)*ft,   -1.8541*ft, (-4.0417-2*spacing)*ft, -1.4791*ft, (-3.6667-2*spacing)*ft, -1.4791*ft, (-5.-2*spacing)*ft],
                    ['patch', 'quad', matTag,  2, 2, -3.4375*ft, (-4.625-2*spacing)*ft,   -3.4375*ft, (-4.0417-2*spacing)*ft, -1.8541*ft, (-4.0417-2*spacing)*ft, -1.8541*ft, (-4.625-2*spacing)*ft],
                    ['patch', 'quad', matTag,  2, 2, -4.0635*ft, (-5.25-2*spacing)*ft,    -4.0625*ft, (-3.4167-2*spacing)*ft, -3.4375*ft, (-4.0417-2*spacing)*ft, -3.4375*ft, (-4.625-2*spacing)*ft],
                    ['patch', 'quad', matTag,  2, 2, -4.6458*ft, (-5.25-2*spacing)*ft,    -4.6458*ft, (-3.4167-2*spacing)*ft, -4.0625*ft, (-3.4167-2*spacing)*ft, -4.0635*ft, (-5.25-2*spacing)*ft],
                 
                    ['patch', 'quad', matTag,  2, 2, -0.8958*ft,  3.6667*ft,  -0.8958*ft, 5*ft,      -0.6875*ft,  5.*ft,      -0.6875*ft,  3.6667*ft],
                    ['patch', 'quad', matTag,  2, 2, -1.4791*ft,  3.6667*ft,  -1.4791*ft, 5*ft,      -0.8958*ft,  5.*ft,      -0.8958*ft,  3.6667*ft],
                    ['patch', 'quad', matTag,  2, 2, -1.8541*ft,  4.0417*ft,  -1.8541*ft, 4.625*ft,  -1.4791*ft,  5.*ft,      -1.4791*ft,  3.6667*ft],
                    ['patch', 'quad', matTag,  2, 2, -3.4375*ft,  4.0417*ft,  -3.4375*ft, 4.625*ft,  -1.8541*ft,  4.625*ft,   -1.8541*ft,  4.0417*ft],
                    ['patch', 'quad', matTag,  2, 2, -4.0625*ft,  3.4167*ft,  -4.0635*ft, 5.25*ft,   -3.4375*ft,  4.625*ft,   -3.4375*ft,  4.0417*ft],
                    ['patch', 'quad', matTag,  2, 2, -4.6458*ft,  3.4167*ft,  -4.6458*ft, 5.25*ft,   -4.0635*ft,  5.25*ft,    -4.0625*ft,  3.4167*ft],
                    
                    ['patch', 'quad', matTag,  2, 2, -0.8958*ft,  (3.6667+spacing)*ft,  -0.8958*ft, (5+spacing)*ft,      -0.6875*ft,  (5.+spacing)*ft,      -0.6875*ft,  (3.6667+spacing)*ft],
                    ['patch', 'quad', matTag,  2, 2, -1.4791*ft,  (3.6667+spacing)*ft,  -1.4791*ft, (5+spacing)*ft,      -0.8958*ft,  (5.+spacing)*ft,      -0.8958*ft,  (3.6667+spacing)*ft],
                    ['patch', 'quad', matTag,  2, 2, -1.8541*ft,  (4.0417+spacing)*ft,  -1.8541*ft, (4.625+spacing)*ft,  -1.4791*ft,  (5.+spacing)*ft,      -1.4791*ft,  (3.6667+spacing)*ft],
                    ['patch', 'quad', matTag,  2, 2, -3.4375*ft,  (4.0417+spacing)*ft,  -3.4375*ft, (4.625+spacing)*ft,  -1.8541*ft,  (4.625+spacing)*ft,   -1.8541*ft,  (4.0417+spacing)*ft],
                    ['patch', 'quad', matTag,  2, 2, -4.0625*ft,  (3.4167+spacing)*ft,  -4.0635*ft, (5.25+spacing)*ft,   -3.4375*ft,  (4.625+spacing)*ft,   -3.4375*ft,  (4.0417+spacing)*ft],
                    ['patch', 'quad', matTag,  2, 2, -4.6458*ft,  (3.4167+spacing)*ft,  -4.6458*ft, (5.25+spacing)*ft,   -4.0635*ft,  (5.25+spacing)*ft,    -4.0625*ft,  (3.4167+spacing)*ft],
                    
                    ['patch', 'quad', matTag,  2, 2, -0.8958*ft,  (3.6667+2*spacing)*ft,  -0.8958*ft, (5+2*spacing)*ft,      -0.6875*ft,  (5.+2*spacing)*ft,      -0.6875*ft,  (3.6667+2*spacing)*ft],
                    ['patch', 'quad', matTag,  2, 2, -1.4791*ft,  (3.6667+2*spacing)*ft,  -1.4791*ft, (5+2*spacing)*ft,      -0.8958*ft,  (5.+2*spacing)*ft,      -0.8958*ft,  (3.6667+2*spacing)*ft],
                    ['patch', 'quad', matTag,  2, 2, -1.8541*ft,  (4.0417+2*spacing)*ft,  -1.8541*ft, (4.625+2*spacing)*ft,  -1.4791*ft,  (5.+2*spacing)*ft,      -1.4791*ft,  (3.6667+2*spacing)*ft],
                    ['patch', 'quad', matTag,  2, 2, -3.4375*ft,  (4.0417+2*spacing)*ft,  -3.4375*ft, (4.625+2*spacing)*ft,  -1.8541*ft,  (4.625+2*spacing)*ft,   -1.8541*ft,  (4.0417+2*spacing)*ft],
                    ['patch', 'quad', matTag,  2, 2, -4.0625*ft,  (3.4167+2*spacing)*ft,  -4.0635*ft, (5.25+2*spacing)*ft,   -3.4375*ft,  (4.625+2*spacing)*ft,   -3.4375*ft,  (4.0417+2*spacing)*ft],
                    ['patch', 'quad', matTag,  2, 2, -4.6458*ft,  (3.4167+2*spacing)*ft,  -4.6458*ft, (5.25+2*spacing)*ft,   -4.0635*ft,  (5.25+2*spacing)*ft,    -4.0625*ft,  (3.4167+2*spacing)*ft],
                  
                    ]
    matcolor = ['lightgrey', 'grey', 'w', 'gold', 'w', 'w', 'w']
    # opsv.plot_fiber_section(fib_sec_6, matcolor=matcolor)
    # plt.axis('equal')
    # plt.grid(True,linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    # plt.savefig(section_dir+'H-Section.png')
    # plt.show()
    opsv.fib_sec_list_to_cmds(fib_sec_6)
    
    return SecTag

#%%

def DiaphragmSection (WSec,  DSec, SecTag, IDconcCore):
    
    
    
    #if not os.path.isdir(results_dir):
        #os.makedirs(results_dir)
        
#     # Define the fiber section
    # print('=======================================================')
    print('Diaphragm Fiber Section : SecTag', SecTag)
    fib_sec_ = [['section', 'Fiber', SecTag , '-GJ', 1.0e6],
                  ['patch','rect', IDconcCore, 1, 1, 0., WSec/2, -DSec, -WSec/2], # Define the core patch
                  
                  ]
    
    matcolor = ['lightgrey', 'grey', 'w', 'w', 'gold', 'w', 'w']
    # opsv.plot_fiber_section(fib_sec_, matcolor=matcolor)
    # plt.axis('equal')
    # plt.grid(True,linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    # plt.savefig(section_dir+'Diaphragm-Section.png')
    # plt.show()
    opsv.fib_sec_list_to_cmds(fib_sec_)
    
    return SecTag
    