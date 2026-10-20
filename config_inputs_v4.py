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
from getSection2_v4 import DiaphragmSection as _define_diaphragm_section
from section_properties_v4 import calculate_longitudinal_reinforcement

# m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi, lbf, pcf, psf, inch2, inch4, g, Ubig, Usmall= get_units2()
m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi = get_units()
maxY = -1480*inch # left corner
minY = 520*inch # right corner

#%% Soil Model inputs
width = [[108*inch, maxY, 0*ft],    [-108*inch, maxY, 0*ft],   [-324*inch, maxY, 0*ft],  [-1000*inch, maxY, 0*ft],   [324*inch, maxY, 0*ft], [-654*inch, 2600*inch, 0*ft]]
# depth = 0*ft
length = [[324*inch, minY, 102*ft],  [108*inch, minY, 102*ft],  [-108*inch, minY, 102*ft], [-324*inch, minY, 102*ft],  [1000*inch, minY, 102*ft], [654*inch, 2200*inch, 102*ft]]

#%% Even number elemet in x and y
width_elements = 8    # Element in x direction
length_elements =40   # Element in y direction 
depth_elements = 51   # element in z direction

fixed_distance = 1800*inch  

# print(f'number of elements in soil {width_elements*depth_elements*length_elements}')

# Pile properties
IntBent_horizontal_space = 18*ft
IntBent_logtitudinal_space = 80*ft
# x_location = [logtitudinal_space/2, logtitudinal_space*2, logtitudinal_space* 3, logtitudinal_space* 4 -logtitudinal_space/2, -logtitudinal_space*2, -logtitudinal_space* 3, -logtitudinal_space* 4]
# y_location = [logtitudinal_space/2, logtitudinal_space/2+logtitudinal_space,  -logtitudinal_space/2, -logtitudinal_space/2-logtitudinal_space, -logtitudinal_space/2-logtitudinal_space* 2]
y_location = [-IntBent_logtitudinal_space/2-IntBent_logtitudinal_space* 2, -IntBent_logtitudinal_space/2-IntBent_logtitudinal_space, -IntBent_logtitudinal_space/2, IntBent_logtitudinal_space/2, IntBent_logtitudinal_space/2+IntBent_logtitudinal_space, IntBent_logtitudinal_space/2+IntBent_logtitudinal_space*2]

x_coords = [-IntBent_horizontal_space, 0, IntBent_horizontal_space]
# hole_centers = [(0, 0)]  # List of (x, y) coordinates for piles
# print(y_location, x_coords)
hole_centers = [(x, y) for x in x_coords for y in y_location]
# hole_centers = [(0,-480*inch)]
PileDia = 20*inch
hole_depth = 25*ft     # Depth up to which the pile should be created "add+1ft"
outer_radius = 60*inch  # Outer radius for nodes around the holes

#%% Geometrical location input
#TODO instead of using actual sign use spacing only 
# xlocation = np.array([-18., 0., 18.])*ft      #Location of Column in Lateral direction
xlocation = np.array(x_coords)
# TODO: instead of using cordinate use spacing
bearingLocation = np.array([-288., -260., -216., -156., -52., 0., 52., 156., 216., 260., 288.])*inch # location of bearing on cap beam

# TODO instead of puting distance use c/c spacing b/w pile 
pileLocation = np.array([-22.5, -13.5, -4.5, 4.5, 13.5, 22.5])*ft

# SpanLength = np.array([80., 160, 240, 320, 400])*ft     # length of the span
SpanLength = np.array(y_location)

# SpanLength = np.array([80., 160, 240])*ft     # length of the span
PileYloc = [y_location[0], y_location[-1]]

PileHoles = [(x, y) for x in pileLocation for y in PileYloc]

for x in PileHoles:
    # print(x)
    hole_centers.append(x) 
# print(hole_centers)
#%% Soil layer properties for spring
# =========================================================================
# GLOBAL 9-LAYER SOIL PROFILE DEFINITION
# =========================================================================
g = 9.81*m/sec**2
global_i95_soil_profile = [
    {'layer_id': 1, 'uscs': 'SM', 'top': 102*ft, 'bottom': 94*ft, 'gamma': 1821.*kg/m**3 * g, 'phi': 31.0, 'su': 5*kPa, 'eps50': 0.00, 'G': (8*MPa)/(2*(1+0.27))},
    {'layer_id': 2, 'uscs': 'SM', 'top': 94*ft,  'bottom': 92*ft, 'gamma': 1807.*kg/m**3 * g, 'phi': 30.0, 'su': 5*kPa, 'eps50': 0.0, 'G': (15*MPa)/(2*(1+0.30))},
    {'layer_id': 3, 'uscs': 'SM', 'top': 92*ft,  'bottom': 90*ft, 'gamma': 1728.*kg/m**3 * g, 'phi': 28.0, 'su': 5*kPa, 'eps50': 0.0, 'G': (10*MPa)/(2*(1+0.20))},
    {'layer_id': 4, 'uscs': 'SM', 'top': 90*ft,  'bottom': 88*ft, 'gamma': 1807.*kg/m**3 * g, 'phi': 30.0, 'su': 5*kPa, 'eps50': 0.0, 'G': (26*MPa)/(2*(1+0.30))},
    {'layer_id': 5, 'uscs': 'SM', 'top': 88*ft,  'bottom': 80*ft, 'gamma': 1807.*kg/m**3 * g, 'phi': 30.0, 'su': 5*kPa, 'eps50': 0.0, 'G': (18*MPa)/(2*(1+0.20))},
    {'layer_id': 6, 'uscs': 'CH', 'top': 80*ft,  'bottom': 70*ft, 'gamma': 1885.*kg/m**3 * g, 'phi': 10.0, 'su': 84*kPa, 'eps50': 0.01, 'G': (30*MPa)/(2*(1+0.30))},
    {'layer_id': 7, 'uscs': 'SP', 'top': 70*ft,  'bottom': 54*ft, 'gamma': 1935.*kg/m**3 * g, 'phi': 34.0, 'su': 5*kPa, 'eps50': 0.0, 'G': (30*MPa)/(2*(1+0.30))},
    {'layer_id': 8, 'uscs': 'CH', 'top': 54*ft,  'bottom': 50*ft, 'gamma': 1964.*kg/m**3 * g, 'phi': 10.0, 'su': 108*kPa, 'eps50': 0.01, 'G': (50*MPa)/(2*(1+0.30))},
    {'layer_id': 9, 'uscs': 'SP', 'top': 50*ft,  'bottom': 0*ft,  'gamma': 1964.*kg/m**3 * g, 'phi': 35.0, 'su': 5*kPa, 'eps50': 0.0, 'G': (50*MPa)/(2*(1+0.30))}
]



#%% Input Data for structural model
#****************************************************************************** 
#Structure modelling
#******************************************************************************

# Column section properties
DSec = 2*26.*inch    		# Legacy/default column diameter
coverSec = 3.*inch	     	# Column cover to reinforcing steel NA.
numBarsSec = 18    	    	# number of uniformly-distributed longitudinal-reinforcement bars
_default_pl = 100.0 * numBarsSec * (1.56*inch**2) / (math.pi*DSec**2/4.0)
barAreaSec = calculate_longitudinal_reinforcement(DSec, _default_pl, numBarsSec)["barAreaSec"]
ColumnSecTag = 1  	      		# set tag for symmetric section

ZerolengthSecTag = 11
ZerolengthNBars = 22

# shaft section properties
DSecS = DSec     		# Column Diameter
coverSecS = 6.*inch	     	# Column cover to reinforcing steel NA.
numBarsSecS = 18    	    	# number of uniformly-distributed longitudinal-reinforcement bars
barAreaSecS = calculate_longitudinal_reinforcement(DSecS, _default_pl, numBarsSecS)["barAreaSec"]	# area of longitudinal-reinforcement bars
SSecTag = 2  	      		# set tag for symmetric section

# socket section properties
DSecSo = DSec     		# Diameter
coverSecSo = 4.*inch	     	# cover to reinforcing steel NA.
numBarsSecSo = 18    	    	# number of uniformly-distributed longitudinal-reinforcement bars
barAreaSecSo = calculate_longitudinal_reinforcement(DSecSo, _default_pl, numBarsSecSo)["barAreaSec"]	    # area of longitudinal-reinforcement bars
SoSecTag = 3  	      		    # set tag for symmetric section

# Cap Beam section properties

Width_CapBeam = 1.68*m          		          # Beam width
Depth_CapBeamSec = 1.37*m                       # Beam depth 
coverSec_CapBeam = 2.*inch	     	             # Column cover to reinforcing steel NA.
barAreaSec_CapBeam = 1.56*inch**2	        # area of longitudinal-reinforcement bars #11 bar
CapBeamSecTag = 4	      		            # set tag for symmetric section
CapBeamLength = 14.6304*m                  # Cap Beam Length

# End bent capbeam section properties
WidthEcapBeam = 5.6667*ft          		          # Beam width
DepthEcapBeamSec = 3.5*ft                      # Beam depth 
coverSecEcapBeam = 2.*inch	     	             # Column cover to reinforcing steel NA.
barAreaEcapBeamTopBott = 1.27*inch**2	        # area of longitudinal-reinforcement bars #10 bar
EndBentSecTag = 5
EcapBeamLength = 51*ft+7/2*inch

# HPile section properties
PileSectionTag          = 6
HsectionDepth           = 13.6*inch
TopFlangeWidth          = 14.6*inch
TopFlangeThickness      = 0.505*inch
BottomFlangeWidth       = 14.6*inch
BottomFlangeThickness   = 0.505*inch
WebThickness            = 0.505*inch
HArea                   = 21.4*inch**2*2
# Dec Section 

DecSecTag = 7

# Diaphragm scetion

DiaPhrgmThickness  = 10*inch
DiaPhrgmDepth      = 38*inch
DisphragmSection   = 8
def configure_default_sections():
    """Create the original default materials and sections.

    This is intentionally called explicitly after ``op.model(...)``.  The
    original function calls are preserved; they are no longer executed while
    this module is being imported.
    """
    global IDconcCore, IDconcCover, IDreinf, IDreinf1
    global IDElastomer, IDdowel_F, IDdowel_E, earth_P, CapPile
    global SoSecTag, SSecTag, ColumnSecTag, ZerolengthSecTag
    global CapBeamSecTag, EBCapBeamSecTag, HsecTag, DecSecTag
    global DiaphragmSection, SecTags

    # defining material
    IDconcCore, IDconcCover, IDreinf, IDreinf1, IDElastomer, IDdowel_F, IDdowel_E, earth_P, CapPile = getMat_v4()

    # defining sections
    SoSecTag = getSocketSection(DSecSo, coverSecSo, numBarsSecSo, barAreaSecSo, SoSecTag, IDconcCore, IDconcCover, IDreinf)
    SSecTag = getShaftSection(DSecS, coverSecS, numBarsSecS, barAreaSecS, SSecTag, IDconcCore, IDconcCover, IDreinf, IDreinf1)
    ColumnSecTag = getColumnSection(DSec, coverSec, numBarsSec, barAreaSec, ColumnSecTag, IDconcCore, IDconcCover, IDreinf)
    ZerolengthSecTag = getZerolengthSection(DSec, coverSec, ZerolengthNBars, barAreaSec, ZerolengthSecTag, IDconcCore, IDconcCover, IDreinf)

    CapBeamSecTag = getBeamSection(Width_CapBeam, Depth_CapBeamSec, coverSec_CapBeam, barAreaSec_CapBeam, CapBeamSecTag, IDconcCore, IDconcCover, IDreinf)
    EBCapBeamSecTag = getEndBeamSection(WidthEcapBeam, DepthEcapBeamSec, coverSecEcapBeam, barAreaEcapBeamTopBott, EndBentSecTag, IDconcCore, IDconcCover, IDreinf)
    HsecTag = HSection1(PileSectionTag, IDreinf1, HsectionDepth, TopFlangeWidth, TopFlangeThickness, BottomFlangeWidth, BottomFlangeThickness, WebThickness)
    DecSecTag = DeckSection(DecSecTag, IDreinf)
    DiaphragmSection = _define_diaphragm_section(DiaPhrgmThickness, DiaPhrgmDepth, DisphragmSection, IDconcCover)
    SecTags = np.array([ColumnSecTag, CapBeamSecTag, HsecTag, SoSecTag, SSecTag, DecSecTag, DiaphragmSection, ZerolengthSecTag])
# EndCapBeamSec

# print(Hsection)

# TODO: instead of different variable use dctionary for input parameters
ColumnLength = 103*ft-96*ft                      # column length
HPileLength = 104*ft-48*ft                        # H Pile length
ColumnElementType = 'dispBeamColumn'        # column element type
SocketElementType = 'dispBeamColumn'        # Socket Element Type
SocketLength  = 58*ft-25*ft                      # Length of Column
ShaftLength = 96*ft-58*ft                        # Length of Column
ShaftElementType = 'dispBeamColumn'         # Shaft
CapBeamElementType = 'elasticBeamColumn'    # Cap Beam 
GamaConc = 24*kN/m**3                       # unit weight of Concrete
GamaSteel = 78.5*kN/m**3                    # unit weight of steel

#%% Mass Calculation
g = 9.81*m/sec**2
columnMass   = ((math.pi*DSec**2/4)*ColumnLength*GamaConc)/g
columnWt     = columnMass*g/ColumnLength
print('-----------------------------------------------------------')
print('Weight calculation of structure')
print(f'Column Wt {columnWt} N/m')
ShaftMass    = ((math.pi*DSecS**2/4)*ShaftLength*GamaConc)/g + ((math.pi*(DSecS**2-(DSecS-4*mm)**2)/4)*ShaftLength*GamaSteel)/g
ShaftWt      =  ShaftMass*g/ShaftLength
print(f'Shaft wt  {ShaftWt} N/m')
SocketMass   = ((math.pi*DSecSo**2/4)*SocketLength*GamaConc)/g
SocketWt     = SocketMass*g/SocketLength
print(f'Socket wt {SocketWt} N/m')
PileMass     = (HArea*HPileLength*GamaSteel)/g
PileWt       = PileMass*g/HPileLength
print(f'Pile wt   {PileWt} N/m')
ECapBeamMass = (WidthEcapBeam*DepthEcapBeamSec*EcapBeamLength*GamaConc)/g
ECapBeamWt   =  ECapBeamMass*g/EcapBeamLength
print(f'ECapBeam wt {ECapBeamWt} N/m ')
CapBeamMass  = (Width_CapBeam*Depth_CapBeamSec*CapBeamLength*GamaConc)/g
CapBeamWt    = CapBeamMass*g/CapBeamLength
print(f'CapBeam wt {CapBeamWt} N/m ')

DiaPhrgmArea = (10*inch)*(38.0*inch)
DeckArea     = 6.5169*m**2
ParapetMass  = (2*7.0051*kN/m*80*ft)/g
DiaphrgamMass = (DiaPhrgmArea*(48.25*ft)*GamaConc)/g
DeckMass = (DeckArea*(80*ft)*GamaConc)/g
SStMass = (DiaphrgamMass+DeckMass+ParapetMass)
DiaPhragmLoad = DiaPhrgmArea*GamaConc*100

#additional loads
FWS = 0.7182*kN/m**2 * 48.25*ft # per m length
SIPF = 0.6224*kN/m**2 * 48.25*ft
AdditionalLoad = 201.8632*kN/m**2*1*ft
# SSTWT = (DeckArea)*GamaConc + 2*7.0051*kN/m + DiaPhragmLoad*3 + FWS + (ParapetMass)*g/(80*ft)
SSTWT = (DeckArea)*GamaConc + 2*7.0051*kN/m + FWS + (ParapetMass)*g/(80*ft)+ SIPF

print(f'SSTWT wt {SSTWT} N/m')
print('-----------------------------------------------------------')
# global DeckArea, g, DiaPhrgmArea, GamaConc

    
#%% input for bridge model
IntBent_horizontal_space = 18*ft
IntBent_logtitudinal_space = 80*ft
y_location = [-IntBent_logtitudinal_space/2-IntBent_logtitudinal_space* 2, -IntBent_logtitudinal_space/2-IntBent_logtitudinal_space, -IntBent_logtitudinal_space/2, IntBent_logtitudinal_space/2, IntBent_logtitudinal_space/2+IntBent_logtitudinal_space, IntBent_logtitudinal_space/2+IntBent_logtitudinal_space*2]
# y_location = [-IntBent_logtitudinal_space/2]

x_coords = [-IntBent_horizontal_space, 0, IntBent_horizontal_space]
# x_coords = [0.]
#TODO instead of using actual sign use spacing only 
# xlocation = np.array([-18., 0., 18.])*ft      #Location of Column in Lateral direction
xlocation = np.array(x_coords)
# TODO: instead of using cordinate use spacing
bearingLocation = np.array([-288., -260, -216., -156., -52., 0., 52., 156., 216., 260., 288.])*inch # location of bearing on cap beam

# TODO instead of puting distance use c/c spacing b/w pile 
pileLocation = np.array([-22.5, -13.5, -4.5, 4.5, 13.5, 22.5])*ft
# pileLocation = np.array([-22.708, -13.625, -4.542, 4.542, 13.625, 22.708])*ft
# SpanLength = np.array([80., 160, 240, 320, 400])*ft     # length of the span
SpanLength = np.array(y_location)
# SpanLength = np.array([80., 160, 240])*ft     # length of the span


def configure_case(case):
    """Apply one matrix row while retaining the original model builders.

    This function only reconnects inputs to the existing material and section
    functions.  Calling it after ``op.wipe()`` creates a fresh OpenSees model
    for the supplied row.
    """
    global DSec, DSecS, DSecSo, pl, fc, fy
    global barAreaSec, barAreaSecS, barAreaSecSo
    global IDconcCore, IDconcCover, IDreinf, IDreinf1, IDElastomer, IDdowel_F, IDdowel_E, earth_P, CapPile
    global SoSecTag, SSecTag, ColumnSecTag, ZerolengthSecTag, CapBeamSecTag, EBCapBeamSecTag, HsecTag, DecSecTag, DiaphragmSection, SecTags
    global ColumnLength, ShaftLength, SocketLength, HPileLength
    global xlocation, y_location, SpanLength, PileYloc, PileHoles, pileLocation, hole_centers
    global columnMass, columnWt, ShaftMass, ShaftWt, SocketMass, SocketWt, PileMass, PileWt

    DSec = DSecS = DSecSo = float(case["D"])
    pl = float(case["pl"])
    fc = float(case["fc"])
    fy = float(case["fy"])
    reinforcement = calculate_longitudinal_reinforcement(DSec, pl, 18)
    numBarsSec = numBarsSecS = numBarsSecSo = 18
    barAreaSec = barAreaSecS = barAreaSecSo = reinforcement["barAreaSec"]

    ColumnLength = float(case["H"])
    remaining_length = float(case["L_P"]) - ColumnLength
    ShaftLength = remaining_length * (38.0 / 70.0)
    SocketLength = remaining_length * (32.0 / 70.0)
    HPileLength = float(case["HPileLength"])

    IDconcCore, IDconcCover, IDreinf, IDreinf1, IDElastomer, IDdowel_F, IDdowel_E, earth_P, CapPile = getMat_v4(fc, fy)
    SoSecTag = getSocketSection(DSecSo, coverSecSo, numBarsSecSo, barAreaSecSo, 3, IDconcCore, IDconcCover, IDreinf)
    SSecTag = getShaftSection(DSecS, coverSecS, numBarsSecS, barAreaSecS, 2, IDconcCore, IDconcCover, IDreinf, IDreinf1)
    ColumnSecTag = getColumnSection(DSec, coverSec, numBarsSec, barAreaSec, 1, IDconcCore, IDconcCover, IDreinf)
    ZerolengthSecTag = getZerolengthSection(DSec, coverSec, ZerolengthNBars, barAreaSec, 11, IDconcCore, IDconcCover, IDreinf)
    CapBeamSecTag = getBeamSection(Width_CapBeam, Depth_CapBeamSec, coverSec_CapBeam, barAreaSec_CapBeam, 4, IDconcCore, IDconcCover, IDreinf)
    EBCapBeamSecTag = getEndBeamSection(WidthEcapBeam, DepthEcapBeamSec, coverSecEcapBeam, barAreaEcapBeamTopBott, 5, IDconcCore, IDconcCover, IDreinf)
    HsecTag = HSection1(PileSectionTag, IDreinf1, HsectionDepth, TopFlangeWidth, TopFlangeThickness, BottomFlangeWidth, BottomFlangeThickness, WebThickness)
    DecSecTag = DeckSection(DecSecTag, IDreinf)
    DiaphragmSection = _define_diaphragm_section(DiaPhrgmThickness, DiaPhrgmDepth, DisphragmSection, IDconcCover)
    SecTags = np.array([ColumnSecTag, CapBeamSecTag, HsecTag, SoSecTag, SSecTag, DecSecTag, DiaphragmSection, ZerolengthSecTag])

    number_of_spans = int(case["number_of_spans"])
    span_length = float(case["span_length"])
    y_location = (np.arange(number_of_spans + 1) - number_of_spans / 2.0) * span_length
    SpanLength = np.array(y_location)
    PileYloc = [y_location[0], y_location[-1]]
    number_of_piles = int(case["number_of_piles"])
    pile_spacing = float(case["pile_spacing"])
    pileLocation = (np.arange(number_of_piles) - (number_of_piles - 1) / 2.0) * pile_spacing
    # The matrix defines the number and transverse spacing of piles per bent.
    # The same centered locations are used for interior bent shafts/columns
    # and end-bent H-piles so every bent has the requested pile count.
    xlocation = np.array(pileLocation)
    PileHoles = [(x, y) for x in pileLocation for y in PileYloc]
    hole_centers = [(x, y) for x in xlocation for y in y_location] + PileHoles

    from structural_properties_v4 import calculate_structural_loads
    loads = calculate_structural_loads(
        D=DSec, ColumnLength=ColumnLength, ShaftLength=ShaftLength, SocketLength=SocketLength,
        HPileLength=HPileLength, HArea=HArea, Width_CapBeam=Width_CapBeam,
        Depth_CapBeamSec=Depth_CapBeamSec, CapBeamLength=CapBeamLength,
        WidthEcapBeam=WidthEcapBeam, DepthEcapBeamSec=DepthEcapBeamSec,
        EcapBeamLength=EcapBeamLength, DiaPhrgmArea=DiaPhrgmArea, DeckArea=DeckArea,
        ParapetMass=ParapetMass, FWS=FWS, SIPF=SIPF, AdditionalLoad=AdditionalLoad,
        GamaConc=GamaConc, GamaSteel=GamaSteel, g=9.81*m/sec**2, kN=kN, m=m, mm=mm, ft=ft)
    for name, value in loads.items():
        globals()[name] = value
    return {"loads": loads}
