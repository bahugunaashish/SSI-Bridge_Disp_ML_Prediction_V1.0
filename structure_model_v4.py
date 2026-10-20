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
import time as tt
from matplotlib.animation import FuncAnimation
from units_v4 import get_units, CreateEqDof, CollectMaxNodes
from getMat_v4 import *
from getSection2_v4 import *

from config_inputs_v4 import *
from helpers_v4 import rotate_node, transverse_capbeam_locations


def _soil_layer_for_model_elevation(model_z, soil_layers, profile_to_model_offset, tolerance=1.0e-8):
    """Find the profile layer, extending the bottom layer below profile elevation 0 m."""
    profile_z = float(model_z) - float(profile_to_model_offset)
    if profile_z < 0.0 - tolerance:
        return soil_layers[-1], profile_z, True

    for layer_index, layer in enumerate(soil_layers):
        is_last_layer = layer_index == len(soil_layers) - 1
        at_profile_bottom = is_last_layer and math.isclose(
            profile_z, float(layer['bottom']), rel_tol=0.0, abs_tol=tolerance
        )
        if float(layer['top']) >= profile_z > float(layer['bottom']) or at_profile_bottom:
            return layer, profile_z, False
    return None, profile_z, False


#%% Structure Model Class
class Define_str:
    def __init__(self, L1, AllSoilNodes, SocketLength, SocketElementType, xlocation, ylocation, SecTag, ShaftLength, 
                   ShaftElementType, ColumnLength, ColumnElementType, bearingLocation, CapBeamElementType, Depth_CapBeam, HPileLength, pileLocation,
                   soil_profile=None, load_data=None, geometry=None):
    # def __init__(self, SocketLength, SocketElementType, xlocation, ylocation, SecTag, ShaftLength, 
                  # ShaftElementType, ColumnLength, ColumnElementType, bearingLocation, CapBeamElementType, Depth_CapBeam, HPileLength, pileLocation):
        global m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi, degrees
        global gamma_rc, gamma_c, gamma_s, Ubig, Usmall, g
        # m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi, lbf, pcf, psf, inch2, inch4, g, Ubig, Usmall= get_units2()

        m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi = get_units()   
        #%% define variables
        self.theta = 0. # in degree
        self.L1               = L1
        self.AllSoilNodes     = AllSoilNodes
        self.soil_profile     = soil_profile
        self.load_data        = load_data
        self.geometry         = geometry or {}
        self.column_top_z     = self.geometry.get('column_top_z', 104*ft)
        self.column_bottom_z  = self.geometry.get('column_bottom_z', 98*ft)
        self.shaft_top_z      = self.geometry.get('shaft_top_z', 96*ft)
        self.shaft_bottom_z   = self.geometry.get('shaft_bottom_z', 58*ft)
        self.socket_top_z     = self.geometry.get('socket_top_z', 56*ft)
        self.socket_bottom_z  = self.geometry.get('socket_bottom_z', 24*ft)
        self.pile_top_z       = self.geometry.get('pile_top_z', 104*ft)
        self.pile_bottom_z    = self.geometry.get('pile_bottom_z', 48*ft)
        self.ground_surface_z = self.geometry.get('ground_surface_z', 102*ft)
        self.pile_diameter    = self.geometry.get('pile_diameter', 2*29.75*inch)
        loads = self.load_data or {}
        self.columnMass = loads.get('columnMass', columnMass)
        self.ShaftMass = loads.get('ShaftMass', ShaftMass)
        self.SocketMass = loads.get('SocketMass', SocketMass)
        self.PileMass = loads.get('PileMass', PileMass)
        # self.L2               = L2 # Soil defined 
        self.SocketLength       = SocketLength                                              # socket length
        self.SocketElementType  = SocketElementType                                         # socket element type
        self.xlocation          = xlocation                                                 # location of column in interior bent
        self.ylocation          = ylocation                                                 # cumulative span lengths
        self.EndBentLocation    = np.array([self.ylocation[0], self.ylocation[-1]])        # Location of end bent
        self.SecTag             = SecTag                                                    # Section Tags
        self.ShaftLength        = ShaftLength                                               # Shaft length
        self.ShaftElementType   = ShaftElementType                                          # shaft element type
        self.ColumnLength       = ColumnLength                                              # Column length
        self.ColumnElementType  = ColumnElementType                                         # Column element type
        self.CapBeamElementType = CapBeamElementType                                        # Cap beam element type
        self.PileElementType    = ColumnElementType
        self.SocketShaftLength  = self.ShaftLength + self.SocketLength      # Length of socket + shaft
        self.PierLength         = self.ColumnLength+self.SocketShaftLength  # Total length of pier
        self.bearingLocation    = bearingLocation                           # rigid element type
        self.capBeamLocations   = transverse_capbeam_locations(self.xlocation, self.bearingLocation)
        self.ColumnBeamRigidLnk = Depth_CapBeam/2                           # rigid link length at interior bent
        self.ColumnGeomTranType = 'PDelta'                                  # Column geometric transform type
        self.bearingdepth       = 15/4*inch                                    # bearingdepth
        self.BearingElementType = CapBeamElementType                        # Bearing element type
        self.DiaphElementType   = CapBeamElementType                        # Diaphragm element type
        self.RigidElementType   = CapBeamElementType                        # rigid element type
        self.HplieLength        = HPileLength                               # Pile length
        self.NumPiles           = int(len(pileLocation))                    # number of piles in a row
        self.EndCapBeamLength   = 616.5*inch                                # Length of end bent cap beam
        self.pileLocation       = pileLocation
        self.IntCapBeamDepth    = 4.5*ft                                    # depth of interior cap beam section
        self.topCapBeam         = self.PierLength + self.IntCapBeamDepth    # length from top of interior cap beam to bottom of socket
        self.EndCapBeamDepth    = 3.5*ft
        self.EffPileLength      = self.HplieLength-self.EndCapBeamDepth/2
        self.EndPierLength      = self.HplieLength+self.EndCapBeamDepth/2
        self.BottomPileNodes    = np.array([])
        self.fixNodesTags       = np.array([])
        self.AllDiaphTags       = np.array([])
        self.IntDiaphEleTag     = np.array([])
        self.EndDiaphEleTag     = np.array([])
        self.connection_warnings = []
        #%% Large values to provide rigid elements
        
        self.Ei = 40*GPa
        self.Ai = 9000000.0*m**2
        self.Ii = 900000000.0*m**4
        self.Ji = 900000000.0*m**4
        self.Gi = 900000.0*N/m**2
        
        self.RigidProp = [self.Ai, self.Ei, self.Gi, self.Ji, self.Ii, self.Ii]
        #defining mass
        
        # print(self.topCapBeam , self.EffPileLength+self.EndCapBeamDepth/2)
        #%% 1- Node Tags
        self.tagSo       = 1000        # Socket node tag identifier
        self.tagS        = 3000        # Shaft node tag identifier
        self.tagC        = 5000        # Column node tag identifier
        self.tagCB       = 7000        # Cap Beam node tag identifier
        self.rigidTag    = 9000        # rigid node tag
        self.BearingTag  = 14000        # Bearing node tag
        self.tagPile     = 16000
        self.tagECB      = 18000        # End Cap Beam node tag identifier
        self.tagD        = 0
        self.eqDofnodes  = 19000
        #%% 2- Element Tags  
        self.DeckEleTag       = 0
        self.SocketEleTag     = 1000        # Socket element tag identifier
        self.ShaftEleTag      = 3000        # Shaft element tag identifier
        self.ColumnEleTag     = 5000        # Column element tag identifier
        self.CapBeamEleTag    = 7000        # Cap Beam element tag identifier
        self.RigidEleTag      = 9000        # Rigid link element tag identifier
        self.BearingEleTag    = 40000        # Bearing element tag
        self.DiaphEleTag      = 41000        # Diaphragm element tag
        self.PileEleTag       = 42000
        self.ECapBeamEleTag   = 43000        # Cap Beam element tag identifier
        self.zeroEleTag       = 45000        # zerolength element tags  
        self.EpTag            = 48000        # earthpressure nodes
        self.zeroEleTags = np.array([], dtype=int) 
        
        self.geomTTag      = 0           # Geometrtic transformation tag 
        self.InteTag       = 0           # Beam Integration Tag identifer
        
        self.DeckTags = np.array([], dtype=int)     # array for all Deck Nodes
        self.GirderNodeTags = np.array([], dtype=int) # array for all Girder node Tags
        self.ColumnTopTags = np.array([], dtype=int)
        self.PileTopTags = np.array([], dtype=int)
        self.DeckEleTags = np.array([], dtype=int)
        self.SocketEleTags = np.array([], dtype=int)
        self.ShaftEleTags = np.array([], dtype=int)
        self.ColumnEleTags = np.array([], dtype=int)
        self.CapBeamEleTags =  np.array([], dtype=int)
        self.ECapBeamEleTags = np.array([], dtype=int)
        self.PileElemTags = np.array([], dtype=int)
        self.BottomNodesSo = np.array([], dtype=int)
        self.ShaftColumnNodeTags = np.array([], dtype=int)
        self.EndbentPileNodes = np.array([], dtype=int)
        self.zeroEleTags = np.array([], dtype=int)
        self.EndbentNodeTags = np.array([], dtype=int)
        self.EndbentElemTags = np.array([], dtype=int)
        #%%call all major function (Methods)
        self.initialize_model()
        self._buildStructure_()
        
        # self._Connect_Soil_Pile()
        self._JoinSoilPile()

    def _warn_unmatched_capbeam_nodes(self, source_nodes, capbeam_nodes, label):
        """Warn when a pile/column top has no matching cap-beam node."""
        tolerance = 1.0e-6
        capbeam_x = [op.nodeCoord(int(node))[0] for node in capbeam_nodes]
        unmatched = []
        for source_node in source_nodes:
            source_x = op.nodeCoord(int(source_node))[0]
            if not any(np.isclose(source_x, cap_x, rtol=0.0, atol=tolerance) for cap_x in capbeam_x):
                unmatched.append(source_x)

        if unmatched:
            message = (
                f"{label}: {len(unmatched)} pile/column top node(s) have no "
                f"matching cap-beam node at x={unmatched}"
            )
            self.connection_warnings.append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        return unmatched
        self._eleload()
        # if len(self.AllSoilNodes)==0:
          
            # self.boundary_()  # acitvate if only superstructre needs to analyzed
    
    def getPileElem(self):
        return self.SocketEleTags.tolist(), self.ShaftEleTags.tolist(), self.ColumnEleTags.tolist(), self.BottomNodesSo.tolist()
        
    def getPileNodes(self):
        
        return self.ShaftColumnNodeTags.tolist()
        
        #%%   Initalization ofmodel     
    def initialize_model(self):
        print('Structure model initialized >>')
        op.model('basic', '-ndm', 3, '-ndf', 6)
        
    #%% boundary for strcture without soil     
    def boundary_(self):
        for ii in self.BottomNodesSo:
            print('fix nodes', ii)
            op.fix(int(ii), 1,1,1,1,1,1)
        
    #%% Connecting soil and pile
    def _Connect_Soil_Pile(self):
                
        for L in self.L1:
            #------------------------------------------------------------------
            #NodeSet = L.create_new_nodes_around_holes()
            # NodeSet = L.getNodes()
            # newNodes = self.create_new_nodes_around_holes(NodeSet)            use this when holes are cretaed
            # print(newNodes)     
            #------------------------------------------------------------------
            print('Nodes to connect the pile and soil are generated...')
            # print('Shaft and Column nodes',self.ShaftColumnNodeTags)
            
            # self._JoinNodes(newNodes)  # use this when self.create_new_nodes_around_holes is used 
            self._JoinNodes() 
            
    #%% create_new_nodes_around_holes
    def create_new_nodes_around_holes(self, NodesSet):
        """
        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        op.model('basic', '-ndm', 3, '-ndf', 6)
        print('Creating new nodes around Piles')
        new_node_id = op.getNodeTags()[-1]+1
        new_nodes = []
        # print(f'new node id {self.node_id}')
        # print('soil nodes around pile',self.nodes_around_holes)
        for node_id in NodesSet:
            # print(node_id)
            x, y, z = op.nodeCoord(node_id)
            
            if z != 25*ft :
              op.node(new_node_id, x, y, z)
              new_nodes.append(new_node_id)
              op.equalDOF(new_node_id, node_id, 1,2,3)
              # print(self.new_node_id, node_id, x, y, z)
              # print(new_node_id, node_id)
              new_node_id += 1
        # print('New Nodes around Piles', self.new_nodes)      
        return new_nodes
    
   
                
    #%% joint node when soil and pile connected through rigid elements
    def _JoinNodes(self, NodeList):
        # print(f'Node List {NodeList}')
        
        nodeList = []
        # print(NodeList)
        for Snode in NodeList:                    # Node List should be All soil Nodes
            for Cnode in self.ShaftColumnNodeTags:
                Scord = op.nodeCoord(int(Snode))
                Ccord = op.nodeCoord(int(Cnode))
                
                if np.linalg.norm(Ccord[2] - Scord[2]) < 1e-6 and np.linalg.norm(Ccord[1] - Scord[1])<70*inch and np.linalg.norm(Ccord[0] - Scord[0])<70*inch:
                
                    # print('Connecting ', Snode, Scord, Cnode, Ccord)
                    xaxis = np.subtract(Ccord, Scord)
                    vecxy = [0,0,1] # What you want (vecxy is global vertical)
                    vecxz = np.cross(xaxis,vecxy) # What OpenSees expects
                    self.geomTTag = self.geomTTag+1
                    op.geomTransf('Linear', self.geomTTag, *vecxz)
                    self.RigidEleTag = self.RigidEleTag+1
                    op.element('elasticBeamColumn', int(self.RigidEleTag), int(Cnode), int(Snode),  *self.RigidProp, self.geomTTag)
                # elif np.linalg.norm(Ccord[1] - Scord[1])<50 and np.linalg.norm(Ccord[0] - Scord[0])<50:
                #     # print('Not Connecting ', Snode, Scord, Cnode, Ccord)
                #     # sdf =  [Snode, Scord, Cnode, Ccord]
                #     nodeList.append([Snode, Scord, Cnode, Ccord])
                else:
                    
                    nodeList.append([Snode, Scord, Cnode, Ccord])
                    # print('not Connecting ', Snode, Scord, Cnode, Ccord)
        output_file = 'NodeList.txt'
        # Save nodeList to a text file
        # with open(output_file, 'w') as file:
        #     for node in nodeList:
        #         file.write(f'{node[0]} {node[1]} {node[2]} {node[3]}\n')
                
        #Connect Bottom nodes of the drilled shaft
        print('Connecting Bottom nodes of Shaft to Soil nodes')
        BottomnodeList = []
        for nJp in self.BottomNodesSo:
            pcord = op.nodeCoord(int(nJp))
            
            for nIs in self.AllSoilNodes:
                scord = op.nodeCoord(int(nIs))
                if np.linalg.norm(pcord[2] - scord[2]) <= 1e-5 and np.linalg.norm(pcord[1] - scord[1])<=1e-5 and np.linalg.norm(pcord[0] - scord[0])<=1e-5:
                    # op.equalDOF(int(nIs),int(nJp), 1 , 2, 3)
                    op.fix(int(nJp), 1, 1, 1, 1, 1, 1)
                    print( 'pile and soil node at the bottom',nJp, pcord, nIs, scord) 
                else:
                    # print(nJp, pcord, nIs, scord)
                    BottomnodeList.append([nJp, pcord, nIs, scord])
    
        output_file = 'BottomNodeList.txt'
        
        # Save nodeList to a text file
        # with open(output_file, 'w') as file:
        #     for node in BottomnodeList:
        #         file.write(f'{node[0]} {node[1]} {node[2]} {node[3]}\n')      
    
        return  BottomnodeList#, nodeList              
                    # RigidElements = np.append(RigidElements, RigidElementid)
        
       
    #%% strcutre building
    def _buildStructure_(self):
        print('modelling started...')
        if len(self.ylocation)>=2:
            print(f'number of bents: {len(self.ylocation)}')
            for ylocation in range (len(self.ylocation)):
                self.SocketLastTags  = np.array([])  
                self.ShaftLastTags   = np.array([])
                self.ColumnLastTags  = np.array([])
                self.CapBeamLastTags = np.array([])
                    
                    #%% create Socket nodes and elements in transverse direction
                    # condition to check for interior bent
                 # condition to check for interior bent
                if self.ylocation[ylocation] != self.ylocation[0] and self.ylocation[ylocation] != self.ylocation[-1]:    
                    #print(self.ylocation[ylocation],'True')
                    
                    # # Create the nodes
                    for Clocation in range (len(self.xlocation)):
                        # print(self.xlocation[Clocation], self.ylocation[ylocation])
                        Col_xT, Col_yT = rotate_node(self.xlocation[Clocation], self.ylocation[ylocation],  self.theta)
                        top_coord = [Col_xT, Col_yT, self.socket_top_z]
                        bottom_coord = [Col_xT, Col_yT, self.socket_bottom_z]
                        # top_coord = [self.xlocation[Clocation], self.ylocation[ylocation], 56*ft] #58ft
                        # bottom_coord = [self.xlocation[Clocation], self.ylocation[ylocation], 24*ft] #25ft
                        total_length = top_coord[2] - bottom_coord[2]  # Since Z-coordinates are negative for depth
                        # print(total_length)
                        # # Calculate the spacing between each node
                        self.num_elements = 16
                        spacing = total_length / self.num_elements
            
                        # # Generate the coordinates for each node
                        z_coords = [bottom_coord[2] + i * spacing for i in range(self.num_elements + 1)]
                        # print('Length of ',len(z_coords))
                        self.Sotag1 = np.array([])
                        SocketNodeMass = self.SocketMass/(self.num_elements+1)
                        for z in z_coords:
                            self.tagSo += 1
                            self.Sotag1 = np.append(self.Sotag1, self.tagSo ) 
                            self.ShaftColumnNodeTags = np.append(self.ShaftColumnNodeTags, self.tagSo )
                            # print(f'Node ID: {self.tagSo} Coordinates: ({Clocation,ylocation,z})')


                            op.node(int(self.tagSo), Col_xT, Col_yT, z, '-mass', SocketNodeMass, SocketNodeMass, SocketNodeMass, 0., 0., 0.) # with roation
                            # op.node(int(self.tagSo), self.xlocation[Clocation], self.ylocation[ylocation], z, '-mass', SocketNodeMass, SocketNodeMass, SocketNodeMass, 0., 0., 0.) # orginal
                            # print(self.tagSo, z)
                        self.SocketLastTags = np.append(self.SocketLastTags, self.tagSo)
                        # print('last node tag for socket',self.SocketLastTags)
                        self.BottomNodesSo = np.append(self.BottomNodesSo , self.Sotag1[0])
                        # print('Bottom node tag for socket',self.BottomNodesSo)
                        #creating geomteric transformation tag
                        self.geomTTag = self.geomTTag+1
                        
                        #creating geomteric transformation
                        op.geomTransf('PDelta', self.geomTTag, 1., 0., 0.)
                        
                        #creating Integration type
                        IntegrationType = 'Lobatto'
                        
                        #creating Integration tag
                        self.InteTag = self.InteTag+1
                        
                        #number of Intergation points
                        NumIP = 5
                        
                        #creating Integration 
                        op.beamIntegration(IntegrationType, self.InteTag, int(self.SecTag[4]), NumIP)
                        
                        self.SocketEleTagC = np.array([])
                        for nodes in range(len(self.Sotag1)-1):
                            
                            self.SocketEleTagC = np.append(self.SocketEleTagC,self.SocketEleTag)
                            # #element('dispBeamColumn', eleTag, *eleNodes, transfTag, integrationTag, '-cMass', '-mass', mass=0.0)
                            op.element(self.SocketElementType, int(self.SocketEleTag), int(self.Sotag1[nodes]), int(self.Sotag1[nodes+1]), self.geomTTag, self.InteTag)
                            self.SocketEleTags = np.append(self.SocketEleTags, self.SocketEleTag)
                            self.SocketEleTag += 1
                        # return  self.SocketEleTags, self.SocketLastTags  
                        
                    #%% Shaft nodes and elements   
                    for Clocation in range (len(self.xlocation)):
                        Col_xT, Col_yT = rotate_node(self.xlocation[Clocation], self.ylocation[ylocation],  self.theta)
                        top_coord = [Col_xT, Col_yT, self.shaft_top_z]
                        bottom_coord = [Col_xT, Col_yT, self.shaft_bottom_z]
                        # print(self.xlocation[Clocation], self.ylocation[ylocation])
                        # top_coord = [self.xlocation[Clocation], self.ylocation[ylocation], 96*ft]
                        # bottom_coord = [self.xlocation[Clocation], self.ylocation[ylocation], 58*ft]  #59
                        total_length = top_coord[2] - bottom_coord[2]  # Since Z-coordinates are negative for depth
                        # print(total_length)
                        # # Calculate the spacing between each node
                        self.num_elements = 19 #37
                        self.spacing = total_length / self.num_elements
        
                        # # Generate the coordinates for each node
                        z_coords = [bottom_coord[2] + i * self.spacing for i in range(self.num_elements + 1)]
                        
                        self.Stag1 = np.array([])
                        ShaftNodeMass = self.ShaftMass/(self.num_elements+2)
                        #creating nodes with tags
                        
                        for z in z_coords:
                            self.tagS += 1
                            self.Stag1 = np.append(self.Stag1,self.tagS) #to store all nodes in one array
                            op.node(self.tagS, Col_xT, Col_yT, z, '-mass', ShaftNodeMass, ShaftNodeMass, ShaftNodeMass, 0., 0., 0.) # with rotation
                            # op.node(self.tagS, self.xlocation[Clocation], self.ylocation[ylocation], z, '-mass', ShaftNodeMass, ShaftNodeMass, ShaftNodeMass, 0., 0., 0.) # original
                            self.ShaftColumnNodeTags = np.append(self.ShaftColumnNodeTags, self.tagS )
                            
                        #inserting socket last node tags
                        # print('check ...', Clocation)
                        # print(self.Stag1)
                        op.mass(int(self.SocketLastTags[Clocation]),  ShaftNodeMass+SocketNodeMass, ShaftNodeMass+SocketNodeMass, ShaftNodeMass+SocketNodeMass, 0., 0., 0.)
                        # print('Last tag of socket',self.SocketLastTags[Clocation])
                        self.Stag1 = np.insert(self.Stag1, 0, self.SocketLastTags[Clocation])
                        self.ShaftLastTags = np.append(self.ShaftLastTags,self.tagS)
                        
                        #creating geomteric transformation tag
                        self.geomTTag = self.geomTTag+1
                        
                        #creating geomteric transformation
                        op.geomTransf('PDelta', self.geomTTag, 1., 0., 0.)
                        
                        #creating Integration type
                        IntegrationType = 'Lobatto'
                        
                        #creating Integration tag
                        self.InteTag = self.InteTag+1
                        
                        #number of Intergation points
                        NumIP = 5
                        
                        #creating Integration 
                        op.beamIntegration(IntegrationType, self.InteTag, int(self.SecTag[4]), NumIP)
                        
                        self.ShaftEleTagC = np.array([])
                        
                        #creating elements
                        for nodes in range(len(self.Stag1)-1):
                            self.ShaftEleTag += 1
                            self.ShaftEleTagC = np.append(self.ShaftEleTagC,self.ShaftEleTag)
                            # #element('dispBeamColumn', eleTag, *eleNodes, transfTag, integrationTag, '-cMass', '-mass', mass=0.0)
                            op.element(self.ShaftElementType, int(self.ShaftEleTag), int(self.Stag1[nodes]), int(self.Stag1[nodes+1]), self.geomTTag, self.InteTag)
                            self.ShaftEleTags = np.append(self.ShaftEleTags, self.ShaftEleTag)
                            
                        # print (Stag1)
                    # print(ShaftEleTags)
                    
                    #%%Column nodes and elements
                    for Clocation in range (len(self.xlocation)):
                        Col_xT, Col_yT = rotate_node(self.xlocation[Clocation], self.ylocation[ylocation],  self.theta)
                        top_coord = [Col_xT, Col_yT,self.column_top_z]
                        bottom_coord = [Col_xT, Col_yT, self.column_bottom_z]
                        # print(self.xlocation[Clocation], self.ylocation[ylocation])
                        # top_coord = [self.xlocation[Clocation], self.ylocation[ylocation], 104.0*ft]  #103ft
                        # bottom_coord = [self.xlocation[Clocation], self.ylocation[ylocation], 98*ft]  #97ft
                        total_length = top_coord[2] - bottom_coord[2]  # Since Z-coordinates are negative for depth
                        
                        # print(total_length)
                        # # Calculate the spacing between each node
                        
                        self.num_elements = 3
                        self.spacing = total_length / self.num_elements
            
                        # # Generate the coordinates for each node
                        z_coords = [bottom_coord[2] + i * self.spacing for i in range(self.num_elements + 1)]
                        
                        self.Ctag1 = np.array([])
                        ColumnNodeMass = self.columnMass/(self.num_elements+2)
                        
                        #creating nodes with tags
                        for z in z_coords:
                            
                            self.tagC +=1 
                            self.Ctag1 = np.append(self.Ctag1,self.tagC) #to store all nodes in one array
                            # CTAG = self.Ctag1
                            op.node(self.tagC, Col_xT, Col_yT, z, '-mass', ColumnNodeMass, ColumnNodeMass, ColumnNodeMass, 0., 0., 0.)# with rotation
                            # op.node(self.tagC, self.xlocation[Clocation], self.ylocation[ylocation], z, '-mass', ColumnNodeMass, ColumnNodeMass, ColumnNodeMass, 0., 0., 0.) # original
                            self.ShaftColumnNodeTags = np.append(self.ShaftColumnNodeTags, self.tagC )
                            # print(self.tagC)
                        # print('Column Last node tags',CTAG)
                        op.mass(int(self.ShaftLastTags[Clocation]), ShaftNodeMass+ColumnNodeMass, ShaftNodeMass+ColumnNodeMass, ShaftNodeMass+ColumnNodeMass, 0., 0., 0.)
                        #inserting shaft last node tags 
                        self.Ctag1 = np.insert(self.Ctag1, 0, self.ShaftLastTags[Clocation])
                        #storing the last node tags of each socket
                        
                        self.ColumnTopTags = np.append(self.ColumnTopTags,self.tagC)
                        self.ColumnLastTags = np.append(self.ColumnLastTags,self.tagC)
                        
                        # print(f'Column Last tag {self.ColumnLastTags}')
                        #creating geomteric transformation tag
                        self.geomTTag = self.geomTTag+1
                        
                        #creating geomteric transformation
                        op.geomTransf('PDelta', self.geomTTag, 1., 0., 0.)
                        
                        #creating Integration type
                        IntegrationType = 'Lobatto'
                        
                        #creating Integration tag
                        self.InteTag = self.InteTag+1
                        
                        #number of Intergation points
                        NumIP = 5
                        
                        #creating Integration 
                        op.beamIntegration(IntegrationType, self.InteTag, int(self.SecTag[4]), NumIP)
                        
                        self.ColumnEleTagC = np.array([])
                        
                        #creating elements
                        for nodes in range(len(self.Ctag1)-1):
                            self.ColumnEleTag += 1
                            self.ColumnEleTagC = np.append(self.ColumnEleTagC, self.ColumnEleTag)
                            # #element('dispBeamColumn', eleTag, *eleNodes, transfTag, integrationTag, '-cMass', '-mass', mass=0.0)
                            op.element(self.ColumnElementType, int(self.ColumnEleTag), int(self.Ctag1[nodes]), int(self.Ctag1[nodes+1]), self.geomTTag, self.InteTag)
                            self.ColumnEleTags = np.append(self.ColumnEleTags, self.ColumnEleTag)
                                
                    #%% Cap Beam and Rigid connection b/w Cap beam and Column
                    self.nodexCB = np.array([])
                    self.nodeyCB = np.array([])
                    self.nodezCB = np.array([])
                    for lo in range (len(self.capBeamLocations)):
                        
                        self.nodexCB = np.append(self.nodexCB, self.capBeamLocations[lo])                     
                        self.nodeyCB = np.append(self.nodeyCB, self.ylocation[ylocation])
                        self.nodezCB = np.append(self.nodezCB, 103*ft+self.ColumnBeamRigidLnk)
                    
                    self.CBtag1 = np.array([])
                    self.CapBeamNodeMass = CapBeamMass/(len(self.nodexCB))
                    #creating nodes with tags
                    for ii in range (len(self.nodexCB)):
                        self.tagCB = int(self.tagCB+1) #temporary
                        self.CBtag1 = np.append(self.CBtag1,self.tagCB) #to store all nodes in one array
                        XCB_T, YCB_T = rotate_node(self.nodexCB[ii], self.nodeyCB[ii],  self.theta)
                        op.node(self.tagCB, XCB_T, YCB_T, self.nodezCB[ii], '-mass', self.CapBeamNodeMass, self.CapBeamNodeMass, self.CapBeamNodeMass, 0., 0., 0.) # with rotation
                        # op.node(self.tagCB, self.nodexCB[ii], self.nodeyCB[ii], self.nodezCB[ii], '-mass', self.CapBeamNodeMass, self.CapBeamNodeMass, self.CapBeamNodeMass, 0., 0., 0.) # original
                        self.CapBeamLastTags = np.append(self.CapBeamLastTags,self.tagCB)
                    
                    # print('cap beam nodes:',CBtag1)
                    #creating geomteric transformation tag
                    self.geomTTag = self.geomTTag+1
                    
                    #creating geomteric transformation
                    op.geomTransf('Linear', self.geomTTag, 0., -1., 0.)
                    self.CapBeamEleTagC = np.array([])
                    
                    #creating elements
                    for nodes in range(len(self.CBtag1)-1):
                        self.CapBeamEleTag = int(self.CapBeamEleTag+1)
                        self.CapBeamEleTagC = np.append(self.CapBeamEleTagC,self.CapBeamEleTag)
                        op.element(self.CapBeamElementType, int(self.CapBeamEleTag), int(self.CBtag1[nodes]), int(self.CBtag1[nodes+1]), int(self.SecTag[1]), self.geomTTag)
                        self.CapBeamEleTags = np.append(self.CapBeamEleTags, self.CapBeamEleTag)
                    # print(self.CapBeamEleTagC)
                    
                    #%%
                    self.jointNodetag1 = np.array([])
                    
                    # print(self.ColumnLastTags)
                    # print(self.ColumnLastTags[-1] )
                    
                    # print(op.nodeCoord(self.ColumnLastTags[-1]),self.ColumnLastTags[-1] )
                    for nonde in self.ColumnLastTags:
                        # print('column node', nonde)
                        jointNode = op.nodeCoord(int(nonde))
                        self.tagPile += 1
                        op.node(self.tagPile, jointNode[0], jointNode[1], jointNode[2])
                        self.jointNodetag1 = np.append(self.jointNodetag1, self.tagPile)

                    self._warn_unmatched_capbeam_nodes(
                        self.jointNodetag1,
                        self.CBtag1,
                        f"Interior bent at y={self.ylocation[ylocation]}"
                    )
                    
                    #%% Connecting/releasing the DOF at cap beam a column joint
                    for ii in range(len(self.jointNodetag1)):
                        # print(self.jointNodetag1[ii], self.ColumnLastTags[ii])
                        op.equalDOF(int(self.ColumnLastTags[ii]),int(self.jointNodetag1[ii]), 1, 2, 3, 4, 5, 6)  # rigid connection 
                        
                        #TODO add zero length element
                        #op.element('zeroLength', int(self.zeroEleTag), *[int(self.ColumnLastTags[ii]),int(self.jointNodetag1[ii])], '-mat', *[CapPile, CapPile, CapPile], '-dir', *[4,5, 6])
                        #op.element('zeroLength', int(self.zeroEleTag), *[int(self.ColumnLastTags[ii]), int(self.jointNodetag1[ii])], '-mat', *[CapPile, CapPile, CapPile], '-dir', *[4,5,6], '-orient', 0, 0, 1, 0, -1, 0)
                        # op.element('zeroLengthSection', int(self.zeroEleTag), *[int(self.ColumnLastTags[ii]), int(self.jointNodetag1[ii])], int(self.SecTag[-1]), '-orient', 0, 0, 1, 0, -1, 0, '-doRayleigh', 1)
                        # self.zeroEleTag = int(self.zeroEleTag) +1 
                        # self.zeroEleTags = np.append(self.zeroEleTags, self.zeroEleTag)
                        
                        # op.equalDOF(int(self.ColumnLastTags[ii]),int(self.jointNodetag1[ii]), 1, 2, 3, 4, 5, 6) 
                        
                        
                    #%%    
                    self.geomTTag = self.geomTTag+1
                    
                    #creating geomteric transformation
                    op.geomTransf('Linear', self.geomTTag, 1., 0., 0.)
                    
                    self.RigidEleTagC = np.array([])
                    # TODO: instead here define in start 
                    
                    for ColnodeTag in self.jointNodetag1:
                        # print('cordinates of node', ColnodeTag, op.nodeCoord(ColnodeTag))
                        self.colcord = op.nodeCoord(int(ColnodeTag))
                        xC = self.colcord[0]
                        # print('Col cordinates',colcord)
                        # print('x cordinate', xC)
                        
                        for CBnodeTag in self.CBtag1:
                            self.CBcord = op.nodeCoord(int(CBnodeTag))
                            xCB = self.CBcord[0]
                            # print('CAP cordinates',CBcord)
                            # print('x cordinate', xCB)
                            
                            if np.isclose(xC, xCB, rtol=0.0, atol=1.0e-6):
                                # print('True', CBnodeTag, ColnodeTag)
                                self.eqDofnodes += 1
                                op.node(self.eqDofnodes, self.CBcord[0], self.CBcord[1], self.CBcord[2])
                                
                                self.RigidEleTag += 1
                                self.RigidEleTagC = np.append(self.RigidEleTagC,self.RigidEleTag)
                                op.element(self.CapBeamElementType, int(self.RigidEleTag), int(self.eqDofnodes), int(ColnodeTag), *self.RigidProp, self.geomTTag)
                                # op.rigidLink('beam',  ColnodeTag, eqDofnodes)
                                op.equalDOF(int(self.eqDofnodes), int(CBnodeTag), 1, 2, 3, 4, 5, 6) 
                        #%% Bearing Nodes and rigid connection
                                self.RigidTag = np.array([])
                                self.EqualDoFnodes = np.array([])
                                
                                for CBnodeTag in self.CBtag1:
                                    # print('starting', CBnodeTag )
                                    CBcord = op.nodeCoord(int(CBnodeTag))
                                    xCB = CBcord[0]
                                    # print('CAP cordinates',CBcord)
                                    # print('x cordinate', xCB)
                                    
                                    if CBnodeTag == self.CBtag1[0] or CBnodeTag == self.CBtag1[-1]:
                                        # print('first and last nodes', CBnodeTag)
                                        continue
                                        
                                        
                                    for ColnodeTag in self.ColumnLastTags:
                                        # print('cordinates of node', ColnodeTag, op.nodeCoord(ColnodeTag))
                                        colcord = op.nodeCoord(int(ColnodeTag))
                                        xC = colcord[0]
                                        # print('in loop')
                                        # print(xC, xCB)
                                            # continue
                                        if np.isclose(xC, xCB, rtol=0.0, atol=1.0e-6):
                                            # print('True')
                                            
                                            break
                                        
                                            # print('Column nodes', CBnodeTag)
                                            # continue
                                        
                                    if (xC != xCB):
                                        # print('create node', CBnodeTag)
                                        CBcord = op.nodeCoord(int(CBnodeTag))
                                        self.eqDofnodes += 1
                                        op.node(self.eqDofnodes, CBcord[0], CBcord[1], CBcord[2])
                                        op.equalDOF(int(CBnodeTag), int(self.eqDofnodes), 1, 2, 3, 4, 5, 6)
                                        
                                        self.rigidTag += 1
                                        self.RigidTag = np.append(self.RigidTag, self.rigidTag)
                                        # print(rigidTag)
                                        #left side of capBeam rigidnode
                                        op.node(self.rigidTag, CBcord[0], CBcord[1]-0.355*m, CBcord[2])
                                        
                                        self.geomTTag = self.geomTTag+1
                                            
                                        #creating geomteric transformation
                                        op.geomTransf('Linear', self.geomTTag, -1., 0., 0.)
                                        
                                        self.RigidEleTag += 1
                                        #left side of capBeam rigid element
                                        op.element(self.CapBeamElementType, self.RigidEleTag, self.eqDofnodes, self.rigidTag, *self.RigidProp, self.geomTTag)
                                        # op.rigidLink('beam', CBnodeTag, rigidTag)
                                        
                                        # op.node(rigidTag, CBcord[0], CBcord[1]+0.2, CBcord[2])
                                        # print(rigidTag,RigidEleTag )
                                        self.eqDofnodes += 1
                                        op.node(self.eqDofnodes, CBcord[0], CBcord[1], CBcord[2])
                                        op.equalDOF(int(CBnodeTag), int(self.eqDofnodes), 1, 2, 3, 4, 5, 6)
                                        
                                        self.rigidTag += 1
                                        #right side of capBeam rigidnode
                                        self.RigidTag = np.append(self.RigidTag, self.rigidTag)
                                        op.node(self.rigidTag, CBcord[0], CBcord[1]+0.355*m, CBcord[2])
                                        
                                        self.geomTTag = self.geomTTag+1
                                            
                                        #creating geomteric transformation
                                        op.geomTransf('Linear', self.geomTTag, -1., 0., 0.)
                                        
                                        self.RigidEleTag = self.RigidEleTag+1
                                        #left side of capBeam rigid element
                                        op.element(self.CapBeamElementType, self.RigidEleTag, self.eqDofnodes, self.rigidTag,  *self.RigidProp, self.geomTTag)
                                        # op.rigidLink('beam', CBnodeTag, rigidTag)
                                        # print(rigidTag,RigidEleTag )
                                # print(RigidTag)
                                #%%
                                self.GirderTags = np.array([])
                                self.BearingTags = np.array([])
                                for nodetag in self.RigidTag:
                                    
                                    rigidcord = op.nodeCoord(int(nodetag))
                                    self.rigidTag = self.rigidTag+1
                                    # print(rigidTag)
                                    op.node(self.rigidTag, rigidcord[0], rigidcord[1], rigidcord[2]+2.25*ft+0.031)  # 0.031 inch to make adjustment to the hight of girder to make deck at the same level at interior and end bent, 2.25ft depth of capbeam
                                    # print(nodetag,rigidTag)
                                    self.geomTTag = self.geomTTag+1
                                    
                                    #creating geomteric transformation
                                    op.geomTransf('Linear', self.geomTTag, 0., 1., 0.)
                                    
                                    self.RigidEleTag = self.RigidEleTag+1
                                    #rigid element between cap beam and bearing
                                    op.element(self.CapBeamElementType, int(self.RigidEleTag), int(nodetag), int(self.rigidTag), *self.RigidProp, self.geomTTag)
                                    
                                    self.BearingTag = self.BearingTag+1
                                    bearingcord = op.nodeCoord(self.rigidTag)
                                    
                                    
                                    op.node(self.BearingTag, bearingcord[0], bearingcord[1], bearingcord[2]+self.bearingdepth)
                                    self.BearingTags = np.append(self.BearingTags, self.BearingTag)
                                    
                                    self.geomTTag = self.geomTTag+1
                                    #creating geomteric transformation
                                    op.geomTransf('Linear', self.geomTTag, -1., 0., 0.)
                                    
                                    self.BearingEleTag = self.BearingEleTag+1
                                    #-----------------------------------------------------------
                                    #Fixed bearing element
                                    #-----------------------------------------------------------
                                    # op.element(self.BearingElementType, BearingEleTag, rigidTag, BearingTag,  *RigidProp, geomTTag)
                                    # op.element('elastomericBearingPlasticity',  BearingEleTag, *[rigidTag, BearingTag], 2000*kip, 20*kip/inch, 0.1, 0.1, 3.0, '-P', IDrubber, '-T', IDdowel, '-My', IDdowel, '-Mz', IDdowel, '-orient', -1, 0, 0) # for a 3D elastomeric bearing 
                                    # op.element('twoNodeLink',  BearingEleTag, rigidTag, BearingTag, '-mat', 3, 3, 3, '-dir', 1, 2, 3, 4, 5, 6, '-orient', 0, -1, 0) # for a 3D elastomeric bearing 
                                    op.element('twoNodeLink',  self.BearingEleTag, self.rigidTag, self.BearingTag, '-mat',  *[IDElastomer, IDElastomer, IDElastomer, IDdowel_F, IDdowel_F, IDdowel_F], '-dir', *[1, 2, 3, 4, 5, 6], '-pDelta', *[0.5, 0.5, 0.5, 0.5],'-shearDist', *[0.5, 0.5], '-orient', -1, 0, 0) # for a 3D elastomeric bearing 
                                    # op.element('ElastomericX', BearingEleTag, rigidTag, BearingTag, Fy_h, alpha, G, K, D1, D2, ts, tr, n, 0, 0, 1, 1, 0, 0, kc, PhiM, ac, sDratio, mb, cd, tc)
                                    
                                    # print(rigidTag,RigidEleTag )
                                    # print(BearingTag)
                                    BearingTagCord = op.nodeCoord(self.BearingTag)
                                    
                                    # print('interior Bent level', BearingTagCord[2])
                                    self.rigidTag = self.rigidTag+1
                                    #Girder nodes
                                    op.node(self.rigidTag, BearingTagCord[0], BearingTagCord[1], BearingTagCord[2]+5*ft)
                                    self.GirderTags = np.append(self.GirderTags, self.rigidTag)
                                    # print('Girder tags',GirderTags)
                                    self.geomTTag = self.geomTTag+1
                                    #creating geomteric transformation
                                    op.geomTransf('Linear', self.geomTTag, -1., 0., 0.)
                                    
                                    self.RigidEleTag = self.RigidEleTag+1
                                    #bearing element
                                    op.element(self.RigidElementType, self.RigidEleTag, self.BearingTag, self.rigidTag,  *self.RigidProp, self.geomTTag)
                                    
                                    # print(rigidTag, BearingTag)
                                    # print(BearingTags)
                                    # print(GirderTags)
                                    
                                #%% Diaphragm element and node
                                # print('CapBeam Tag',  CapBeamLastTags)
                                # print('Girder Tags Int Bent', GirderTags)
                                self.LeftGirderTags  = np.array([])
                                self.RightGirderTags  = np.array([])
                                for ii in range (int(len(self.GirderTags))):
                                    if (ii+1)%2==0:
                                        # print('even place', GirderTags[ii])
                                        self.LeftGirderTags = np.append(self.LeftGirderTags, self.GirderTags[ii])
                                        self.AllDiaphTags = np.append(self.AllDiaphTags, self.GirderTags[ii])
                                        
                                    
                                    else:
                                        # print('odd place', GirderTags[ii])
                                        self.RightGirderTags  = np.append(self.RightGirderTags, self.GirderTags[ii])
                                        self.AllDiaphTags    = np.append(self.AllDiaphTags, self.GirderTags[ii])
                                        
                                # print('left nodes', LeftGirderTags)
                                # print('right nodes', RightGirderTags)
                                
                                cordNodeLeft = op.nodeCoord(int(self.LeftGirderTags[0]))
                                cordNodeRight = op.nodeCoord(int(self.RightGirderTags[0]))
                                
                                self.rigidTag = self.rigidTag+1
                                
                                op.node(self.rigidTag, 0., cordNodeLeft[1], cordNodeLeft[2] )
                                self.GirderNodeTags = np.append(self.GirderNodeTags, self.rigidTag)
                                self.AllDiaphTags = np.append(self.AllDiaphTags, self.rigidTag)
                                self.LeftGirderTags = np.insert(self.LeftGirderTags, int(len(self.LeftGirderTags)/2), self.rigidTag)
                                self.rigidTag = self.rigidTag+1
                                op.node(self.rigidTag, 0., cordNodeRight[1], cordNodeRight[2] )
                                self.GirderNodeTags = np.append(self.GirderNodeTags, self.rigidTag)
                                self.AllDiaphTags = np.append(self.AllDiaphTags, self.rigidTag)
                                self.RightGirderTags = np.insert(self.RightGirderTags, int(len(self.RightGirderTags)/2), self.rigidTag)
                                
                                for ii in range (int(len(self.RightGirderTags)-1)):
                                    self.geomTTag = self.geomTTag+1
                                    
                                    #creating geomteric transformation
                                    op.geomTransf('Linear', self.geomTTag, 0., -1., 0.)
                                    self.DiaphEleTag = self.DiaphEleTag+1
                                    # op.element(self.DiaphElementType, DiaphEleTag, RightGirderTags[ii], RightGirderTags[ii+1], int(self.SecTag[6]), geomTTag)
                                    op.element(self.DiaphElementType, int(self.DiaphEleTag), int(self.RightGirderTags[ii]), int(self.RightGirderTags[ii+1]), *self.RigidProp, self.geomTTag)
                                    self.EndDiaphEleTag = np.append(self.EndDiaphEleTag, self.DiaphEleTag)
                                    
                                    self.geomTTag = self.geomTTag+1
                                    #creating geomteric transformation
                                    op.geomTransf('Linear', self.geomTTag, 0., -1., 0.)
                                    self.DiaphEleTag = self.DiaphEleTag+1
                                    # op.element(self.DiaphElementType, DiaphEleTag, LeftGirderTags[ii], LeftGirderTags[ii+1], int(self.SecTag[6]), geomTTag)
                                    op.element(self.DiaphElementType, int(self.DiaphEleTag), int(self.LeftGirderTags[ii]), int(self.LeftGirderTags[ii+1]), *self.RigidProp, self.geomTTag)
                                    self.EndDiaphEleTag = np.append(self.EndDiaphEleTag, self.DiaphEleTag)
                    
                                #     # print(GirderTags[ii], GirderTags[ii+2])
        #%% 12-pile row End bent
             
        #TODO: instead of puting here enterd in input sheet 
        pilelocationX = self.pileLocation
        # print('pile x',pilelocationX)
        # print('pile y', self.EndBentLocation)
        if len(self.EndBentLocation)==2:                  # to create both use ==2
            for yy in range (len(self.EndBentLocation)):
                     
                HplileLastTagsL = np.array([])
                HplileLastTagsR = np.array([])
                self.ECapBeamLastTags = np.array([])
                
                for XX in range (len(pilelocationX)): # (self.NumPiles-1):
                    Col_xT, Col_yT = rotate_node(pilelocationX[XX], self.EndBentLocation[yy], self.theta)
                    top_coord = [Col_xT, Col_yT, self.pile_top_z]
                    bottom_coord = [Col_xT, Col_yT, self.pile_bottom_z]
                    # top_coord = [pilelocationX[XX], self.EndBentLocation[yy], 103.94737085*ft]
                    # top_coord = [pilelocationX[XX], self.EndBentLocation[yy], 104*ft]
                    # bottom_coord = [pilelocationX[XX], self.EndBentLocation[yy], 48*ft]
                    total_length = top_coord[2] - bottom_coord[2]  # Since Z-coordinates are negative for depth
                    
                    # print(total_length)
                    # # Calculate the spacing between each node
                    self.num_elements = 56//2
                    spacing = total_length / self.num_elements
        
                    # # Generate the coordinates for each node
                    z_coords = [bottom_coord[2] + i * spacing for i in range(self.num_elements + 1)]
                    
                    self.Piletag1 = np.array([])
                    PileNodeMass = self.PileMass/(self.num_elements+1)
                    for z in z_coords:
                        self.tagPile += 1
                        self.Piletag1 = np.append(self.Piletag1, self.tagPile ) 
                        # print(f'Node ID: {self.tagSo} Coordinates: ({Clocation,ylocation,z})')
                        op.node(int(self.tagPile), Col_xT, Col_yT, z, '-mass', PileNodeMass, PileNodeMass, PileNodeMass, 0., 0., 0.) # with rotation
                        # op.node(int(self.tagPile), pilelocationX[XX], self.EndBentLocation[yy], z, '-mass', PileNodeMass, PileNodeMass, PileNodeMass, 0., 0., 0.) # original
                        self.ShaftColumnNodeTags = np.append(self.ShaftColumnNodeTags, self.tagPile)
                        self.EndbentNodeTags = np.append(self.EndbentNodeTags, self.tagPile)
                        
                    self.BottomPileNodes = np.append(self.BottomPileNodes, self.tagPile)
                    # print('Bottom pile nodes ',self.BottomPileNodes )
                    self.PileTopTags = np.append(self.PileTopTags, self.tagPile)
                    HplileLastTagsL = np.append(HplileLastTagsL, self.Piletag1[-1])
                    
                    # need for fixing boundary at the bottom of pile if  soil domin is not there
                    self.BottomNodesSo = np.append(self.BottomNodesSo , self.Piletag1[0]) 
                    # print(self.Piletag1[-1], self.Piletag1[0])
                    # print('Pile top Nodes ',HplileLastTagsL)
                        
                    #inserting shaft last node tags 
                    # self.Sotag1 = np.insert(self.Sotag1, 0, self.ShaftLastTags[Clocation])
                    
                    # print(self.Sotag1)
                    
                    self.geomTTag = self.geomTTag+1
                    
                    #creating geomteric transformation
                    # op.geomTransf('PDelta', self.geomTTag, 0., -1., 0.)
                    op.geomTransf('PDelta', self.geomTTag, 1., 0., 0.)
                    #creating Integration type
                    IntegrationType = 'Lobatto'
                    
                    #creating Integration tag
                    self.InteTag = self.InteTag+1
                    
                    #number of Intergation points
                    NumIP = 5
                    
                    #creating Integration 
                    op.beamIntegration(IntegrationType, self.InteTag, int(self.SecTag[2]), NumIP)
                    
                    self.PileEleTagC = np.array([])
                    #creating elements
                    for nodes in range(len(self.Piletag1)-1):
                        self.PileEleTag = int(self.PileEleTag+1)
                        self.PileEleTagC = np.append(self.PileEleTagC,self.PileEleTag)
                        # #element('dispBeamColumn', eleTag, *eleNodes, transfTag, integrationTag, '-cMass', '-mass', mass=0.0)
                        op.element(self.PileElementType, int(self.PileEleTag), int(self.Piletag1[nodes]), int(self.Piletag1[nodes+1]), self.geomTTag, self.InteTag)
                        self.PileElemTags = np.append(self.PileElemTags, self.PileEleTag)
                        self.EndbentElemTags = np.append(self.EndbentElemTags, self.PileEleTag)
                        # print(self.PileEleTag)
                #%% 14- end bent cap beam                2                4              6                            6                4              2
                self.Endcapbeam = transverse_capbeam_locations(self.pileLocation, self.bearingLocation)
                # print('last pile node tag ', tagPile)
                
                self.ECBtag1 = np.array([])
                ECapBeamNodeMass = ECapBeamMass/(len(self.Endcapbeam))
                for point in self.Endcapbeam:
                    Col_xT1, Col_yT1 = rotate_node(point, self.EndBentLocation[yy], self.theta)
                    pilecord = op.nodeCoord(int(self.Piletag1[-1]))
                    self.tagECB   = int(self.tagECB+1)
                    self.ECBtag1 = np.append(self.ECBtag1,self.tagECB) #to store all nodes in one array
                    # op.node(int(self.tagECB), point, self.EndBentLocation[yy], pilecord[2]+self.EndCapBeamDepth/2, '-mass', ECapBeamNodeMass, ECapBeamNodeMass, ECapBeamNodeMass, 0., 0., 0.)# original
                    op.node(int(self.tagECB), Col_xT1, Col_yT1, pilecord[2]+self.EndCapBeamDepth/2, '-mass', ECapBeamNodeMass, ECapBeamNodeMass, ECapBeamNodeMass, 0., 0., 0.) # with rotation
                    
                    self.ECapBeamLastTags = np.append(self.ECapBeamLastTags,self.tagECB)
                    
                    self.EpTag = self.EpTag+1  # adding node  for earthpressure
                    self.zeroEleTag = int(self.zeroEleTag)+1 
                    op.node(int(self.EpTag), Col_xT1, Col_yT1, pilecord[2]+self.EndCapBeamDepth/2) # with rotation
                    op.fix(int(self.EpTag), 1,1,1,1,1,1)
                    op.element('zeroLength', int(self.zeroEleTag), int(self.tagECB), int(self.EpTag), '-mat', earth_P, earth_P,earth_P, '-dir',1, 2,3)

                
                # fixNodesTags = np.append(fixNodesTags, [ECapBeamLastTags[0], ECapBeamLastTags[-1]] )
                #creating geomteric transformation tag
                self.geomTTag = self.geomTTag+1
                
                #creating geomteric transformation
                op.geomTransf('Linear', self.geomTTag, 0., -1., 0.)
                
                #creating Integration type
                # IntegrationType = 'Lobatto'
                
                #creating Integration tag
                # InteTag = InteTag+1
                
                #number of Intergation points
                # NumIP = 5
                self.ECapBeamEleTagC = np.array([])
                for nodes in range(len(self.ECBtag1)-1):
                    self.ECapBeamEleTag = int(self.ECapBeamEleTag+1)
                    self.ECapBeamEleTagC = np.append(self.ECapBeamEleTagC,self.ECapBeamEleTag)
                    op.element(self.CapBeamElementType, int(self.ECapBeamEleTag), int(self.ECBtag1[nodes]), int(self.ECBtag1[nodes+1]), int(self.SecTag[1]), self.geomTTag)
                    self.ECapBeamEleTags = np.append(self.ECapBeamEleTags, self.ECapBeamEleTag)
                # print('right pile tag',ECapBeamEleTags)
                #%% Creating Joint node for connecting cap and pile
                self.jointNodetag = np.array([])
                for nonde in HplileLastTagsL:
                    jointNode = op.nodeCoord(int(nonde))
                    self.tagPile = int(self.tagPile+1)
                    op.node(int(self.tagPile), jointNode[0], jointNode[1], jointNode[2])
                    self.jointNodetag = np.append(self.jointNodetag, self.tagPile)

                self._warn_unmatched_capbeam_nodes(
                    self.jointNodetag,
                    self.ECapBeamLastTags,
                    f"End bent at y={self.EndBentLocation[yy]}"
                )
                # print("New tags", self.jointNodetag)
                
                #%% Connecting or releasing the DOF
                for ii in range(len(self.jointNodetag)):
                    #print(self.jointNodetag[ii], HplileLastTagsL[ii])
                    
                    #op.equalDOF(int(HplileLastTagsL[ii]), int(self.jointNodetag[ii]), 1, 2, 3, 4, 5, 6)
                    
                    # op.element('zeroLengthSection', int(self.zeroEleTag), *[int(HplileLastTagsL[ii]), int(self.jointNodetag[ii])], int(self.SecTag[-1]), '-orient', 0, 0, 1, 0, -1, 0, '-doRayleigh', 1)
                    # self.zeroEleTag = int(self.zeroEleTag) +1 
                    # self.zeroEleTags = np.append(self.zeroEleTags, self.zeroEleTag)
                    op.equalDOF(int(int(HplileLastTagsL[ii])),int(self.jointNodetag[ii]), 1, 2, 3, 4, 5, 6) 
                #%% 15- Connecting End capBeam with pile    
                for nodeTagP in self.jointNodetag:
                    for nodeTagECB in self.ECapBeamLastTags:
                        #print(nodeTagP, nodeTagECB)
                        pilecord   = op.nodeCoord(int(nodeTagP))
                        ECBeamcord = op.nodeCoord(int(nodeTagECB))
                        # print('Pile',nodeTagP, pilecord, 'beam', nodeTagECB, ECBeamcord )
                        if np.isclose(pilecord[0], ECBeamcord[0], rtol=0.0, atol=1.0e-6):
                            # print(nodeTagP, nodeTagECB, 'True')
                            self.geomTTag = self.geomTTag+1
                                
                            #creating geomteric transformation
                            op.geomTransf('Linear', self.geomTTag, -1., 0., 0.)
                            
                            self.RigidEleTag = self.RigidEleTag+1
                            #left side of capBeam rigid element
                            op.element(self.CapBeamElementType, int(self.RigidEleTag), int(nodeTagECB), int(nodeTagP),  *self.RigidProp, self.geomTTag)
                            # op.rigidLink('beam', nodeTagECB, nodeTagP)
                            # print('Element tag b/w endcap and pile',self.RigidEleTag, nodeTagECB, nodeTagP)
                #%% 17- End CapBeam to bearing link
                self.GirderTags = np.array([])
                self.BearingTags = np.array([])
                
                for nodeTagECB in self.ECapBeamLastTags:
                    # print('start', nodeTagECB)
                    ECBeamcord = op.nodeCoord(int(nodeTagECB))
                    if nodeTagECB == self.ECapBeamLastTags[0] or nodeTagECB == self.ECapBeamLastTags[-1] :
                        # print('first and last node', nodeTagECB)
                        continue
                    
                    if ECBeamcord[0] == 0.0:
                        
                        continue
                        
                    for nodeTagP in HplileLastTagsL:
                        # print(nodeTagP, nodeTagECB)
                        pilecord   = op.nodeCoord(int(nodeTagP))
                        ECBeamcord = op.nodeCoord(int(nodeTagECB))
                            
                        if np.isclose(pilecord[0], ECBeamcord[0], rtol=0.0, atol=1.0e-6):
                            # print('Pile nodes', nodeTagECB)
                            break
                        
                    if not np.isclose(pilecord[0], ECBeamcord[0], rtol=0.0, atol=1.0e-6):
                        # print(nodeTagECB, 'Bearing connection')
                        # create node rigid
                        self.rigidTag = int(self.rigidTag+1)
                        op.node(self.rigidTag, ECBeamcord[0], ECBeamcord[1], ECBeamcord[2]+1.75*ft) # 1.75ft half of cap beam depth           
                        
                        self.geomTTag = self.geomTTag+1
                            
                        #creating geomteric transformation
                        op.geomTransf('Linear', self.geomTTag, 0., 1., 0.)
                        
                        self.RigidEleTag = self.RigidEleTag+1
                        #left side of capBeam rigid element
                        op.element(self.CapBeamElementType, int(self.RigidEleTag), int(nodeTagECB), int(self.rigidTag), *self.RigidProp, self.geomTTag)
                        
                        self.BearingTag = int(self.BearingTag+1)
                        bearingcord = op.nodeCoord(self.rigidTag)
                        
                        
                        op.node(self.BearingTag, bearingcord[0], bearingcord[1], bearingcord[2]+self.bearingdepth)
                        self.BearingTags = np.append(self.BearingTags, self.BearingTag)
                        
                        self.geomTTag = self.geomTTag+1
                        #creating geomteric transformation
                        op.geomTransf('Linear', self.geomTTag, -1., 0., 0.)
                        
                        self.BearingEleTag = int(self.BearingEleTag+1)
                        #-----------------------------------------------------------
                        #Expansion bearing element
                        #-----------------------------------------------------------
                        # element('ElastomericX', eleTag, *eleNodes, Fy, alpha, Gr, Kbulk, D1, D2, ts, tr, n, <<x1, x2, x3>, y1, y2, y3>, <kc>, <PhiM>, <ac>, <sDratio>, <m>, <cd>, <tc>, <tag1>, <tag2>, <tag3>, <tag4>)
                        # element('elastomericBearingPlasticity', eleTag, *eleNodes, kInit, qd, alpha1, alpha2, mu, '-P', PMatTag, '-T', TMatTag, '-My', MyMatTag, '-Mz', MzMatTag, <'-orient', <x1, x2, x3>, y1, y2, y3>, <'-shearDist', sDratio>, <'-doRayleigh'>, <'-mass', m>)
                        # op.element('elastomericBearingPlasticity',  BearingEleTag, *[rigidTag, BearingTag], 2000*kip, 20*kip/inch, 0.0, 0.0, 3.0, '-P', IDrubber, '-T', IDdowel, '-My', IDdowel, '-Mz', IDdowel, '-orient', -1, 0, 0) # op.element('ElastomericX', BearingEleTag, rigidTag, BearingTag, 69.4*kN, 0.33, 0.64*MPa, 2000*MPa, 0., D2, ts, tr, n, <<x1, x2, x3>, y1, y2, y3>, <kc>, <PhiM>, <ac>, <sDratio>, <m>, <cd>, <tc>, <tag1>, <tag2>, <tag3>, <tag4>)
                        op.element('twoNodeLink',  self.BearingEleTag, self.rigidTag, self.BearingTag, '-mat',  *[IDElastomer, IDElastomer, IDElastomer, IDdowel_F, IDdowel_F, IDdowel_F], '-dir', *[1, 2, 3, 4, 5, 6], '-pDelta', *[0.5, 0.5, 0.5, 0.5],'-shearDist', *[0.5, 0.5], '-orient', -1, 0, 0) # for a 3D elastomeric bearing 
                        # op.element ('ElastomericX', BearingEleTag, rigidTag, BearingTag, Fy_h, alpha, G, K, D1, D2, ts, tr, n, 0, 0, 1, 1, 0, 0, kc, PhiM, ac, sDratio, mb, cd, tc)
                        # op.element(self.BearingElementType, BearingEleTag, rigidTag, BearingTag, *RigidProp, geomTTag)
                        # print(rigidTag,RigidEleTag )
                        # print(BearingTag)
                        
                        BearingTagCord = op.nodeCoord(self.BearingTag)
                        # print('End Bent level', BearingTagCord[2], rigidTag)
                        self.rigidTag = self.rigidTag+1
                        #Girder nodes
                        op.node(self.rigidTag, BearingTagCord[0], BearingTagCord[1], BearingTagCord[2]+5*ft)
                        self.GirderTags = np.append(self.GirderTags, self.rigidTag)
                        self.AllDiaphTags = np.append(self.AllDiaphTags, self.rigidTag)
                        self.geomTTag = self.geomTTag+1
                        #creating geomteric transformation
                        op.geomTransf('Linear', self.geomTTag, -1., 0., 0.)
                        
                        self.RigidEleTag = self.RigidEleTag+1
                        #bearing element
                        op.element(self.RigidElementType, self.RigidEleTag, self.BearingTag, self.rigidTag, *self.RigidProp, self.geomTTag)
                        
                        self.EpTag = self.EpTag+1  # adding node  for earthpressure
                        self.zeroEleTag = int(self.zeroEleTag)+1 
                        op.node(self.EpTag, BearingTagCord[0], BearingTagCord[1], BearingTagCord[2]+5*ft)
                        op.fix(self.EpTag, 1,1,1,1,1,1)
                        op.element('zeroLength', int(self.zeroEleTag), int(self.rigidTag), int(self.EpTag), '-mat', earth_P, earth_P, earth_P,'-dir', 1, 2, 3)
                
                # print('location of bent',EndbentlocationY)
                # print('last rigid node', rigidTag)
                # print('Girder Tags', GirderTags)
                if len(self.GirderTags)%2==0:
                    
                    lastrigidcord = op.nodeCoord(self.rigidTag)
                    self.rigidTag = int(self.rigidTag+1)
                    op.node(self.rigidTag, 0.0, self.EndBentLocation[yy], lastrigidcord[2])
                    
                    self.GirderNodeTags = np.append(self.GirderNodeTags, self.rigidTag)
                    self.AllDiaphTags = np.append(self.AllDiaphTags, self.rigidTag)
                    self.GirderTags = np.insert(self.GirderTags, int(len(self.GirderTags)/2), self.rigidTag)
                    
                    self.EpTag = int(self.EpTag+1)  # adding node  for earthpressure  ===============================================================================
                    self.zeroEleTag = int(self.zeroEleTag)+1 
                    op.node(self.EpTag, 0.0, self.EndBentLocation[yy], lastrigidcord[2]) 
                    op.fix(self.EpTag, 1,1,1,1,1,1)
                    op.element('zeroLength', int(self.zeroEleTag), int(self.rigidTag), int(self.EpTag), '-mat', earth_P, earth_P, earth_P,'-dir', 1, 2, 3)
                
                # print('New Girder Tags', GirderTags)
                
                for ii in range (int(len(self.GirderTags)-1)):
                    self.geomTTag = self.geomTTag+1
                    
                    #creating geomteric transformation
                    op.geomTransf('Linear', self.geomTTag, 0., -1., 0.)
                    
                    self.DiaphEleTag = int(self.DiaphEleTag+1)
                    op.element(self.DiaphElementType, self.DiaphEleTag, int(self.GirderTags[ii]), int(self.GirderTags[ii+1]), *self.RigidProp, self.geomTTag)
                    self.EndDiaphEleTag = np.append(self.EndDiaphEleTag, self.DiaphEleTag)
            # print('Ends diaphram',EndDiaphEleTag)
                    # print('Girder Element tags ',DiaphEleTag, GirderTags[ii], GirderTags[ii+1])
                #     # print(op.nodeCoord(rigidTag))
            #%% Interior diaphrgam
            i=0
            for j in range(int(len(self.ylocation)-1)):
                self.IntDiaphTag = np.array([])
                for ntags in self.GirderTags:
                    
                    tagcord = op.nodeCoord(int(ntags))
                    self.rigidTag = self.rigidTag+1
                    # print(ntags, tagcord[0], tagcord[1]-(40 + 80*i), tagcord[2])
                    
                    span_start_y = self.ylocation[j]
                    span_end_y = self.ylocation[j + 1]
                    midspan_y = 0.5 * (span_start_y + span_end_y)
                    op.node(self.rigidTag, tagcord[0], midspan_y, tagcord[2])
                    self.IntDiaphTag = np.append(self.IntDiaphTag, self.rigidTag)
                    self.AllDiaphTags = np.append(self.AllDiaphTags, self.rigidTag)
                
                for ii in range (int(len(self.IntDiaphTag)-1)):
                    self.geomTTag = self.geomTTag+1
                    
                    #creating geomteric transformation
                    op.geomTransf('Linear', self.geomTTag, 0., -1., 0.)
                    
                    self.DiaphEleTag = self.DiaphEleTag+1
                    op.element(self.DiaphElementType, int(self.DiaphEleTag), int(self.IntDiaphTag[ii]), int(self.IntDiaphTag[ii+1]), *self.RigidProp, self.geomTTag)
                    self.IntDiaphEleTag = np.append(self.IntDiaphEleTag, self.DiaphEleTag)   
                i = i+1
            
            # print('Int Diaphram',self.IntDiaphEleTag)
            #%% 18- Deck elements
            # print(AllDiaphTags)       
            xCord = op.nodeCoord(self.rigidTag)
            
            if len(self.ylocation)<2:
                print('single span')
                self.ylocation = np.insert(self.ylocation, 0, 0.)
                nodexD =([])
                nodeyD =([])
                nodezD =([])
                for ii in range (len(self.ylocation)-1):
                    
                    # print('deck nodes', np.array([self.ylocation[ii]-0.3,  self.ylocation[ii],  self.ylocation[ii+1]-0.3,  self.ylocation[ii+1]+0.3]))
                    # print(self.ylocation[ii]-1*ft,  self.ylocation[ii],  self.ylocation[ii+1],  self.ylocation[ii+1]+1*ft)
                    nodexD = np.array([0.,                      0.,                  0.,                                                             0.,            0.,   0.,        0. ])
                    nodeyD = np.array([self.ylocation[ii]-0.355,  self.ylocation[ii],  self.ylocation[ii]+(self.ylocation[ii+1]-self.ylocation[ii])/4, self.ylocation[ii]+(self.ylocation[ii+1]-self.ylocation[ii])/2,  self.ylocation[ii]+(self.ylocation[ii+1]-self.ylocation[ii])*3/4, self.ylocation[ii+1],  self.ylocation[ii+1]+0.355])
                    nodezD = np.array([xCord[2],              xCord[2],          xCord[2],                                                     xCord[2],    xCord[2],    xCord[2],         xCord[2]         ])
                    
                    Dtag1 = np.array([])
                    #creating nodes with tags
                    SStNodeMass = SStMass/(len(nodexD))
                    for ii in range (len(nodexD)):
                        tagD = int(tagD+1) #temporary
                        Dtag1 = np.append(Dtag1,tagD) #to store all nodes in one array
                        # DTAG = Dtag1
                        
                        op.node(tagD, nodexD[ii], nodeyD[ii], nodezD[ii], '-mass', SStNodeMass, SStNodeMass, SStNodeMass, 0., 0., 0.)
                        # print(tagD)
                        DeckTags = np.append(DeckTags,tagD)
                        
                    
                    # print('cap beam nodes:',CBtag1)
                    #creating geomteric transformation tag
                    geomTTag = geomTTag+1
                    
                    #creating geomteric transformation
                    op.geomTransf('Linear', geomTTag, 1., 0., 0.)
                    DeckEleTagC = np.array([])
                    
                    #creating elements
                    for nodes in range(len(Dtag1)-1):
                        DeckEleTag = int(DeckEleTag+1)
                        DeckEleTagC = np.append(DeckEleTagC,DeckEleTag)
                        op.element(self.CapBeamElementType, DeckEleTag, Dtag1[nodes], Dtag1[nodes+1], int(self.SecTag[5]), geomTTag)
                        DeckEleTags = np.append(DeckEleTags, DeckEleTag)
                        # connecting dof for all nodes of span
                        # op.equalDOF(Dtag1[nodes],Dtag1[nodes+1],1,2,3,4,5,6)
            #%%
            else:
                print('Multi-span')
                # self.ylocation = np.insert(self.ylocation, 0, 0.)
                nodexD =np.array([])
                nodeyD =np.array([])
                nodezD =np.array([])
                for ii in range (len(self.ylocation)-1):
                    
                    if self.ylocation[ii]==self.ylocation[0] :
                        
                        # print('deck nodes', np.array([self.ylocation[ii]-0.3,  self.ylocation[ii],  self.ylocation[ii+1]-0.3,  self.ylocation[ii+1]]))
                        
                        # print(self.ylocation[ii]-1*ft,  self.ylocation[ii],  self.ylocation[ii+1],  self.ylocation[ii+1]+1*ft)
                        nodexD = np.array([                      0.,                  0.,                                                0,             0.,             0.,       0. ])
                        nodeyD = np.array([self.ylocation[ii],  self.ylocation[ii]+(self.ylocation[ii+1]-self.ylocation[ii])/4, self.ylocation[ii]+(self.ylocation[ii+1]-self.ylocation[ii])/2, self.ylocation[ii]+(self.ylocation[ii+1]-self.ylocation[ii])*3/4, self.ylocation[ii+1]-0.355*m,  self.ylocation[ii+1]])
                        nodezD = np.array([               xCord[2],            xCord[2],                                            xCord[2],           xCord[2],    xCord[2],          xCord[2]         ])
                        
                            
                    elif self.ylocation[ii]==self.ylocation[-2]:
                        # print('deck nodes', np.array([self.ylocation[ii]-0.3,  self.ylocation[ii],  self.ylocation[ii+1]-0.3,  self.ylocation[ii+1]+0.3]))
                        
                        # print(self.ylocation[ii]-1*ft,  self.ylocation[ii],  self.ylocation[ii+1],  self.ylocation[ii+1]+1*ft)
                        nodexD = np.array([0.,                      0.,                  0.,   0., 0.,                                                           0.,                   ])
                        nodeyD = np.array([self.ylocation[ii],  self.ylocation[ii]+0.355*m, self.ylocation[ii]+(self.ylocation[ii+1]-self.ylocation[ii])/4, self.ylocation[ii]+(self.ylocation[ii+1]-self.ylocation[ii])/2, self.ylocation[ii]+(self.ylocation[ii+1]-self.ylocation[ii])*3/4, self.ylocation[ii+1]])
                        nodezD = np.array([xCord[2],              xCord[2],          xCord[2],    xCord[2],  xCord[2],                                                  xCord[2],            xCord[2]         ])
                        
                    else:
                        # print(self.ylocation[ii]-1*ft,  self.ylocation[ii],  self.ylocation[ii+1],  self.ylocation[ii+1]+1*ft)
                        nodexD = np.array([0.,                      0.,                  0.,                                                             0.,              0.,  0.,      0. ])
                        nodeyD = np.array([self.ylocation[ii],  self.ylocation[ii]+0.355*m,  self.ylocation[ii]+(self.ylocation[ii+1]-self.ylocation[ii])/4, self.ylocation[ii]+(self.ylocation[ii+1]-self.ylocation[ii])/2, self.ylocation[ii]+(self.ylocation[ii+1]-self.ylocation[ii])*3/4, self.ylocation[ii+1]-0.355*m,  self.ylocation[ii+1]])
                        nodezD = np.array([xCord[2],              xCord[2],          xCord[2],                                                     xCord[2],       xCord[2],   xCord[2],      xCord[2]         ])
                                         
                    self.Dtag1 = np.array([])
                    #creating nodes with tags
                    SStNodeMass = SStMass/(len(nodexD))
                    for ii in range (len(nodexD)):
                        self.tagD = int(self.tagD+1) #temporary
                        self.Dtag1 = np.append(self.Dtag1,self.tagD) #to store all nodes in one array
                        # DTAG = Dtag1
                        op.node(self.tagD, nodexD[ii], nodeyD[ii], nodezD[ii], SStNodeMass, SStNodeMass, SStNodeMass, 0., 0., 0. )
                        # print(tagD)
                        
                        self.DeckTags = np.append(self.DeckTags,self.tagD)
                    print(f'Deck Nodes : ----------------- {self.DeckTags}')
                    # print('cap beam nodes:',CBtag1)
                    #creating geomteric transformation tag
                    self.geomTTag = self.geomTTag+1
                    
                    #creating geomteric transformation
                    op.geomTransf('Linear', self.geomTTag, 1., 0., 0.)
                    self.DeckEleTagC = np.array([])
                    
                    #creating elements
                    for nodes in range(len(self.Dtag1)-1):
                        self.DeckEleTag = int(self.DeckEleTag+1)
                        self.DeckEleTagC = np.append(self.DeckEleTagC,self.DeckEleTag)
                        op.element(self.CapBeamElementType, self.DeckEleTag, int(self.Dtag1[nodes]), int(self.Dtag1[nodes+1]), int(self.SecTag[5]), self.geomTTag)
                        self.DeckEleTags = np.append(self.DeckEleTags, self.DeckEleTag)
                    
# ======================================================================================================================================================
# Creating Rigid Deck for dynamic analysis
#*=======================================================================================================================================================
                        # print('deck connection',self.Dtag1[nodes],self.Dtag1[nodes+1])
                        # op.equalDOF(Dtag1[nodes],Dtag1[nodes+1], 1, 2, 3, 4, 5, 6)
                        # print(self.Dtag1[nodes],self.Dtag1[nodes+1])

                        op.equalDOF(int(self.Dtag1[nodes]), int(self.Dtag1[nodes+1]), 1,2,3,4,5,6)
                        
#%%
            for DeckNode in self.DeckTags:
                DeckCord = op.nodeCoord(int(DeckNode))
                print(DeckNode, DeckCord)
                
                self.GirderDeckNodes = np.array([]) 
                for GirderNode in self.AllDiaphTags:
                    GirderCord = op.nodeCoord(int(GirderNode))
                    # if  DeckNode == 4:
                        # print(op.nodeCoord(DeckNode), GirderNode, op.nodeCoord(GirderNode))
                    # if GirderCord[1] == DeckCord[1]:
                    if np.linalg.norm(GirderCord[1]-DeckCord[1]) < 1e-6:
                        self.GirderDeckNodes = np.append(self.GirderDeckNodes, GirderNode)
                        # print(DeckNode, GirderNode)
                        op.equalDOF(int(DeckNode), int(GirderNode), *[1, 2, 3, 4, 5, 6])
                    else:
                        # print(GirderNode)
                        continue
        #%%
        
    def _eleload(self): 
      loads = self.load_data or {
          "SSTWT": SSTWT,
          "SocketWt": SocketWt,
          "ShaftWt": ShaftWt,
          "columnWt": columnWt,
          "CapBeamWt": CapBeamWt,
          "ECapBeamWt": ECapBeamWt,
          "PileWt": PileWt,
          "DiaPhragmLoad": DiaPhragmLoad,
          "AdditionalLoad": AdditionalLoad,
      }
      endbentdiaph = self.EndDiaphEleTag[48:61]
      op.timeSeries('Linear',2000)
      op.pattern('Plain', 2000, 2000)
      print('element load', loads["ECapBeamWt"], loads["PileWt"])
      op.eleLoad('-ele', *self.DeckEleTags,     '-type', '-beamUniform', loads["SSTWT"], 0., 0)
      op.eleLoad('-ele', *self.SocketEleTags,   '-type', '-beamUniform', 0, 0., -loads["SocketWt"])
      op.eleLoad('-ele', *self.ShaftEleTags,    '-type', '-beamUniform', 0, 0., -loads["ShaftWt"])
      op.eleLoad('-ele', *self.ColumnEleTags,   '-type', '-beamUniform', 0, 0., -loads["columnWt"])
      op.eleLoad('-ele', *self.CapBeamEleTags,  '-type', '-beamUniform', -loads["CapBeamWt"], 0., 0.)
      op.eleLoad('-ele', *self.ECapBeamEleTags, '-type', '-beamUniform', -loads["ECapBeamWt"], 0., 0.)
      op.eleLoad('-ele', *self.PileElemTags,    '-type', '-beamUniform', 0., 0., -loads["PileWt"])
      op.eleLoad('-ele', *self.EndDiaphEleTag,  '-type', '-beamUniform', -loads["DiaPhragmLoad"], 0., 0.)
      op.eleLoad('-ele', *endbentdiaph,         '-type', '-beamUniform', -loads["AdditionalLoad"], 0., 0.)
    
      return self.DeckEleTags, self.SocketEleTags, self.ShaftEleTags, self.ColumnEleTags, self.CapBeamEleTags, self.ECapBeamEleTags, self.PileElemTags 
           

    def _JoinSoilPile(self):
        """
        Automated Beam-on-Nonlinear-Winkler-Foundation (BNWF).
        Generates fixed soil nodes at the exact coordinates of the pile nodes
        and connects them with zeroLength p-y, t-z, and q-z springs.
        Includes Parallel Viscous materials for Radiation Damping.
        """
              
        print('\n--- Building Winkler Foundation ---')
        
        # Initialize storage for the master spring plots
        self.spring_curves = {'py': [], 'tz': [], 'qz': []}
        
        # Initialize array to hold the base soil nodes for dynamic excitation
        self.BaseSoilNodes = np.array([], dtype=int)
        self.bottom_node_soil_assignments = []
        
        # Ensure units_v4 are defined in your global scope, or define them here if needed
     
        ground_surface_z = self.ground_surface_z
        bottom_of_cap_z = 102.0 * ft  # No soil springs at or above this elevation
        pile_diameter = self.pile_diameter
        
        # Tag counters
        soil_node_tag = 80000 
        mat_tag_base  = 100000 
        
        col_coords = {int(n): tuple(op.nodeCoord(int(n))) for n in self.ShaftColumnNodeTags}

        # Keep bottom-node checks tag based. BottomNodesSo contains both the
        # interior socket bases and the end H-pile tips; these belong to
        # different tag ranges, so do not infer tips from a single prefix.
        bottom_node_tags = {int(n) for n in self.BottomNodesSo}
        missing_bottom_nodes = bottom_node_tags.difference(col_coords)
        if missing_bottom_nodes:
            raise RuntimeError(
                "Bottom node tags are missing from ShaftColumnNodeTags: "
                f"{sorted(missing_bottom_nodes)}"
            )

        active_soil_profile = (
            self.soil_profile if self.soil_profile is not None
            else global_i95_soil_profile
        )
        if not active_soil_profile:
            raise ValueError("No soil profile is available for pile-soil spring generation")

        # Soil profile elevations use a local datum. Align the profile surface
        # with the model ground surface, then compare actual node elevations
        # in that common coordinate system.
        profile_surface_z = float(active_soil_profile[0]['top'])
        profile_bottom_z = float(active_soil_profile[-1]['bottom'])
        profile_to_model_offset = ground_surface_z - profile_surface_z
        # The soil profile datum is 0 m at its bottom. Keep model node
        # elevations unchanged and translate them into that local datum.
        model_profile_bottom_z = profile_to_model_offset
        profile_tolerance = 1.0e-8
        bottom_nodes_below_profile = [
            (node_tag, float(col_coords[node_tag][2]))
            for node_tag in sorted(bottom_node_tags)
            if float(col_coords[node_tag][2]) < model_profile_bottom_z - profile_tolerance
        ]
        if bottom_nodes_below_profile:
            warnings.warn(
                "Bottom nodes extend below the soil profile; the deepest soil "
                "layer properties will be used for those nodes. "
                f"Model soil-bottom elevation={model_profile_bottom_z:.9g}; "
                f"node tags/elevations={bottom_nodes_below_profile}",
                RuntimeWarning,
                stacklevel=2,
            )

        bottom_tz_nodes = set()
        bottom_qz_nodes = set()
      
        
        # Group nodes into individual piles based on (X, Y)
        piles = {}
        for node, (x, y, z) in col_coords.items():
            loc_key = (round(x, 4), round(y, 4))
            # x, y = round(x, 1), round(y, 1)  # ← add this one line
            # loc_key = (x, y)   
            if loc_key not in piles:
                piles[loc_key] = {}
            piles[loc_key][node] = (x, y, z)

        # Loop through each pile individually
        for loc_key, pile_nodes in piles.items():
            
            # Sort ONLY this specific pile's nodes by depth (highest elevation to lowest)
            sorted_pile_nodes = sorted(pile_nodes.items(), key=lambda item: item[1][2], reverse=True)
            
            for i, (Cnode, (x, y, z)) in enumerate(sorted_pile_nodes):
                
                # Skip nodes above the groudn surfcae
                if z >= bottom_of_cap_z:
                    continue
                    
                # 1. CREATE THE FIXED SOIL NODE
                Snode = soil_node_tag
                op.node(Snode, x, y, z)
                op.fix(Snode, 1, 1, 1, 1, 1, 1) 
                # print(Cnode, '-------------------------' )
                
                # 2. CALCULATE TRUE TRIBUTARY LENGTH FOR THIS PILE
                if i == 0: 
                    z_below = sorted_pile_nodes[i+1][1][2]
                    trib_length = (z - z_below) / 2.0
                elif i == len(sorted_pile_nodes) - 1: 
                    z_above = sorted_pile_nodes[i-1][1][2]
                    trib_length = (z_above - z) / 2.0
                else: 
                    z_above = sorted_pile_nodes[i-1][1][2]
                    z_below = sorted_pile_nodes[i+1][1][2]
                    trib_length = (z_above - z_below) / 2.0
                    
                trib_length = max(trib_length, 0.001) 
                depth_from_surf = ground_surface_z - z 
                
                # 3. GET SOIL PROPERTIES
                current_layer, profile_z, used_bottom_layer_fallback = (
                    _soil_layer_for_model_elevation(z, active_soil_profile, profile_to_model_offset)
                )
                
                if not current_layer:
                    if int(Cnode) in bottom_node_tags:
                        message = (
                            f"No soil layer covers bottom node {int(Cnode)} at "
                            f"model z={z:.9g} (profile z={profile_z:.9g}); "
                            f"soil profile bounds are [{profile_bottom_z:.9g}, "
                            f"{profile_surface_z:.9g}]."
                        )
                        warnings.warn(message, RuntimeWarning, stacklevel=2)
                        raise ValueError(message)
                    soil_node_tag += 1
                    continue 

                if int(Cnode) in bottom_node_tags:
                    self.bottom_node_soil_assignments.append({
                        "node_tag": int(Cnode),
                        "model_z": float(z),
                        "depth_below_ground": float(depth_from_surf),
                        "profile_z": float(profile_z),
                        "layer_id": current_layer.get("layer_id"),
                        "uscs": current_layer.get("uscs"),
                        "soil_source": "bottom-layer fallback" if used_bottom_layer_fallback else "profile layer",
                    })
                    
                layer_type = current_layer['uscs']
                mat_type = 1 if layer_type[0].upper() in ['C','M'] else 2
                
                gamma = current_layer.get('gamma', 18000.0) # Fallback to 18 kN/m3
                sigV = max(depth_from_surf * gamma, 0.01)
                soil_logic = 'sand' if layer_type[0].upper() in ['S', 'G'] else 'clay'
                # if Cnode <3000:
                #     print(f" Linked Pile Node {Cnode} <---> Soil Node {Snode} | Coords: (X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}) | Soil Type: {layer_type} | Trib Length: {trib_length:.2f}")
                
                # ---------------------------------------------------------
                # 4. P-Y Springs (Lateral X & Y) with Parallel Radiation Damping
                # ---------------------------------------------------------
                pult, y50 = self._get_pile_soil_params('py', current_layer, depth_from_surf, pile_diameter, trib_length)
                self.store_spring_curve('py', pult/trib_length, y50, depth_from_surf, sigV, soil_logic)

                # A. Calculate Radiation Damping
                G_modulus = current_layer.get('G', 5000.0) # Fallback to 50e6 Pa if not found
                rho = gamma / 9.81  # Mass density
                Vs = math.sqrt(G_modulus / rho) # Shear wave velocity
                C_rad_per_m = 2.0 * pile_diameter * rho * Vs 
                
                # --- THE FIX: SCALE DOWN THE DAMPER ---
                # Theoretical damping is too aggressive against a fixed base.
                # Drop it to 2% just to kill the high-frequency ringing.
                damping_modifier = 0.0001
                C_rad = C_rad_per_m * trib_length * damping_modifier
                
                # B. Define Sequential Tags (Fits safely inside the +10 increment)
                spring_tag_x   = mat_tag_base + 1
                dashpot_tag_x  = mat_tag_base + 2   
                parallel_tag_x = mat_tag_base + 3  
                
                spring_tag_y   = mat_tag_base + 4
                dashpot_tag_y  = mat_tag_base + 5   
                parallel_tag_y = mat_tag_base + 6  
                
                tz_tag         = mat_tag_base + 7
                qz_tag         = mat_tag_base + 8

                # C. Build Parallel Materials (X and Y)
                op.uniaxialMaterial('PySimple1', spring_tag_x, mat_type, pult, y50, 0.3) 
                # op.uniaxialMaterial('Viscous', dashpot_tag_x, C_rad, 1.0) 
                # op.uniaxialMaterial('Parallel', parallel_tag_x, spring_tag_x, dashpot_tag_x)
                
                op.uniaxialMaterial('PySimple1', spring_tag_y, mat_type, pult, y50, 0.3) 
                # op.uniaxialMaterial('Viscous', dashpot_tag_y, C_rad, 1.0) 
                # op.uniaxialMaterial('Parallel', parallel_tag_y, spring_tag_y, dashpot_tag_y)
                
                # ---------------------------------------------------------
                # 5. T-Z Spring (Vertical Skin Friction)
                # ---------------------------------------------------------
                tult, z50 = self._get_pile_soil_params('tz', current_layer, depth_from_surf, pile_diameter, trib_length)
                self.store_spring_curve('tz', tult/trib_length, z50, depth_from_surf, sigV, soil_logic)

                op.uniaxialMaterial('TzSimple1', tz_tag, mat_type, tult, z50, 0.1)
                
                # Connect the fixed Snode to the moving Cnode (DOFs 1, 2, 3)
                # Assign parallel wrappers to lateral DOFs 1 & 2, and the standard T-Z spring to vertical DOF 3
                self.zeroEleTag += 1
                # op.element('zeroLength', int(self.zeroEleTag), Snode, Cnode, '-mat', parallel_tag_x, tz_tag, '-dir', 1,  3)
                op.element('zeroLength', int(self.zeroEleTag), Snode, Cnode, '-mat', spring_tag_x, spring_tag_y, tz_tag, '-dir', 1, 2, 3)
                self.zeroEleTags = np.append(self.zeroEleTags, self.zeroEleTag)
                if int(Cnode) in bottom_node_tags:
                    bottom_tz_nodes.add(int(Cnode))
                
                
                # ---------------------------------------------------------
                # 6. Q-Z Spring (Pile Tip Only)
                # ---------------------------------------------------------
                # Use the explicit bottom-node tags; socket bases and H-pile
                # tips do not share one elevation or one node-tag range.
                if int(Cnode) in bottom_node_tags:
                    qult, qz50 = self._get_pile_soil_params('qz', current_layer, depth_from_surf, pile_diameter, trib_length)
                    self.store_spring_curve('qz', qult, qz50, depth_from_surf, sigV, soil_logic)

                    op.uniaxialMaterial('QzSimple1', qz_tag, mat_type, qult, qz50, 0.1)
                    
                    # Apply strictly in the Z direction (DOF 3)
                    self.zeroEleTag += 1
                    op.element('zeroLength', int(self.zeroEleTag), Snode, Cnode, '-mat', qz_tag, '-dir', 3)
                    self.zeroEleTags = np.append(self.zeroEleTags, self.zeroEleTag)
                    bottom_qz_nodes.add(int(Cnode))
                    
                    
                    # Record this base node for dynamic excitation
                    self.BaseSoilNodes = np.append(self.BaseSoilNodes, Snode)
                    # 
                    print(f"    -> Added Q-Z End Bearing Spring at true Pile Tip (Z: {op.nodeCoord(Cnode)[2]:2f})")
                    print(f"    -> Added Q-Z End Bearing Spring at true Pile Tip Node ({Cnode}")
                    
                # Increment by 10 safely covers all 8 tags used above
                mat_tag_base += 10 
                soil_node_tag += 1

        missing_tz = bottom_node_tags.difference(bottom_tz_nodes)
        missing_qz = bottom_node_tags.difference(bottom_qz_nodes)
        if missing_tz or missing_qz:
            raise RuntimeError(
                "Bottom spring audit failed: "
                f"missing T-Z at {sorted(missing_tz)}; "
                f"missing Q-Z at {sorted(missing_qz)}"
            )
        print(
            f"Verified T-Z and Q-Z springs at all {len(bottom_node_tags)} "
            f"bottom nodes: {sorted(bottom_node_tags)}"
        )

        # Generates the Master Plots once all piles are finished
        # print(f'\n--- Generating Master Spring Plots  {self.BottomNodesSo} ---')
        self.plot_master_spring_curves()
       
       

   
    def _get_pile_soil_params(self, curve_type, layer_data, depth, b, pEleLength, gwtSwitch=1, loading_type='cyclic', pci_to_units=271447.1, psfin_to_units=1885.05, p_mult=0.5, seismic_softening_factor=0.15):
        """ 
        Calculates p-y, t-z, and q-z parameters strictly in N, m, s.
        Includes direct calibration hooks to match 3D target displacements.
        """
        import math
        import numpy as np

        # =====================================================================
        # CALIBRATION HOOKS - TUNE THESE TO MATCH YOUR 2.3 INCH TARGET
        # =====================================================================
        # To get more displacement, lower p_modifier and increase y_modifier.
        p_modifier = 0.40  # Drops ultimate soil strength to 30%
        y_modifier = 40.0   # Makes the soil 3x softer initially (increases y50)
        # =====================================================================

        uscs = layer_data.get('uscs', 'SM')
        gamma = layer_data.get('gamma', 0.01) # N/m^3
        phi = layer_data.get('phi', 0.0)      # degrees
        su = layer_data.get('su', 0.0)        # N/m^2 (Pa)
        eps50 = layer_data.get('eps50', 0.0)
        G = layer_data.get('G', 0.0)          # N/m^2 (Pa)
        
        k_py_user = layer_data.get('k_py', None)
        k_tz_user = layer_data.get('k_tz', None)

        pi = math.pi
        sigV = max(depth * gamma, 0.01)
        uscs_main = uscs[0].upper()
        soil_logic = 'sand' if uscs_main in ['S', 'G'] else 'clay'
        
        c_type = curve_type.strip().lower()

        if c_type == 'py':
            if soil_logic == 'sand':
                phi_rad = math.radians(phi)
                A = 0.9 
                K0 = 0.4
                Ka = math.tan(pi/4 - phi_rad/2)**2
                
                c1 = K0 * math.tan(phi_rad) * math.sin(pi/4 + phi_rad/2) / (math.tan(pi/4 - phi_rad/2) * math.cos(phi_rad/2))
                c2 = (math.tan(pi/4 + phi_rad/2)**2) * math.tan(phi_rad/2)
                c3 = K0 * math.tan(pi/4 + phi_rad/2) * (math.tan(phi_rad) * math.sin(pi/4 + phi_rad/2) - math.tan(phi_rad/2))
                c4 = (math.tan(pi/4 + phi_rad/2) / math.tan(pi/4 - phi_rad/2)) - Ka
                
                pst = gamma * depth * (depth * (c1 + c2 + c3) + b * c4)
                psd = b * gamma * depth * (Ka * (math.tan(pi/4 + phi_rad/2)**8 - 1) + K0 * math.tan(phi_rad) * math.tan(pi/4 + phi_rad/2)**4)
                
                pult_stress = A * min(pst, psd) if depth > 0 else 0.01
                pult_stress = pult_stress * p_mult
                
                if k_py_user is not None:
                    k_val = k_py_user * seismic_softening_factor
                else:
                    k_pci = 61.0 if gwtSwitch == 1 else 42.0
                    k_val = (k_pci * pci_to_units) * seismic_softening_factor
                
                y50 = 0.5 * (pult_stress / A) / (k_val * max(depth, 0.01)) * 0.5 * math.log((1+0.5)/(1-0.5))
                # print(f'y50 : {y50* y_modifier}, and p: {(pult_stress * pEleLength) * p_modifier}, at depth {depth} for sand')
                # --- APPLY CALIBRATION ---
                return (pult_stress * pEleLength) * p_modifier, y50 #* y_modifier
                
            else:
                pu_shallow = (3 + (sigV/su) + (0.5 * depth / b)) * su * b
                pu_deep = 9 * su * b
                p_stress = min(pu_shallow, pu_deep)
                p_stress = p_stress * p_mult
                
                if loading_type.lower() == 'cyclic' and depth < (2.5 * b):
                    p_stress *= 0.75 
                    
                y50 = 2.5 * eps50 * b
                # print(f'y50 : {y50* y_modifier}, and p: {(p_stress * pEleLength) * p_modifier}, at depth {depth} for sand')
                # --- APPLY CALIBRATION ---
                return (p_stress * pEleLength) * p_modifier, y50 #* y_modifier

        elif c_type == 'tz':
            if soil_logic == 'sand':
                delta = math.radians(0.8 * phi)
                tult = (0.4 * sigV * pi * b * math.tan(delta)) * pEleLength
                
                if k_tz_user is not None:
                    kf = k_tz_user
                else:
                    kf_psfin = np.interp(phi, [28, 31, 32, 34, 35, 38], [6000, 10000, 10000, 14000, 14000, 18000])
                    kf = kf_psfin * psfin_to_units
                
                z50 = tult / kf
                
                # --- APPLY CALIBRATION ---
                return tult * p_modifier, z50 #* y_modifier
                
            else:
                tult = (0.5 * su) * (pi * b) * pEleLength
                z50 = 0.01 * b  
                
                # --- APPLY CALIBRATION ---
                return tult * p_modifier, z50 #* y_modifier

        elif c_type == 'qz':
            if soil_logic == 'sand':
                phi_rad = math.radians(phi)
                Ir = G / (sigV * math.tan(phi_rad)) if G > 0 else 250
                Nq = (1+2*(1-math.sin(phi_rad)))*(1/(3-math.sin(phi_rad)))*math.exp(pi/2-phi_rad)*(math.tan(pi/4+phi_rad/2)**2)*(Ir**((4*math.sin(phi_rad))/(3*(1+math.sin(phi_rad)))))
                
                qult = (Nq * sigV) * (pi * b**2 / 4)
                qz50 = 0.125 * (0.05 * b)
                
                # --- APPLY CALIBRATION ---
                return qult * p_modifier, qz50 #* y_modifier
                
            else:
                qult = (9 * su) * (pi * b**2 / 4)
                qz50 = 0.05 * b
                
                # --- APPLY CALIBRATION ---
                return qult * p_modifier, qz50 #* y_modifier
                
        return 0.0, 0.0


    def store_spring_curve(self, curve_type, ult_force, disp50, depth, sigV, soil_logic):
        """ Calculates the backbone curve arrays and stores them in memory for later plotting. """
        import numpy as np

        if ult_force <= 0 or disp50 <= 0: 
            return

        ult_kN = ult_force / 1000.0
        disp50_mm = disp50 * 1000.0
        sigV_kPa = sigV / 1000.0

        disp_mm = np.linspace(0, 100, 100) # 0 to 100 mm displacement
        disp_m = disp_mm / 1000.0
        force_kN = np.zeros_like(disp_mm)
        c_type = curve_type.strip().lower()

        if c_type == 'py':
            if soil_logic == 'sand':
                force_kN = ult_kN * np.tanh(0.5493 * (disp_m / disp50))
            else:
                for i, y in enumerate(disp_m):
                    f = 0.5 * ult_kN * (y / disp50)**(1/3)
                    force_kN[i] = min(f, ult_kN)
        elif c_type in ['tz', 'qz']:
            for i, z in enumerate(disp_m):
                force_kN[i] = ult_kN * (z / (disp50 + z))

        self.spring_curves[c_type].append({
            'disp': disp_mm,
            'force': force_kN,
            'depth': depth,
            'sigV': sigV_kPa
        })


    def plot_master_spring_curves(self):
        """ Draws all stored curves onto 3 master plots (PY, TZ, QZ) and annotates them. """
        import os
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np

        plot_dir = "Spring_Plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        for c_type, curves in self.spring_curves.items():
            if not curves: 
                continue

            plt.figure(figsize=(10, 8))
            
            # Sort curves by depth so they plot nicely top-to-bottom
            curves = sorted(curves, key=lambda x: x['depth'])
            
            # Create a custom gradient from Pure Blue (shallow) to Pure Red (deep)
            from matplotlib.colors import LinearSegmentedColormap
            blue_to_red_cmap = LinearSegmentedColormap.from_list("blue_red", ["blue", "red"])
            colors = blue_to_red_cmap(np.linspace(0.0, 1.0, len(curves)))

            for i, curve in enumerate(curves):
                plt.plot(curve['disp'], curve['force'], color=colors[i], linewidth=1.5)
                
                annotation_text = f"{curve['depth']:.1f}m ({curve['sigV']:.0f}kPa)"
                plt.annotate(annotation_text, 
                             xy=(curve['disp'][-1], curve['force'][-1]), 
                             xytext=(5, 0),
                             textcoords="offset points", 
                             fontsize=8, 
                             color=colors[i],
                             verticalalignment='center')

            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('Displacement (mm)', fontsize=16)
            plt.tick_params(axis='both', which='major', labelsize=14, length=8, width=1.5)
            plt.tick_params(axis='both', which='minor', labelsize=14, length=4, width=1)
            
            if c_type == 'py':
                plt.ylabel('Lateral Resistance, p (kN)', fontsize=16)
                plt.title('P-Y Curves', fontsize=16, fontweight='bold')
            elif c_type == 'tz':
                plt.ylabel('Vertical Skin Friction, t (kN)', fontsize=16)
                plt.title('T-Z Curves', fontsize=16, fontweight='bold')
            else:
                plt.ylabel('Tip Resistance, q (kN)', fontsize=16)
                plt.title('Q-Z Curves', fontsize=16, fontweight='bold')

            # plt.xlim(0, 115) 
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"Master_{c_type.upper()}_Curves.png"), dpi=200)
            plt.close()
            print(f" -> Saved Master {c_type.upper()} plot with {len(curves)} curves.")
    

    def getBaseSoilNodes(self):
        """Returns the list of fixed soil nodes at the pile tips."""
        return self.BaseSoilNodes.tolist()
