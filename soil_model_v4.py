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
class _ModelSoil_:
    global_node_map = {}
    ALL_NODES = []
    def __init__(self,  width, length, fixed_distance, fixed_length, fixed_width, width_elements, hole_centers, PileDia, hole_depth, outer_radius, LayerEndNodeid, LayerEndEleid, counter, flag):
        
        """
        
        Parameters
        ----------
        width : TYPE
            DESCRIPTION.
        depth : TYPE
            DESCRIPTION.
        length : TYPE
            DESCRIPTION.
        width_elements : TYPE
            DESCRIPTION.
        hole_centers : TYPE
            DESCRIPTION.
        PileDia : TYPE
            DESCRIPTION.
        hole_depth : TYPE
            DESCRIPTION.
        outer_radius : TYPE
            DESCRIPTION.
        LayerEndNodeid : TYPE
            DESCRIPTION.
        LayerEndEleid : TYPE
            DESCRIPTION.
        counter : TYPE
            DESCRIPTION.
        User_range : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        #%% variables
        # self.range = User_range
        # self.width = width
        
        self.bottom_left_corner = width
        self.depth_b = width[2]
        self.top_right_corner = length
        self.flag = flag
        # self.length = length
        self.width_elements = width_elements
        self.depth_elements = depth_elements
        self.length_elements = length_elements
        self.fixed_distance =  fixed_distance 
        self.counter = counter
        # self.fixed_length =  fixed_length
        # self.fixed_width = fixed_width
        # Pile properties
        self.hole_centers = hole_centers  # List of (x, y) coordinates for piles
        self.hole_radius = PileDia
        self.hole_depth = hole_depth  # Depth up to which the pile should be created
        self.outer_radius = outer_radius  # Outer radius for nodes around the holes

        # Starting ID for soil nodes and elements
        self.starting_node_id = int(LayerEndNodeid)
        self.starting_element_id = int(LayerEndEleid)
        global m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi, g, Ubig, Usmall
        global gamma_rc, gamma_c, gamma_s, Ubig, Usmall, g
        # m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi, lbf, pcf, psf, inch2, inch4, g, Ubig, Usmall= get_units2()
        # get material id based on layer
        m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi = get_units()
        self.SoilNodes = np.array([])      
        
        self.zi = 24*ft              # the depth where spacing becomes loose
        self.n_dense = 6             # number of dense vertical layers above zi
        #%% Calling methods/functions
        self.initialize_model()
        if self.counter == 1:
            
            self.define_materials()
        
        self.node_coords, self.node_id, self.minx_nodes, self.maxx_nodes, self.depthnodes, self.miny_nodes, self.maxy_nodes = self.create_nodes()
        self.mDensity = self.mden()
        self._soilNodes = self.create_elements()
       
        # self.delete_nodes()
        # self.nodes_around_holes = self.get_nodes_around_holes()
        
        # self.CreateEqDof()
        
        self.fixBoundary()
        # self.newNodes = self.create_new_nodes_around_holes()
    
    def mden(self):
        mDen = np.array([1821., 1807., 1728., 1807., 1807., 1885., 1935., 1964., 1964.])*kg/m**3
        self.mDensity = mDen
        return self.mDensity
           
    #%%   
    
        
    #%% model initialize   
    def initialize_model(self):
        """
        

        Returns
        -------
        None.

        """
        
        op.model('basic', '-ndm', 3, '-ndf', 3)
        print('soil model initialized')
        
   
    #%%  Create fixed boundary  
    def fixBoundary(self):
        print('fixed boundary at bottom and constraing disp in Y direction ...')
    
                
        print('fix boundary at depth',self.depth_b)
        op.fixZ(self.depth_b, 1, 1, 1) # fix 
       
        # op.fixY(self.bottom_left_corner[1], 0, 1, 0)
        # op.fixY(self.bottom_left_corner[1], 1, 1, 1)
        # op.fixX(self.bottom_left_corner[1], 1, 1, 1)
        # op.fixY(self.top_right_corner[1], 0, 1, 0)
        # op.fixY(self.top_right_corner[1], 1, 1, 1)
        # op.fixX(self.top_right_corner[1], 1, 1, 1)

   #%%
    
    def non_linear_spacing(self, elements, exponent=1.2,
                       start_coord=0.0, end_coord=1.0, center_coord=None):
        """
        elements      : number of elements (intervals)
        exponent      : controls clustering intensity (higher = tighter)
        start_coord   : physical start coordinate
        end_coord     : physical end coordinate
        center_coord  : physical coordinate where density is highest
        """
    
        # -------------------------------------------------------------
        # Input validation
        # -------------------------------------------------------------
        if not isinstance(elements, (int, float)):
            raise TypeError(f"elements must be a number, got {type(elements)}")
    
        if center_coord is None:
            center_coord = (start_coord + end_coord) / 2.0   # default: geometric center
    
        # -------------------------------------------------------------
        # Map physical coordinate to fractional element index
        # -------------------------------------------------------------
        if end_coord == start_coord:
            raise ValueError("start_coord and end_coord cannot be equal")
    
        domain_length = end_coord - start_coord
    
        # Center position in index space (0 to elements)
        center_index = (center_coord - start_coord) / domain_length * elements
    
        # Clamp center_index into valid range
        center_index = max(0.0, min(center_index, float(elements)))
    
        # -------------------------------------------------------------
        # Generate non-linear cluster shape
        # -------------------------------------------------------------
        positions = []
    
        for i in range(elements + 1):
            # distance from center in index units_v4
            factor = i - center_index
    
            # scale factor by maximum possible distance (for stability)
            denom = max(center_index, elements - center_index, 1e-12)
            factor /= denom
    
            # apply exponent bias
            val = math.copysign(abs(factor) ** exponent, factor)
    
            positions.append(val)
    
        # -------------------------------------------------------------
        # Normalize 0–1 so spacing is consistent
        # -------------------------------------------------------------
        min_pos = min(positions)
        max_pos = max(positions)
    
        # convert to normalized spacings
        return [(p - min_pos) / (max_pos - min_pos) for p in positions]
    #%% detph spacing

    
    def custom_depth_spacing(self):
        """
        Generate depth spacing with dense nodes near top (max_z → zi)
        and loose nodes below zi (toward bottom, min_z).
        Works for bottom-to-top node creation.
        """
        min_z = self.bottom_left_corner[2]
        max_z = self.top_right_corner[2]
        total_H = max_z - min_z
    
        # Transition depth (from bottom)
        zi = np.clip(self.zi, min_z + 1e-6, max_z - 1e-6)  # prevent exact 0 or max
    
        # Fractional position of zi from bottom
        zi_frac = (zi - min_z) / total_H
    
        N = self.depth_elements
    
        # At least 1 element in loose portion
        N_loose = max(1, int(np.round(N * zi_frac)))
        N_dense = N - N_loose
    
        # -------------------------------------------------------------
        # Dense spacing above zi (closer to top) - unchanged
        # -------------------------------------------------------------
        if N_dense > 0:
            frac_dense = np.linspace(0, 1, N_dense + 1)[1:]  # avoid duplicate at zi_frac
            z_dense = zi_frac + (1 - zi_frac) * (frac_dense ** 1.)  # exponent = 1 for dense
        else:
            z_dense = np.array([])
    
        # -------------------------------------------------------------
        # Loose spacing below zi (toward bottom) - fixed
        # -------------------------------------------------------------
        frac_loose = np.linspace(0, 1, N_loose + 1)
        z_loose = (zi_frac) * (frac_loose ** 0.7)  # exponent <1: longer elements at bottom
        z_loose[-1] = zi_frac  # ensure last loose node exactly at transition
    
        # -------------------------------------------------------------
        # Combine bottom-to-top
        # -------------------------------------------------------------
        spacing = np.concatenate([z_loose, z_dense])
    
        return spacing.tolist()


    #%% FINAL: create_nodes (NO Pile Hole, Unique Nodes, Exact Return Signature)
    def create_nodes(self):
        print('Creating soil Nodes ...')
        
        min_x, min_y, min_z = self.bottom_left_corner
        max_x, max_y, max_z = self.top_right_corner
        
        # REQUIRED FOR RETURN SIGNATURE. Initialized as empty set.
        self.nodetodelete = set() 
        
        self.width = max_x - min_x
        self.length = max_y - min_y
        self.depth = max_z - min_z
        
        self.fixed_length = 24*inch 
        self.fixed_width = 24*inch

        # Define Spacing
        if self.counter < 4: 
            self.width_spacing = self.non_linear_spacing(self.width_elements, exponent=1.7, center_coord =0.5)
            self.length_spacing = self.non_linear_spacing(self.length_elements, exponent=1.7, center_coord = 0.5)
            self.depth_spacing = [k / self.depth_elements for k in range(self.depth_elements + 1)]
        else:
            self.width_spacing = self.non_linear_spacing(self.width_elements, exponent=1.05, center_coord =0.2)
            self.length_spacing = self.non_linear_spacing(self.length_elements, exponent=1.7, center_coord = 0.5)
            self.depth_spacing = [k / self.depth_elements for k in range(self.depth_elements + 1)]

        self.node_coords = {}
        self.local_to_global_tags = [] 
        
        # =========================================================================
        # ID LOGIC: Robust Tagging (Starts at max(Input, Last Used Tag + 1, 50001))
        # =========================================================================
        START_NODE_TAG = 50001
        self.node_id = self.starting_node_id
        
        try:
            max_existing_tag = max(op.getNodeTags()) + 1
            self.node_id = max(self.starting_node_id, max_existing_tag, START_NODE_TAG)
        except:
            self.node_id = max(self.starting_node_id, START_NODE_TAG)
        # =========================================================================
        
        self.depth_nodes = {}
        self.min_x_nodes = []
        self.max_x_nodes = []
        self.min_y_nodes = []
        self.max_y_nodes = []
        
        TOL = 1e-6
        ROUND_DIGITS = 6 

        for j in range(self.length_elements + 1):
            for i in range(self.width_elements + 1):
                for k in range(self.depth_elements + 1):
                    x = min_x + self.width * self.width_spacing[i]
                    y = min_y + self.length * self.length_spacing[j]
                    z = min_z + self.depth * self.depth_spacing[k]
                    
                    key = (round(x, ROUND_DIGITS), round(y, ROUND_DIGITS), round(z, ROUND_DIGITS))
                    
                    # CORE LOGIC: CHECK FOR EXISTING NODE
                    node_tag = _ModelSoil_.global_node_map.get(key)

                    if node_tag is None:
                        # NODE IS NEW: Create it using the next sequential tag
                        node_tag = self.node_id
                        op.node(node_tag, x, y, z)
                        
                        _ModelSoil_.global_node_map[key] = node_tag
                        self.node_id += 1 # Increment only for NEW nodes
                    else:
                        # NODE EXISTS: Use the existing tag
                        pass
                    
                    self.local_to_global_tags.append(node_tag)

                    # REMOVED: Pile hole logic (no population of self.nodetodelete)

                    coords = np.array([x, y, z])
                    self.node_coords[node_tag] = coords
                    
                    if k not in self.depth_nodes:
                        self.depth_nodes[k] = []
                    self.depth_nodes[k].append((node_tag, x))

                    if abs(x - self.bottom_left_corner[0]) < TOL:
                        self.min_x_nodes.append((node_tag, x, z))
                    elif abs(x - self.top_right_corner[0]) < TOL:
                        self.max_x_nodes.append((node_tag, x, z))
                    
                    if abs(y - self.bottom_left_corner[1]) < TOL:
                        self.min_y_nodes.append((node_tag, y,z))
                    elif abs(y - self.top_right_corner[1]) < TOL:
                        self.max_y_nodes.append((node_tag, y,z))
                    
        print('Nodes Created ...')
        # Returns the exact 8-variable signature, with self.nodetodelete being an empty set.
        return self.node_coords, self.node_id, self.min_x_nodes, self.max_x_nodes, self.depth_nodes, self.min_y_nodes, self.max_y_nodes
    #%% define materials    
    def define_materials(self):
        """
        
        Returns
        -------
        None.

        """
        
#        self.materials = [
#           (20, 5*MPa, 0.2, 1800*kg/m**3),
#           (21, 100*MPa, 0.35, 1800*kg/m**3),
#           (22, 100*MPa, 0.35, 1800*kg/m**3),
#           (23, 100*MPa, 0.35, 1800*kg/m**3)
#        ]
#       

        #%% Drucker Prager material
        nm= 1
        Es = np.array([8, 15, 10, 26, 18, 30, 30, 50, 50])* MPa                        # intermediate
        # Es = np.array([51, 86, 149,215, 171, 225, 461, 615, 516])* MPa               # Vs 
       
        # Es = np.array([5*nm, 14, 6*nm, 26, 15, 22, 20, 28, 27])* MPa                 # SPT
        mu = np.array([0.27, 0.3, 0.2, 0.3, 0.2, 0.3, 0.3, 0.3, 0.3])
        Gs = np.ceil(Es/(2*(1+mu)))
        ki = np.ceil(Es/(3*(1-2*mu)))
        ci = np.array([5, 5, 5, 5, 5, 84., 5, 108., 5])*kPa
        phi = np.array([31, 30, 28, 30, 30,10, 34, 10, 35]) # degree
        phi = np.radians(phi)
        rho = 2*np.sin(phi)/(np.sqrt(3)*(3-np.sin(phi)))
        # rho = 2*np.sqrt(2)*math.sin(math.radians(phi))/(np.sqrt(3)*(3-math.sin(math.radians(phi))))
        sigY =   6*ci*np.cos(phi) / (np.sqrt(2)*(3-np.sin(phi)))   
        rhoBar = 2/3*rho # 0.2*rho
        #------------------------------------------------------------------------------------------------------------ change this based on the properties used above
        Kinf = np.array([178, 170, 153, 166, 166, 112, 200, 137, 222])*kPa
#        Kinf = 100.*kPa
        Ko = 10.*kPa # 30.*kPa
        delta1 = np.array([200., 200., 200., 200., 200., 200., 200., 100., 200.])
#        delta1 = np.array([250., 250., 120., 250., 250., 120., 100., 300., 70.])
        
        delta2 = 0.
        H = 1000.*kPa
        #theta = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) #0.5 Assume for numerical stability
        theta = np.array([0.43, 0.43, 0.5, 0.5, 0.5, 0.5, 0.4, 0.51, 0.2]) #0.5 intermediate 
        #--------------------------------------------------------------------------------------------------------------
        mDen = np.array([1821., 1807., 1728., 1807., 1807., 1885., 1935., 1964., 1964.])*kg/m**3
        self.mDensity = mDen
        patm = 101*kPa
        print(f'Soil Mat E : {Es}, \nmu : {mu}, \nGs :{Gs}, \nK: {ki}')
        
       
        self.materials = [
                               (20, ki[0], Gs[0], sigY[0],  rho[0],  rhoBar[0],  Kinf[0],  Ko,  delta1[0],  delta2,  H,  theta[0],  mDen[0], patm),
                               (21, ki[1], Gs[1], sigY[1],  rho[1],  rhoBar[1],  Kinf[1],  Ko,  delta1[1],  delta2,  H,  theta[0],  mDen[1], patm),
                               (22, ki[2], Gs[2], sigY[2],  rho[2],  rhoBar[2],  Kinf[2],  Ko,  delta1[2],  delta2,  H,  theta[0],  mDen[2], patm),
                               (23, ki[3], Gs[3], sigY[3],  rho[3],  rhoBar[3],  Kinf[3],  Ko,  delta1[3],  delta2,  H,  theta[0],  mDen[3], patm),
                               (24, ki[4], Gs[4], sigY[4],  rho[4],  rhoBar[4],  Kinf[4],  Ko,  delta1[4],  delta2,  H,  theta[0],  mDen[4], patm),
                               (25, ki[5], Gs[5], sigY[5],  rho[5],  rhoBar[5],  Kinf[5],  Ko,  delta1[5],  delta2,  H,  theta[0],  mDen[5], patm),
                               (26, ki[6], Gs[6], sigY[6],  rho[6],  rhoBar[6],  Kinf[6],  Ko,  delta1[6],  delta2,  H,  theta[0],  mDen[6], patm),
                               (27, ki[7], Gs[7], sigY[7],  rho[7],  rhoBar[7],  Kinf[7],  Ko,  delta1[7],  delta2,  H,  theta[0],  mDen[7], patm),
                               (28, ki[8], Gs[8], sigY[8],  rho[8],  rhoBar[8],  Kinf[8],  Ko,  delta1[8],  delta2,  H,  theta[0],  mDen[8], patm),
                               
                                
                           ]
        

#        self.materials = [
#                               (20, Es[0], mu[0], mDen[0]),
#                               (21, Es[1], mu[1], mDen[1]),
#                               (22, Es[2], mu[2], mDen[2]),
#                               (23, Es[3], mu[3], mDen[3]),
#                               (24, Es[4], mu[4], mDen[4]),
#                               (25, Es[5], mu[5], mDen[5]),
#                               (26, Es[6], mu[6], mDen[6]),
#                               (27, Es[7], mu[7], mDen[7]),
#                               (28, Es[8], mu[8], mDen[8]),      
#                           ]
#        
        
        # print('material for soil layers', self.materials)
        
        for mat in self.materials:
        
            #nDMaterial ('DruckerPrager', matTag, bulkMod, shearMod, sigY, rho, rhoBar, Kinf, Ko, delta1, delta2, H, theta, mDen)
            op.nDMaterial ('DruckerPrager', *mat)
            
            #op.nDMaterial('ElasticIsotropic', *mat)
        print('soil material defined') 
        op.nDMaterial ('InitialStateAnalysisWrapper', 30, 20, 3)
        op.nDMaterial ('InitialStateAnalysisWrapper', 31, 21, 3)
        op.nDMaterial ('InitialStateAnalysisWrapper', 32, 22, 3)
        op.nDMaterial ('InitialStateAnalysisWrapper', 33, 23, 3)
        op.nDMaterial ('InitialStateAnalysisWrapper', 34, 24, 3)
        op.nDMaterial ('InitialStateAnalysisWrapper', 35, 25, 3)
        op.nDMaterial ('InitialStateAnalysisWrapper', 36, 26, 3)
        op.nDMaterial ('InitialStateAnalysisWrapper', 37, 27, 3)
        op.nDMaterial ('InitialStateAnalysisWrapper', 38, 28, 3)   


    #%% Get Material ID    
    def get_material_id(self, z):
        """
        
        Parameters
        ----------
        z : TYPE
            DESCRIPTION.
    
        Raises
        ------
        ValueError
            DESCRIPTION.
    
        Returns
        -------
        int
            DESCRIPTION.
    
        """
        
        if 102*ft >= z >= 94*ft:
            return 30, self.mDensity[0]
        elif 94*ft > z >= 92*ft:
            return 31, self.mDensity[1]
        elif 92*ft > z >= 90*ft:
            return 32, self.mDensity[2]
        elif 90*ft > z >= 88*ft:
            return 33, self.mDensity[3]
        elif 88*ft > z >= 80*ft:
            return 34, self.mDensity[4]
        elif 80*ft > z >= 70*ft:
            return 35, self.mDensity[5]
        elif 70*ft > z >= 54*ft:
            return 36, self.mDensity[6]
        elif 54*ft > z >= 50*ft:
            return 37, self.mDensity[7]
        elif 50*ft > z >= 0.:
            return 38, self.mDensity[8]
        else:
            raise ValueError(f"Depth out of range for material assignment: {z}")

#        if 102*ft >= z >= 94*ft:
#            return 27
#        elif 94*ft > z >= 92*ft:
#            return 27
#        elif 92*ft > z >= 90*ft:
#            return 27
#        elif 90*ft > z >= 88*ft:
#            return 27
#        elif 88*ft > z >= 80*ft:
#            return 27
#        elif 80*ft > z >= 70*ft:
#            return 27
#        elif 70*ft > z >= 54*ft:
#            return 27
#        elif 54*ft > z >= 50*ft:
#            return 27
#        elif 50*ft > z >= 0.:
#            return 27
#        else:
#            raise ValueError(f"Depth out of range for material assignment: {z}")
        
#        if 102*ft >= z >= 94*ft:
#            return 37
#        elif 94*ft > z >= 92*ft:
#            return 37
#        elif 92*ft > z >= 90*ft:
#            return 37
#        elif 90*ft > z >= 88*ft:
#            return 37
#        elif 88*ft > z >= 80*ft:
#            return 37
#        elif 80*ft > z >= 70*ft:
#            return 37
#        elif 70*ft > z >= 54*ft:
#            return 37
#        elif 54*ft > z >= 50*ft:
#            return 37
#        elif 50*ft > z >= 0.:
#            return 37
#        else:
#            raise ValueError(f"Depth out of range for material assignment: {z}")  
                 
    #%% Creta soil elements

    def create_elements(self):
        
        print('defining soil element ...')
        # self.elements_to_delete is not needed now
        element_aspect_data = []
        
        # =========================================================================
        # ELEMENT ID LOGIC: Robust Tagging 
        # Start element ID at max(self.starting_element_id, max(existing tags)+1, 50001)
        # =========================================================================
        START_ELEMENT_TAG = 50001
        self.element_id = self.starting_element_id
        
        try:
            max_existing_tag = max(op.getEleTags()) + 1
            self.element_id = max(self.starting_element_id, max_existing_tag, START_ELEMENT_TAG)
        except:
            self.element_id = max(self.starting_element_id, START_ELEMENT_TAG)
        # =========================================================================
        
        width_elements = self.width_elements
        length_elements = self.length_elements
        depth_elements = self.depth_elements
        
        # INDEXING FUNCTION: Maps the 3D grid position (i, j, k) to the 1D index
        def get_local_index(i, j, k):
             # k-index varies fastest (depth), then i (width), then j (length)
             return j * (width_elements + 1) * (depth_elements + 1) + i * (depth_elements + 1) + k
             
        for j in range(self.length_elements):
            for i in range(self.width_elements):
                for k in range(self.depth_elements):
                    
                    # CRITICAL CHANGE: Fetching the unique node tag from the list for guaranteed connectivity
                    # The list index is calculated based on the local grid indices (i, j, k)
                    self.nodes = [
                        self.local_to_global_tags[get_local_index(i, j, k)],      # 1
                        self.local_to_global_tags[get_local_index(i+1, j, k)],    # 2
                        self.local_to_global_tags[get_local_index(i+1, j+1, k)],  # 3
                        self.local_to_global_tags[get_local_index(i, j+1, k)],    # 4
                        self.local_to_global_tags[get_local_index(i, j, k+1)],    # 5
                        self.local_to_global_tags[get_local_index(i+1, j, k+1)],  # 6
                        self.local_to_global_tags[get_local_index(i+1, j+1, k+1)],# 7
                        self.local_to_global_tags[get_local_index(i, j+1, k+1)],  # 8
                    ]
                    
                    
                   
                  
                    self.skip_element = False
                            
                    coords = [self.node_coords[n] for n in self.nodes if n in self.node_coords]
                    if len(coords) != 8:
                        self.skip_element = True
                    
                    if not self.skip_element:
                        dx = max(coord[0] for coord in coords) - min(coord[0] for coord in coords)
                        dy = max(coord[1] for coord in coords) - min(coord[1] for coord in coords)
                        dz = max(coord[2] for coord in coords) - min(coord[2] for coord in coords)
                        max_edge = max(dx, dy, dz)
                        min_edge = min(dx, dy, dz)
                        aspect_ratio = max_edge / min_edge if min_edge > 0 else float('inf')
                        element_aspect_data.append([self.element_id, max_edge, min_edge, aspect_ratio])

                        self.material_id, mDens = self.get_material_id(self.node_coords[self.nodes[0]][2])
                        
                        op.element('SSPbrick', self.element_id, *self.nodes, self.material_id, 0., 0., -mDens*9.81)
                        
                        element_data_row = [self.element_id] + self.nodes+ [mDens]
                        _ModelSoil_.ALL_NODES.append(element_data_row)
                        self.element_id += 1
                    
        self.SoilNodes = np.append(self.SoilNodes, self.element_id)
        print('soil element defined')
        element_aspect_data = np.array(element_aspect_data)
        # The element function originally returned only self.SoilNodes. This remains unchanged.
        return self.SoilNodes

    @staticmethod
    def save_all(filename='all_element_nodes.csv'):
        """Converts the collected node data to a 2D NumPy array and saves it to a file."""
        if not _ModelSoil_.ALL_NODES:
            print("Warning: No node data was collected.")
            return

        # Convert the list of node lists into a 2D NumPy array
        final_node_array = np.array(_ModelSoil_.ALL_NODES)
        
        # Save the array to a file (CSV format is easy to read)
        # fmt='%d' ensures the node tags (integers) are saved cleanly.
        # np.savetxt(filename, final_node_array, fmt='%d', delimiter=',',
                   # header='Node Tags for all Elements (8 nodes per row)', comments='# ')
        np.savetxt(filename, final_node_array, fmt='%d', delimiter=',')
        
