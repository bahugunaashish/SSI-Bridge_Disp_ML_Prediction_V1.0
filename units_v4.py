# -*- coding: utf-8 -*-   
"""
Created on Wed Sep 13 12:15:36 2023

@author: abahugu
"""
import numpy as np
from datetime import *
import openseespy.opensees as op




def merge_close_nodes(Nodes, tol=1e-8, debug=False):

    """
    Find nodes that are very close (interface nodes) and merge them in connected elements
    without deleting nodes globally.
    """

    # Step 1: collect nodes and coordinates
    node_tags = Nodes
    coords = {nd: np.array(op.nodeCoord(nd), dtype=float) for nd in node_tags}

    # Step 2: identify duplicates
    remap = {}  # duplicate node -> survivor
    tags = list(node_tags)

    for i in range(len(tags)):
        ndI = tags[i]
        if ndI in remap:
            continue
        Xi = coords[ndI]

        for j in range(i+1, len(tags)):
            ndJ = tags[j]
            if ndJ in remap:
                continue
            Xj = coords[ndJ]

            if np.linalg.norm(Xi - Xj) < tol:
                survivor = ndI  # choose first node as survivor
                remap[ndJ] = survivor
                if debug:
                    print(f"Node {ndJ} will be merged into {survivor}")

    if not remap:
        print("No interface duplicates found.")
        return

    print(f"Found {len(remap)} duplicate nodes at interfaces.")

    # Step 3: find elements connected to duplicate nodes and replace them
    for ele in op.getEleTags():
        conn = op.eleNodes(ele)
        new_conn = [remap.get(nd, nd) for nd in conn]

        if new_conn != conn:
            # Remove old element and recreate with new connectivity
            etype = op.eleType(ele)
            try:
                params = op.eleResponse(ele, 'parameters') or []
            except Exception:
                params = []
            params = [float(p) if isinstance(p, np.ndarray) else p for p in params]

            # Remove old element
            op.remove("element", ele)
            # Recreate element with updated node connectivity
            try:
                op.element(etype, ele, *new_conn, *params)
            except Exception:
                # fallback: recreate without params if needed
                op.element(etype, ele, *new_conn)

            if debug:
                print(f"Element {ele} updated nodes: {new_conn}")

    print("Interface nodes merged in elements successfully.")
    
    
def CreateEqDof2D (depth_elements, maxx_nodes, minx_nodes, depth_b,
                 maxy_nodes, miny_nodes, leftcorner, rightcorner):
    print('periodic boundary applied at the boundary elements in the direction of motion...')
    # if self.Forcedirection.lower()== 'x':
    #equaldof in x-direction  
    # for i in range (len(self.maxx_nodes)):
    for i in range (depth_elements+1):
        Side_NodesSet = np.array([])
        ndI = maxx_nodes[i][0]  # Extract node ID from maxx_nodes
        cord_ndI = op.nodeCoord(ndI)
        Side_NodesSet = np.append(Side_NodesSet, ndI)
        
        for j in range (len(maxx_nodes)):
            ndJ = maxx_nodes[j][0]  # Extract node ID from maxx_nodes
            cord_ndJ = op.nodeCoord(ndJ)
            # MODIFICATION: In 2D, cord_ndI[1] is Z. Dropped Y comparison.
            if cord_ndI[1] == cord_ndJ[1] and abs(cord_ndI[0]) == abs(cord_ndJ[0]):
                Side_NodesSet = np.append(Side_NodesSet, ndJ)
        for j in range (len(minx_nodes)):
            ndJ = minx_nodes[j][0]  # Extract node ID from maxx_nodes
            cord_ndJ = op.nodeCoord(ndJ)
            # MODIFICATION: In 2D, cord_ndI[1] is Z.
            if cord_ndI[1] == cord_ndJ[1] and abs(cord_ndI[0]) == abs(cord_ndJ[0]):
                Side_NodesSet = np.append(Side_NodesSet, ndJ)
        
        # print(self.Side_NodesSet)
        
        NodeSetCord = op.nodeCoord(int(Side_NodesSet[0])) # collecting the Z-coord of nodeset
          
        if NodeSetCord[1] != depth_b:       # check if z-coord is equal to depth of the soil box
            for Eqnode in Side_NodesSet:
                
                if Side_NodesSet[0] == Eqnode:
                    continue
                else:
                    # MODIFICATION: Changed to 1, 2 for 2D DOFs
                    op.equalDOF(int(Side_NodesSet[0]),int(Eqnode), 1, 2)
                    #print('equalDOF Node X:',int(Side_NodesSet[0]),int(Eqnode))
    
    # Y-direction loop removed as it is not applicable in 2D space

def CreateEqDof (depth_elements, maxx_nodes, minx_nodes, depth_b,
                 maxy_nodes, miny_nodes, leftcorner, rightcorner):
    print('periodic boundary applied at the boundary elements in the direction of motion...')
    # if self.Forcedirection.lower()== 'x':
    #equaldof in x-direction  
    # for i in range (len(self.maxx_nodes)):
    for i in range (depth_elements+1):
        Side_NodesSet = np.array([])
        ndI = maxx_nodes[i][0]  # Extract node ID from maxx_nodes
        cord_ndI = op.nodeCoord(ndI)
        Side_NodesSet = np.append(Side_NodesSet, ndI)
        
        for j in range (len(maxx_nodes)):
            ndJ = maxx_nodes[j][0]  # Extract node ID from maxx_nodes
            cord_ndJ = op.nodeCoord(ndJ)
            if cord_ndI[2] == cord_ndJ[2] and abs(cord_ndI[0]) == abs(cord_ndJ[0]) and cord_ndI[1] != cord_ndJ[1]:
                Side_NodesSet = np.append(Side_NodesSet, ndJ)
        for j in range (len(minx_nodes)):
            ndJ = minx_nodes[j][0]  # Extract node ID from maxx_nodes
            cord_ndJ = op.nodeCoord(ndJ)
            if cord_ndI[2] == cord_ndJ[2] and abs(cord_ndI[0]) == abs(cord_ndJ[0]):
                Side_NodesSet = np.append(Side_NodesSet, ndJ)
        
        # print(self.Side_NodesSet)
        
        NodeSetCord = op.nodeCoord(int(Side_NodesSet[0])) # collecting the Z-coord of nodeset
          
        if NodeSetCord[2] != depth_b:       # check if z-coord is equal to depth of the soil box
            for Eqnode in Side_NodesSet:
                
                if Side_NodesSet[0] == Eqnode:
                    continue
                else:
                    op.equalDOF(int(Side_NodesSet[0]),int(Eqnode), 1, 2, 3)
                    # print('equalDOF Node X:',int(Side_NodesSet[0]),int(Eqnode))
    
    # elif self.Forcedirection.lower()== 'y':    
    # equaldof in y-direction
    for i in range (depth_elements+1):
        Side_NodesSet = np.array([])
        ndI = maxy_nodes[i][0]  # Extract node ID from maxx_nodes
        cord_ndI = op.nodeCoord(int(ndI))
        Side_NodesSet = np.append(Side_NodesSet, ndI)
        
        for j in range (len(maxy_nodes)):
            ndJ = maxy_nodes[j][0]  # Extract node ID from maxy_nodes
            cord_ndJ = op.nodeCoord(int(ndJ))
            if cord_ndI[2] == cord_ndJ[2] and abs(cord_ndI[1]) == abs(cord_ndJ[1]) and cord_ndI[0] != cord_ndJ[0]:
                Side_NodesSet = np.append(Side_NodesSet, ndJ)
        # print('without min_y:',self.Side_NodesSet)
        for j in range (len(miny_nodes)):
            ndJ = miny_nodes[j][0]  # Extract node ID from maxy_nodes
            cord_ndJ = op.nodeCoord(int(ndJ))
            # if cord_ndI[2] == cord_ndJ[2] and abs(cord_ndI[1] - cord_ndJ[1]) == abs(bottom_left_corner[1]- top_right_corner[1]):
            if cord_ndI[2] == cord_ndJ[2] and abs(cord_ndI[1] - cord_ndJ[1]) == abs(leftcorner- rightcorner):
                # print('true')
                Side_NodesSet = np.append(Side_NodesSet, ndJ)
        
        # print('Side Y nodes:',self.Side_NodesSet)
        
        NodeSetCord = op.nodeCoord(int(Side_NodesSet[0])) # collecting the Z-coord of nodeset
          
        if NodeSetCord[2] != depth_b:       # check if z-coord is equal to depth of the soil box
            for Eqnode in Side_NodesSet:
                
                if Side_NodesSet[-1] == Eqnode:   # takes the first node or last node based on the requirement
                    continue
                else:
                    op.equalDOF(int(Side_NodesSet[-1]),int(Eqnode), 1, 2, 3)
                    # print('equalDOF Node Y:', int(Side_NodesSet[-1]),int(Eqnode))

def CollectMaxNodes2D(Nodes, leftCorner, rightCorner):
    
    # --- Tolerance Definition ---
    # Define a small value to treat numbers as equal if they are within this range
    TOLERANCE = 1e-5 
    
    # Helper function using np.isclose for robust comparison
    def is_close(a, b):
        return np.isclose(a, b, atol=TOLERANCE)
    # ----------------------------

    min_x_nodes = []
    max_x_nodes = []
    min_y_nodes = []
    max_y_nodes = []

    # NOTE: To make this code run, you need to ensure 'op' and 'inch' are defined 
    # and that 'op.nodeCoord' returns coordinates that are compatible with comparison.
    # I assume 'inch' is defined as a unit factor (e.g., 0.0254 meters).
    
    # Calculate the target coordinate values outside the loop
    # assuming 'inch' is a defined constant (e.g., in a structural analysis library)
    min_x_target = -1000 * inch 
    max_x_target = 1000 * inch

    for nnode in Nodes:
        # MODIFICATION: In 2D, ndcord is [x, z]
        ndcord = op.nodeCoord(nnode)
        
        # Comparison with Tolerance for x-coordinate
        # Check if ndcord[0] is close to the minimum x boundary
        if is_close(ndcord[0], min_x_target):  # at minimum x
            # The structure of the appended tuple is: (node_ID, x_coord, z_coord)
            # MODIFICATION: In 2D, z is index [1]
            min_x_nodes.append((nnode, ndcord[0], ndcord[1]))
        
        # Check if ndcord[0] is close to the maximum x boundary
        elif is_close(ndcord[0], max_x_target):  # at maximum x
            max_x_nodes.append((nnode, ndcord[0], ndcord[1]))
            
        # Comparison with Tolerance for y-coordinate - Omitted for 2D logic
        # but returning empty lists to maintain return signature
            
    # print(min_x_nodes, max_x_nodes, min_y_nodes, max_y_nodes)
    return min_x_nodes, max_x_nodes, min_y_nodes, max_y_nodes


# def CollectMaxNodes(Nodes, leftCorner, rightCorner):
    
#     min_x_nodes = []
#     max_x_nodes = []
#     min_y_nodes = []
#     max_y_nodes = []

#     for nnode in Nodes:
#         ndcord = op.nodeCoord(nnode)
#         if ndcord[0] == -1000*inch:  # at minimum x
#             min_x_nodes.append((nnode, ndcord[0], ndcord[2]))
        
#         elif ndcord[0] == 1000*inch:  # at maximum x
        
#             max_x_nodes.append((nnode, ndcord[0], ndcord[2]))
        
#         if ndcord[1] == leftCorner:  # at left corner y
#             min_y_nodes.append((nnode, ndcord[0], ndcord[2]))
#         elif ndcord[1] == rightCorner:  # at right corner y
#             max_y_nodes.append((nnode, ndcord[0], ndcord[2]))
#     # print(min_x_nodes, max_x_nodes, min_y_nodes,  max_y_nodes)   
#     return min_x_nodes, max_x_nodes, min_y_nodes,  max_y_nodes


def CollectMaxNodes(Nodes, leftCorner, rightCorner, min_x_target, max_x_target):
    
    # --- Tolerance Definition ---
    # Define a small value to treat numbers as equal if they are within this range
    TOLERANCE = 1e-5 
    
    # Helper function using np.isclose for robust comparison
    def is_close(a, b):
        return np.isclose(a, b, atol=TOLERANCE)
    # ----------------------------

    min_x_nodes = []
    max_x_nodes = []
    min_y_nodes = []
    max_y_nodes = []

    # NOTE: To make this code run, you need to ensure 'op' and 'inch' are defined 
    # and that 'op.nodeCoord' returns coordinates that are compatible with comparison.
    # I assume 'inch' is defined as a unit factor (e.g., 0.0254 meters).
    
    # Calculate the target coordinate values outside the loop
    # assuming 'inch' is a defined constant (e.g., in a structural analysis library)
    min_x_target = min_x_target # -324* inch # change based 
    max_x_target = max_x_target # 324 * inch

    for nnode in Nodes:
        # Assuming ndcord is a list or array of 3 coordinates [x, y, z]
        ndcord = op.nodeCoord(nnode)
        
        # Comparison with Tolerance for x-coordinate
        # Check if ndcord[0] is close to the minimum x boundary
        if is_close(ndcord[0], min_x_target):  # at minimum x
            # The structure of the appended tuple is: (node_ID, x_coord, z_coord)
            min_x_nodes.append((nnode, ndcord[0], ndcord[2]))
        
        # Check if ndcord[0] is close to the maximum x boundary
        elif is_close(ndcord[0], max_x_target):  # at maximum x
            max_x_nodes.append((nnode, ndcord[0], ndcord[2]))
        
        # Comparison with Tolerance for y-coordinate
        # Check if ndcord[1] is close to the leftCorner (min y)
        if is_close(ndcord[1], leftCorner):  # at left corner y
            # NOTE: This check is independent of the x-check (it uses 'if', not 'elif')
            min_y_nodes.append((nnode, ndcord[0], ndcord[2]))
            
        # Check if ndcord[1] is close to the rightCorner (max y)
        elif is_close(ndcord[1], rightCorner):  # at right corner y
            max_y_nodes.append((nnode, ndcord[0], ndcord[2]))
            
    # print(min_x_nodes, max_x_nodes, min_y_nodes, max_y_nodes)
    return min_x_nodes, max_x_nodes, min_y_nodes, max_y_nodes

# def tie_constrain(Nodes):
#     print('tieing nodes togather of partitions..')
#     for ndI in Nodes:
#         XYZI = np.array(op.nodeCoord(ndI))
#         for ndJ in op.getNodeTags():
#             if ndI <= ndJ:
#                 continue
#             XYZJ = np.array(op.nodeCoord(ndJ))
#             if np.linalg.norm(XYZJ-XYZI) < 1e-8:
#                 print(f'connecting {ndI} {XYZI} and {ndJ} {XYZJ}')
#                 op.equalDOF(ndI,ndJ,1,2,3)
                
from scipy.spatial import cKDTree

def tie_constrain(Nodes, tol=1e-8):
    coords = np.array([op.nodeCoord(nd) for nd in Nodes])
    tree = cKDTree(coords)

    pairs = tree.query_pairs(tol)  # very fast
    for i,j in pairs:
        ndI, ndJ = Nodes[i], Nodes[j]
        op.rigidLink('beam',ndI, ndJ)
        # op.equalDOF(ndI, ndJ, 1,2,3)
# from scipy.spatial import cKDTree

# def tie_constrain(Nodes):
#     print("Tying nodes together of partitions...")
    
#     # Get coordinates of all nodes in one go
#     # all_nodes = np.array(op.getNodeTags())  # all nodes in the model
#     coords = np.array([op.nodeCoord(nd) for nd in Nodes])

#     # Build a KDTree for fast spatial search
#     tree = cKDTree(coords)

#     # Search for all pairs within tolerance
#     tol = 1e-8
#     pairs = tree.query_pairs(r=tol)

#     # Tie only the nodes in the provided Nodes list
#     nodes_set = set(Nodes)
#     for i, j in pairs:
#         ndI = Nodes[i]
#         ndJ = Nodes[j]
#         if ndI in nodes_set and ndJ in nodes_set:
#             op.equalDOF(ndI, ndJ, 1, 2, 3)


def program_info(print_flag=0):
    """
    -------------------
    PROGRAM INFORMATION
    -------------------
    The method saves the basic software information as string

    Parameters
    ----------
    print_flag: int, optional (The default is 0)
        If 1, prints the output on the screen

    Returns
    -------
    text: str
        String describes the software

    """
    today = date.today()
    d1 = today.strftime("%d/%m/%Y")
    # Sample program info to print
    text = ("|-----------------------------------------------------------------------|\n" +
            "|    BridgeAnalyze: Program to perform dynamic analysis                 |\n" +
            "|    and Seismic Risk assessment of bridges                             |\n" +
            "|    Version: 1.0                                                       |\n" +
            "|    Date: {d1}                                                         |\n" +
            "|                                                                       |\n" +
            "|    started on 23/08/2023                                              |\n" +
            "|    Author: Ashish Bahuguna                                            |\n" +
            "|    Affiliation: Clemson University SC                                 |\n" +
            "|    Glenn Department of Civil Engineering PhD Candidate                |\n" +
            "|                                                                       |\n" +
            "|-----------------------------------------------------------------------|\n")

    if print_flag == 1:
        print(text)

    return text

# define units_v4 of the modelling
def get_units(print_flag=0):
    """
    -------------------
    DEFINITION OF UNITS
    -------------------
    This method defines units_v4 and their conversions to the basic units_v4: m, kN, sec

    Parameters
    ----------
    print_flag; int
        Flag to print allowed units_v4 on the screen

    Returns
    -------
    m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi
        Value of the all allowed units_v4 in terms of basic units_v4 (m, kN, sec)
    """

    # Basic Units
    m = 1.0
    N = 1.0
    sec = 1.0

    # Length
    mm = m / 1000.0
    cm = m / 100.0
    inch = 25.4 * mm
    ft = 12.0 * inch

    # Force
    kN = N * 1000.0
    kip = N * 4448.2216

    # Mass (tonnes)
    tonne = kN * sec ** 2 / m
    kg = N * sec ** 2 / m

    # Stress (kN/m2 or kPa)
    Pa = N / (m ** 2)
    kPa = Pa * 1.0e3
    MPa = Pa * 1.0e6
    GPa = Pa * 1.0e9
    ksi = 6.8947573 * MPa
    psi = 1e-3 * ksi
    
    
    
    # Basic Units
    # m = 1.0
    # kN = 1.0
    # sec = 1.0

    # # Length
    # mm = m / 1000.0
    # cm = m / 100.0
    # inch = 25.4 * mm
    # ft = 12.0 * inch

    # # Force
    # N = kN / 1000.0
    # kip = kN * 4.448221615

    # # Mass (tonnes)
    # tonne = kN * sec ** 2 / m
    # kg = N * sec ** 2 / m

    # # Stress (kN/m2 or kPa)
    # Pa = N / (m ** 2)
    # kPa = Pa * 1.0e3
    # MPa = Pa * 1.0e6
    # GPa = Pa * 1.0e9
    # ksi = 6.8947573 * MPa
    # psi = 1e-3 * ksi

    if print_flag == 1:
        print('Values are returned for the following units_v4:')
        print('m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi')
    return m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi


def find_nearest(array, value):
    """
    Details
    -------
    Finds the row where the nearest value is located

    Parameters
    ----------
    array : numpy.ndarray
        array in which the row of the closest value is sought
    value: float, int
        value used to find closest value in array

    Returns
    -------
    idx: int, list, tuple
        index of the closest value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_run_time(start_time, print_flag=0):
    """
    -----------
    PASSED TIME
    -----------
    The method computes the total amount of time passed since start_time

    Parameters
    ----------
    print_flag: int, optional (The default is 0)
        If 1, prints the output on the screen
    start_time: float
        Reference time to calculate passed time until now. time.perf_counter()

    Returns
    -------
    text: str
        String describing total amount of time passed since start_time in hr,min,sec

    """

    # Procedure to obtained elapsed time in Hr, Min, and Sec
    finish_time = time.perf_counter()
    time_seconds = finish_time - start_time
    time_minutes = int(time_seconds / 60)
    time_hours = int(time_seconds / 3600)
    time_minutes = int(time_minutes - time_hours * 60)
    time_seconds = time_seconds - time_minutes * 60 - time_hours * 3600
    text = "Run time: %d hours: %d minutes: %.2f seconds" % (time_hours, time_minutes, time_seconds)

    if print_flag == 1:
        print(text)

    return text


def get_distance(coord1, coord2):
    """
    ---------------------------
    DISTANCE BETWEEN TWO POINTS
    ---------------------------

    Parameters
    ----------
    coord1: list, numpy.ndarray
        Coordinates of the first point
    coord2: list, numpy.ndarray
        Coordinates of the second point

    Returns
    -------
    distance: float
        The distance between two points on 3-D cartesian system

    """

    dist = ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2 + (coord1[2] - coord2[2]) ** 2) ** 0.5
    return dist


def get_current_time():
    """
    ------------
    CURRENT TIME
    ------------

    Parameters
    ----------

    Returns
    -------
    time.perf_counter()
        The point where the time starts in computer

    """

    return time.perf_counter()


def create_dir(output_path):
    """
    ----------------
    CREATE DIRECTORY
    ----------------
    The method creates an empty directory. It will delete it if the directory exists, and recreate it.

    Parameters
    ----------
    output_path : str
        output directory to create.

    Returns
    -------

    """

    def handle_remove_readonly(func, path, exc):
        exc_value = exc[1]
        if func in (os.rmdir, os.remove) and exc_value.errno == errno.EACCES:
            os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777
            func(path)
        else:
            raise

    if os.path.exists(output_path):
        shutil.rmtree(output_path, ignore_errors=False, onerror=handle_remove_readonly)
    os.makedirs(output_path)

m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi = get_units()



