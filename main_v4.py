# -*- coding: utf-8 -*-
"""Entry point for building and analyzing the OpenSees model."""
import sys
import csv
import json
from pathlib import Path
from config_inputs_v4 import *
import config_inputs_v4 as config
from soil_model_v4 import _ModelSoil_
from structure_model_v4 import Define_str
from analysis_v4 import ModalAnalysis
from soil_profile_reader_v4 import read_soil_profile
from simulation_matrix_reader_v4 import write_case_input_file
from structural_properties_v4 import calculate_structural_loads
import math


def _write_ml_model_sidecars(directory, op, node_tags, element_tags,
                             response_node_tags, response_element_tags,
                             simulation_no=None):
    """Write stable ID maps and graph connectivity next to each case's TXT files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'node_coordinates.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['node_id', 'x', 'y', 'z'])
        for tag in node_tags:
            coords = list(op.nodeCoord(int(tag)))
            writer.writerow([int(tag), *coords[:3]])
    with (directory / 'element_connectivity.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['element_id', 'element_type', 'node_ids...'])
        for tag in element_tags:
            try:
                ele_type = str(op.eleType(int(tag)))
            except Exception:
                ele_type = 'unknown'
            writer.writerow([int(tag), ele_type, *[int(n) for n in op.eleNodes(int(tag))]])
    mapping = {
        'simulation_no': simulation_no,
        'all_node_ids': [int(x) for x in node_tags],
        'all_element_ids': [int(x) for x in element_tags],
        'response_node_ids_in_recorder_column_order': [int(x) for x in response_node_tags],
        'response_element_ids_in_recorder_column_order': [int(x) for x in response_element_tags],
        'node_disp_components': ['Ux', 'Uy', 'Uz', 'Rx', 'Ry', 'Rz'],
        'node_accel_components': ['Ax', 'Ay', 'Az'],
        'recorder_node_sets': {
            'structure_displacement': [int(x) for x in response_node_tags],
            'structure_element_force': [int(x) for x in response_element_tags],
            'capbeam_acceleration': [7003, 7006, 5001, 3001],
            'pilebottom_acceleration': [1001],
            'capbeam_displacement': [7003, 7006],
        },
        'element_force_channel_definition': 'OpenSees element-type-dependent response; retained as raw flattened values',
        'recorder_specs': [
            {'key': 'structure_displacement', 'kind': 'Node', 'response': 'disp', 'dofs': [1, 2, 3, 4, 5, 6], 'node_ids': [int(x) for x in response_node_tags], 'dt_s': 0.005},
            {'key': 'structure_element_force', 'kind': 'Element', 'response': 'force', 'element_ids': [int(x) for x in response_element_tags], 'dt_s': 0.005},
            {'key': 'capbeam_acceleration', 'kind': 'Node', 'response': 'accel', 'dofs': [1, 2, 3], 'node_ids': [7003, 7006, 5001, 3001], 'dt_s': 0.005},
            {'key': 'pilebottom_acceleration', 'kind': 'Node', 'response': 'accel', 'dofs': [1, 2, 3], 'node_ids': [1001], 'dt_s': 0.01},
            {'key': 'capbeam_displacement', 'kind': 'Node', 'response': 'disp', 'dofs': [1, 2, 3], 'node_ids': [7003, 7006], 'dt_s': 0.005},
        ],
        'node_disp_dt_s': 0.005,
        'element_force_dt_s': 0.005,
        'files': {
            'node_coordinates': 'node_coordinates.csv',
            'element_connectivity': 'element_connectivity.csv',
        },
    }
    (directory / 'recorder_manifest.json').write_text(
        json.dumps(mapping, indent=2), encoding='utf-8'
    )


def main(soil_profile_file=None, case=None, mpi_mode=False, plot_results=True, feature_only=False):
    #%% Configure logging to save to a file
    logging.basicConfig(filename='output.log', level=logging.INFO)

    #%% Directory Setup for Figures and Results
    path = os.getcwd() #get the current directory path
    Model_dir = os.path.join(path, 'ModelFigures/')

    startT = tt.time()
    #if not os.path.isdir(Model_dir):

    try:    
      os.makedirs(Model_dir)
    except:
      print('skipping')
    
    plt.close('all')
    op.wipe()
    op.wipeAnalysis()
    op.model('basic', '-ndm', 3, '-ndf', 6)

    if case is not None:
        config.configure_case(case)
        config_values = {name: value for name, value in vars(config).items() if not name.startswith('_')}
        globals().update(config_values)
    else:
        config.configure_default_sections()
        config_values = {name: value for name, value in vars(config).items() if not name.startswith('_')}
        globals().update(config_values)

    # The dependent modules use the original ``from config_inputs import *``
    # pattern. Refresh their module namespaces after the selected case has
    # created its materials and sections.
    for module_name in ('structure_model_v4', 'soil_model_v4', 'analysis_v4'):
        module = sys.modules.get(module_name)
        if module is not None:
            module.__dict__.update(config_values)

    pid =op.getPID()
    nps = op.getNP() 
    print(pid, nps)
    if case is not None:
        write_case_input_file(case, os.path.join(os.getcwd(), 'input_params.txt'), mpi_rank=pid)

    #%%
    #%% Soil model generation
    SoilLayers = []
    counter = 1
    LayerEndNodeid = 50001
    LayerEndEleid = 50001
    fixed_distance1 = 36*ft  
    fixed_length = 24*inch
    fixed_width = 24*inch
    ElementFlag = 1 # 1 for fullGauss, 0 for single point

    print('creating inerior bent soil domain...')

    for ii in range(0):
        #soil = _ModelSoil_(width[ii], length[ii], width_elements, hole_centers, PileDia, hole_depth, outer_radius, LayerEndNodeid, LayerEndEleid, counter)
        if ii<4:
            soil = _ModelSoil_(width[ii], length[ii], fixed_distance1, fixed_length, fixed_width, width_elements, hole_centers, PileDia, hole_depth, outer_radius, LayerEndNodeid, LayerEndEleid, counter, ElementFlag)
        else:
            soil = _ModelSoil_(width[ii], length[ii], fixed_distance1, fixed_length, fixed_width, width_elements, hole_centers, PileDia, hole_depth, outer_radius, LayerEndNodeid, LayerEndEleid, counter, ElementFlag)
   
        SoilLayers.append(soil)
        LayerEndNodeid =  op.getNodeTags()[-1]+1   
        LayerEndEleid = op.getEleTags()[-1]+1  
        counter = 1+ counter
    _ModelSoil_.save_all()    # save the collected node for each soil box
    AllSoilNodes = op.getNodeTags()  
    AllSoilElem = op.getEleTags()
    ReactionNode = np.array([], dtype = int)
    # for node in AllSoilNodes:
    #   nodcord = op.nodeCoord(node)
    #   if nodcord[2]==0:
    #     ReactionNode = np.append(ReactionNode, node)
    # ReactionNode = ReactionNode.tolist()

    #%% COnstraint in soil nodes
    # tie_constrain(AllSoilNodes)
    # merge_close_nodes(AllSoilNodes)

    # min_x_nodes, max_x_nodes, min_y_nodes,  max_y_nodes = CollectMaxNodes(AllSoilNodes, maxY, minY)
    # print( max_y_nodes)
    # CreateEqDof (depth_elements, max_x_nodes, min_x_nodes, 0., max_y_nodes, min_y_nodes, maxY, minY)
    #%%
    gamma = 0.5
    beta  = 0.25
    #%% Gravity analysis soil
    #=======================================================
    # Check if soil is included 
    #======================================================

    # Feature-only runs build the model but must never execute a solver step.
    if (not feature_only) and len(AllSoilNodes) != 0:
       print('soil is included..')


       op.constraints ('Transformation')
       op.constraints ('Penalty', 1.e18, 1.e18)
       op.test('NormDispIncr', 1e-5, 150, 2) 
       #op.algorithm('KrylovNewton')
       alogr = ['KrylovNewton', 'Newton', 'NewtonLineSearch', 'ModifiedNewton', 'SecantNewton', 'RaphsonNewton'] 
       op.algorithm(alogr[0])
       op.numberer('ParallelRCM')
       op.system('Mumps')
  
      #integrator of transisent analysis
      #                    alpha, gamma, beta
       #op.integrator('HHT', 0.0, 1.0,  1.5) # HHT-aplha method, alpha =0.67 for higher numeical damping
       op.integrator('Newmark', gamma, beta)
       op.integrator('LoadControl', 1e-1)
   
       op.analysis('Transient')
       op.analysis('Static')
       op.InitialStateAnalysis('on')
       ok = op.analyze(5)
       op.InitialStateAnalysis('off')
  
       sok = op.analyze(5) 
  
  
    if not feature_only:
        op.loadConst ('-time', 0.0)
    op.wipeAnalysis()

    op.remove('recorders')
    print('recorder removed')

    startT = tt.time()
    
    g=9.81

    op.barrier()
    #%% Add structure   
    soil_profile = (
        case.get('soil_layers') if case is not None else
        (read_soil_profile(soil_profile_file) if soil_profile_file else None)
    )
    load_data = calculate_structural_loads(
        D=DSec,
        ColumnLength=ColumnLength,
        ShaftLength=ShaftLength,
        SocketLength=SocketLength,
        HPileLength=HPileLength,
        HArea=HArea,
        Width_CapBeam=Width_CapBeam,
        Depth_CapBeamSec=Depth_CapBeamSec,
        CapBeamLength=CapBeamLength,
        WidthEcapBeam=WidthEcapBeam,
        DepthEcapBeamSec=DepthEcapBeamSec,
        EcapBeamLength=EcapBeamLength,
        DiaPhrgmArea=DiaPhrgmArea,
        DeckArea=DeckArea,
        ParapetMass=ParapetMass,
        FWS=FWS,
        SIPF=SIPF,
        AdditionalLoad=AdditionalLoad,
        GamaConc=GamaConc,
        GamaSteel=GamaSteel,
        g=g,
        kN=kN,
        m=m,
        mm=mm,
        ft=ft,
    )
    geometry = None
    if case is not None:
        column_top_z = 104 * ft
        column_bottom_z = column_top_z - ColumnLength
        shaft_top_z = column_bottom_z - 2 * ft
        shaft_bottom_z = shaft_top_z - ShaftLength
        socket_top_z = shaft_bottom_z - 2 * ft
        socket_bottom_z = socket_top_z - SocketLength
        geometry = {
            'column_top_z': column_top_z,
            'column_bottom_z': column_bottom_z,
            'shaft_top_z': shaft_top_z,
            'shaft_bottom_z': shaft_bottom_z,
            'socket_top_z': socket_top_z,
            'socket_bottom_z': socket_bottom_z,
            'pile_top_z': column_top_z,
            'pile_bottom_z': column_top_z - HPileLength,
            'ground_surface_z': column_top_z - 2 * ft,
            'pile_diameter': DSec,
        }
    bridge = Define_str(SoilLayers, AllSoilNodes, SocketLength, SocketElementType, xlocation, SpanLength, SecTags, ShaftLength, ShaftElementType, ColumnLength, ColumnElementType, bearingLocation, CapBeamElementType, Depth_CapBeamSec, HPileLength, pileLocation, soil_profile=soil_profile, load_data=load_data, geometry=geometry)

    if case is not None:
        bottom_assignments = getattr(bridge, 'bottom_node_soil_assignments', [])
        with open('input_params.txt', 'a', encoding='utf-8') as input_stream:
            input_stream.write('\nBottom node soil-layer assignments: '
                               f'{len(bottom_assignments)}\n')
            for assignment in bottom_assignments:
                input_stream.write(
                    "  Node {node_tag}: model_z={model_z:.9g}, "
                    "depth_below_ground={depth_below_ground:.9g}, "
                    "profile_z={profile_z:.9g}, layer_id={layer_id}, "
                    "uscs={uscs}, source={soil_source}\n".format(**assignment)
                )

    SocketEleTags, ShaftEleTags, ColumnEleTags, BottomNodesSo = bridge.getPileElem()
        
    ShaftColumnNodeTags = bridge.getPileNodes()

    allelement = op.getEleTags()
    AllNodes = op.getNodeTags()
    StructElement = list(set(allelement)^set (AllSoilElem))
    StructNode = list(set(AllNodes)^set (AllSoilNodes))
    _write_ml_model_sidecars(
        Path.cwd(), op, AllNodes, allelement, StructNode, StructElement,
        simulation_no=case.get('simulation_no') if case is not None else None,
    )

    if feature_only:
        # This return is deliberately before all analysis configuration and
        # dynamic loading below. Model-building routines above contain only
        # OpenSees model-definition commands (no analyze/eigen calls).
        from ssi_ml_surrogate_v1.opensees_adapter import graph_from_opensees
        graph = graph_from_opensees(
            op, case, node_tags=AllNodes, element_tags=allelement,
            pile_node_tags=bridge.getPileNodes(),
            bottom_node_tags=BottomNodesSo,
            response_node_tags=StructNode,
            response_element_tags=StructElement,
            dt=float(case.get('ground_motion_dt_s', 0.005)),
        )
        return graph

    #
    print('Modelling step complete...\nTotal no. of nodes', len(op.getNodeTags())) 
    #
    print('Total number of nodes',len(op.getNodeTags()))
    #%%

    Soil_element_data = []
    for Ele in op.getEleTags():
        Ele_Nodes = op.eleNodes(Ele)
        Soil_element_data.append([Ele] + Ele_Nodes)

    # Find maximum row length (some elements may have optional density/property)
    max_len = max(len(row) for row in Soil_element_data)

    # Convert to NumPy array, padding shorter rows with np.nan
    arr = np.full((len(Soil_element_data), max_len), np.nan, dtype=float)
    for i, row in enumerate(Soil_element_data):
        arr[i, :len(row)] = row

    # Save using numpy.savetxt (fast, no brackets)

    Soil_node_data = []
    for nodes in op.getNodeTags():
        nodecord = op.nodeCoord(nodes)
        row_node = [nodes] + nodecord             # node ID + coords
        Soil_node_data.append(row_node)
    
    Soil_Node_array = np.array(Soil_node_data, dtype=object)
    # Save as CSV
    # np.savetxt("SoilElementNodes.csv", Soil_element_array, fmt='%s', delimiter=',')
    np.savetxt("SoilElementNodes.csv", arr, delimiter=",", fmt='%g')
    np.savetxt("SoilNodesCoords.csv", Soil_Node_array, fmt='%s', delimiter=',')
    # print("Element-node connectivity saved to ElementNodes.csv using NumPy")
    #%%

    #%% loadin
    op.constraints ('Transformation')
    # op.constraints ('Penalty', 1.e18, 1.e18)
    op.test('NormDispIncr', 1e-5, 50, 2) 
    #op.algorithm('KrylovNewton')
    alogr = ['KrylovNewton', 'Newton', 'NewtonLineSearch', 'ModifiedNewton', 'SecantNewton', 'RaphsonNewton'] 
    op.algorithm(alogr[0])
    if mpi_mode:
        op.numberer('ParallelRCM')
        op.system('Mumps')
    else:
        op.numberer('RCM')
        op.system('SuperLU')

    #integrator of transisent analysis
    #                    alpha, gamma, beta
    #op.integrator('HHT', 0.0, 1.0,  1.5) # HHT-aplha method, alpha =0.67 for higher numeical damping
    op.integrator('Newmark', gamma, beta)
    op.integrator('LoadControl', 1e-3)

    op.analysis('Transient')
    # op.analysis('Static')
    # sok = op.analyze(10) 



    Nmodes = 10  # Number modes to be calculated
    ExictationDir = [1]  # 0: X; 1: Y; 2: Z   # mass participation factor calculation direction

    model = 'ModalAnalysis-p-y curves'
    # Phi = ModalAnalysis(Nmodes, ExictationDir, model)

    #%% pushover Analysis
  
    # op.barrier()
    sok = 0 if case is not None else 1
    #_PushAna([7003, 7006, 7009], 1, 1*m, 1e-2, 30, ReactionNode, ZerolengthNBars, SocketEleTags, ShaftEleTags, ColumnEleTags, ShaftColumnNodeTags)
   
    #%% Dynmaic analysis
    #-----------------------------------------------------------------------------------------------------------------------------------------------------
    if sok==0:
        dof =  2    # direction of motion
        EQName = case['EQ'] if case is not None else 'EQ18'
        target_pga = case['target_pga'] if case is not None else 0.4
        EQPath = str(case['EQPath']) if case is not None else 'E:/I95-results/' + EQName + '.out'
        aa = np.loadtxt(EQPath)
        numsteps = len(aa)
        R_angle_1 = (
            float(case['incident_angle_deg'])
            if case is not None else 300.0
        )
        R_ang = f'_{R_angle_1:g}deg_'
        # 1. Find the current max PGA of the raw record
        current_pga = max(abs(aa))
        print(f'Original EQ PGA: {current_pga:.3f} g')
    
        # 2. Calculate the Scale Factor (fact)
        fact = target_pga / current_pga
        print(f'Target PGA: {target_pga:.3f} g | Calculated Scale Factor: {fact:.4f}')
    
        try:
           # 3. Define directory using the TARGET PGA so your output folders are organized
          dirname = f"I95_CompleteBridge_3D_PY-EP_{EQName}_{target_pga:.4f}g_{R_ang}deg"
          os.makedirs(dirname)
        except:
          print('skipping')   
    #-----------------------------------------------------------------------------------------------------------------------------------------------------
    #if sok==0:
        print('gravity analysis complete...')
        op.wipeAnalysis()
        op.remove('recorders')
        print('recorder removed')
        op.loadConst ('-time', 0.0)
        stepp = float(case.get('ground_motion_dt_s', 0.005))
        pga = f'_{target_pga:.3f}g' 
        
        
    
        # Text-recorder paths and exact column maps for ML conversion.
        accel_capbeam_file = f"{dirname}/I95_CompleteBridge_{dof}_Dir_{EQName}_Accel_XCAPBEAM.txt"
        accel_pilebottom_file = f"{dirname}/I95_CompleteBridge_{dof}_Dir_{EQName}_Accel_XPILEBOTTOM.txt"
        force_file = f"{dirname}/I95_CompleteBridge_{dof}_Dir_{EQName}Structure_Force_{R_ang}{pga}.txt"
        disp_file = f"{dirname}/I95_CompleteBridge_{dof}_Dir_{EQName}Structure_Disp_{R_ang}{pga}.txt"
        capbeam_disp_file = f"{dirname}/I95_CompleteBridge_{dof}_Dir_{EQName}_Disp_XCAPBEAM.txt"
        op.recorder('Node', '-file', accel_capbeam_file, '-time', '-dT', stepp, '-node', *[7003, 7006, 5001, 3001], '-dof', 1, 2, 3, 'accel')
        op.recorder('Node', '-file', accel_pilebottom_file, '-time', '-dT', 0.01, '-node', 1001, '-dof', 1, 2, 3, 'accel')
        op.recorder('Element', '-file', force_file, '-time', '-dT', stepp, '-ele', *StructElement, 'force')
        op.recorder('Node', '-file', disp_file, '-time', '-dT', stepp, '-node', *StructNode, '-dof', 1, 2, 3, 4, 5, 6, 'disp')
        op.recorder('Node', '-file', capbeam_disp_file, '-time', '-dT', stepp, '-node', *[7003, 7006], '-dof', 1, 2, 3, 'disp')
        manifest_path = Path('recorder_manifest.json')
        recorder_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        recorder_manifest['files'].update({
            'structure_displacement': disp_file,
            'structure_element_force': force_file,
            'capbeam_acceleration': accel_capbeam_file,
            'pilebottom_acceleration': accel_pilebottom_file,
            'capbeam_displacement': capbeam_disp_file,
        })
        recorder_manifest['time_step_s'] = {
            'structure_displacement': stepp, 'structure_element_force': stepp,
            'capbeam_acceleration': stepp, 'pilebottom_acceleration': 0.01,
            'capbeam_displacement': stepp,
        }
        manifest_path.write_text(json.dumps(recorder_manifest, indent=2), encoding='utf-8')

        filename = 'IntBent_'+dirname+'.pvd'
        path_pvd = os.path.join(dirname, filename)
        try:
          os.makedirs(path_pvd)
    
        except FileExistsError:
    
          print(f"Directory already exists, skipping creation.")
        op.recorder('PVD', dirname+'/'+filename, 'disp', '-dT', stepp)

        # damping ratio
        pi = 3.1415
        damp =    0.05
        # lower frequency
        #omega1 = 2*pi/0.4
        omega1 = 10
        # upper frequency
        #omega2 = 2*pi/0.04
        omega2 = 0.1
        # damping coefficients
        a0 = 2*damp*omega1*omega2/(omega1 + omega2)
        a1 = 2*damp/(omega1 + omega2)
    
        print('applying rayleigh damping...')
        op.rayleigh (a0, a1, 0.0,  0.0)
        # op.region(1, '-ele', *StructElement, '-rayleigh', a0, a1, 0.0, 0.0)
        # op.rayleigh (x[0], x[1], 0.0, 0.0)
    
       # intial time step and and number of steps
        dtOrig = float(case.get('ground_motion_dt_s', 0.005))
        fact =  fact
        R_angle_rad = math.radians(R_angle_1)
        factr_x = math.sin(R_angle_rad) * fact
        factr_y = math.cos(R_angle_rad) * fact
        numSteps = numsteps#5998  numSteps
        g=9.81*m/sec**2
    
        print('Dynamic Analysis running ....')
        op.loadConst ('-time', 0.0)
        op.timeSeries('Path', 8, '-filePath', EQPath, '-dt', dtOrig, '-factor', 9.81*factr_x)
        op.timeSeries('Path', 9, '-filePath', EQPath, '-dt', dtOrig, '-factor', 9.81*factr_y)

        op.pattern('UniformExcitation', 11, 1, '-accel', 8)
        op.pattern('UniformExcitation', 12, 2, '-accel', 9)

        # ---------------------------------------------------------
        # Dynamic Base Excitation Setup
        # ---------------------------------------------------------
        # op.timeSeries('Path', 8, '-filePath', EQPath, '-dt', dtOrig, '-factor', 9.81*fact)

        # Create the MultipleSupport Pattern
        # op.pattern('MultipleSupport', 11)
        # op.groundMotion(101, 'Plain', '-accel', 8)

        print('Applying base excitation exclusively to the recorded pile tip soil nodes...')

        # 3. Get the exact base soil nodes from your class
        # base_soil_nodes = bridge.getBaseSoilNodes()

        # Loop through and apply the earthquake to each one
        # for snode in base_soil_nodes:
            # op.imposedMotion(int(snode), dof, 101)
    
        # z_coord = op.nodeCoord(int(snode))[2]
        # print(f" -> Earthquake successfully applied to Base Soil Node: {snode} (Depth: {z_coord:.2f})")

        #op.pattern('Plain', 10, 3)
        #op.load(50001,  1.0, 0.0, 0.0, 0.0)
        # op.constraints ('Penalty', 1.e10, 1.e10)
        op.constraints ('Transformation')
        op.test('NormDispIncr', 1.0e-3, 10, 2)
        op.algorithm('KrylovNewton')
        #op.numberer('ParallelPlain') #ParallelRCM
        op.numberer('RCM')
        op.system('SuperLU')
        #op.system('Mumps',  '-ICNTL14', 200.0)
        alpha = 0.67
        gamma=1.5-alpha 
        beta=(2-alpha)**2/4
        op.integrator('HHT',0.6, gamma, beta) # HHT-aplha method
        # op.integrator('Newmark', gamma, beta) #Newmark mathod
        op.analysis('Transient')
       
        # Automatically calculate the total simulation time
        tFinal = numSteps * dtOrig
        print(f"Total calculated simulation time: {tFinal} seconds")
    
        # Initialize the working time step
        dt = dtOrig
    
        # Loop until the current time reaches the calculated final time
        # (Subtracting a tiny tolerance prevents floating-point rounding errors at the final step)
        while op.getTime() < (tFinal*0.8 - 1.0e-6):
        
            # 1. Attempt the analysis step at the current dt
            ok = op.analyze(1, dt)
        
            # If it succeeds, explicitly fall back to the original time step
            if ok == 0:
                dt = dtOrig
            
            # 2. FIRST NESTED LEVEL: If it fails, cut step in half
            if ok != 0:
                print(f"Failed at time {op.getTime():.5f}. Reducing dt to half...")
                dt = dtOrig / 2.0
                ok = op.analyze(1, dt)
            
                # If it succeeds at half step, explicitly fall back to original
                if ok == 0:
                    dt = dtOrig
                
                # 3. SECOND NESTED LEVEL: If it fails again, cut step to a quarter
                if ok != 0:
                    print(f"Failed again. Reducing dt to a quarter...")
                    dt = dtOrig / 4.0
                    ok = op.analyze(1, dt)
                
                    # If it succeeds at quarter step, explicitly fall back to original
                    if ok == 0:
                        dt = dtOrig
                    
                    # 4. THIRD NESTED LEVEL: If it fails a third time, cut step to an eighth
                    if ok != 0:
                        print(f"Failed again. Reducing dt to an eighth...")
                        dt = dtOrig / 8.0
                        ok = op.analyze(1, dt)
                    
                        # If it succeeds at eighth step, explicitly fall back to original
                        if ok == 0:
                            dt = dtOrig
                        
                        # If it STILL fails here, break the loop
                        if ok != 0:
                            print("Analysis failed completely at dt/8. Stopping.")
                            break 
    
        if ok == 0:
            print("Analysis completed successfully!")
               
    else:
        print('Dynamic Analysis bypass...')

    endT = tt.time()
    print("Analysis execution time: ",(endT-startT))
    # # vfo.plot_model(model="3DSoil-Ple",show_nodes='yes',show_nodetags='yes')     
    if plot_results:
        fig_wi_he = 100., 100.
        opsv.plot_model(node_labels=0, element_labels=0, axis_off=1, gauss_points=False, fig_wi_he=fig_wi_he, local_axes=True,
                       fmt_model={'color': 'black', 'linestyle': 'solid', 'linewidth': 0.8, 'marker': '.', 'markersize': 6}, node_supports=True)
        # #plt.savefig(Model_dir+'Bridge_pushModel.png')
        plt.show()
        opsv.plot_defo(sfac=5e2, fmt_defo={'color': 'blue', 'linestyle': 'solid', 'linewidth': 1, 'marker': '', 'markersize': 2}, 
                       fmt_undefo={'color': 'red', 'linestyle': (0, (1, 1)),'linewidth': 1, 'marker': '.', 'markersize': 0.5}, 
                       fmt_nodes={'color': 'red', 'linestyle': 'None', 'linewidth': 1.5, 'marker': 's', 'markersize': 0.5}, 
                       fig_wi_he= fig_wi_he, node_supports=False, ax=False, )
        # plt.savefig(Model_dir+'Bridge_push12.png')
        ops.vis.plotly.plot_model(odb_tag=None, show_node_numbering=True, show_ele_numbering=False, 
                                    show_ele_hover=True, style='surface', color=None, show_bc=True, bc_scale=1.0, show_link=True, show_mp_constraint=True, 
                                    show_constraint_dofs=True, show_nodal_loads=False, 
                                    show_ele_loads=False, load_scale=1.0, show_local_axes=True, local_axes_scale=1.0, show_outline=True)
        plt.show()
    # op.partition()

    op.wipeAnalysis()
    op.wipe()

    # 
    # vfo.plot_modeshape(modenumber=1, scale=3000, model=model)



if __name__ == "__main__":
    main()
