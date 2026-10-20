# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 18:45:04 2023

@author: abahugu
"""
import openseespy.opensees as op
from units_v4 import get_units

m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi = get_units()
# Ubig = 1.e10; 			# a really large number
# Usmall = 1./Ubig 		# a really small number
# m, kN, sec, mm, cm, inch, ft, N, kip, tonne, kg, Pa, kPa, MPa, GPa, ksi, psi, lbf, pcf, psf, inch2, inch4, g, Ubig, Usmall= get_units2()

def getMat_v4(fc_value=None, fy_value=None):
    """Create OpenSees materials, optionally using simulation fc and fy.

    When omitted, the original hard-coded material values are retained.
    fc_value and fy_value are expected in the model base units_v4.
    """
    # MATERIAL parameters -------------------------------------------------------------------
    IDconcCore  = 1 				# material ID tag -- confined core concrete
    IDconcCover = 2 				# material ID tag -- unconfined cover concrete
    IDreinf     = 3 				# material ID tag -- reinforcement
    IDreinf1    = 4 				# material ID tag -- steel cover
    # pile cap connection material id
    cappile     = 30

    # nominal concrete compressive strength
    fc_reference = -5.1996*ksi
    fc = fc_reference if fc_value is None else -abs(fc_value)
    Ec = 57*ksi*(-fc/psi)**0.5	# Concrete Elastic Modulus
    # Ec = 5000*(-fc/MPa)**0.5*MPa 
    # confined concrete
    # Kfc = 1.3;			# ratio of confined to unconfined concrete strength
    fc1C_reference = -43.7*MPa
    fc2C_reference = 35.12*MPa
    fc_scale = fc / fc_reference
    fc1C = fc1C_reference * fc_scale  # preserve original confined/concrete ratio
    eps1C = -0.0044	     # strain at maximum stress 
    fc2C  = fc2C_reference * fc_scale  # preserve original ultimate-stress ratio
    eps2C = -0.0119		 # strain at ultimate stress 

    # unconfined concrete
    fc1U =	fc			# UNCONFINED concrete (todeschini parabolic model), maximum stress
    eps1U =	-0.0022		# strain at maximum strength of unconfined concrete
    fc2U = 0.		# ultimate stress
    eps2U =	-0.005		# strain at ultimate stress
    lamb =  0.01			# ratio between unloading slope at $eps2 and initial slope $Ec

    # tensile-strength properties
    ftC = 0.34*(-fc1C)**0.5	 #-0.14*fc1C			# tensile strength +tension xiao et al 2022)
    ftU = 0.34*(-fc1U)**0.5			# tensile strength +tension
    Ets = ftU/0.002			# tension softening stiffness

    # -----------
    Fy_reference = 413.7*MPa
    Fy = Fy_reference if fy_value is None else fy_value  # STEEL yield stress
    Es =  29000.*ksi	    # modulus of steel
    Esh = 5.09*GPa
    fu = 0.621*GPa
    eps_sh = 0.0115
    eps_ult = 0.08
    
    Bs =  0.06#0.0073 #0.06		    # strain-hardening ratio 
    R0 =  18 		  	       # control the transition from elastic to plastic branches
    cR1 = 0.925				# control the transition from elastic to plastic branches
    cR2 = 0.15				# control the transition from elastic to plastic branches
    # -----------
    Fy1 = Fy if fy_value is not None else 344.7*MPa  # secondary reinforcement steel
    Es1 =  29000.*ksi	    # modulus of steel
    Esh1 = 4.09*GPa
    fu1 = 0.421*GPa
    eps_sh1 = 0.0115
    eps_ult1 = 0.155
    
    Bs1 =  0.02 #0.0073			    # strain-hardening ratio 
    R01 =  18 		  	       # control the transition from elastic to plastic branches
    cR11 = 0.925				# control the transition from elastic to plastic branches
    cR21 = 0.15				# control the transition from elastic to plastic branches
    
    
    Fy30 = 400*kN
    Es30 = 300*MPa
    Bs30 = 0.007
    R030 = 18
    cR130 = 0.925
    cR230 = 0.15
    
    ##  uniaxialMaterial('Concrete02', matTag, fpc, epsc0, fpcu, epsU, lambda, ft, Ets)
    op.uniaxialMaterial ('Concrete01', IDconcCore, fc1C, eps1C, fc2C, eps2C)#, lamb, ftC, Ets)	# build core concrete (confined)
    op.uniaxialMaterial ('Concrete01', IDconcCover, fc1U, eps1U, fc2U, eps2U)#, lamb, ftU, Ets)	# build cover concrete (unconfined)
    op.uniaxialMaterial ('Steel02', IDreinf, Fy, Es, Bs, R0, cR1, cR2)				# build reinforcement material
    op.uniaxialMaterial ('Steel02', IDreinf1, Fy1, Es1, Bs1 ,R01, cR11, cR21)
    op.uniaxialMaterial('Steel02', cappile, Fy1, Es1, Bs1 ,R01, cR11, cR21) # for cap pile connection zerolength element
    
    
    print('Concrete01 for core', IDconcCore, fc1C, eps1C, fc2C, eps2C)#, lamb, ftC, Ets)	# build core concrete (confined)
    print('Concrete01 for cover', IDconcCover, fc1U, eps1U, fc2U, eps2U)#, lamb, ftU, Ets)	# build cover concrete (unconfined)
    print('Steel02 for refinforemcent', IDreinf, Fy, Es, Bs, R0, cR1, cR2)				# build reinforcement material
    print('Steel02 for steel cover', IDreinf1, Fy1, Es1, Bs1 ,R01, cR11, cR21)
    #  uniaxialMaterial('ReinforcingSteel', matTag, fy, fu, Es, Esh, eps_sh, eps_ult, '-GABuck', lsr, beta, r, gamma, '-DMBuck', lsr, alpha=1.0, '-CMFatigue', Cf, alpha, Cd, '-IsoHard', a1=4.3, limit=1.0, '-MPCurveParams', R1=0.333, R2=18.0, R3=4.0)
    
    # op.uniaxialMaterial('ReinforcingSteel', IDreinf, Fy, fu, Es, Esh, eps_sh, eps_ult)
    # op.uniaxialMaterial('ReinforcingSteel', IDreinf1, Fy1, fu1, Es1, Esh1, eps_sh1, eps_ult1)
    
    #%%  
    #  ----------------------------------------------------------------------------
    #  MATERIAL DEFINITIONS USED BY BEARING ELEMENTS
    #  ----------------------------------------------------------------------------
    IDElastomer = 5   # material id for elastomeric rubber
    IDdowel_f = 6    # material id for steel dowel as hysteretic
    IDdowel_E = 7
    # Geometry of the bearing
    Lb = 700 * mm  # length of the bearing in longitudinal direction
    Wb = 500 * mm  # width of the bearing in transverse direction
    Hb = 20 * mm  # total height of the bearing

    # Material properties obtained via realizations
    Eb = 30 * ksi  # bearing young's modulus (from AASHTO)
    mub = 0.4  # friction coefficient between rubber and concrete (from realization)
    Gb = 1 * MPa  # shear modulus of the rubber (from realization)
    Pb = 827 * kN / 5
    #%%
    # #hysteretic material
    # k1 = 92*kN/mm
    # u1 = 1.2*mm
    # k2 = 1*kN/mm
    # u2 = 5.3*mm
    # k3 = -45*kN/mm
    # mu1 = 0.2
    # beta = 0.
    # k4 = mu1**(-beta)*k1
    # d1 = 0.
    # d2 = 0.
    # px = 1.
    # py = 0.
    #%%
    #Fixed bearing
    #BoucWen material (hysteretic material)
    alphaF = 0.2 
    # koF = 857.47*kPa/m
    koF = 92000*kN/m
    n = 1
    gamma = 0
    beta = 100
    Ao = 1
    deltaA = 0
    deltaNu = 0
    deltaEta = 0

    # Mechanical properties
    # Kh = Gb * (Wb * Lb) / Hb
    # Kv = (30 * ksi) * (Wb * Lb) / Hb  # assume 30ksi for E (from AASHTO)
    # Krotz = (30 * ksi) * (1/12 * Wb**3 * Lb) / Hb  # assume 30ksi for E
    # Kroty = (30 * ksi) * (1/12 * Wb * Lb**3) / Hb  # assume 30ksi for E
    # Ktor = (1 * ksi) / (30 * ksi) * (Krotz + Kroty)   # assume 1 ksi for G
    #  uniaxialMaterial('Hysteretic', matTag, *p1, *p2, *p3=p2, *n1, *n2, *n3=n2, pinchX, pinchY, damage1, damage2, beta=0.0)
    #op.uniaxialMaterial('Hysteretic', IDdowel, *[k1*u1, u1], *[k2*u2, u2], *[k1*u1, u1], *[k2*u2, u2], px, py, d1, d2)
    op. uniaxialMaterial('BoucWen', IDdowel_f, alphaF, koF, n, gamma, beta, Ao, deltaA, deltaNu, deltaEta)
    print('BoucWen for fixed bearing', IDdowel_f, alphaF, koF, n, gamma, beta, Ao, deltaA, deltaNu, deltaEta)
    
    #steel01 material
    Ke = 100000*kN/mm
    Kp = 0.
    Fy = 4000*kN 
    op.uniaxialMaterial ('Steel02', IDElastomer, Fy, Ke, 0.)
    print('Steel02 for elastomer', IDElastomer, Fy, Ke, 0.)
    # op.uniaxialMaterial ('Steel02', IDElastomer, Fy, Ke, 0.)
    
    #Expansion Bearing
    alphaE = 0.8 
    koE = 10400*kN/m
    n = 1
    gamma = 100
    beta = 100
    Ao = 1
    deltaA = 0
    deltaNu = 0
    deltaEta = 0
    op. uniaxialMaterial('BoucWen', IDdowel_E, alphaE, koE, n, gamma, beta, Ao, deltaA, deltaNu, deltaEta)
    print('BoucWen Material for dowel', IDdowel_E, alphaE, koE, n, gamma, beta, Ao, deltaA, deltaNu, deltaEta)
    
    #%% Backfill wall material (Recomended Values:)
    Backfill_mat = 9
    Kmax =  20300*kN/m  # initial stiffness (of abutment width)
    Kur  =  Kmax        # unloading/reloading stiffness
    Rf   =  0.          # failure ratio
    Fult = -326*kN      # ultimate (maximum) passive resistance (per meter of abutment width)
    gap  = -2.54*cm      # initial gap
    op.uniaxialMaterial('HyperbolicGapMaterial', Backfill_mat, Kmax, Kur, Rf, Fult, gap)
    
    return IDconcCore, IDconcCover, IDreinf, IDreinf1, IDElastomer, IDdowel_f, IDdowel_E, Backfill_mat, cappile
