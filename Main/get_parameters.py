import numpy as np

#######################################################################
def get_parameters(self):
    """Set some basic model constants and parameters"""
#######################################################################

    #################################
    # Time/space discretization 
    #################################
    self.yrs = 365.0*24.0*60.0*60.0   # 1 year in seconds
    self.maxtime =  4.55e9*self.yrs  # Simulation time
    self.dt = 1e6*self.yrs            # Time stepping
    self.n_steps = int(self.maxtime/self.dt)   # Number of time steps
    self.n_layers = 100               # Number of mantle layers

#   Method for fixed point calculation. See root method of scipy.optimize for details. Possibilities include:
#  'broyden1', broyden2', 'anderson', linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane'
    self.rootmethod = 'lm' 
    
    #################################
    # Various settings
    """
    tectonics:        'SL' for stagnant lid, 'ML' for mobile lid tectonics
    core_cooling:      consider core cooling ("yes") or neglect it ("no")
    var_alpha:         turn thermal expansivity (alpha) calculations off ('no') or on ('yes')
    partitioning_calc: 'no' for no local, P-T-X dependent partition coefficient calculations; 'yes' enabling it
    fixed_bulk_Ds:      set fixed_bulk_Ds ='no', with 'yes', all D's will be 0.002 (for D_H2O=0.01)
    eclogite:      turn eclogite dripping on ("yes") or off ("no"); limits crustal growth up to eclogire formation
    crust_delamination:'yes' for delamination (crust is not allowed to be thicker than lithosphere), 
                       'no' for no delamination (lithosphere is not allowed to become thinner than crust)
    hydrous_melting:   'yes': takes for T_sol calculations (H2O dependent melting from Katz et al) into account
    Tcr_cut:           'yes': does not allow Tcr to become larger than solidus T, 'no': Tcr <= Tl
    meltcomp:          choose between basaltic melt compositions 'Mercury', 'Earth', 'Mars'
    mineralogy:        choose mineral compisition for the upper mantle, choice between 'Mercury','Earth', 'Mars'
    """
    #################################
    self.tectonics = 'SL'
    self.core_cooling = 'yes'
    self.var_alpha = 'no'
    self.partitioning_calc = 'yes' #'no' for no partition coefficient calculations; 'yes' enabling it
    self.fixed_bulk_Ds ='no'       # set fixed_bulk_Ds ='no', with 'yes', all D's will be 0.002 (for D_H2O=0.01)
    self.eclogite ='yes'        # set 'no' if delamination is 'yes'
    self.crust_delamination = 'no' #only eclogite or delamination can be 'yes', not both
    if self.eclogite =='yes' and self.crust_delamination == 'yes':
        print('WARNING: Eclogite dripping and crust delamination both activated!!!')
    self.hydrous_melting = 'yes'   # yes: takes for T_sol calculations H2O dependent melting into account
    self.Tcr_cut = 'no'          # yes: does not allow Tcr to become larger than solidus T, 'no': Tcr <= Tl
    self.meltcomp = 'Earth'         # Earth, Mercury, or Mars
    self.mineralogy = 'Earth'       # Earth, Mercury, or Mars
    
    #################################
    # Various constants
    #################################
    self.Racrit   = 450.0         # Critical Rayleigh number Morschhauser: 450, Stamenkovic 2012: 500
    self.beta     = 1./3.         # Nu-Ra scaling exponent
    self.aa       = 0.5           # Prefactor for Nu-Ra stagnant-lid scaling
    self.delta_s0 = 10e3          # Initial thickness of top boundary layer     
    self.delta_c0 = 10e3          # Initial thickness of bottom boundary layer        
    self.Tref     = 1600.0        # Reference temperature (K)
    self.Pref     = 3e9           # Reference pressure (Pa)
    self.E        = 3e5           # Activation energy (J/mol), 3e5 for olivine
    self.Rg       = 8.3144        # Gas constant (J/(mol K))
    self.u0       = 2e-12         # Convection speed scale (m/s) (Morschhauser 2011)
    self.Dcr0     =  30000        # initial crust thickness (m); if set too small, the surface heat flux qs is initially too high
    self.Dl0      = 50000         # initial lithosphere thickness (m)
    self.Dl_min   = 1000          # minimum lithosphere thickness (m) must be >= self.Dl_cr
    self.Dl_cr    = 1000          # differenz zwischen lithosphäre und Kruste
    self.Mscr0    = 0             # initial mass of secondary crust [kg]
    self.oeff_H2O = 0.4           # outgassing efficiency H2O (Morschhauser 2011), only used if magma_outgassing is not used
    self.oeff_CO2 = 0.4           # outgassing efficiency CO2 (Morschhauser 2011), only used if magma_outgassing is not used
    self.theta    = 2.9           # Frank-Kamenetskii-Parameter; 2.9 by Morschhauser 2011 for spherical symmetry
    self.R_gas    = 8.31448       #J/(K*mol), gas constant 
    self.NA       = 6.0221409e+23 #Avogadros Number
        
    #CI abundances for REE (Anders and Grevesse 1981)
    self.X0_La = 0.2347e-6
    self.X0_Ce = 0.6032e-6
    self.X0_Sm = 0.1471e-6
    self.X0_Eu = 0.056e-6
    self.X0_Lu = 0.0243e-6
    
    #################################
    # Melt 
    #SiO2 TiO2 Al2O3 Cr2O3 FeO MnO MgO CaO Na2O K2O
    #################################
    self.wt_perc_molm = np.array([60.083, 79.865, 101.961, 151.989, 71.844, 70.937, 40.304, 56.077, 61.979, 94.195]) 
    self.n_oxy = np.array([2, 2, 3, 3, 1, 1, 1, 1, 1, 1])
    self.n_cat = np.array([1, 1, 2, 2, 1, 1, 1, 1, 2, 2])
    
    #################################
    # Compositional options
    #################################
    # Melt composition (wt%)
    #SiO2 TiO2 Al2O3 Cr2O3 FeO MnO MgO CaO Na2O K2O
    if self.meltcomp == 'Earth':
        self.wt_perc_comp = np.array([45.0, 0.201, 4.45, 0.384, 8.05, 0.135, 37.8, 3.55, 0.36, 0.029])  # McDonough 1995
    elif self.meltcomp =='Mercury':
        self.wt_perc_comp = np.array([57.71, 1.35, 13.46, 0.79, 5.2, 0.7, 15.2, 5.59, 1e-10, 1e-10]) # Vander Kaaden and McCubbin 2015; considered  broadly representative of Mercurian melts
    elif self.meltcomp == 'Mars':
        self.wt_perc_comp = np.array([44.4, 0.14, 3.02, 0.76, 17.9, 0.46, 30.2, 2.45, 0.5, 0])  # Waenke & Dreibus 1988
    else:
        print('ERROR! Choose melt composition!')
        
    # Upper mantle mineralogy (wt%)
    if self.mineralogy == 'Earth':
        self.Ol  = 40 # wt%, pyrolite from Duffy and Anderson 1989; otherwise matched well with seismology: 40
        self.Opx = 10
        self.Cpx = 37
        self.Grt = 13
    elif self.mineralogy =='Mercury':
        #Mantle mineralogy: Padovan et al 2014; building blocks for Mercury are matched compositionally by the chondrules of two metal-rich chondrites (MC model) [Taylor and Scott, 2005]
        self.Ol  = 27.2   # wt% # 26 vol%
        self.Opx = 47.8         # 50
        self.Cpx = 8.9          # 9
        self.Grt = 16           # 15
    elif self.mineralogy =='Mars':
        self.Ol  = 60   # wt%   # 58.7 vol%
        self.Opx = 19.2         # 20.6
        self.Cpx = 11           # 11.4
        self.Grt = 9.8          # 9.3
    else:
        print('ERROR! Choose a mineralogy!')
        
    
    #################################
    # Mineral/Melt Partition Coefficients  
    #################################
    if self.fixed_bulk_Ds =='yes':  # very simplistic approach where all Ds for HPE & REE are 0.002 (except D_Ce=0.01)
        self.D_K_Ol  = 0.002 
        self.D_K_Opx = 0.002
        self.D_K_Grt = 0.002
        self.D_K_Cpx_const = 0.002

        self.D_U_Ol  = 0.002
        self.D_U_Opx = 0.002
        self.D_U_Grt =0.002
        self.D_U_Cpx_const = 0.002

        self.D_Th_Ol  = 0.002
        self.D_Th_Opx =0.002
        self.D_Th_Grt = 0.002
        self.D_Th_Cpx_const =0.002

        self.D_Ce_Ol = 0.01 
        self.D_Ce_Opx =  0.01 
        self.D_Ce_Grt =  0.01
        self.D_Ce_Cpx_const = 0.01

        self.D_La_Ol = 0.002
        self.D_La_Opx =0.002
        self.D_La_Grt = 0.002
        self.D_La_Cpx_const =  0.002

        self.D_Sm_Ol = 0.002
        self.D_Sm_Opx = 0.002
        self.D_Sm_Grt = 0.002
        self.D_Sm_Cpx_const =  0.002

        self.D_Eu_Ol =  0.002
        self.D_Eu_Opx =0.002
        self.D_Eu_Grt = 0.002
        self.D_Eu_Cpx_const =  0.002
        
        self.D_Lu_Ol =  0.002
        self.D_Lu_Opx = 0.002
        self.D_Lu_Grt =  0.002
        self.D_Lu_Cpx_const = 0.002
    else:   # partitioning according to experimental literature values. Mineral ratios are used for bulk mineral/melt partitioning. If self.partitioning_calc = 'yes', the values for Cpx will be calculated in the code.
        self.D_K_Ol  = 0.0056 # Philpotts&Schnetzler(slightly overestimated)
        self.D_K_Opx = 0.005  # estimated, no data
        self.D_K_Grt = 0.0006 # Gaetani 2003
        self.D_K_Cpx_const = 0.007  # Hauri 1994

        self.D_U_Ol  = 0.00001 # Beattie 1993
        self.D_U_Opx = 0.0006  # Klemme 2006 
        self.D_U_Grt = 0.0015  # Klemme 2002
        self.D_U_Cpx_const = 0.01    # Hauri, Wood & Trigila 2001, Salters & Longhi 1999

        self.D_Th_Ol  = 0.00001 # Beattie 1993
        self.D_Th_Opx = 0.0001  # Klemme 2006
        self.D_Th_Grt = 0.006   # Klemme 2002
        self.D_Th_Cpx_const = 0.01    # Hauri 1994 (Klemme 2002: 0.007)

        self.D_Ce_Ol = 0.001  # Adam & Green 2009 
        self.D_Ce_Opx = 0.002 # Salters & Longhi 1999
        self.D_Ce_Grt =  0.01  # Gaetani 2003
        self.D_Ce_Cpx_const =  0.1   # Hauri 1994 

        self.D_La_Ol = 0.0001 # Nielsen 1991
        self.D_La_Opx = 0.0007 # Klemme 2006 #evtl 0.0037, Kennedy et al 1993
        self.D_La_Grt =  0.0016 # Johnson 1999
        self.D_La_Cpx_const =  0.0515 # Hauri 1994

        self.D_Sm_Ol = 0.0011 # Nielsen 1991
        self.D_Sm_Opx =0.015 # Klemme 2006
        self.D_Sm_Grt = 0.27 # Van Westrenen 1999
        self.D_Sm_Cpx_const =  0.462 #Hauri 1994

        self.D_Eu_Ol =  0.00075 # Kennedy et al 1993
        self.D_Eu_Opx = 0.0036 # Kennedy et al 1993
        self.D_Eu_Grt = 0.4 # Johnson 1999
        self.D_Eu_Cpx_const =  0.002#0.458 # Hauri 1994

        self.D_Lu_Ol =  3.9e-2 # Kennedy 1993
        self.D_Lu_Opx = 0.4 #Klemme 2006 #4.7e-2 #Kennedy 1993
        self.D_Lu_Grt = 3.79 #Hauri 1994
        self.D_Lu_Cpx_const = 0.458 # Hauri 1994

    #################################
    # Radioactive elements 
    #################################
    # Half lives (s)
    self.tau_U238  = 4.47e9*self.yrs
    self.tau_U235  = 0.704e9*self.yrs   
    self.tau_Th232 = 14.5e9*self.yrs    
    self.tau_K40   = 1.2483e9*self.yrs  
    # Isotope abundaces
    self.f_U238  = 0.9928
    self.f_U235  = 7.2e-3
    self.f_Th232 = 1.0
    self.f_K40   = 1.19e-4 # take 1.0 only if whole K content
    # Present-day heat productions (W/kg)
    self.H_U238  = 9.46e-5
    self.H_U235  = 5.69e-4
    self.H_K40   = 2.92e-5
    self.H_Th232 = 2.54e-5
    # Ionic radii (Source: http://abulafia.mt.ic.ac.uk/shannon/radius.php)
    self.R_Na = 1.18e-10
    self.R_K  = 1.51e-10
    self.R_Th = 1.041e-10  # Thorium 4+ VIII
    self.R_U  = 0.983e-10   # Uranium 5+(VII) Wood 1999
    self.R_Ce = 1.143e-10  # Cerium 3+ (VIII)
    self.R_La = 1.16e-10   # Lanthanium 3+ (VIII)
    self.R_Sm = 1.079e-10  # Samarium 3+ (VIII)
    self.R_Eu = 1.066e-10  # Europium 3+ (VIII)
    self.R_Lu = 0.977e-10  # Lutetium
    
    return
