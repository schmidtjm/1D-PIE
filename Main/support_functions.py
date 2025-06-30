import numpy as np
import pandas as pd

#####################################################


#####################################################
def initialize_arrays(self):
    """Initialize arrays for thermal evolution"""
#####################################################

    self.t       = np.linspace(0, self.maxtime, self.n_steps+1)
    self.Tprofile= np.zeros((self.n_steps+1,self.n_layers))
    self.Tcr     = np.zeros((self.n_steps+1))
    self.Tl      = np.zeros((self.n_steps+1))
    self.Tm      = np.zeros((self.n_steps+1))
    self.Tb      = np.zeros((self.n_steps+1))
    self.Tc      = np.zeros((self.n_steps+1))
    self.etam    = np.zeros((self.n_steps+1))
    self.etac    = np.zeros((self.n_steps+1))
    self.etab    = np.zeros((self.n_steps+1))
    self.Dl      = np.zeros((self.n_steps+1))
    self.Dcr     = np.zeros((self.n_steps+1))
    self.Dm      = np.zeros((self.n_steps+1))
    self.delta_s = np.zeros((self.n_steps+1))  
    self.delta_c = np.zeros((self.n_steps+1))
    self.qs      = np.zeros((self.n_steps+1))
    self.qc      = np.zeros((self.n_steps+1))
    self.ql      = np.zeros((self.n_steps+1))
    self.Q_U238  = np.zeros((self.n_steps+1))
    self.Q_U235  = np.zeros((self.n_steps+1))
    self.Q_Th232 = np.zeros((self.n_steps+1))
    self.Q_K40   = np.zeros((self.n_steps+1))
    self.Q_tot   = np.zeros((self.n_steps+1))
    self.Ra      = np.zeros((self.n_steps+1))
    self.Ur      = np.zeros((self.n_steps+1))
    self.Rl      = np.zeros((self.n_steps+1))
    self.Rcr     = np.zeros((self.n_steps+1))
    self.Pl      = np.zeros((self.n_steps+1))
    self.Pcr     = np.zeros((self.n_steps+1))
    self.depl    = np.zeros((self.n_steps+1))
    self.St      = np.zeros((self.n_steps+1))
    
    self.Pmelt_mean = np.zeros((self.n_steps+1))
    self.Pmelt_min = np.zeros((self.n_steps+1))
    self.Pmelt_max = np.zeros((self.n_steps+1))
    self.Tmelt_mean = np.zeros((self.n_steps+1))
    self.Tmelt_min = np.zeros((self.n_steps+1))
    self.Tmelt_max = np.zeros((self.n_steps+1))
    
    self.X_U238_liq = np.zeros((self.n_steps+1))
    self.X_U235_liq = np.zeros((self.n_steps+1))
    self.X_K40_liq  = np.zeros((self.n_steps+1))
    self.X_Th232_liq= np.zeros((self.n_steps+1))
    self.X_H2O_liq  = np.zeros((self.n_steps+1))
    self.X_H2O_liq_av = np.zeros((self.n_steps+1))
    
    self.X_K_t      = np.zeros((self.n_steps+1))
    self.X_K_liq    = np.zeros((self.n_steps+1))
    self.Xm_K       = np.zeros((self.n_steps+1))
    self.Mscr_K     = np.zeros((self.n_steps+1))
    self.Mcr_K      = np.zeros((self.n_steps+1))
    self.X_K40      = np.zeros((self.n_steps+1))
      
    self.X_U238_av  = np.zeros((self.n_steps+1))
    self.X_U235_av  = np.zeros((self.n_steps+1))
    self.X_K40_av   = np.zeros((self.n_steps+1))
    self.X_Th232_av = np.zeros((self.n_steps+1))
    self.X_tot_av   = np.zeros((self.n_steps+1))
    self.Mcr        = np.zeros((self.n_steps+1))
    self.Mcr_U238   = np.zeros((self.n_steps+1))
    self.Mcr_U235   = np.zeros((self.n_steps+1))
    self.Mcr_Th232  = np.zeros((self.n_steps+1))
    self.Mcr_K40    = np.zeros((self.n_steps+1))
    self.Mcr_HPE    = np.zeros((self.n_steps+1))
    self.Mcr_H2O    = np.zeros((self.n_steps+1))
    self.Mcr_CO2    = np.zeros((self.n_steps+1))
    self.Mscr_U238  = np.zeros((self.n_steps+1))
    self.Mscr_U235  = np.zeros((self.n_steps+1))
    self.Mscr_K40   = np.zeros((self.n_steps+1))
    self.Mscr_Th232 = np.zeros((self.n_steps+1))
    self.Mscr_HPE   = np.zeros((self.n_steps+1))
    self.Mscr_H2O   = np.zeros((self.n_steps+1))
    self.Pcr        = np.zeros((self.n_steps+1))
    
    self.Mm_U238    = np.zeros((self.n_steps+1))
    self.Mm_U235    = np.zeros((self.n_steps+1))
    self.Mm_Th232   = np.zeros((self.n_steps+1))
    self.Mm_K40     = np.zeros((self.n_steps+1))
    self.Mm_K       = np.zeros((self.n_steps+1))
    self.Mm_HPE     = np.zeros((self.n_steps+1))
    self.Mm_H2O     = np.zeros((self.n_steps+1))
    self.Mm_CO2     = np.zeros((self.n_steps+1))
    self.X_gas_H2O  = np.zeros((self.n_steps+1))
    
    self.Xcrust_U238 = np.zeros((self.n_steps+1)) 
    self.Xcrust_U235 = np.zeros((self.n_steps+1)) 
    self.Xcrust_Th232 = np.zeros((self.n_steps+1)) 
    self.Xcrust_K40 = np.zeros((self.n_steps+1)) 
    self.Xcrust_HPE = np.zeros((self.n_steps+1)) 
    self.Xcrust_K   = np.zeros((self.n_steps+1)) 
    self.Xcrust_H2O = np.zeros((self.n_steps+1)) 
    self.Xcrust_CO2 = np.zeros((self.n_steps+1)) 
    self.Xmantle_HPE = np.zeros((self.n_steps+1)) 
    self.Xmantle_H2O = np.zeros((self.n_steps+1)) 
    self.Xmantle_K = np.zeros((self.n_steps+1)) 
    
    self.Mscr_CO2   = np.zeros((self.n_steps+1))
    self.Mm_evol    = np.zeros((self.n_steps+1)) 
    self.Xm_U238    = np.zeros((self.n_steps+1))
    self.Xm_U235    = np.zeros((self.n_steps+1))
    self.Xm_Th232   = np.zeros((self.n_steps+1))
    self.Xm_K40     = np.zeros((self.n_steps+1))
    self.Xm_HPE     = np.zeros((self.n_steps+1))
    self.Xm_H2O     = np.zeros((self.n_steps+1))
    self.Xm_CO2     = np.zeros((self.n_steps+1))
    self.X_sat_H2O  = np.zeros((self.n_steps+1))
    self.X_melt_H2O = np.zeros((self.n_steps+1))
    self.Qcr        = np.zeros((self.n_steps+1))
    self.Qm         = np.zeros((self.n_steps+1))

    self.X_carbonate_melt= np.zeros((self.n_steps+1))
    self.X_CO2_melt      = np.zeros((self.n_steps+1)) # concentration calculated depending on melt volume
    self.X_CO2_liq = np.zeros((self.n_steps+1))     # average concentration in melt
    self.M_H2O_gas  = np.zeros((self.n_steps+1))
    self.M_CO2_gas  = np.zeros((self.n_steps+1))
    
    self.M_degas_CO2_dt= np.zeros((self.n_steps+1))
    self.M_degas_H2O_dt= np.zeros((self.n_steps+1))
    self.M_degas_H2_dt = np.zeros((self.n_steps+1))
    self.M_degas_CO_dt = np.zeros((self.n_steps+1))
    self.M_degas_CH4_dt= np.zeros((self.n_steps+1))
    
    self.M_H2_gas   = np.zeros((self.n_steps+1))
    self.M_H2O_gas  = np.zeros((self.n_steps+1))
    self.M_CH4_gas  = np.zeros((self.n_steps+1))
    self.M_CO_gas   = np.zeros((self.n_steps+1))
    self.M_CO2_gas  = np.zeros((self.n_steps+1))
    
    self.M_degas_H2_dt_ext  = np.zeros((self.n_steps+1))
    self.M_degas_H2O_dt_ext = np.zeros((self.n_steps+1))
    self.M_degas_CH4_dt_ext = np.zeros((self.n_steps+1))
    self.M_degas_CO_dt_ext  = np.zeros((self.n_steps+1))
    self.M_degas_CO2_dt_ext = np.zeros((self.n_steps+1))
    
    self.M_degas_H2_dt_int  = np.zeros((self.n_steps+1))
    self.M_degas_H20_dt_int = np.zeros((self.n_steps+1))
    self.M_degas_CH4_dt_int = np.zeros((self.n_steps+1))
    self.M_degas_CO_dt_int  = np.zeros((self.n_steps+1))
    self.M_degas_CO2_dt_int = np.zeros((self.n_steps+1))
    
    self.P_H2       = np.zeros((self.n_steps+1))
    self.P_CH4      = np.zeros((self.n_steps+1))
    self.P_CO       = np.zeros((self.n_steps+1))
    self.P_CO2      = np.zeros((self.n_steps+1))
    
    self.EGL        = np.zeros((self.n_steps+1))
    
    self.D_K_Cpx     = np.zeros((self.n_steps+1))
    self.D_Th_Cpx    = np.zeros((self.n_steps+1))
    self.D_U_Cpx     = np.zeros((self.n_steps+1))
    self.D_K         = np.zeros((self.n_steps+1))
    self.D_U         = np.zeros((self.n_steps+1))
    self.D_Th        = np.zeros((self.n_steps+1))
    self.D_Ce        = np.zeros((self.n_steps+1))
    self.D_Ce_Cpx    = np.zeros((self.n_steps+1))
    self.F_av   = np.zeros((self.n_steps+1)) 
    
    self.D_U_min = np.ones((self.n_steps+1))
    self.D_U_max = np.zeros((self.n_steps+1))
    self.D_U_mean = np.zeros((self.n_steps+1))
    self.D_Ce_min = np.ones((self.n_steps+1))
    self.D_Ce_max = np.zeros((self.n_steps+1))
    self.D_Ce_mean = np.zeros((self.n_steps+1))
    self.D_K_min = np.ones((self.n_steps+1))
    self.D_K_max = np.zeros((self.n_steps+1))
    self.D_K_mean = np.zeros((self.n_steps+1))
    self.D_Th_min = np.ones((self.n_steps+1))
    self.D_Th_max = np.zeros((self.n_steps+1))
    self.D_Th_mean = np.zeros((self.n_steps+1))
    
    self.D_U_Cpx_min = np.ones((self.n_steps+1))
    self.D_U_Cpx_max = np.zeros((self.n_steps+1))
    self.D_U_Cpx_mean = np.zeros((self.n_steps+1))
    self.D_Ce_Cpx_min = np.ones((self.n_steps+1))
    self.D_Ce_Cpx_max = np.zeros((self.n_steps+1))
    self.D_Ce_Cpx_mean = np.zeros((self.n_steps+1))
    self.D_K_Cpx_min = np.ones((self.n_steps+1))
    self.D_K_Cpx_max = np.zeros((self.n_steps+1))
    self.D_K_Cpx_mean = np.zeros((self.n_steps+1))
    self.D_Th_Cpx_min = np.ones((self.n_steps+1))
    self.D_Th_Cpx_max = np.zeros((self.n_steps+1))
    self.D_Th_Cpx_mean = np.zeros((self.n_steps+1))
    
    self.D_K_Cpx_mean= np.zeros((self.n_steps+1))
    self.D_K_Cpx_min= np.zeros((self.n_steps+1))
    self.D_K_Cpx_max = np.zeros((self.n_steps+1))
    self.Dcr_prod = np.zeros((self.n_steps+1))
    self.Vcr_prod = np.zeros((self.n_steps+1))
    
    self.Mcr_La      = np.zeros((self.n_steps+1))
    self.Mcr_Ce      = np.zeros((self.n_steps+1))
    self.Mcr_Sm      = np.zeros((self.n_steps+1))
    self.Mcr_Eu      = np.zeros((self.n_steps+1))
    self.Mcr_Lu      = np.zeros((self.n_steps+1))

    self.Xm_La       = np.zeros((self.n_steps+1))
    self.Xm_Ce       = np.zeros((self.n_steps+1))
    self.Xm_Sm       = np.zeros((self.n_steps+1))
    self.Xm_Eu       = np.zeros((self.n_steps+1))
    self.Xm_Lu       = np.zeros((self.n_steps+1))

    self.D_La_Cpx    = np.zeros((self.n_steps+1))
    self.D_La        = np.zeros((self.n_steps+1))
    self.D_Sm_Cpx    = np.zeros((self.n_steps+1))
    self.D_Sm        = np.zeros((self.n_steps+1))
    self.D_Eu_Cpx    = np.zeros((self.n_steps+1))
    self.D_Eu        = np.zeros((self.n_steps+1))
    self.D_Lu_Cpx    = np.zeros((self.n_steps+1))
    self.D_Lu        = np.zeros((self.n_steps+1))

    self.D_La_min = np.ones((self.n_steps+1))
    self.D_La_max = np.zeros((self.n_steps+1))
    self.D_La_mean = np.zeros((self.n_steps+1))
    self.D_Sm_min = np.ones((self.n_steps+1))
    self.D_Sm_max = np.zeros((self.n_steps+1))
    self.D_Sm_mean = np.zeros((self.n_steps+1))
    self.D_Eu_min = np.ones((self.n_steps+1))
    self.D_Eu_max = np.zeros((self.n_steps+1))
    self.D_Eu_mean = np.zeros((self.n_steps+1))
    self.D_Lu_min = np.ones((self.n_steps+1))
    self.D_Lu_max = np.zeros((self.n_steps+1))
    self.D_Lu_mean = np.zeros((self.n_steps+1))

    self.D_La_Cpx_min = np.ones((self.n_steps+1))
    self.D_La_Cpx_max = np.zeros((self.n_steps+1))
    self.D_La_Cpx_mean = np.zeros((self.n_steps+1))
    self.D_Sm_Cpx_min = np.ones((self.n_steps+1))
    self.D_Sm_Cpx_max = np.zeros((self.n_steps+1))
    self.D_Sm_Cpx_mean = np.zeros((self.n_steps+1))
    self.D_Eu_Cpx_min = np.ones((self.n_steps+1))
    self.D_Eu_Cpx_max = np.zeros((self.n_steps+1))
    self.D_Eu_Cpx_mean = np.zeros((self.n_steps+1))
    self.D_Lu_Cpx_min = np.ones((self.n_steps+1))
    self.D_Lu_Cpx_max = np.zeros((self.n_steps+1))
    self.D_Lu_Cpx_mean = np.zeros((self.n_steps+1))

    self.X_La_liq = np.zeros((self.n_steps+1))
    self.X_Ce_liq = np.zeros((self.n_steps+1))
    self.X_Sm_liq = np.zeros((self.n_steps+1))
    self.X_Eu_liq = np.zeros((self.n_steps+1))
    self.X_Lu_liq = np.zeros((self.n_steps+1))

    self.Mscr_La  = np.zeros((self.n_steps+1))
    self.Mscr_Ce  = np.zeros((self.n_steps+1))
    self.Mscr_Sm  = np.zeros((self.n_steps+1))
    self.Mscr_Eu  = np.zeros((self.n_steps+1))
    self.Mscr_Lu  = np.zeros((self.n_steps+1))

    self.Mm_La    = np.zeros((self.n_steps+1))
    self.Mm_Ce    = np.zeros((self.n_steps+1))
    self.Mm_Sm    = np.zeros((self.n_steps+1))
    self.Mm_Eu    = np.zeros((self.n_steps+1))
    self.Mm_Lu    = np.zeros((self.n_steps+1))

    self.Xm_La   = np.zeros((self.n_steps+1))
    self.Xm_Ce   = np.zeros((self.n_steps+1))
    self.Xm_Sm   = np.zeros((self.n_steps+1))
    self.Xm_Eu   = np.zeros((self.n_steps+1))
    self.Xm_Lu   = np.zeros((self.n_steps+1))

    self.Xcrust_La = np.zeros((self.n_steps+1))
    self.Xcrust_Ce = np.zeros((self.n_steps+1))
    self.Xcrust_Sm = np.zeros((self.n_steps+1))
    self.Xcrust_Eu = np.zeros((self.n_steps+1))
    self.Xcrust_Lu = np.zeros((self.n_steps+1))
    
    return

#####################################################
def set_initial_conditions(self):
    """Set initial
       upper mantle temperature Tm0
       core temperature Tc0
       thickness of the upper thermal boundary layer delta_s0
       thickness of the lower thermal boundary layer delta_c0
    """
#####################################################

    D = self.Rp - self.Rc

    self.t[0]     = 0

    if self.tectonics =='ML':
        if self.Dcr0!=0:
            print('Warning! self.Dcr0 was set to 0 because we are in the mobile lid case')
            self.Dcr0 = 0
        self.Tl[0] = self.Ts
    else:
        self.Tl[0]     = self.Tm0 - 2.9*(self.R_gas*self.Tm0**2/self.E)
    self.Dcr[0] = self.Dcr0 
    self.Dl[0] =  self.Dl0
    self.Rl[0]      =self.Rp-self.Dl[0]  
    self.Rcr[0] = self.Rp-self.Dcr[0] 
    
    self.depl[0] = (self.Rp**3 - (self.Rp - self.Dcr0)**3)/(self.Rp**3 - self.Rc**3)
    self.Xm_H2O[0] = self.X0_H2O # for Tcr_cut. Set correctly in interior evolution.
    
    self.Tcr[0]     = self.Ts + (self.Tl[0]-self.Ts)*self.Dcr[0]/self.Dl[0] 
    if (self.Tcr_cut == 'yes'): # temperature is not allowed to become larger than solidus
        self.Tcr[0] = Tcr_cut_melt(self,self.Rcr[0],self.X0_H2O,self.depl[0],self.Tcr[0],0) 
        
    self.Tm[0]     = self.Tm0
    self.Tb[0]      = self.Tm[0]
    self.Tc[0]   = self.Tc0           # Initial core temperature
   
    self.Mcr[0] = 4./3.*np.pi*(self.Rp**3 - (self.Rp - self.Dcr[0])**3)*self.rhocr
    self.Pcr[0] = self.rhocr*self.g*self.Dcr[0]
    self.Pl[0] = self.rhom*self.g*(self.Dl[0]-self.Dcr[0]) +  self.Pcr[0]
    
    self.etam[0]    = calculate_viscosity(self, self.Tm[0], self.Pl[0])   
    self.etac[0]    = calculate_viscosity(self, self.Tc[0], self.rhom*self.g*D) 
    self.etab[0]    = calculate_viscosity(self, self.Tb[0], self.rhom*self.g*D) 
    self.Dm[0] = D-self.Dl[0]
    if self.core_cooling =='yes':
        self.delta_c[0] = self.delta_c0   # Initial thickness of the bottom TBL
        self.qc[0]      = self.km*(self.Tc[0] - self.Tb[0])/self.delta_c[0]
    else:
        self.delta_c[0] = 0.0
        self.qc[0]      = 0.0
        
    self.delta_s[0] = self.delta_s0   # Initial thickness of the upper TBL
    
    if (self.Dcr0!=0):
        self.qs[0]      = self.kcr*((self.Tcr[0] - self.Ts)/self.Dcr[0])
    else:
        self.qs[0]      = self.kcr*((self.Tl[0] - self.Ts)/self.Dl[0])
    
    self.ql[0] = self.km*((self.Tm[0] - self.Tl[0])/self.delta_s[0]) 

    return

#####################################################
def calculate_viscosity(s, T, P):
    """Calculate T- and P-dependent viscosity based on Arrhenius law for diffusion creep"""
#####################################################

    if s.tectonics =='SL':
        eta = s.etaref*np.exp( (s.E + P*s.V)/(s.Rg*T) - (s.E + s.Pref*s.V)/(s.Rg*s.Tref) )
    else: #ML; chosen Morschhauser viscosity because for mobile lid tectonics, the viscosity of the upper mantle becomes pressure-independent (Stamenkovic 2012), might needs to change for super-Earths planets
        eta = s.etaref*np.exp( s.E*(s.Tref-T)/(s.Rg*s.Tref*T)) # Morschhauser 2011 

    return eta 

#####################################################
def calculate_viscosity_CMB(s, T, P): #, V
    """Calculate T- and P-dependent viscosity based on Arrhenius law for diffusion creep, for Pb (upper lower thermal boundary layer) 
    >5e10 Pa"""
#####################################################

    V_down = (1.38+2.15*np.exp(-0.065*(P/1e9+10)**0.485))/1e6  #activation volume for lower mante in larger planets
    eta = s.etaref*np.exp(s.E/s.Rg*(1/T-1/s.Tref)+1/s.Rg*( P*V_down/T - s.Pref*V_down/s.Tref) ) # Stamenkovic 2011 

    return eta

#####################################################################################
def calculate_viscosity_karato(s, T, P):
    """Calculate T- and P-dependent viscosity according to Karato & Wu (1993)"""
#####################################################################################

    B = 6.1e-19
    m = 2.5
    d = 2e-3
    E = 3e5
    V = 2.5e-6
    eta = 1/(2*B) * d**m * np.exp( (E+P*s.V)/(s.Rg*T) )  

    return eta 

###################################################################################################
def calculate_thermal_expansivity(T, P):
    """Calculate P- and T-dependent thermal expansivity using the lower mantle parametrization of 
    Tosi et al. (PEPI, 2013).
    T: temperature in K
    P: pressure in GPa
    alpha: thermal expansivity in 1/K
    """
###################################################################################################

    a0 = 2.68e-5  # 1/K
    a1 = 2.77e-9  # 1/K**2
    a2 = -1.21    # K
    a3 = 8.63e-3  # 1/GPa
    
    alpha = (a0 + a1*T + a2/T**2) * np.exp(-a3*P)

    return alpha

###################################################################################################

def calculate_adiabat(s, n_layers, Tm, Pm, Pb):
    """"""
###################################################################################################
     
    Tprof = np.zeros((n_layers))
    Pprof = np.linspace(Pm, Pb, n_layers)
    dP = (np.diff(Pprof))[0]
    dP = Pprof[1]-Pprof[0]
    
    Tprof[0] = Tm
    Pprof[0] = Pm

    if (s.var_alpha == 'yes'):        
        for i in np.arange(0, n_layers-1):         
            alpha = calculate_thermal_expansivity(Tprof[i], Pprof[i]/1e9)
            Tprof[i+1] = Tprof[i] + alpha*Tprof[i]/s.rhom/s.cm * dP
    else:
        Tprof = Tm * np.exp(s.alpha * Pprof / s.rhom /s.cm)
            
    return Tprof

#####################################################
def initialize_heatproduction(tbp, tau):
    """Scale present-day heat production back by tbp"""
#####################################################

    f = np.exp(np.log(2.)*tbp/tau)

    return f

##############################################################
def calculate_radiodecay(X0, H, tau, t):
    """Calculate radioactive decay for a specific isotope"""
##############################################################

    g = X0*H*np.exp(-np.log(2.)*t/tau)

    return g

########################################################################
def calculate_radiodecay_simple(Q0, lam, t):
    """Calculate radioactive decay based on a single decay constant"""
########################################################################

    g = Q0*np.exp(-lam*t)

    return g

###########################################################################
def calculate_conductive_two_layers(k1,k2,Q1,Q2,T1,T2,R1,R2,RX):
    """calculate the conductive temperature at R2, note: Q is in W/m^3"""
####################################################################
   
    A = [[-1/(k1*R1),1,0,0],[0,0,-1/(k2*R2),1],[1/(k1*RX),-1,-1/(k2*RX),1],[-1/(k1*RX**2),0,1/(k2*RX**2),0]]
    b = [T1+(Q1*R1**2/(6*k1)),T2+(Q2*R2**2/(6*k2)),-(Q1*RX**2)/(6*k1) + (Q2*RX**2)/(6*k2),-(Q1*RX)/(3*k1)+(Q2*RX)/(3*k2)]
    C = np.linalg.solve(A, b)
         
    TX = -Q1*RX**2/(6*k1) - C[0]/(k1*RX) + C[1]
    TX = -Q2*RX**2/(6*k2) - C[2]/(k2*RX) + C[3]
    
    q1l = -k2*(-Q2*RX/(3*k2)+C[2]/(k2*RX**2))
    q1lb = -k1*(-Q1*RX/(3*k1)+C[0]/(k1*RX**2))
    q2l = (T2-T1)/((R1-RX)/k1+(RX-R2)/k2)
    
    qu = -k1*((T1-TX)/(R1-RX))
    qd = -k2*((TX-T2)/(RX-R2))

    return TX  

######################################################################################
def calculate_dry_solidus(P):
    """Calculate dry solidus of Noack et al (PEPI 2017) (for P>=15 GPa) and Morschhauser and Grott (2011) (for P<15 GPa).
    Solidus in K, input pressure in GPa
    Account for H2O influence in interior_evolution"""
######################################################################################
    
    Tsol = np.where(P>15, 1761+36.918*P-0.065444*P**2+7.6686e-5*P**3-3.09272e-8*P**4,1409+134.2*P-6.581*P**2+0.1054*P**3)
    
    return Tsol

######################################################################################
def calculate_dry_liquidus(P):
    """Calculate dry liquidus of Noack et al (PEPI 2017) (for P>=15 GPa) and Morschhauser and Grott (2011) (for P<15 GPa).
    Liquidus in K, input pressure in GPa
    Account for H2O influence in interior_evolution"""
######################################################################################
    
    Tliq= np.where(P>15,1761+75+36.918*P-0.065444*P**2+7.6686e-5*P**3-3.09272e-8*P**4,2035.0+57.46*P-3.487*P**2+0.0769*P**3)
    
    return Tliq
 
######################################################################################
def Tcr_cut_melt(s,Rcr,Xm_H2O,depl,Tcr,i):
#################################
    
    Prcr = s.rhom*s.g*(s.Rp - Rcr)/1e9  # GPa
    Prcr = min(Prcr,9.0) # equation valid until 9 GPa
    delta_T_sol = 1.36*np.minimum(Xm_H2O*1e6*(Prcr*0+1),12*Prcr**0.6+Prcr)**0.75# Katz et al 2003
    Tsolr_ini = calculate_dry_solidus(Prcr) - delta_T_sol
    Tliqr = calculate_dry_liquidus(Prcr) 
    Tsolr = Tsolr_ini + depl*(Tliqr - Tsolr_ini)
    Tcrust = min(Tcr, Tsolr)
    
    return Tcrust
    
######################################################################################
def melting_idx(s):
    """Determine the indeces of the timeseries where the mantle temperature is above
       and below the solidus"""
######################################################################################    
    rho = s.rhom
    g = s.g
    ds = s.delta_s
    # Pressure (in Pa) at the base of the lid
    P = rho*g*ds
    # Indices where Tm >= Tsol and Tm < Tsol
    idx_above_solidus = np.where( s.Tm[:-1] >= calculate_dry_solidus(P[:-1]/1e9) )
    idx_below_solidus = np.where( s.Tm[:-1] < calculate_dry_solidus(P[:-1]/1e9) )    
    
    return idx_above_solidus, idx_below_solidus

######################################################################################
def melting_range(s):
    """"""
######################################################################################    
    rho = s.rhom
    g = s.g
    ds = s.delta_s
    # Pressure (in Pa) at the base of the lid
    P = rho*g*ds
    
    # Arrays containing temperature above and below the solidus
    # For Stagnant lid, consider solidus at the base of the lithosphere
    if (s.tectonics == 'SL'):
        T_below_solidus = np.ma.masked_where(s.Tm[:-1] >= calculate_dry_solidus(P[:-1]/1e9), s.Tm[:-1])
        T_above_solidus = np.ma.masked_where(s.Tm[:-1] < calculate_dry_solidus(P[:-1]/1e9), s.Tm[:-1])
    # For Mobile lid, consider surface solidus
    elif (s.tectonics == 'ML'):
        T_below_solidus = np.ma.masked_where(s.Tm[:-1] >= calculate_dry_solidus(0), s.Tm[:-1])
        T_above_solidus = np.ma.masked_where(s.Tm[:-1] < calculate_dry_solidus(0), s.Tm[:-1])

    return T_below_solidus, T_above_solidus

######################################################################################
def calculate_initial_CMBtemperature(Pcmb):
    """ """
######################################################################################
    X_0 = 0.21
    Tcmb = 5400*(Pcmb/140)**0.48 / (1 - np.log(1. - X_Fe)) # Stixrude 2014

    return Tcmb

######################################################################################
def melt_composition_factor(s):
    """Determine the factor you need for calcualting carbonate """
######################################################################################
   
    fwm = 36.596
    
    return fwm

######################################################################################
def write_output_file(s, outfile):
    """ """
######################################################################################

    yrs = 365.0*24.0*60.0*60.0   # 1 year in seconds
    
    if s.partitioning_calc == 'no':
        outdata = {'time[Myrs]':s.t/yrs/1e6, 'Qm[W/kg]':s.Qm, 'Qcr[W/kg]':s.Qcr, 'Tcr[K]':s.Tcr, 'Tm[K]':s.Tm, 'Tc[K]':s.Tc, 'etam[Pas]':s.etam, 'etab[Pas]':s.etab, 'etac[Pas]':s.etac, 'qs[W/m3]':s.qs, 'ql[W/m3]':s.ql, 'qc[W/m3]':s.qc, 'delta_s[m]':s.delta_s, 'delta_c[m]':s.delta_c, 'Dcr[m]':s.Dcr, 'Dl[m]':s.Dl, 'Dm[m]':s.Dm, 'Pcr[Pa]':s.Pcr, 'Pl[Pa]':s.Pl, 'F_av':s.F_av, 'depletion':s.depl,'Dcr_prod[km3/yr]':s.Dcr_prod*s.yrs/1e9, 'Xm_HPE[wt%]':s.Xm_HPE, 'Xm_K[wt%]':s.Xm_K, 'Xm_H2O[wt%]':s.Xm_H2O, 'Mcr[kg]':s.Mcr,  'Mm_evol':s.Mm_evol, 'M_degas_H2O[kg]':s.M_degas_H2O_dt, 'M_H2O_gas[kg]':s.M_H2O_gas, 'EGL[m]':s.EGL, 'Mcr_U238[kg]':s.Mcr_U238,  'Mcr_U235[kg]':s.Mcr_U235, 'Mcr_Th232[kg]':s.Mcr_Th232, 'Mcr_K40[kg]':s.Mcr_K40, 'Mcr_HPE[kg]':s.Mcr_HPE, 'Mcr_K[kg]':s.Mcr_K, 'Mcr_La[kg]':s.Mcr_La, 'Mcr_Ce[kg]':s.Mcr_Ce, 'Mcr_Sm[kg]':s.Mcr_Sm, 'Mcr_Eu[kg]':s.Mcr_Eu, 'Mcr_Lu[kg]':s.Mcr_Lu, 'Pmelt_min':s.Pmelt_min, 'Pmelt_mean':s.Pmelt_mean, 'Pmelt_max':s.Pmelt_max, 'Tmelt_min':s.Tmelt_min, 'Tmelt_mean':s.Tmelt_mean, 'Tmelt_max':s.Tmelt_max, 'Xmantle_HPE':s.Xmantle_HPE, 'Xcrust_HPE':s.Xcrust_HPE, 'Xmantle_K':s.Xmantle_K, 'Xcrust_K':s.Xcrust_K, 'Xcrust_H2O':s.Xcrust_H2O, 'Xmantle_H2O':s.Xmantle_H2O,  'X_gas_H2O':s.X_gas_H2O}

        
        columns = ('time[Myrs]', 'Qm[W/kg]', 'Qcr[W/kg]', 'Tcr[K]', 'Tm[K]', 'Tc[K]', 'etam[Pas]', 'etab[Pas]', 'etac[Pas]', 'qs[W/m3]', 'ql[W/m3]', 'qc[W/m3]', 'delta_s[m]', 'delta_c[m]', 'Dcr[m]', 'Dl[m]', 'Dm[m]', 'Pcr[Pa]', 'Pl[Pa]', 'F_av', 'depletion', 'Dcr_prod[km3/yr]', 'Xm_HPE[wt%]', 'Xm_K[wt%]', 'Xm_H2O[wt%]', 'Mcr[kg]', 'Mm_evol[kg]',  'M_degas_H2O[kg]', 'M_H2O_gas[kg]', 'EGL[m]', 'Mcr_U238[kg]', 'Mcr_U235[kg]', 'Mcr_Th232[kg]', 'Mcr_K40[kg]', 'Mcr_HPE[kg]', 'Mcr_K[kg]', 'Mcr_La[kg]', 'Mcr_Ce[kg]', 'Mcr_Sm[kg]', 'Mcr_Eu[kg]', 'Mcr_Lu[kg]', 'Pmelt_min[Pa]', 'Pmelt_mean[Pa]', 'Pmelt_max[Pa]', 'Tmelt_min[K]', 'Tmelt_mean[K]', 'Tmelt_max[K]','Xmantle_HPE[%]', 'Xcrust_HPE[%]', 'Xmantle_K[%]', 'Xcrust_K[%]', 'Xcrust_H2O[%]', 'Xmantle_H2O[%]', 'X_gas_H2O[%]')
        
     
    else:  # outdata with partition coefficient calculaitons
        outdata = {'time[Myrs]':s.t/yrs/1e6, 'Qm[W/kg]':s.Qm, 'Qcr[W/kg]':s.Qcr, 'Tcr[K]':s.Tcr, 'Tm[K]':s.Tm, 'Tc[K]':s.Tc, 'etam[Pas]':s.etam, 'etab[Pas]':s.etab, 'etac[Pas]':s.etac, 'qs[W/m3]':s.qs, 'ql[W/m3]':s.ql, 'qc[W/m3]':s.qc, 'delta_s[m]':s.delta_s, 'delta_c[m]':s.delta_c, 'Dcr[m]':s.Dcr, 'Dl[m]':s.Dl, 'Dm[m]':s.Dm, 'Pcr[Pa]':s.Pcr, 'Pl[Pa]':s.Pl, 'F_av':s.F_av, 'depletion':s.depl,'Dcr_prod[km3/yr]':s.Dcr_prod*s.yrs/1e9, 'Xm_HPE[wt%]':s.Xm_HPE, 'Xm_K[wt%]':s.Xm_K, 'Xm_H2O[wt%]':s.Xm_H2O, 'Mcr[kg]':s.Mcr, 'M_degas_H2O[kg]':s.M_degas_H2O_dt, 'M_H2O_gas[kg]':s.M_H2O_gas, 'EGL[m]':s.EGL, 'Mcr_U238[kg]':s.Mcr_U238,  'Mcr_U235[kg]':s.Mcr_U235, 'Mcr_Th232[kg]':s.Mcr_Th232, 'Mcr_K40[kg]':s.Mcr_K40, 'Mcr_HPE[kg]':s.Mcr_HPE, 'Mcr_K[kg]':s.Mcr_K, 'Mcr_La[kg]':s.Mcr_La, 'Mcr_Ce[kg]':s.Mcr_Ce, 'Mcr_Sm[kg]':s.Mcr_Sm, 'Mcr_Eu[kg]':s.Mcr_Eu, 'Mm_evol':s.Mm_evol, 'Mcr_Lu[kg]':s.Mcr_Lu, 'Pmelt_min':s.Pmelt_min, 'Pmelt_mean':s.Pmelt_mean, 'Pmelt_max':s.Pmelt_max, 'Tmelt_min':s.Tmelt_min, 'Tmelt_mean':s.Tmelt_mean, 'Tmelt_max':s.Tmelt_max, 'Xmantle_HPE':s.Xmantle_HPE, 'Xcrust_HPE':s.Xcrust_HPE, 'Xmantle_K':s.Xmantle_K, 'Xcrust_K':s.Xcrust_K, 'Xcrust_H2O':s.Xcrust_H2O, 'Xmantle_H2O':s.Xmantle_H2O,  'X_gas_H2O':s.X_gas_H2O, 'D_K_min':s.D_K_min, 'D_K_max':s.D_K_max, 'D_Th_min':s.D_Th_min, 'D_Th_max':s.D_Th_max, 'D_U_min':s.D_U_min, 'D_U_max':s.D_U_max, 'D_Ce_min':s.D_Ce_min, 'D_Ce_max':s.D_Ce_max, 'D_La_min':s.D_La_min, 'D_La_max':s.D_La_max, 'D_Sm_min':s.D_Sm_min, 'D_Sm_max':s.D_Sm_max, 'D_Eu_min':s.D_Eu_min, 'D_Eu_max':s.D_Eu_max, 'D_Lu_min':s.D_Lu_min, 'D_Lu_max':s.D_Lu_max, 'D_K_Cpx_min':s.D_K_Cpx_min, 'D_K_Cpx_max':s.D_K_Cpx_max, 'D_Th_Cpx_min':s.D_Th_Cpx_min, 'D_Th_Cpx_max':s.D_Th_Cpx_max, 'D_U_Cpx_min':s.D_U_Cpx_min, 'D_U_Cpx_max':s.D_U_Cpx_max, 'D_Ce_Cpx_min':s.D_Ce_Cpx_min, 'D_Ce_Cpx_max':s.D_Ce_Cpx_max, 'D_La_Cpx_min':s.D_La_Cpx_min, 'D_La_Cpx_max':s.D_La_Cpx_max, 'D_Sm_Cpx_min':s.D_Sm_Cpx_min, 'D_Sm_Cpx_max':s.D_Sm_Cpx_max, 'D_Eu_Cpx_min':s.D_Eu_Cpx_min, 'D_Eu_Cpx_max':s.D_Eu_Cpx_max, 'D_Lu_Cpx_min':s.D_Lu_Cpx_min, 'D_Lu_Cpx_max':s.D_Lu_Cpx_max}

        
        columns = ('time[Myrs]', 'Qm[W/kg]', 'Qcr[W/kg]', 'Tcr[K]', 'Tm[K]', 'Tc[K]', 'etam[Pas]', 'etab[Pas]', 'etac[Pas]', 'qs[W/m3]', 'ql[W/m3]', 'qc[W/m3]', 'delta_s[m]', 'delta_c[m]', 'Dcr[m]', 'Dl[m]', 'Dm[m]', 'Pcr[Pa]', 'Pl[Pa]', 'F_av', 'depletion', 'Dcr_prod[km3/yr]', 'Xm_HPE[wt%]', 'Xm_K[wt%]', 'Xm_H2O[wt%]', 'Mcr[kg]', 'Mm_evol[kg]', 'M_degas_H2O[kg]', 'M_H2O_gas[kg]', 'EGL[m]', 'Mcr_U238[kg]',  'Mcr_U235[kg]', 'Mcr_Th232[kg]', 'Mcr_K40[kg]', 'Mcr_HPE[kg]', 'Mcr_K[kg]', 'Mcr_La[kg]', 'Mcr_Ce[kg]', 'Mcr_Sm[kg]', 'Mcr_Eu[kg]', 'Mcr_Lu[kg]', 'Pmelt_min[Pa]', 'Pmelt_mean[Pa]', 'Pmelt_max[Pa]', 'Tmelt_min[K]', 'Tmelt_mean[K]', 'Tmelt_max[K]','Xmantle_HPE[%]', 'Xcrust_HPE[%]',  'Xmantle_K[%]', 'Xcrust_K[%]', 'Xcrust_H2O[%]', 'Xmantle_H2O[%]', 'X_gas_H2O[%]', 'D_K_min', 'D_K_max', 'D_Th_min', 'D_Th_max', 'D_U_min', 'D_U_max', 'D_Ce_min', 'D_Ce_max', 'D_La_min', 'D_La_max', 'D_Sm_min', 'D_Sm_max', 'D_Eu_min', 'D_Eu_max', 'D_Lu_min', 'D_Lu_max','D_K_Cpx_min', 'D_K_Cpx_max', 'D_Th_Cpx_min', 'D_Th_Cpx_max', 'D_U_Cpx_min', 'D_U_Cpx_max', 'D_Ce_Cpx_min', 'D_Ce_Cpx_max', 'D_La_Cpx_min', 'D_La_Cpx_max', 'D_Sm_Cpx_min',  'D_Sm_Cpx_max', 'D_Eu_Cpx_min', 'D_Eu_Cpx_max', 'D_Lu_Cpx_min', 'D_Lu_Cpx_max')
     
    
    df = pd.DataFrame(data=outdata)
    
    df.to_csv('outfile.csv')
    df.to_excel('outfile.xlsx')
   
    return

######################################################################################
def mass_radius_relations_withFe(Mr, X_Fe):
    """Given the mass planetary mass to Earth mass ratio Mr = Mp/M_E, and iron
    mass fraction, calculate planetary radius (Rp), 
    core radius (Rc), surface gravity (g), mantle density (rhom) and core density (rhoc) 
    using mass-radius relations from L. Noack (unpublished)
    """
######################################################################################

    M_E = 5.972e24  # Earth mass
    G = 6.67e-11    # Gravitational constant

    Rp = (7e3 - 1.8e3*X_Fe)*Mr**0.3
    rhoc = 12300.*Mr**0.2
    Rc = 1e-3*( (X_Fe*Mr*M_E)/ (4./3*np.pi*rhoc) )**(1./3) 
    rhom = (1 - X_Fe)*Mr*M_E/ (4./3*np.pi*((Rp*1e3)**3 - (Rc*1e3)**3))
    g = G*M_E*Mr / (Rp*1e3)**2
    
    print('Rp = ', Rp, 'km')
    print('Rc = ', Rc, 'km')
    print('g  = ', g, 'm/s^2')
    print('rhom = ', rhom, 'kg/m^3')
    print('rhoc = ', rhoc, 'kg/m^3')

    return Rp, Rc, g, rhom, rhoc

######################################################################################
def mass_radius_relations(Mr):
    """Given the mass planetary mass to Earth mass ratio Mr = Mp/M_E,
    calculate planetary radius (Rp), core radius (Rc), surface gravity (g),
    mantle density (rhom) and core density (rhoc) using mass-radius relations
    from Valencia et al. (Icarus, 2006) 
    """
######################################################################################

    Rp_E = 6371e3   # Earth raidus
    Rc_E = 3480e3   # Earth's core radius
    M_E = 5.972e24  # Earth mass
    g_E = 9.81      # Earth surface gravity
    X_Fe = 0.326    # Earth core mass fraction

    #Rp = Rp_E*Mr**0.27
    #Rc = Rc_E*Mr**0.247
    #g = g_E*Mr**0.46
    rhom = Mr*M_E*(1-X_Fe) / (4*np.pi/3*(Rp**3 - Rc**3))
    rhoc = Mr*M_E*X_Fe / (4*np.pi/3*Rc**3)
    
    print('Rp = ', Rp/1e3, 'km')
    print('Rc = ', Rc/1e3, 'km')
    print('g  = ', g, 'm/s^2')
    print('rhom = ', rhom, 'kg/m^3')
    print('rhoc = ', rhoc, 'kg/m^3')

    return Rp, Rc, g, rhom, rhoc

