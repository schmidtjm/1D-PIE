"""
This code is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license." --> alles in Readme?

@author: Julia M. Schmidt
@date: 16.08.2023, last updated: 26.05.2025
"""

#################################################################
import numpy as np
from scipy.optimize import fsolve, fixed_point, brentq, root
from matplotlib import pyplot as plt
from statistics import mean
from get_input import *
from get_parameters import *
import support_functions as suppf
import plotting_functions as plotf
import json
#################################################################

class interior_evolution:

####################################################################################
    def __init__(self, body='Mars', inpfile='input.json'):        
        """Set initial conditions and model parameters"""
####################################################################################

        # Set planet parameters and initial conditions
        get_input(self, body=body, inpfile=inpfile)

        # Set model parameters
        get_parameters(self)

######################################################################
    def calculate_evolution(self, outfile=True):
        """Compute evolution of mantle and core temperatures"""
######################################################################
        #############################################################
        # Initialize arrays
        suppf.initialize_arrays(self)
        
        # Set initial conditions
        suppf.set_initial_conditions(self)

        # Set/compute some parameters
        D  = self.Rp - self.Rc                                                         # Mantle thickness                                
        M  = 4./3.*np.pi*((self.Rp**3.-self.Rc**3)*self.rhom + (self.Rc**3)*self.rhoc) # Planet mass
        Mm = 4./3.*np.pi*((self.Rp**3.-self.Rc**3)*self.rhom)                          # Mantle mass
        Mcm = 4./3.*np.pi*((self.Rl[0]**3-self.Rc**3)*self.rhom)                       # convecting mantle mass
        Vm = 4./3.*np.pi*((self.Rp**3.-self.Rc**3))                                    # Mantle volume
        Vcm = 4./3.*np.pi*(self.Rl[0]**3-self.Rc**3)                                   # convecting mantle volume
        Mc = 4./3.*np.pi*(self.Rc**3*self.rhoc)                                        # Core mass
        Ap = 4.*np.pi*self.Rp**2                                                       # Surface area             
        Ac = 4.*np.pi*self.Rc**2                                                       # Core area 
        Acm = 4.*np.pi*self.Rl[0]**2                                                   # convecting mantle surface area     
        kappa = self.km/self.rhom/self.cm                                              # Thermal diffusivity 
        kappa_CMB =  self.kc/self.rhom/self.cc                                         # Thermal diffusivity at CMB
        gCMB = 6.6743e-11*Mc/self.Rc**2                                                # gravity at CMB with gravitational constant
        

           
        # Scale heat production back in time
        X0_U238 = self.f_U238*self.X_U*suppf.initialize_heatproduction(self.maxtime,self.tau_U238)
        X0_U235 = self.f_U235*self.X_U*suppf.initialize_heatproduction(self.maxtime,self.tau_U235)
        X0_Th232 = self.f_Th232*self.X_Th*suppf.initialize_heatproduction(self.maxtime,self.tau_Th232)
        X0_K40 = self.f_K40*self.X_K*suppf.initialize_heatproduction(self.maxtime,self.tau_K40)
        X0_K = self.X_K - self.X_K * self.f_K40 + X0_K40
        X0_HPE = X0_U238 + X0_U235 + X0_Th232 + X0_K40

        
        # Planet averaged heating rate 25.7.
        self.Q_U238[0]  = suppf.calculate_radiodecay(X0_U238, self.H_U238, self.tau_U238, self.t[0])
        self.Q_U235[0]  = suppf.calculate_radiodecay(X0_U235, self.H_U235, self.tau_U235, self.t[0])
        self.Q_Th232[0] = suppf.calculate_radiodecay(X0_Th232, self.H_Th232, self.tau_Th232, self.t[0])
        self.Q_K40[0]   = suppf.calculate_radiodecay(X0_K40, self.H_K40, self.tau_K40, self.t[0])
        self.Q_tot[0]   = self.Q_U238[0] + self.Q_U235[0] + self.Q_Th232[0] + self.Q_K40[0] 
        self.Qm[0] = self.Q_U238[0] + self.Q_U235[0] + self.Q_Th232[0] + self.Q_K40[0] # mantle heating rate
  
        
        if self.tectonics == 'SL':
            # amount of radioactive elements in primordial crust; is zero if no primordial crust considered
            Mpcr_U238 = self.lam * X0_U238 * self.Mcr[0] 
            Mpcr_U235 = self.lam * X0_U235 * self.Mcr[0] 
            Mpcr_Th232 = self.lam * X0_Th232 * self.Mcr[0] 
            Mpcr_K40 = self.lam * X0_K40 * self.Mcr[0]
            Mpcr_K = self.lam * X0_K * self.Mcr[0]
            Mpcr_HPE = Mpcr_U238 + Mpcr_U235 + Mpcr_Th232 +  Mpcr_K40
            Mpcr_H2O = self.lam * self.X0_H2O * self.Mcr[0]
            Mpcr_CO2 = self.lam * self.X0_CO2 * self.Mcr[0]
            Mpcr_La = self.lam * self.X0_La * self.Mcr[0]
            Mpcr_Ce = self.lam * self.X0_Ce * self.Mcr[0]
            Mpcr_Sm = self.lam * self.X0_Sm * self.Mcr[0]
            Mpcr_Eu = self.lam * self.X0_Eu * self.Mcr[0]
            Mpcr_Lu = self.lam * self.X0_Lu * self.Mcr[0]

            # total mass in crust [kg]
            self.Mcr_U238[0] = Mpcr_U238    
            self.Mcr_U235[0] = Mpcr_U235  
            self.Mcr_Th232[0]= Mpcr_Th232
            self.Mcr_K40[0]  = Mpcr_K40                                                   
            self.Mcr_HPE[0] = Mpcr_HPE
            self.Mcr_H2O[0] = Mpcr_H2O     # Wasser that is removed with melt formation
            self.Mcr_CO2[0] = Mpcr_CO2
            self.Mcr_K[0] =  Mpcr_K  
            self.Mcr_La[0] =  Mpcr_La  
            self.Mcr_Ce[0] =  Mpcr_Ce  
            self.Mcr_Sm[0] =  Mpcr_Sm 
            self.Mcr_Eu[0] =  Mpcr_Eu  
            self.Mcr_Lu[0] =  Mpcr_Lu 

        # crust heating rate [W/kg] 
            Qcr_U238  = suppf.calculate_radiodecay(self.Mcr_U238[0]/self.Mcr[0], self.H_U238, self.tau_U238, self.t[0])
            Qcr_U235  = suppf.calculate_radiodecay(self.Mcr_U235[0]/self.Mcr[0], self.H_U235, self.tau_U235, self.t[0])
            Qcr_Th232 = suppf.calculate_radiodecay(self.Mcr_Th232[0]/self.Mcr[0], self.H_Th232, self.tau_Th232, self.t[0])
            Qcr_K40   = suppf.calculate_radiodecay(self.Mcr_K40[0]/self.Mcr[0], self.H_K40, self.tau_K40, self.t[0])
            self.Qcr[0]   = Qcr_U238 + Qcr_U235 + Qcr_Th232 + Qcr_K40

        else:
            
            Mpcr_U238 = 0.0
            Mpcr_U235 = 0.0
            Mpcr_Th232 = 0.0
            Mpcr_K40 = 0.0
            Mpcr_K = 0.0
            Mpcr_HPE = 0.0
            Mpcr_H2O = 0.0
            Mpcr_CO2 = 0.0
            Mpcr_La = 0.0
            Mpcr_Ce = 0.0
            Mpcr_Sm = 0.0
            Mpcr_Eu = 0.0
            Mpcr_Lu = 0.0
            
            self.Qcr[0]   = 0.0
        
        self.Mm_evol[0] = 4./3.*np.pi*((self.Rp-self.Dcr[0])**3-self.Rc**3)*self.rhom   # mantle mass without crust [kg]
        
        # amount of HPE & volatiles in the mantle
        self.Xm_U238[0] = (X0_U238*Mm - Mpcr_U238)/self.Mm_evol[0]
        self.Xm_Th232[0]= (X0_Th232*Mm - Mpcr_Th232)/self.Mm_evol[0]
        self.Xm_K40[0]  = (X0_K40*Mm - Mpcr_K40)/self.Mm_evol[0]                   
        self.Xm_K[0] = (X0_K*Mm - Mpcr_K)/self.Mm_evol[0]
        self.Xm_H2O[0] = (self.X0_H2O*Mm - Mpcr_H2O)/self.Mm_evol[0] 
        self.Xm_CO2[0] = (self.X0_CO2*Mm - Mpcr_CO2)/self.Mm_evol[0]
        self.Xm_La[0] = (self.X0_La*Mm - Mpcr_La)/self.Mm_evol[0]
        self.Xm_Ce[0] = (self.X0_Ce*Mm - Mpcr_Ce)/self.Mm_evol[0]
        self.Xm_Sm[0] = (self.X0_Sm*Mm - Mpcr_Sm)/self.Mm_evol[0]
        self.Xm_Eu[0] = (self.X0_Eu*Mm - Mpcr_Eu)/self.Mm_evol[0]
        self.Xm_Lu[0] = (self.X0_Lu*Mm - Mpcr_Lu)/self.Mm_evol[0]
        
        Vmelt = 0


        ###################################################################
        # Time loop
        ###################################################################
        for i in np.arange(1,self.n_steps):
            # Planet averaged heating rate
            self.Q_U238[i]  = suppf.calculate_radiodecay(X0_U238, self.H_U238, self.tau_U238, self.t[i])
            self.Q_U235[i]  = suppf.calculate_radiodecay(X0_U235, self.H_U235, self.tau_U235, self.t[i])
            self.Q_Th232[i] = suppf.calculate_radiodecay(X0_Th232, self.H_Th232, self.tau_Th232, self.t[i])
            self.Q_K40[i]   = suppf.calculate_radiodecay(X0_K40, self.H_K40, self.tau_K40, self.t[i])
            self.Q_tot[i]   = self.Q_U238[i] + self.Q_U235[i] + self.Q_Th232[i] + self.Q_K40[i] 
            
            # number of HPE elements:
            self.X_K40[i] = suppf.calculate_radiodecay(X0_K40, 1, self.tau_K40, self.t[i])
            self.X_K_t[i] =  self.X_K*(1-self.f_K40) + self.X_K40[i]

            
            # mantle volumetric heating rate
            Qm_U238  = suppf.calculate_radiodecay(self.Xm_U238[i-1], self.H_U238, self.tau_U238, self.t[i])
            Qm_U235 = suppf.calculate_radiodecay(self.Xm_U235[i-1], self.H_U235, self.tau_U235, self.t[i])
            Qm_Th232 = suppf.calculate_radiodecay(self.Xm_Th232[i-1], self.H_Th232, self.tau_Th232, self.t[i])
            Qm_K40  = suppf.calculate_radiodecay(self.Xm_K40[i-1], self.H_K40, self.tau_K40, self.t[i])
            self.Qm[i]   = Qm_U238 + Qm_U235 + Qm_Th232 + Qm_K40 
            
                
            Qcr_U238  = suppf.calculate_radiodecay(self.Mcr_U238[i-1]/(self.Mcr[i-1]+self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr), self.H_U238, self.tau_U238, self.t[i])
            Qcr_U235  = suppf.calculate_radiodecay(self.Mcr_U235[i-1]/(self.Mcr[i-1]+self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr), self.H_U235, self.tau_U235, self.t[i])
            Qcr_Th232 = suppf.calculate_radiodecay(self.Mcr_Th232[i-1]/(self.Mcr[i-1]+self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr), self.H_Th232, self.tau_Th232, self.t[i])
            Qcr_K40   = suppf.calculate_radiodecay(self.Mcr_K40[i-1]/(self.Mcr[i-1]+self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr), self.H_K40, self.tau_K40, self.t[i])
            self.Qcr[i]   = Qcr_U238 + Qcr_U235 + Qcr_Th232 + Qcr_K40 
            

            # Mantle temperature evolution
            self.Tm[i] = self.Tm[i-1] + self.dt/(self.epsm*(1+self.St[i-1]))*(self.Qm[i]/self.cm + self.Qtidal/self.cm -(Acm*(self.ql[i-1]+(self.rhocr*self.L+self.rhocr*self.ccr*(self.Tm[i-1]-self.Tl[i-1]))*self.Dcr_prod[i-1]/self.dt))/(Mcm*self.cm) + (Ac*self.qc[i-1])/(Mcm*self.cm)) 
          
            
            # Core temperature evolution
            if self.core_cooling == 'yes': 
                self.Tc[i] = self.Tc[i-1] - (self.dt/self.epsc*Ac*self.qc[i-1])/(Mc*self.cc) 
            else:
                self.Tc[i] = self.Tb[i-1]

            if self.tectonics == 'ML':
                self.Tl[i] =self.Tm[i] - 2.9*(self.R_gas*self.Tm[i]**2/self.E)
                self.Tcr[i] = self.Ts
                
                #heat fluxes
                self.qs[i] = self.kcr*(self.Tl[i] - self.Ts)/self.Dl[i-1]    # surface heat flux
                self.ql[i] = self.km*((self.Tm[i] - self.Tl[i])/self.delta_s[i-1])  # Lithosphere heat flux

            else: # SL
                self.Tl[i] =self.Tm[i] - 2.9*(self.R_gas*self.Tm[i]**2/self.E)
                if (self.Dcr[i-1] == 0):
                    self.Tcr[i] = self.Ts
                    self.qs[i] = self.km*((self.Tl[i] - self.Ts)/self.Dl[i-1])
                    self.qs[i] = max(self.qs[i], 0.02)
                else:
                    self.Tcr[i] = suppf.calculate_conductive_two_layers(self.kcr,self.km,self.Qcr[i]*self.rhocr,self.Qm[i]*self.rhom,
                                                                        self.Ts,self.Tl[i],self.Rp,self.Rl[i-1],self.Rcr[i-1]) 
                    
                    if (self.Tcr_cut == 'yes'): # Tcr is not allowed to become larger than Tsol
                        self.Tcr[i] = suppf.Tcr_cut_melt(self, i) 
                    else: # Method used by Morschhauser et al. 2011, Tcr is not allowed to become larger than Tl
                        self.Tcr[i] = min(self.Tcr[i], self.Tl[i])
                    self.qs[i] = self.kcr*((self.Tcr[i] - self.Ts)/self.Dcr[i-1]) 
                
                self.ql[i] = self.km*((self.Tm[i] - self.Tl[i])/self.delta_s[i-1])  # Lithosphere heat flux

                                           
            self.Ur[i] = self.Q_tot[i]*Mm/(self.qs[i]*Ap) # Urey ratio
   
            # Pressure at the mantle (Pm), top of the bottom TBL (Pb) and at the CMB (Pc)
            self.Pcr[i] = self.rhocr*self.g*self.Dcr[i-1] 
            self.Pl[i] = self.rhom*self.g*self.Dl[i-1] - self.Dcr[i-1]*self.rhom*self.g  +  self.Pcr[i]
            Pm = self.rhom*self.g*(self.Dl[i-1]+self.delta_s[i-1]) - self.Dcr[i-1]*self.rhom*self.g + self.Dcr[i-1]*self.rhocr*self.g
            zb = self.Rp - (self.Rc-self.delta_c[i-1])
            Pb = self.rhom*zb*self.g - self.Dcr[i-1]*self.rhom*self.g + self.Dcr[i-1]*self.rhocr*self.g
            Pc = self.rhom*self.g*(self.Rp - self.Rc) - self.Dcr[i-1]*self.rhom*self.g + self.Dcr[i-1]*self.rhocr*self.g
            self.Tprofile[i,:] = suppf.calculate_adiabat(self, self.n_layers, self.Tm[i], Pm, Pb)
            self.Tb[i] = self.Tprofile[i][-1] 


            # Average temperatures and pressures in the bottom thermal boundary layer
            if self.core_cooling =='yes':
                 
                Tbmean = (self.Tb[i] + self.Tc[i])/2
                Pbmean = (Pb + Pc)/2

            else:
                Tbmean = self.Tb[i]
                Pbmean = Pb

            # Viscosities   
            if (Pc<25e+9): #Pa
                self.etam[i] = suppf.calculate_viscosity(self, self.Tm[i], Pm) #changed location back because Pc would not influence all Tm
            else:
                self.etam[i] = suppf.calculate_viscosity_CMB(self, self.Tm[i], Pm) #, Vdown
                

            # Average viscosity near the bottom TBL (etab) and at the CMB (etac)
            if (Pc<25e+9): #Pa
                self.etac[i] = suppf.calculate_viscosity(self, self.Tc[i], Pc)
            else:
                self.etac[i] = suppf.calculate_viscosity_CMB(self, self.Tc[i], Pc) #, Vdown
                
            if (Pbmean<25e+9):
                self.etab[i] = suppf.calculate_viscosity(self, Tbmean, Pbmean)
            else:
                self.etab[i] = suppf.calculate_viscosity_CMB(self, Tbmean, Pbmean)

            dTc = max(self.Tc[i]-self.Tb[i],1)
            self.Ra[i] = self.rhom*self.g*self.alpha*(self.Tm[i]-self.Tl[i]+dTc)*(self.Rp-self.Rc-self.Dl[i-1])**3.0/(kappa*self.etam[i]) # Morschhauser et al. 2011 Eq.6
            self.delta_s[i] = (self.Rp-self.Rc- self.Dl[i-1])*(self.Racrit/self.Ra[i])**self.beta           # Thickness of the upper TBL 
            self.delta_s[i] = min(self.delta_s[i], self.Rp-self.Rc-self.Dl[i-1])
   
            # consider core cooling
            if self.core_cooling =='yes':

                # Internal and critical Rayleigh number
                Ra_int = self.rhom*self.g*self.alpha*(self.Tm[i]-self.Ts+dTc)*D**3.0/(kappa*self.etam[i]) # Morschhauser et al. 2011 Eq.14
                Racrit_int = 0.28*Ra_int**0.21  # Morschhauser et al. 2011

                # Update bottom TBL thickness
                self.delta_c[i] = (kappa_CMB*self.etab[i]*Racrit_int/(self.rhom*self.alpha*gCMB*dTc))**self.beta   
                self.delta_c[i] = min(self.delta_c[i],2*self.delta_s[i])
                
                # CMB heat flux
                self.qc[i] = self.km*(self.Tc[i] - self.Tb[i])/self.delta_c[i]

            else:
                self.delta_c[i] = 0.0
                self.qc[i] = 0.0
                        
            #############################################################################
            """melt calculations"""
            #############################################################################
            
            # Vertical resolution for the temperature profile (higher resolution variant for larger planets)
           # nt = np.size(self.t)-1
           # npta = 7000 #3000 #1000 # 6000 #3000             # adiabatc mantle
           # nptb = 200 #200 #300 #150               # lower boundary layer
           # npts = 1000 #200 #600 #300 #150          # upper boundary layer
           # nptl = 1000 #500 #1200 #600 #300         # Lithosphere
           # nptc = 500 #200 #400 #200               # crust      
           # npt = nptb + npta + npts + nptl + nptc
        
            nt = np.size(self.t)-1
            npta = 5000 #3000 #1000 # 6000 #3000             # adiabatc mantle
            nptb = 200 #200 #300 #150               # lower boundary layer
            npts = 600 #200 #600 #300 #150          # upper boundary layer
            nptl = 1500 #500 #1200 #600 #300         # Lithosphere
            nptc = 500 #200 #400 #200               # crust      
            npt = nptb + npta + npts + nptl + nptc
            
           #  Vertical resolution for the temperature profile
           # nt = np.size(self.t)-1
           # npta = 2000 #3000 #1000 # 6000 #3000             # adiabatc mantle
           # nptb = 200 #200 #300 #150               # lower boundary layer
           # npts = 150 #200 #600 #300 #150          # upper boundary layer
           # nptl = 500 #500 #1200 #600 #300         # Lithosphere
           # nptc = 100 #200 #400 #200               # crust      
           # npt = nptb + npta + npts + nptl + nptc
           
            # calculate solidus and liquidus temperatures; solidus dependent on H2O content              
            r = np.linspace(self.Rc, self.Rp, npt)
            Pr = np.linspace(self.rhom*self.g*(self.Rp - self.Rc)/1e9, 0., npt)  
            if self.hydrous_melting =="yes":
                delta_T_sol =  43*np.minimum(self.Xm_H2O[i-1]*1e-4*(Pr*0+1),12*Pr**0.6+Pr)**0.75 # revised version of Katz et al. 2003 by Vulpius et al. 2024
            else:
                delta_T_sol =  0.0
                
            Tsolr_ini = suppf.calculate_dry_solidus(Pr) - delta_T_sol
            Tliqr = suppf.calculate_dry_liquidus(Pr) 
            Tsolr = Tsolr_ini + self.depl[i-1]*(Tliqr - Tsolr_ini)   

            ra = np.linspace(self.Rc + self.delta_c[i], self.Rp - self.Dl[i-1], npta) 
            Ta = suppf.calculate_adiabat(self, npta, self.Tm[i], Pm, Pb)
            Ta = np.flipud(Ta)
            Pa = self.rhom*self.g*(self.Rp - ra)      

            # Lower boundary layer
            rb = np.linspace(self.Rc, self.Rc + self.delta_c[i], nptb)
            Trb = np.linspace(self.Tc[i], Ta[0], nptb)
            Prb = self.rhom*self.g*(self.Rp - rb)

            # Upper boundary layer
            rs = np.linspace(self.Rp - self.delta_s[i] - self.Dl[i-1], self.Rp - self.Dl[i-1], npts) 
            Trs = np.linspace(Ta[npts-1], self.Tl[i], npts)
            Prs = self.rhom*self.g*(self.Rp - rs)

            # Lithosphere without crust
            rl = np.linspace(self.Rp - self.Dl[i-1], self.Rp - self.Dcr[i-1], nptl)
            Trl = np.linspace(self.Tl[i],self.Tcr[i], nptl)        
            Prl = self.rhom*self.g*(self.Rp - rl)

            # Crust
            rcr = np.linspace(self.Rp - self.Dcr[i-1], self.Rp, nptc)    
            Trcr = np.linspace(self.Tcr[i], self.Ts, nptc)
            Prcr = self.rhocr*self.g*(self.Rp - rcr)         

            Ttemp = np.concatenate((Trb, Ta, Trs, Trl, Trcr)) 
            rtemp = np.concatenate((rb, ra, rs, rl, rcr))                     
            Ptemp = np.concatenate((Prb, Pa, Prs, Prl, Prcr))   
            
            # interpolating P and T
            Tprof = np.interp(r, rtemp, Ttemp) 
            Pprof = np.interp(r, rtemp, Ptemp)      
            
            # Partial melt zone
            idx_meltzone = np.argwhere((Pprof<15e9) & (Tprof > Tsolr)) 
            
            if idx_meltzone.size > 0:
                self.Tmelt_min[i]=Tprof[idx_meltzone[-1]]
                self.Tmelt_max[i]=Tprof[idx_meltzone[0]]
                self.Tmelt_mean[i]=np.mean(Tprof[idx_meltzone])

                self.Pmelt_min[i]=Pprof[idx_meltzone[-1]]
                self.Pmelt_max[i]=Pprof[idx_meltzone[0]]
                self.Pmelt_mean[i]=np.mean(Pprof[idx_meltzone])
            
            
            Vmelt = 0  
            dVmelt_dTm = 0
            Vmeltzone = 0
            it=0
            self.X_U238_liq[i] = 0  
            self.X_U235_liq[i] = 0
            self.X_Th232_liq[i] = 0 
            self.X_K40_liq[i] = 0 
            self.X_H2O_liq[i] = 0
            self.X_CO2_liq[i] = 0 
            self.X_K_liq[i] = 0
            self.X_Sm_liq[i] = 0
            self.X_Lu_liq[i] = 0
            self.X_Ce_liq[i] = 0
            self.X_La_liq[i] = 0
            self.X_Eu_liq[i] = 0
            
            for idx in idx_meltzone: 
                if (Pprof[idx] < self.Pcrossover) and (idx < npt -1):  
                    it=it+1
                    F =  (Tprof[idx] - Tsolr[idx])/(Tliqr[idx] - Tsolr_ini[idx])#**1.5 #determine melt fraction
                    if (F > 0.0):
                        F = min(F,1.0)
                        V = 4./3.*np.pi*(((r[idx]+r[idx+1])/2)**3 - ((r[idx]+r[idx-1])/2)**3) # volume of the meltzone
                    else:
                        F = 0.0
                        V = 0.0

                    Vmelt = Vmelt + F*V                  # Volume of melt in the meltzone
                    Vmeltzone = Vmeltzone + V             # volume of the meltzone over time
                    dVmelt_dTm = dVmelt_dTm + V/(Tliqr[idx] - Tsolr_ini[idx]) 
                    
                    self.Pmelt_mean[i]=self.Pmelt_mean[i]+np.mean(Pprof[idx])
                    self.Tmelt_mean[i]=self.Tmelt_mean[i]+np.mean(Tprof[idx])
                   
                    # Carbonate and CO2 reditribution in melt 
                    self.KI = 40.07639-((2.53932*10.0**(-2))*Tprof[idx])+(5.27096*10.0**(-6))*(Tprof[idx]**2)+0.0267*((Pprof[idx]*1e-5)-1.0)/Tprof[idx]
                    self.KII = -6.24736 - (282.56/Tprof[idx])-(0.119242*((Pprof[idx]*1e-5-1000.0)/Tprof[idx]))
                    self.fO2 = (6.899-(27714.0/Tprof[idx]))+(0.05*((Pprof[idx]*1e-5)-1.0)/Tprof[idx])+self.dIW

                    fwm = suppf.melt_composition_factor(self) 

                    self.Kf = 10.0**(self.KI+self.KII+self.fO2)
                    self.X_carbonate_melt[i] = self.Kf/(1.0+self.Kf) 
                    self.X_CO2_melt[i] =((44.01/fwm)*self.X_carbonate_melt[i])/(1.0-(1.0-(44.01/fwm))* self.X_carbonate_melt[i]) # with 44.01 representing the MM of CO2 (12+2*16)
                    self.X_CO2_melt[i] = min(self.X_CO2_melt[i],self.Xm_CO2[i-1]/F) # concentration in melt 
                    self.X_CO2_liq[i] = self.X_CO2_liq[i] + self.X_CO2_melt[i]*F*V # to be averaged concentration in melt
                    
                    
                    if self.partitioning_calc == 'no':
                        # bulk partition coefficients (single parameters from literature)
                        D_U_bulk = self.Ol/100*self.D_U_Ol + self.Opx/100*self.D_U_Opx + self.Cpx/100*self.D_U_Cpx_const + self.Grt/100*self.D_U_Grt
                        D_K_bulk = self.Ol/100*self.D_K_Ol + self.Opx/100*self.D_K_Opx + self.Cpx/100*self.D_K_Cpx_const + self.Grt/100*self.D_K_Grt
                        D_Th_bulk = self.Ol/100*self.D_Th_Ol + self.Opx/100*self.D_Th_Opx + self.Cpx/100*self.D_Th_Cpx_const + self.Grt/100*self.D_Th_Grt
                        D_Ce_bulk = self.Ol/100*self.D_Ce_Ol + self.Opx/100*self.D_Ce_Opx + self.Cpx/100*self.D_Ce_Cpx_const + self.Grt/100*self.D_Ce_Grt 
                        D_La_bulk = self.Ol/100*self.D_La_Ol + self.Opx/100*self.D_La_Opx + self.Cpx/100*self.D_La_Cpx_const + self.Grt/100*self.D_La_Grt
                        D_Sm_bulk = self.Ol/100*self.D_Sm_Ol + self.Opx/100*self.D_Sm_Opx + self.Cpx/100*self.D_Sm_Cpx_const + self.Grt/100*self.D_Sm_Grt
                        D_Eu_bulk = self.Ol/100*self.D_Eu_Ol + self.Opx/100*self.D_Eu_Opx + self.Cpx/100*self.D_Eu_Cpx_const + self.Grt/100*self.D_Eu_Grt
                        D_Lu_bulk = self.Ol/100*self.D_Lu_Ol + self.Opx/100*self.D_Lu_Opx + self.Cpx/100*self.D_Lu_Cpx_const + self.Grt/100*self.D_Lu_Grt 

                        # heat producing elements 
                        self.X_U238_liq[i] = self.X_U238_liq[i] + (self.Xm_U238[i-1]/F)*(1-(1-F)**(1/D_U_bulk))*F*V 
                        self.X_U235_liq[i] = self.X_U235_liq[i] +(self.Xm_U235[i-1]/F)*(1-(1-F)**(1/D_U_bulk))*F*V
                        self.X_K40_liq[i] = self.X_K40_liq[i] + (self.Xm_K40[i-1]/F)*(1-(1-F)**(1/D_K_bulk))*F*V
                        self.X_Th232_liq[i] = self.X_Th232_liq[i] + (self.Xm_Th232[i-1]/F)*(1-(1-F)**(1/D_Th_bulk))*F*V 
                        self.X_H2O_liq[i] = self.X_H2O_liq[i] + (self.Xm_H2O[i-1]/F)*(1-(1-F)**(1/D_Ce_bulk))*F*V #concentration of water in the melt
                        self.X_K_liq[i] = self.X_K_liq[i] + (self.Xm_K[i-1]/F)*(1-(1-F)**(1/D_K_bulk))*F*V
                        self.X_La_liq[i] = self.X_La_liq[i] + (self.Xm_La[i-1]/F)*(1-(1-F)**(1/D_La_bulk))*F*V
                        self.X_Ce_liq[i] = self.X_Ce_liq[i] + (self.Xm_Ce[i-1]/F)*(1-(1-F)**(1/D_Ce_bulk))*F*V
                        self.X_Sm_liq[i] = self.X_Sm_liq[i] + (self.Xm_Sm[i-1]/F)*(1-(1-F)**(1/D_Sm_bulk))*F*V
                        self.X_Eu_liq[i] = self.X_Eu_liq[i] + (self.Xm_Eu[i-1]/F)*(1-(1-F)**(1/D_Eu_bulk))*F*V
                        self.X_Lu_liq[i] = self.X_Lu_liq[i] + (self.Xm_Lu[i-1]/F)*(1-(1-F)**(1/D_Lu_bulk))*F*V
                        
                    else:
                        # Partition coefficient calculations for HPE, Ce (here: implicity H2O), La, Sm, Eu, Lu
                        mol_ox = np.divide(self.wt_perc_comp, self.wt_perc_molm )
                        oxy = np.multiply(mol_ox, self.n_oxy)
                        S_ox = sum(self.n_oxy)  #Sum of mole oxygen
                        cat = np.multiply(mol_ox, self.n_cat)
                        S_cat = sum(self.n_cat) #sum of mole cations
                
                        norm = 6  #calculated to 6 oxygens
                        mol2 = np.multiply(norm,self.n_cat)
                        mol = (norm*cat)/S_ox
                
                        JD = mol[8]; #NaAlSi2O6, Na 
                        KT = mol[9]; #KAlSi2O6, K
                        CT = mol[1]; #CaTiSi2O6, Ti 
                        CATS = mol[2] - 2*CT - JD - KT; #CaAl2SiO6, Al - 2Ti - Na - K
                        DI = mol[7] - CT; #Ca(Mg,Fe,Mn)Si2O6, Ca-CaAl2SiO6-CT
                        OL = 1/3*(mol[1] + mol[4] + mol[5] - DI); #(Mg,Fe,Mn)3Si1.5O6, 1/3(Mg + Fe + Mn - Ca(Mg,Fe,Mn)Si2O6
                        QZ = 1/3*(mol[0] - 2*KT - 2*JD - 2*DI - CATS - 1.5*OL); #Si3O6, 1/3*(Si - 2*K - 2*Na - 2Ca(Mg,Fe,Mn)Si2O6 - CaAl2SiO6 - 1.5*(Mg,Fe,Mn)3Si1.5O6)
                    
                        sum_melt = JD+KT+CT+CATS+DI+OL+QZ
                        JD   = JD/sum_melt
                        KT   = KT/sum_melt
                        CT   = CT/sum_melt
                        CATS = CATS/sum_melt
                        DI   = DI/sum_melt
                        OL   = OL/sum_melt
                        QZ   = QZ/sum_melt
                        
                        #mole fraction of NaAlSi2O6 in the crystal
                        X_Jd = JD*np.exp((10367 + 2100*Pprof[idx]/1e9 - 165*(Pprof[idx]/1e9)**2)/Tprof[idx] - 10.27 + 0.358*Pprof[idx]/1e9 - 0.0184*(Pprof[idx]/1e9)**2)
                        
                        #Ti content of cpx on M1 site 
                        X_Ti = CT*(0.374 + 1.5*X_Jd)
                        
                        #Cr in crystal M1 site
                        X_Cr = 5*mol[3]
                        
                        #Al in crytal M1 site
                        X_Al = mol[2]
                        
                        #calculate Mg_cpx and magnesium number Mg_nr
                        Mg_nr = self.wt_perc_comp[6]/(self.wt_perc_comp[6] + self.wt_perc_comp[4])
                        A = Mg_nr/(1 - Mg_nr)
                        B = 0.108 + 0.323*Mg_nr
                        Mg_cpx = A/(A + B)
                        
                        #activity of CaAl2SiO6 component in cpx
                        a_cats = (CT + CATS + DI)*(CATS + JD + KT)*np.exp((76469 - 62.078*Tprof[idx] + 12430*Pprof[idx]/1e9 - 870*(Pprof[idx]/1e9)**2)/(self.R_gas*Tprof[idx]))
                        
                        if a_cats > 0.4:
                            a_cats = 0.4
                        else:
                            a_cats
                        
                        #activity of CaMgSi2O6
                        a_Di = DI*Mg_nr*np.exp((132650 - 82.152*Tprof[idx] + 13860*Pprof[idx]/1e9 - 1130*(Pprof[idx]/1e9)**2)/(self.R_gas*Tprof[idx]))
                        
                        #amount of Ca in M2 site
                        X_Ca = (a_cats*Mg_nr + a_Di)/(Mg_cpx*(1 - X_Cr - X_Ti))
                        
                        #Mg on M1 site
                        X_Mg = Mg_nr*(1 - X_Cr - X_Ti - (a_cats/X_Ca))
                        
                        # Cpx partition coefficients
                        D_Na = np.exp(((2508 + 2333*Pprof[idx]/1e9 - 138.5*Pprof[idx]/1e9**2)/Tprof[idx]) - 4.514 - 0.4791*Pprof[idx]/1e9 + 0.0415*(Pprof[idx]/1e9)**2) # "reference coefficient", Eq 25 along solidus, Schmidt and Noack
                       # D_Na = np.exp(((2183 + 2517*Pprof[idx]/1e9 - 157*Pprof[idx]/1e9**2)/Tprof[idx]) - 4.575 - 0.5149*Pprof[idx]/1e9 + 0.0475*(Pprof[idx]/1e9)**2) # "reference coefficient", Eq 24 for general use, Schmidt and Noack
                             
                        # K
                        R0_K = (((0.974 + 0.067*X_Ca) - 0.051*a_cats/X_Ca)+0.12)*1e-10 #1+ = 3+ + 0.12
                        E_K = (318.6 + 6.9*Pprof[idx]/1e9 - 0.036*Tprof[idx])/3 #GPa
                        self.D_K_Cpx[i] = D_Na*np.exp((-4*np.pi*self.NA*E_K*1e9*(0.5*R0_K*(self.R_Na**2-self.R_K**2) + 1/3*(self.R_K**3 - self.R_Na**3)))/(self.R_gas*Tprof[idx]))  #Wood 2014 1+ charge ion
                        self.D_K[i] = self.Ol/100*self.D_K_Ol + self.Opx/100*self.D_K_Opx + self.Cpx/100*self.D_K_Cpx[i] + self.Grt/100*self.D_K_Grt
                        
                        self.D_K_Cpx_min[i] = self.D_K_Cpx[i]
                        if (self.D_K_Cpx[i]<self.D_K_Cpx_min[i]):
                            self.D_Th_Cpx_min[i] = self.D_K_Cpx[i]
                    
                         # min/max for D_bulk
                        self.D_K_min[i] = self.D_K[i]
                        if (self.D_K[i]<self.D_K_min[i]):
                            self.D_K_min[i] = self.D_K[i]
                            
                        self.D_K_mean[i] = self.D_K_mean[i] + self.D_K[i]*F*V
                        
                        if (self.D_K[i]>self.D_K_max[i]):
                            self.D_K_max[i] = self.D_K[i]

                        
                        # Th
                        R0_Th = (((0.974 + 0.067*X_Ca) - 0.051*a_cats/X_Ca))*1e-10 
                        E_Th = (318.6 + 6.9*Pprof[idx]/1e9 - 0.036*Tprof[idx])*(2/3)
                        gamma_Th= np.exp((4*np.pi*E_Th*self.NA)/(self.R_gas*Tprof[idx])*(R0_Th/2*(self.R_Th - R0_Th)**2 + 1/3*(self.R_Th - R0_Th)**3))
                        gamma_Mg = np.exp(902*(1 - X_Mg)**2/Tprof[idx])
                        self.D_Th_Cpx[i] = Mg_nr/(gamma_Th*X_Mg*gamma_Mg)*np.exp((214790 - 175.7*Tprof[idx] + 16420*Pprof[idx]/1e9-1500*(Pprof[idx]/1e9)**2)/(self.R_gas*Tprof[idx]))
                        self.D_Th[i] = self.Ol/100*self.D_Th_Ol + self.Opx/100*self.D_Th_Opx + self.Cpx/100*self.D_Th_Cpx[i] + self.Grt/100*self.D_Th_Grt
                        
                        
                        self.D_Th_Cpx_min[i]=self.D_Th_Cpx[i]
                        if (self.D_Th_Cpx[i]<self.D_Th_Cpx_min[i]):
                            self.D_Th_Cpx_min[i] = self.D_Th_Cpx[i]
                            
                        self.D_Th_Cpx_mean[i] = self.D_Th_Cpx_mean[i] + self.D_Th_Cpx[i]*F*V
                        
                        if (self.D_Th_Cpx[i]>self.D_Th_Cpx_max[i]):
                            self.D_Th_Cpx_max[i] = self.D_Th_Cpx[i]
                        
                        # min/max for D_bulk
                        self.D_Th_min[i] = self.D_Th[i]
                        if (self.D_Th[i]<self.D_Th_min[i]):
                            self.D_Th_min[i] = self.D_Th[i]
                            
                        self.D_Th_mean[i] = self.D_Th_mean[i] + self.D_Th[i]*F*V
                        
                        if (self.D_Th[i]>self.D_Th_max[i]):
                            self.D_Th_max[i] = self.D_Th[i]
                        
                        # U
                        R0_U = (((0.974 + 0.067*X_Ca) - 0.051*a_cats/X_Ca))*1e-10 
                        E_U = (318.6 + 6.9*Pprof[idx]/1e9 - 0.036*Tprof[idx])*(2/3)
                        self.D_U_Cpx[i] = (self.D_Th_Cpx[i]*np.exp((-4*np.pi*self.NA*E_U*1e9*(0.5*R0_U*(self.R_Th**2-self.R_U**2) + 1/3*(self.R_U**3 - self.R_Th**3)))/(self.R_gas*Tprof[idx])))
                        self.D_U[i] = self.Ol/100*self.D_U_Ol + self.Opx/100*self.D_U_Opx + self.Cpx/100*self.D_U_Cpx[i] + self.Grt/100*self.D_U_Grt
                       
                        # min/max for D_Cpx
                        self.D_U_Cpx_min[i]=self.D_U_Cpx[i]
                        if (self.D_U_Cpx[i]<self.D_U_Cpx_min[i]):
                            self.D_U_Cpx_min[i] = self.D_U_Cpx[i]

                        self.D_U_Cpx_mean[i] = self.D_U_Cpx_mean[i] + self.D_U_Cpx[i]*F*V

                        if (self.D_U_Cpx[i]>self.D_U_Cpx_max[i]):
                            self.D_U_Cpx_max[i] = self.D_U_Cpx[i]

                        # min/max D_bulk
                        self.D_U_min[i]=self.D_U[i]
                        if (self.D_U[i]<self.D_U_min[i]):
                            self.D_U_min[i] = self.D_U[i]

                        self.D_U_mean[i] = self.D_U_mean[i] + self.D_U[i]*F*V
                    
                      #  self.D_U_max[i]=self.D_U[i]
                        if (self.D_U[i]>self.D_U_max[i]):
                            self.D_U_max[i] = self.D_U[i]
                            
                        
                        # Ce (H2O)
                        R0_Ce = (((0.974 + 0.067*X_Ca) - 0.051*a_cats/X_Ca))*1e-10     #3+
                        E_Ce = (318.6 + 6.9*Pprof[idx]/1e9 - 0.036*Tprof[idx])
                        self.D_Ce_Cpx[i] = D_Na*np.exp((-4*np.pi*self.NA*E_Ce*1e9*(0.5*R0_Ce*(self.R_Ce-R0_Ce)**2 + 1/3*(self.R_Ce-R0_Ce)**3))/(self.R_gas*Tprof[idx]))
                        self.D_Ce[i] = self.Ol/100*self.D_Ce_Ol + self.Opx/100*self.D_Ce_Opx + self.Cpx/100*self.D_Ce_Cpx[i] + self.Grt/100*self.D_Ce_Grt

                       
                        # min/max for D_Cpx
                        self.D_Ce_Cpx_min[i]=self.D_Ce_Cpx[i]
                        if (self.D_Ce_Cpx[i]<self.D_Ce_Cpx_min[i]):
                            self.D_Ce_Cpx_min[i] = self.D_Ce_Cpx[i]

                        self.D_Ce_Cpx_mean[i] = self.D_Ce_Cpx_mean[i] + self.D_Ce_Cpx[i]*F*V

                        if (self.D_Ce_Cpx[i]>self.D_Ce_Cpx_max[i]):
                            self.D_Ce_Cpx_max[i] = self.D_Ce_Cpx[i]


                        # min/max D_bulk
                        self.D_Ce_min[i]=self.D_Ce[i]
                        if (self.D_Ce[i]<self.D_Ce_min[i]):
                            self.D_Ce_min[i] = self.D_Ce[i]

                        self.D_Ce_mean[i] = self.D_Ce_mean[i] + self.D_Ce[i]*F*V 


                        if (self.D_Ce[i]>self.D_Ce_max[i]):
                            self.D_Ce_max[i] = self.D_Ce[i]
  
                        
                        # REE without Ce (all 3+ & VIII coordinated)
                        R0_3 = (((0.974 + 0.067*X_Ca) - 0.051*a_cats/X_Ca))*1e-10     #3+
                        E_3 = (318.6 + 6.9*Pprof[idx]/1e9 - 0.036*Tprof[idx])
                        self.D_La_Cpx[i] = D_Na*np.exp((-4*np.pi*self.NA*E_3*1e9*(0.5*R0_3*(self.R_La-R0_3)**2 + 1/3*(self.R_La-R0_3)**3))/(self.R_gas*Tprof[idx]))
                        self.D_La[i] = self.Ol/100*self.D_La_Ol + self.Opx/100*self.D_La_Opx + self.Cpx/100*self.D_La_Cpx[i] + self.Grt/100*self.D_La_Grt

                       
                        # min/max for D_Cpx
                        self.D_La_Cpx_min[i]=self.D_La_Cpx[i]
                        if (self.D_La_Cpx[i]<self.D_La_Cpx_min[i]):
                            self.D_La_Cpx_min[i] = self.D_La_Cpx[i]

                        self.D_La_Cpx_mean[i] = self.D_La_Cpx_mean[i] + self.D_La_Cpx[i]*F*V

                        if (self.D_La_Cpx[i]>self.D_La_Cpx_max[i]):
                            self.D_La_Cpx_max[i] = self.D_La_Cpx[i]

                        # min/max D_bulk
                        self.D_La_min[i]=self.D_La[i]
                        if (self.D_La[i]<self.D_La_min[i]):
                            self.D_La_min[i] = self.D_La[i]

                        self.D_La_mean[i] = self.D_La_mean[i] + self.D_La[i]*F*V 

                        if (self.D_La[i]>self.D_La_max[i]):
                            self.D_La_max[i] = self.D_La[i]
                        
                        # Sm
                        self.D_Sm_Cpx[i] = D_Na*np.exp((-4*np.pi*self.NA*E_3*1e9*(0.5*R0_3*(self.R_Sm-R0_3)**2 + 1/3*(self.R_Sm-R0_3)**3))/(self.R_gas*Tprof[idx]))
                        self.D_Sm[i] = self.Ol/100*self.D_Sm_Ol + self.Opx/100*self.D_Sm_Opx + self.Cpx/100*self.D_Sm_Cpx[i] + self.Grt/100*self.D_Sm_Grt

                       
                        # min/max for D_Cpx
                        self.D_Sm_Cpx_min[i]=self.D_Sm_Cpx[i]
                        if (self.D_Sm_Cpx[i]<self.D_Sm_Cpx_min[i]):
                            self.D_Sm_Cpx_min[i] = self.D_Sm_Cpx[i]

                        self.D_Sm_Cpx_mean[i] = self.D_Sm_Cpx_mean[i] + self.D_Sm_Cpx[i]*F*V

                        if (self.D_Sm_Cpx[i]>self.D_Sm_Cpx_max[i]):
                            self.D_Sm_Cpx_max[i] = self.D_Sm_Cpx[i]

                        # min/max D_bulk
                        self.D_Sm_min[i]=self.D_Sm[i]
                        if (self.D_Sm[i]<self.D_Sm_min[i]):
                            self.D_Sm_min[i] = self.D_Sm[i]

                        self.D_Sm_mean[i] = self.D_Sm_mean[i] + self.D_Sm[i]*F*V 

                        if (self.D_Sm[i]>self.D_Sm_max[i]):
                            self.D_Sm_max[i] = self.D_Sm[i]
                            
                            
                         # Eu
                        self.D_Eu_Cpx[i] = D_Na*np.exp((-4*np.pi*self.NA*E_3*1e9*(0.5*R0_3*(self.R_Eu-R0_3)**2 + 1/3*(self.R_Eu-R0_3)**3))/(self.R_gas*Tprof[idx]))
                        self.D_Eu[i] = self.Ol/100*self.D_Eu_Ol + self.Opx/100*self.D_Eu_Opx + self.Cpx/100*self.D_Eu_Cpx[i] + self.Grt/100*self.D_Eu_Grt

                       
                        # min/max for D_Cpx
                        self.D_Eu_Cpx_min[i]=self.D_Eu_Cpx[i]
                        if (self.D_Eu_Cpx[i]<self.D_Eu_Cpx_min[i]):
                            self.D_Eu_Cpx_min[i] = self.D_Eu_Cpx[i]

                        self.D_Eu_Cpx_mean[i] = self.D_Eu_Cpx_mean[i] + self.D_Eu_Cpx[i]*F*V

                        if (self.D_Eu_Cpx[i]>self.D_Eu_Cpx_max[i]):
                            self.D_Eu_Cpx_max[i] = self.D_Eu_Cpx[i]

                        # min/max D_bulk
                        self.D_Eu_min[i]=self.D_Eu[i]
                        if (self.D_Eu[i]<self.D_Eu_min[i]):
                            self.D_Eu_min[i] = self.D_Eu[i]

                        self.D_Eu_mean[i] = self.D_Eu_mean[i] + self.D_Eu[i]*F*V 

                        if (self.D_Eu[i]>self.D_Eu_max[i]):
                            self.D_Eu_max[i] = self.D_Eu[i]
                            
                        # Lu
                        self.D_Lu_Cpx[i] = D_Na*np.exp((-4*np.pi*self.NA*E_3*1e9*(0.5*R0_3*(self.R_Lu-R0_3)**2 + 1/3*(self.R_Lu-R0_3)**3))/(self.R_gas*Tprof[idx]))
                        self.D_Lu[i] = self.Ol/100*self.D_Lu_Ol + self.Opx/100*self.D_Lu_Opx + self.Cpx/100*self.D_Lu_Cpx[i] + self.Grt/100*self.D_Lu_Grt

                       
                        # min/max for D_Cpx
                        self.D_Lu_Cpx_min[i]=self.D_Lu_Cpx[i]
                        if (self.D_Lu_Cpx[i]<self.D_Lu_Cpx_min[i]):
                            self.D_Lu_Cpx_min[i] = self.D_Lu_Cpx[i]

                        self.D_Lu_Cpx_mean[i] = self.D_Lu_Cpx_mean[i] + self.D_Lu_Cpx[i]*F*V

                        if (self.D_Lu_Cpx[i]>self.D_Lu_Cpx_max[i]):
                            self.D_Lu_Cpx_max[i] = self.D_Lu_Cpx[i]

                        # min/max D_bulk
                        self.D_Lu_min[i]=self.D_Lu[i]
                        if (self.D_Lu[i]<self.D_Lu_min[i]):
                            self.D_Lu_min[i] = self.D_Lu[i]

                        self.D_Lu_mean[i] = (self.D_Lu_mean[i] + self.D_Lu[i]*F*V) 
                      
                        if (self.D_Lu[i]>self.D_Lu_max[i]):
                            self.D_Lu_max[i] = self.D_Lu[i]
                     
                        # total amount of HPE and H2O inside the melt at each time step; Partition coefficients calculated in partitioning_functions
                        self.X_U238_liq[i] = self.X_U238_liq[i] +(self.Xm_U238[i-1]/F)*(1-(1-F)**(1/self.D_U[i]))*F*V
                        self.X_U235_liq[i] = self.X_U235_liq[i] +(self.Xm_U235[i-1]/F)*(1-(1-F)**(1/self.D_U[i]))*F*V
                        self.X_K40_liq[i] = self.X_K40_liq[i] + (self.Xm_K40[i-1]/F)*(1-(1-F)**(1/self.D_K[i]))*F*V
                        self.X_Th232_liq[i] = self.X_Th232_liq[i] + (self.Xm_Th232[i-1]/F)*(1-(1-F)**(1/self.D_Th[i]))*F*V 
                        self.X_H2O_liq[i] = self.X_H2O_liq[i] +(self.Xm_H2O[i-1]/F)*(1-(1-F)**(1/self.D_Ce[i]))*F*V 
                        self.X_K_liq[i] = self.X_K_liq[i] +(self.Xm_K[i-1]/F)*(1-(1-F)**(1/self.D_K[i]))*F*V
                        self.X_La_liq[i] = self.X_La_liq[i] +(self.Xm_La[i-1]/F)*(1-(1-F)**(1/self.D_La[i]))*F*V
                        self.X_Ce_liq[i] = self.X_Ce_liq[i] +(self.Xm_Ce[i-1]/F)*(1-(1-F)**(1/self.D_Ce[i]))*F*V
                        self.X_Sm_liq[i] = self.X_Sm_liq[i] +(self.Xm_Sm[i-1]/F)*(1-(1-F)**(1/self.D_Sm[i]))*F*V
                        self.X_Eu_liq[i] = self.X_Eu_liq[i] +(self.Xm_Eu[i-1]/F)*(1-(1-F)**(1/self.D_Eu[i]))*F*V
                        self.X_Lu_liq[i] = self.X_Lu_liq[i] +(self.Xm_Lu[i-1]/F)*(1-(1-F)**(1/self.D_Lu[i]))*F*V
          

            if (Vmelt > 0):  
                ma =(1/Vmeltzone *Vmelt)  
                # update amount of elements to average concentration of HPE over entire melt zone
                self.X_U238_liq[i] = self.X_U238_liq[i]/Vmelt
                self.X_U235_liq[i] = self.X_U235_liq[i]/Vmelt
                self.X_K40_liq[i] = self.X_K40_liq[i]/Vmelt
                self.X_Th232_liq[i] = self.X_Th232_liq[i]/Vmelt
                self.X_CO2_liq[i] = self.X_CO2_liq[i]/Vmelt
                self.X_H2O_liq[i] = self.X_H2O_liq[i]/Vmelt #average amount of melt over the entire melt zone 
                self.X_K_liq[i] = self.X_K_liq[i]/Vmelt
                self.X_La_liq[i] = self.X_La_liq[i]/Vmelt
                self.X_Ce_liq[i] = self.X_Ce_liq[i]/Vmelt
                self.X_Sm_liq[i] = self.X_Sm_liq[i]/Vmelt
                self.X_Eu_liq[i] = self.X_Eu_liq[i]/Vmelt
                self.X_Lu_liq[i] = self.X_Lu_liq[i]/Vmelt
                
                
                self.F_av[i] = Vmelt/Vmeltzone  # averaged melt fraction in the melt zone at each time step
               
                
                self.D_U_Cpx_mean[i] =  self.D_U_mean[i]/Vmelt  # averaged mean partition coefficient
                self.D_K_Cpx_mean[i] =  self.D_K_mean[i]/Vmelt
                self.D_Th_Cpx_mean[i] =  self.D_Th_mean[i]/Vmelt
                self.D_Ce_Cpx_mean[i] =  self.D_Ce_mean[i]/Vmelt
                self.D_La_Cpx_mean[i] =  self.D_La_mean[i]/Vmelt
                self.D_Sm_Cpx_mean[i] =  self.D_Sm_mean[i]/Vmelt
                self.D_Eu_Cpx_mean[i] =  self.D_Eu_mean[i]/Vmelt
                self.D_Lu_Cpx_mean[i] =  self.D_Lu_mean[i]/Vmelt
                
                self.D_U_mean[i] =  self.D_U_mean[i]/Vmelt  
                self.D_K_mean[i] =  self.D_K_mean[i]/Vmelt
                self.D_Th_mean[i] =  self.D_Th_mean[i]/Vmelt
                self.D_Ce_mean[i] =  self.D_Ce_mean[i]/Vmelt
                self.D_La_mean[i] =  self.D_La_mean[i]/Vmelt
                self.D_Sm_mean[i] =  self.D_Sm_mean[i]/Vmelt
                self.D_Eu_mean[i] =  self.D_Eu_mean[i]/Vmelt
                self.D_Lu_mean[i] =  self.D_Lu_mean[i]/Vmelt
                 
                    
            # total mass for each species HPE in secondary crust
            if self.tectonics=='SL':
                self.Mscr_U238[i] = self.Mscr_U238[i-1] + self.X_U238_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
                self.Mscr_U235[i] = self.Mscr_U235[i-1] + self.X_U235_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
                self.Mscr_Th232[i]= self.Mscr_Th232[i-1] + self.X_Th232_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
                self.Mscr_K40[i]  = self.Mscr_K40[i-1] + self.X_K40_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
                self.Mscr_H2O[i]  = self.Mscr_H2O[i-1] + self.X_H2O_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
                self.Mscr_CO2[i]  = self.Mscr_CO2[i-1] + self.X_CO2_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
                self.Mscr_K[i]  = self.Mscr_K[i-1] + self.X_K_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
                self.Mscr_La[i]  = self.Mscr_La[i-1] + self.X_La_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
                self.Mscr_Ce[i]  = self.Mscr_Ce[i-1] + self.X_Ce_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
                self.Mscr_Sm[i]  = self.Mscr_Sm[i-1] + self.X_Sm_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
                self.Mscr_Eu[i]  = self.Mscr_Eu[i-1] + self.X_Eu_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
                self.Mscr_Lu[i]  = self.Mscr_Lu[i-1] + self.X_Lu_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
            else: 
                self.Mscr_U238[i] = self.X_U238_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
                self.Mscr_U235[i] = self.X_U235_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
                self.Mscr_Th232[i]= self.X_Th232_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
                self.Mscr_K40[i]  = self.X_K40_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
                self.Mscr_H2O[i]  = self.X_H2O_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
                self.Mscr_CO2[i]  = self.X_CO2_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr 
                self.Mscr_K[i]  = self.X_K_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
                self.Mscr_La[i]  = self.X_La_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
                self.Mscr_Ce[i]  = self.X_Ce_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
                self.Mscr_Sm[i]  = self.X_Sm_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
                self.Mscr_Eu[i]  = self.X_Eu_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
                self.Mscr_Lu[i]  = self.X_Lu_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr


            # mantle convection velocity
            u_vel = self.u0 *(self.Ra[i]/self.Racrit)**(2*self.beta)      

            # Stefan number
            self.St[i] = (self.L/self.cm)*(dVmelt_dTm/Vcm)
            
            
            ###############################################################
            """Volatile outgassing
               - simplistic first-order approach using outgassing efficiency factor """
            ###############################################################
            
            self.M_degas_CO2_dt[i] = self.oeff_CO2*self.X_CO2_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
            self.M_degas_H2O_dt[i] = self.oeff_H2O*self.X_H2O_liq[i]*self.Dcr_prod[i-1]*(4*np.pi*self.Rcr[i-1]**2)*self.rhocr
            self.M_degas_H2_dt[i] = 0.0
            self.M_degas_CH4_dt[i] = 0.0
            self.M_degas_CO_dt[i] = 0.0
            
            #sum of all outgassed H2O & CO2, in magma outgassing code we calculate only outgassing with crystallization, not over time
            self.M_H2_gas[i] = self.M_H2_gas[i-1] +  self.M_degas_H2_dt[i]
            self.M_H2O_gas[i] = self.M_H2O_gas[i-1] +  self.M_degas_H2O_dt[i]  
            self.M_CH4_gas[i] = self.M_CH4_gas[i-1] +  self.M_degas_CH4_dt[i]
            self.M_CO_gas[i] = self.M_CO_gas[i-1] +  self.M_degas_CO_dt[i]
            self.M_CO2_gas[i] = self.M_CO2_gas[i-1] + self.M_degas_CO2_dt[i]
            
            # Mass of CO2 in secondary crust
            self.Mscr_H2O[i] = self.Mscr_H2O[i] - self.M_degas_H2O_dt[i] 
            self.Mscr_CO2[i] = self.Mscr_CO2[i] - self.M_degas_CO2_dt[i] 
            
            # CO2 partial pressure
            self.P_H2[i] = (self.M_H2_gas[i]*self.g)/(4*np.pi*self.Rp**2)
            self.P_CH4[i] = (self.M_CH4_gas[i]*self.g)/(4*np.pi*self.Rp**2)
            self.P_CO[i] = (self.M_CO_gas[i]*self.g)/(4*np.pi*self.Rp**2)
            self.P_CO2[i] = (self.M_CO2_gas[i]*self.g)/(4*np.pi*self.Rp**2)
            
            # equivalant global layer thickness (if all outgassed H2O condenses to an ocean)
            self.EGL[i] = (self.M_H2O_gas[i])/(4*np.pi*self.Rp**2*self.rho_H2O) 
                    
            #######################################################################
            """ Update boundary layer thicknesses"""
            #######################################################################

            # Crust production rate 
            self.Dcr_prod[i] = u_vel*Vmelt/(4*np.pi*self.Rp**2*2*D)*self.dt #m  # 2*D for whole-mantle convection (according to Schubert 1990: Rp, since there D=1/2 Rp)
        
            if self.tectonics == 'SL':
                self.Dcr[i] = self.Dcr[i-1] + self.Dcr_prod[i] #m

            else:
                self.Dcr[i] = self.Dcr_prod[i] #m

            self.Mcr[i] = 4./3.*np.pi*(self.Rp**3-(self.Rp-self.Dcr[i])**3)*self.rhocr
            self.Rcr[i] = self.Rp-self.Dcr[i] # crust radius
            self.depl[i] = (self.Rp**3 - (self.Rp - self.Dcr[i])**3)/(self.Rp**3 - self.Rc**3) # depletion
 
            # calculate lithosphere thickness
            if self.tectonics == 'SL': 
                # Lithosphere thickness (derivTrRl: derivatile from crust temperature and lithosphere thickness)
                derivTrRl = (-self.Qm[i]*self.rhom*self.Rl[i-1]/(3.0*self.km) -(self.Tl[i]-self.Tcr[i]-self.Qm[i]*self.rhom*( self.Rcr[i]**2-self.Rl[i-1]**2)/(6.0*self.km))/(self.Rl[i-1]**2.0*(1.0/self.Rl[i-1]-  1.0/ self.Rcr[i])) )


                self.Dl[i] = (self.Dl[i-1] + (self.dt * (-self.ql[i]+(self.rhocr*self.L+self.rhocr*self.ccr*(self.Tm[i]-self.Tl[i]))*self.Dcr_prod[i]/self.dt - self.km*derivTrRl)/(self.rhom*self.cm*(self.Tm[i]-self.Tl[i]))) )
                

                if self.crust_delamination == 'yes': # crust delamination through mantle convection
                    Dcr_old = self.Dcr[i]
                    if (Dcr_old == 0):
                        Dcr_old = 1.0
                    self.Dcr[i] = min(self.Dcr[i],self.Dl[i]-self.Dl_cr)  # with crustal delamination

                    Delam_ratio = (self.Rp**3-(self.Rp-self.Dcr[i])**3)/(self.Rp**3-(self.Rp-Dcr_old)**3)

                    self.Mscr_U238[i] = self.Mscr_U238[i]*Delam_ratio
                    self.Mscr_U235[i] = self.Mscr_U235[i]*Delam_ratio
                    self.Mscr_Th232[i]= self.Mscr_Th232[i]*Delam_ratio
                    self.Mscr_K40[i]  = self.Mscr_K40[i]*Delam_ratio
                    self.Mscr_H2O[i]  = self.Mscr_H2O[i]*Delam_ratio
                    self.Mscr_CO2[i]  = self.Mscr_CO2[i]*Delam_ratio
                    self.Mscr_K[i]  = self.Mscr_K[i]*Delam_ratio
                    self.Mscr_La[i]  = self.Mscr_La[i]*Delam_ratio
                    self.Mscr_Ce[i]  = self.Mscr_Ce[i]*Delam_ratio
                    self.Mscr_Sm[i]  = self.Mscr_Sm[i]*Delam_ratio
                    self.Mscr_Eu[i]  = self.Mscr_Eu[i]*Delam_ratio
                    self.Mscr_Lu[i]  = self.Mscr_Lu[i]*Delam_ratio

                    self.Rcr[i] = self.Rp-self.Dcr[i] 

                elif self.eclogite =='yes': # crust delamination through eclogite dripping: eclogite formation conditions set maximim crust thickness
                    Dcr_old = self.Dcr[i]
                    if (Dcr_old == 0):
                        Dcr_old = 1.0
                    Pcr = self.Dcr[i]*self.rhocr*self.g/1e9 #GPa
                    Dcr_max = 1.2e9/(self.rhocr*self.g)         #12e9 Pa for eclogite formation
                    P_Dcrmax = Dcr_max*self.rhocr*self.g/1e9
                    
                    if self.Tcr[i]>700:

                        self.Dcr[i] = min(self.Dcr[i],Dcr_max)

                        Delam_ratio = (self.Rp**3-(self.Rp-self.Dcr[i])**3)/(self.Rp**3-(self.Rp-Dcr_old)**3)

                        self.Mscr_U238[i] = self.Mscr_U238[i]*Delam_ratio
                        self.Mscr_U235[i] = self.Mscr_U235[i]*Delam_ratio
                        self.Mscr_Th232[i]= self.Mscr_Th232[i]*Delam_ratio
                        self.Mscr_K40[i]  = self.Mscr_K40[i]*Delam_ratio
                        self.Mscr_H2O[i]  = self.Mscr_H2O[i]*Delam_ratio
                        self.Mscr_CO2[i]  = self.Mscr_CO2[i]*Delam_ratio
                        self.Mscr_K[i]  = self.Mscr_K[i]*Delam_ratio
                        self.Mscr_La[i]  = self.Mscr_La[i]*Delam_ratio
                        self.Mscr_Ce[i]  = self.Mscr_Ce[i]*Delam_ratio
                        self.Mscr_Sm[i]  = self.Mscr_Sm[i]*Delam_ratio
                        self.Mscr_Eu[i]  = self.Mscr_Eu[i]*Delam_ratio
                        self.Mscr_Lu[i]  = self.Mscr_Lu[i]*Delam_ratio

                        self.Rcr[i] = self.Rp-self.Dcr[i] 
                        self.Dl[i] = max(self.Dl[i],self.Dl_cr+self.Dcr[i])

                else:  # no crustal delamination 
                    self.Dl[i] = max(self.Dl[i],self.Dl_cr+self.Dcr[i])   
               

            else: #ML 
                self.Dl[i] = self.Dl_cr+self.Dcr[i]
               
            # Lithosphere Radius
            self.Rl[i] = self.Rp - self.Dl[i]

            # Mantle thickness
            self.Dm[i] = D-self.Dl[i]
            
            Mcm = 4./3.*np.pi*((self.Rl[i]**3-self.Rc**3)*self.rhom)     # convecting mantle mass
            Acm = 4.*np.pi*self.Rl[i]**2                                   # convecting mantle surface area                
            Vcm = 4./3.*np.pi*(self.Rl[0]**3-self.Rc**3)             # convecting mantle volume
            Vm = 4./3.*np.pi*(self.Rcr[i]**3-self.Rc**3)

            self.Mscr_HPE[i] = self.Mscr_U238[i] + self.Mscr_U235[i] + self.Mscr_K40[i] + self.Mscr_Th232[i] # total mass of HPE in secondary crust


            # total mass in crust [kg]
            self.Mcr_U238[i] = self.Mcr_U238[i] + self.Mscr_U238[i]
            self.Mcr_U235[i] = self.Mcr_U235[i] + self.Mscr_U235[i] 
            self.Mcr_Th232[i]= self.Mcr_Th232[i] + self.Mscr_Th232[i] 
            self.Mcr_K40[i]  = self.Mcr_K40[i] + self.Mscr_K40[i]                                               
            self.Mcr_HPE[i] = self.Mcr_HPE[i] + self.Mscr_HPE[i]
            self.Mcr_H2O[i] = self.Mcr_H2O[i]  + self.Mscr_H2O[i]     # Wasser, welches mit der Schmelzbildung entfernt wird 
            self.Mcr_CO2[i] = self.Mcr_CO2[i] + self.Mscr_CO2[i]
            self.Mcr_K[i] = self.Mcr_K[i]  + self.Mscr_K[i] 
            self.Mcr_La[i] = self.Mcr_La[i] + self.Mscr_La[i] 
            self.Mcr_Ce[i] = self.Mcr_Ce[i] + self.Mscr_Ce[i] 
            self.Mcr_Sm[i] = self.Mcr_Sm[i] + self.Mscr_Sm[i] 
            self.Mcr_Eu[i] = self.Mcr_Eu[i] + self.Mscr_Eu[i]
            self.Mcr_Lu[i] = self.Mcr_Lu[i] + self.Mscr_Lu[i]

           
            # total mass in mantle [kg]
            self.Mm_U238[i] = X0_U238*Mm - self.Mcr_U238[i]
            self.Mm_U235[i] = X0_U235*Mm - self.Mcr_U235[i]
            self.Mm_Th232[i] = X0_Th232*Mm - self.Mcr_Th232[i]
            self.Mm_K40[i] = X0_K40*Mm - self.Mcr_K40[i]
            self.Mm_HPE[i] = X0_HPE*Mm - self.Mcr_HPE[i]
            self.Mm_K[i]   = X0_K*Mm - self.Mcr_K[i] 
            self.Mm_La[i]   = self.X0_La*Mm - self.Mcr_La[i] 
            self.Mm_Ce[i]   = self.X0_Ce*Mm - self.Mcr_Ce[i] 
            self.Mm_Sm[i]   = self.X0_Sm*Mm - self.Mcr_Sm[i] 
            self.Mm_Eu[i]   = self.X0_Eu*Mm - self.Mcr_Eu[i] 
            self.Mm_Lu[i]   = self.X0_Lu*Mm - self.Mcr_Lu[i]
            
            if (self.tectonics=='SL'):
                self.Mm_H2O[i] = self.X0_H2O*Mm - self.Mcr_H2O[i]
                self.Mm_CO2[i] = self.X0_CO2*Mm - self.Mcr_CO2[i]
            else:
                self.Mm_H2O[i] = self.X0_H2O*Mm - self.M_degas_H2O_dt[i]
                self.Mm_CO2[i] = self.X0_CO2*Mm - self.M_degas_CO2_dt[i]
                
                
            self.Mm_evol[i] = 4./3.*np.pi*((self.Rp-self.Dcr[i])**3-self.Rc**3)*self.rhom   # mantle mass without crust

            # relative concentration in mantle (weight fraction)
            self.Xm_U238[i] = self.Mm_U238[i]/self.Mm_evol[i]
            self.Xm_U235[i] = self.Mm_U235[i]/self.Mm_evol[i]
            self.Xm_Th232[i] =self.Mm_Th232[i]/self.Mm_evol[i]
            self.Xm_K40[i] = self.Mm_K40[i]/self.Mm_evol[i]
            self.Xm_H2O[i] = self.Mm_H2O[i]/self.Mm_evol[i]
            self.Xm_CO2[i] = self.Mm_CO2[i]/self.Mm_evol[i]
            self.Xm_HPE[i] = self.Mm_HPE[i]/self.Mm_evol[i]
            self.Xm_K[i] = self.Mm_K[i]/self.Mm_evol[i]
            self.Xm_La[i] = self.Mm_La[i]/self.Mm_evol[i]
            self.Xm_Ce[i] = self.Mm_Ce[i]/self.Mm_evol[i]
            self.Xm_Sm[i] = self.Mm_Sm[i]/self.Mm_evol[i]
            self.Xm_Eu[i] = self.Mm_Eu[i]/self.Mm_evol[i]
            self.Xm_Lu[i] = self.Mm_Lu[i]/self.Mm_evol[i]

            self.Xmantle_HPE[i] = self.Mm_HPE[i]/(self.Mm_HPE[i]+self.Mcr_HPE[i])
            self.Xmantle_H2O[i] = self.Mm_H2O[i]/(self.Mm_H2O[i]+self.Mcr_H2O[i]+self.M_H2O_gas[i])
            self.Xmantle_K[i] = self.Mm_K[i]/(self.Mm_K[i]+ self.Mcr_K[i])
            
            self.X_gas_H2O[i] = self.M_H2O_gas[i]/(self.Mm_H2O[i]+self.Mcr_H2O[i]+self.M_H2O_gas[i])

            # relative concentration of material that is in the crust vs in the mantle (weight fraction)
            self.Xcrust_U238[i] = self.Mcr_U238[i]/(self.Mm_U238[i]+self.Mcr_U238[i]) 
            self.Xcrust_U235[i] = self.Mcr_U235[i]/(self.Mm_U235[i]+self.Mcr_U235[i])
            self.Xcrust_Th232[i] = self.Mcr_Th232[i]/(self.Mm_Th232[i]+self.Mcr_Th232[i])
            self.Xcrust_K40[i] = self.Mcr_K40[i]/(self.Mm_K40[i]+self.Mcr_K40[i])
            self.Xcrust_H2O[i] = self.Mcr_H2O[i]/(self.Mm_H2O[i]+self.Mcr_H2O[i]+self.M_H2O_gas[i]) 
            self.Xcrust_CO2[i] = self.Mcr_CO2[i]/(self.Mm_CO2[i]+self.Mcr_CO2[i])
            self.Xcrust_K[i] = self.Mcr_K[i]/(self.Mm_K[i]+self.Mcr_K[i])
            self.Xcrust_HPE[i] = self.Mcr_HPE[i]/(self.Mm_HPE[i]+self.Mcr_HPE[i])

        # Write timeseries on file
        if not(outfile):
            print('no output file written')
            pass
        else:
            print('output written in ' + outfile, 'and outfile.xlsx')
            suppf.write_output_file(self, outfile)
                
    
