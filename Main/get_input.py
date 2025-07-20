import json
import numpy as np

#####################################################
"""
This code is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license." --> alles in Readme?

@author: Julia M. Schmidt
@date: 16.08.2023, last updated: 12.11.2024

Input values generated from 1D Interior Structure Code (@Lena Noack) after magma ocean (100% solidified mantle)
Inputs for 1D interior evolution code (@Julia Schmidt & Lena Noack), solar system rocky bodies and varying Earth-masses

"""

#####################################################
def get_input(self, body='Mars', inpfile='input.json'):
        """Set input parameters: 

        Rp:   planet radius (m)
        Rc:   core radius (m) 
        g:    surface gravity (m/s^2)          
        rhom: mantle density (kg/m^3)
        rhocr:crust density (kg/m^3)
        rhoc: core density (kg/m^3)
        rho_H2O: H2O density (kg/m^3)
        Ts:  surface temperature (K)      
        Tm0: initial mantle temperature (K)
        Tc0: initial core temperature (K) 
        V:   Activation volume in the upper mantle (m^3/mol)
        etaref: reference viscosity (Pa*s)
        cc:     Core heat capacity (J/(kg K))    
        cm:     Mantle heat capacity (J/(kg K)) 
        ccr:    Crust heat capacity (J/(kg K))
        alpha:  thermal expansivity (1/K)
        X_U:    bulk abundance of U (weight fraction)
        X_Th:   bulk abundance of Th(weight fraction)
        X_K:    bulk abundance of K (weight fraction)
        X0_H2O: initial water abundance in the mantle (weight fraction)
        X0_CO2: initial CO2 content in the mantle (weight frcaction)
        lam:    enrichment factor (starting enrichment of the crust)
        Qtidal: tidal heating
        L:      latent heat of melting (J/kg)
        epsm:   Ratio of mean and upper mantle temperature
        epsc:   Ratio of mean and upper core temperature
        Pcrossover: density crossover (Pa)
        dIW:    Iron-Wüstite buffer
        km:     Mantle thermal conductivity (W/(mK)) 
        kcr:    Crustal thermal conductivity (W/(mK)) 
        kc:     Core thermal conductivity (W/(mK))

        """
#####################################################

        if body == 'Mercury':

            self.name = 'Mercury'
            # Core radius, core density and mantle density from the representative model of Margot et al. (2019). In: Mercury - The view after MESSENGER. Ch. 4, 85-113, CUP.
            self.Rp   = 2440e3
            self.Rc   = 2024e3   
            self.g    = 4.2           # Ziercke (Master thesis) #3.7         
            self.rhom = 3295.0  
            self.rhocr = 2800.0       # Grott, Breuer, Laneuville 2011
            self.rhoc = 7034.0     
            self.rho_H2O = 1000.0
            self.Ts   = 440.0       
            self.Tm0  = 1650.        
            self.Tc0  = 1900. 
            self.V = 5e-6             # Activation volume in upper mantle (m^3/mol) (Karato & Wu for dry rheologies)
            self.etaref = 1e21 #1e19
            self.cc = 750.0           # Core heat capacity (J/(kg K))    # Breuer et al 2007: 750.0; Hauck et al 2004: 465 .0
            self.cm = 1297.0          # Mantle heat capacity (J/(kg K))  # Breuer et al 2007: 1297.0; Hauck et al 2004: 1212.0
            self.ccr = 1000.0         # Crust heat capacity (J/(kg K))
            self.alpha = 3.197e-5     # Ziercke (Master thesis)
            # Heat sources from Hauck et al. (2019). In: Mercury - The view after MESSENGER. Ch. 19, 516–543, CUP.
            self.X_U  = 25e-9    
            self.X_Th = 44e-9
            self.X_K  = 368e-6 
            self.X0_H2O  = 100e-6 # Mercury heavily reduced, so this is just case study parameter
            self.X0_CO2 = 50e-6       # initial CO2 content [weight frcaction]
            self.lam  = 3             # Tosi et al 2013: 2.5-4.5
            self.Qtidal = 0           # not found in literature yet, might need change
            self.L    = 6e5
            self.epsm = 1.0274        # Ziercke (Master thesis), Spohn 1990: 1.0 
            self.epsc = 1.0807        # Ziercke (Master thesis), Spohn 1990: 1.1 
            self.Pcrossover = 6.8e+9  # Vander Kaaden and McCubbin 2015
            self.dIW = -7 
            self.km = 1.5791          # Mantle thermal conductivity (W/(mK)) (Ziercke (Master thesis)), Morschhauser 2011: 4.0
            self.kcr = 3.0            # Crustal thermal conductivity (W/(mK)), Morschhauser 2011
            self.kc = 50              # Core thermal conductivity, Ziercke (Master thesis)
    
        elif body == 'Venus':
            
            self.name = 'Venus'
            self.Rp   = 6050e3
            self.Rc   = 3186e3
            self.g    = 8.87         
            self.rhom = 4400.0          
            self.rhoc = 10100.0   
            self.rhocr = 2900.0
            self.rho_H2O = 1000.0
            self.Ts   = 730.0       
            self.Tm0  = 1700      
            self.Tc0  = 4200.
            self.V = 5e-6             # Activation volume in upper mantle (m^3/mol) (Karato & Wu for dry rheologies)
            self.etaref = 1e20         
            self.cc = 850.0           # Core heat capacity (J/(kg K))    #O'Rourke & Korenaga 2015: 850.0
            self.cm = 1200.0          # Mantle heat capacity (J/(kg K))  #O'Rourke & Korenaga 2015: 1200.0 
            self.ccr = 1000.0         # Crust heat capacity (J/(kg K))
            self.alpha = 2.072e-5     # thermal expansion coefficient, Sonjas MA
            # Heat sources from Kaula (Icarus, 1999)
            self.X_U  = 21e-9
            self.X_Th = 86e-9 
            self.X_K  = 153e-6 
            self.X0_H2O  = 700e-6            # assumption: similar starting conditions than Earth; Kulikov 2006, Planetary and Space Science: *one hypotheses, that inventory was similar to Earth'
            self.X0_CO2 = 50e-6 #1e-3        # initial CO2 content [weight frcaction]
            self.lam  = 3
            self.Qtidal = 0
            self.L    = 6e5
            self.epsm = 1.2144         # Ziercke (Master thesis) 
            self.epsc = 1.1184         # Ziercke (Master thesis) 
            self.Pcrossover = 12e+9    # Ohtani et al. 1995
            self.dIW = 4               # buffer
            self.km = 4.6              # Ziercke (Master thesis), Morschhauser 2011: 4.0
            self.kcr = 3.0             # Crustal thermal conductivity (W/(mK)) Morschhauser 2011
            self.kc = 50               # Core thermal conductivity, Ziercke (Master thesis)

        elif body == 'Earth':

            self.name = 'Earth'
            self.Rp   = 6370e3
            self.Rc   = 3480e3
            self.g    = 9.8         
            self.rhom = 4460.0          
            self.rhoc = 10640.0 
            self.rhoCMB= 5500.0      # density post-perovskite
            self.rhocr = 2900.0
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1700          
            self.Tc0  = 4000.
            self.V = 5e-6             # Activation volume in upper mantle (m^3/mol) (Karato & Wu for dry rheologies)
            self.etaref = 1e20 #1e20
            self.cc = 800.0           # Core heat capacity (J/(kg K))  Stamenkovi´c & Breuer 2012  # O'Rourke & Korenaga 2015: 850.0
            self.cm = 1250.0          # Mantle heat capacity (J/(kg K))Stamenkovi´c & Breuer 2012   #O'Rourke & Korenaga 2015: 1200.0 
            self.ccr = 1000.0         # Crust heat capacity (J/(kg K))
            self.alpha = 1.979e-5     # thermal expansion coefficient, Ziercke (Master thesis)
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 700e-6 #estimate after Tosi et al 2017, Astronomy and Astrophysics: 500&1000 ppm taken, Peslier et al 2007: 330-1200ppm in primitive mantle
            self.X0_CO2 = 50e-6       # initial CO2 content [weight frcaction]
            self.lam  = 3 
            self.Qtidal = 0
            self.L    = 6e5
            self.epsm = 1.2393         # Ziercke (Master thesis)
            self.epsc = 1.137          # Ziercke (Master thesis)
            self.Pcrossover = 12e+9    # Ohtani et al. 1995
            self.dIW = 4               # buffer
            self.km = 5.1              # Ziercke (Master thesis), Mantle thermal conductivity (W/(mK)) Morschhauser 2011: 4.0 , 5.1 -> Sonja
            self.kcr = 3.0             # Crustal thermal conductivity (W/(mK)) Morschhauser 2011
            self.kc = 50               # Core thermal conductivity, Ziercke (Master thesis)
            
        elif body == 'Moon':

            self.name = 'Moon'
            self.Rp   = 1740e3
            self.Rc   = 330e3
            self.g    = 1.6         
            self.rhom = 3400.0          
            self.rhoc = 7400.0    
            self.rhocr = 2550.0       # Wieczorek, Neumann et al 2012 (Science)
            self.rho_H2O = 1000.0
            self.Ts   = 270.0       
            self.Tm0  = 1550.           
            self.Tc0  = 1998. 
            self.V = 5e-6             # Activation volume in upper mantle (m^3/mol) (Karato & Wu for dry rheologies)
            self.etaref = 1e19        # Laneuville, Taylor, Wieczorek 2018
            self.cc = 850.0           # Core heat capacity (J/(kg K))    # Laneuville, Taylor, Wieczorek 2018
            self.cm = 1000.0          # Mantle heat capacity (J/(kg K))  # Laneuville, Taylor, Wieczorek 2018
            self.ccr = 1000.0         # Crust heat capacity (J/(kg K))
            self.alpha = 3.291e-5     # Sonjas MA   
            # Heat sources from Taylor (1982). Planetary Science: A Lunar Perspective, LPI.
            self.X_U  = 33e-9
            self.X_Th = 125e-9
            self.X_K  = 82.5e-6 
            self.X0_H2O  = 300e-6      #Hauri 2015 water in the Moon's interior, upper end assumption for bulk silicate Moon (BSM)
            self.X0_CO2 = 50e-6        # initial CO2 content [weight frcaction]
            self.lam  = 3
            self.Qtidal = 0     
            self.L    = 3e5           # Laneuville, Taylor, Wieczorek 2018
            self.epsm = 1.0155        # Ziercke (Master thesis)
            self.epsc = 1.0065        # Ziercke (Master thesis)
            self.Pcrossover = 5e+9    # Ohtani et al. 1995 (close to center of the Moon)  
            self.dIW = 4              # buffer, for now Earth-like
            self.km = 1.4789          # Ziercke (Master thesis), Laneuville et al 2013: 3.0
            self.kcr = 1.5            # Crustal thermal conductivity (W/(mK)) Laneuville et al 2013
            self.kc = 50              # Core thermal conductivity # Sonja

            
        elif body == 'Mars':

            self.name = 'Mars'    
            self.Rp   = 3390e3
            self.Rc   = 1850e3        # Core radius from Plesa et al. (2018). Geophys. Res. Lett., 45(22), 12198-12209.  1850e3
            self.g    = 3.7         
            self.rhom = 3500.0 
            self.rhoCMB= 4400.0
            self.rhoc = 7200.0  
            self.rhocr= 2900.0
            self.rho_H2O=1000.0
            self.Ts   = 220.0       
            self.Tm0  = 1750            
            self.Tc0  = 2160.  
            self.V = 5e-6 
            self.etaref = 1e19 
            self.cc = 840.0           # Core heat capacity (J/(kg K)) Morschhauser 2011
            self.cm = 1142.0          # Mantle heat capacity (J/(kg K))  
            self.ccr = 1000.0         # Crust heat capacity (J/(kg K))
            self.alpha = 2.712e-5     # Ziercke (Master thesis), Morschhauser 2011: 2.5e-5
            # Heat sources from Waenke & Dreibus (1994). Philos. Trans. R. Soc. London, A349, 2134–2137.
            self.X_U  = 16e-9 
            self.X_Th = 56e-9
            self.X_K  = 305e-6 
            self.X0_H2O  = 300e-6     # initial water content [weight fraction]. Wänke & Dreibus: 36e-6 (assumption Grott2011: 100 ppm -> between 800(plume model)-2500(global melt layer model) ppm H2O in melt, Taylor 2013: min. 300 ppm
            self.X0_CO2 = 50e-6       # initial CO2 content [weight frcaction]
            self.lam  = 3             # Morschhauser 2011: 5  
            self.Qtidal = 0
            self.L    = 6e5           # latent heat of melting (J/kg)
            self.epsm = 1.0638        # Ziercke (Master thesis), Morschauser 2011: 1.0 
            self.epsc = 1.0514        # Ziercke (Master thesis), Morschauser 2011: 1.1       
            self.eps  = 0.6           # Katz et al.,2003
            self.chi_a = 12           # wt%/GPa, Katz et al.,2003
            self.chi_b = 1            # wt%/GPa, Katz et al.,2003
            self.Pcrossover = 7e+9    # Ohtani et al. 1995
            self.dIW = 1              # for Mars ~0.5-1 # buffer
            self.km = 4               # Mantle thermal conductivity (W/(mK)) Morschhauser 2011:4.0 ; Ziercke (Master thesis: 2.3033)
            self.kcr = 3              # Crustal thermal conductivity (W/(mK)) Morschhauser 2011:3, Plesa et al 2015: 2.3
            self.kc = 50              # Core thermal conductivity # Sonja

       
        elif body == '0.1_M_Earth':

            self.name = '0.1_M_Earth'
            self.Rp   = 3224e3
            self.Rc   = 1651e3   
            self.g    = 3.88        
            self.rhom = 3456.0  
            self.rhocr = 2900.0      
            self.rhoc = 9378.0     
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  =  1600.0       # assumed 
            self.Tc0  = 2636. 
            self.V = 5e-6          
            self.etaref = 1e20
            self.cc = 1008.0          
            self.cm = 1191.0         
            self.ccr = 1000.0     
            self.alpha = 2.594E-05   
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6  
            self.X0_CO2 = 50e-6
            self.lam  = 1 #3             
            self.Qtidal = 0          
            self.L    = 6e5 
            self.epsm = 1.219436693
            self.epsc = 1.043560597
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km = 2.24276        
            self.kcr = 3.0          
            self.kc = 50             

            
            
        elif body == '0.2_M_Earth':

            self.name = '0.2_M_Earth'
            self.Rp   = 3980e3
            self.Rc   = 2044e3   
            self.g    = 5.095        
            self.rhom = 3677.6
            self.rhocr = 2900.0      
            self.rhoc = 9894.0
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1600.            
            self.Tc0  = 3245.0
            self.V = 5e-6             
            self.etaref = 1e20
            self.cc = 1082.432857          
            self.cm = 1205.617274      
            self.ccr = 1000.0        
            self.alpha = 2.48E-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3         
            self.Qtidal = 0        
            self.L    = 6e5
            self.epsm = 1.060
            self.epsc = 1.058
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km = 2.6402      
            self.kcr = 3.0            
            self.kc = 50              

        elif body == '0.3_M_Earth':

            self.name = '0.3_M_Earth'
            self.Rp   = 4493e3
            self.Rc   = 2308e3   
            self.g    = 6.009        
            self.rhom = 3836.92
            self.rhocr = 2900.0      
            self.rhoc = 10305.67
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1600.            
            self.Tc0  = 3729.4
            self.V = 5e-6            
            self.etaref = 1e20
            self.cc = 1129.729    
            self.cm =1207.351
            self.ccr = 1000.0        
            self.alpha = 2.35e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3            
            self.Qtidal = 0         
            self.L    = 6e5
            self.epsm = 1.347
            self.epsc = 1.068
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km = 3.012        
            self.kcr = 3.0           
            self.kc = 50            
            
        elif body == '0.4_M_Earth':

            self.name = '0.4_M_Earth'
            self.Rp   = 4894e3
            self.Rc   = 2511e3   
            self.g    = 6.76817      
            self.rhom = 3955.77
            self.rhocr = 2900.0      
            self.rhoc = 10676.58
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1600.            
            self.Tc0  = 4103.45
            self.V = 5e-6           
            self.etaref = 1e20
            self.cc =  1151.62         
            self.cm = 1212.05       
            self.ccr = 1000.0        
            self.alpha = 2.28e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3          
            self.Qtidal = 0    
            self.L    = 6e5
            self.epsm = 1.283
            self.epsc = 1.075
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km = 3.311           
            self.kcr = 3.0        
            self.kc = 50         
            
        elif body == '0.5_M_Earth':

            self.name = '0.5_M_Earth'
            self.Rp   = 5227e3
            self.Rc   = 2674e3   
            self.g    = 7.4376      
            self.rhom = 4054.49
            self.rhocr = 2900.0      
            self.rhoc = 11050.03
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1650.0 
            self.Tc0  = 4310.0
            self.V = 5e-6            
            self.etaref = 1e20
            self.cc =  1135.41      
            self.cm = 1213.87        
            self.ccr = 1000.0     
            self.alpha = 2.20e-5
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6
            self.lam  = 1 #3           
            self.Qtidal = 0       
            self.L    = 6e5
            self.epsm = 1.236
            self.epsc = 1.081
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km = 3.596 
            self.kcr = 3.0       
            self.kc = 50            

            
        elif body == '0.6_M_Earth':

            self.name = '0.6_M_Earth'
            self.Rp   = 5507e3
            self.Rc   = 2810e3   
            self.g    = 8.06     
            self.rhom = 4153.23
            self.rhocr = 2900.0      
            self.rhoc = 11431.64
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1650.            
            self.Tc0  = 4356.35
            self.V = 5e-6            
            self.etaref = 1e20
            self.cc =  1090.07      
            self.cm = 1216.0       
            self.ccr = 1000.0       
            self.alpha = 2.14e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3        
            self.Qtidal = 0         
            self.L    = 6e5
            self.epsm = 1.201
            self.epsc = 1.086
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km = 3.87
            self.kcr = 3.0      
            self.kc = 50        
            
            
        elif body == '0.7_M_Earth':

            self.name = '0.7_M_Earth'
            self.Rp   = 5758e3
            self.Rc   = 2930e3  
            self.g    = 8.62    
            self.rhom = 4235.70
            self.rhocr = 2900.0      
            self.rhoc = 11765.24
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1650.          
            self.Tc0  = 4464.28
            self.V = 5e-6           
            self.etaref = 1e20
            self.cc =  1066.26    
            self.cm = 1217.27      
            self.ccr = 1000.0        
            self.alpha = 2.08e-5
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6
            self.X0_CO2 = 50e-6     
            self.lam  = 1 #3          
            self.Qtidal = 0         
            self.L    = 6e5
            self.epsm = 1.172
            self.epsc = 1.090
            self.Pcrossover =12e+9   
            self.dIW = 4 
            self.km = 4.13           
            self.kcr = 3.0           
            self.kc = 50             
            
            
        elif body == '0.8_M_Earth':

            self.name = '0.8_M_Earth'
            self.Rp   = 5992e3
            self.Rc   = 3038e3   
            self.g    = 9.12     
            self.rhom = 4286.1
            self.rhocr = 2900.0      
            self.rhoc = 12060.71
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1650.            
            self.Tc0  = 4631.7
            self.V = 5e-6            
            self.etaref = 1e20
            self.cc =  1058.4     
            self.cm = 1216.0      
            self.ccr = 1000.0         
            self.alpha = 2.01e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3         
            self.Qtidal = 0       
            self.L    = 6e5
            self.epsm = 1.144
            self.epsc = 1.093
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km = 4.375 
            self.kcr = 3.0          
            self.kc = 50            
            
            
        elif body == '0.9_M_Earth':

            self.name = '0.9_M_Earth'
            self.Rp   = 6193e3
            self.Rc   = 3136e3   
            self.g    = 9.62    
            self.rhom = 4367.14
            self.rhocr = 2900.0      
            self.rhoc = 12337.25
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1650.      
            self.Tc0  = 4825.54
            self.V = 5e-6          
            self.etaref = 1e20
            self.cc =  1056.2       
            self.cm = 1210.7      
            self.ccr = 1000.0        
            self.alpha = 1.92e-5
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6      
            self.lam  = 1 #3           
            self.Qtidal = 0         
            self.L    = 6e5
            self.epsm = 1.932
            self.epsc = 1.097
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km = 4.714 
            self.kcr = 3.0          
            self.kc = 50              

           
        elif body == '1.0_M_Earth':

            self.name = '1.0_M_Earth'
            self.Rp   = 6377e3
            self.Rc   = 3225e3   
            self.g    = 10.0    
            self.rhom = 4440.86
            self.rhocr = 2900.0      
            self.rhoc = 12595.47
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1700.            
            self.Tc0  = 5020.19
            self.V =   5e-6            
            self.etaref = 1e20
            self.cc =  1055.93  
            self.cm = 1213.84         
            self.ccr = 1000.0         
            self.alpha = 1.92e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3            
            self.Qtidal = 0        
            self.L    = 6e5
            self.epsm = 1.92
            self.epsc = 1.099
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km = 4.966        
            self.kcr = 3.0           
            self.kc = 50             
            
            
        elif body == '1.1_M_Earth':

            self.name = '1.1_M_Earth'
            self.Rp   = 6558e3
            self.Rc   = 3308e3   
            self.g    = 10.5   
            self.rhom = 4487.8
            self.rhocr = 2900.0      
            self.rhoc = 12843.58
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1700.        
            self.Tc0  = 5211.7
            self.V =   5e-6           
            self.etaref = 1e20
            self.cc =  1056.05        
            self.cm = 1213.15         
            self.ccr = 1000.0        
            self.alpha = 1.87e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 #1e-3       
            self.lam  = 1 #3            
            self.Qtidal = 0           
            self.L    = 6e5
            self.epsm = 1.897
            self.epsc = 1.102
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km = 5.211 
            self.kcr = 3.0           
            self.kc = 50            

            
        elif body == '1.2_M_Earth':

            self.name = '1.2_M_Earth'
            self.Rp   = 6725e3
            self.Rc   = 3385e3   
            self.g    = 10.94   
            self.rhom = 4533.97
            self.rhocr = 2900.0      
            self.rhoc = 13081.88
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1700.            
            self.Tc0  = 5399.70
            self.V = 5e-6             
            self.etaref = 1e20
            self.cc =  1056.29      
            self.cm = 1212.56       
            self.ccr = 1000.0     
            self.alpha = 1.82e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6       
            self.lam  = 1 #3            
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.877
            self.epsc = 1.104
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km = 5.452 
            self.kcr = 3.0           
            self.kc = 50             
            
            
        elif body == '1.25_M_Earth':

            self.name = '1.25_M_Earth'
            self.Rp   = 6806e3
            self.Rc   = 3421e3 
            self.g    = 11.13  
            self.rhom = 4555.29
            self.rhocr = 2900.0      
            self.rhoc = 13197.22
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1700.       
            self.Tc0  = 5491.34
            self.V = 5e-6            
            self.etaref = 1e20
            self.cc =  1056.38       
            self.cm = 1212.00        
            self.ccr = 1000.0        
            self.alpha = 1.797e-5
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3             
            self.Qtidal = 0         
            self.L    = 6e5
            self.epsm = 1.867
            self.epsc = 1.105
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km = 5.572          
            self.kcr = 3.0           
            self.kc = 50            
            
            
        elif body == '1.3_M_Earth':

            self.name = '1.3_M_Earth'
            self.Rp   = 6883e3
            self.Rc   = 3456e3   
            self.g    = 11.33  
            self.rhom = 4578.45
            self.rhocr = 2900.0      
            self.rhoc = 13312.10
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1700.          
            self.Tc0  = 5583.56
            self.V = 5e-6            
            self.etaref = 1e20
            self.cc =  1056.46          
            self.cm = 1212.19       
            self.ccr = 1000.0        
            self.alpha = 1.78e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3             
            self.Qtidal = 0     
            self.L    = 6e5
            self.epsm = 1.859
            self.epsc = 1.107
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km = 5.691           
            self.kcr = 3.0           
            self.kc = 50             
      
            
        elif body == '1.4_M_Earth':

            self.name = '1.4_M_Earth'
            self.Rp   = 7015e3
            self.Rc   = 3523e3   
            self.g    = 11.75 
            self.rhom = 4657.083
            self.rhocr = 2900.0      
            self.rhoc = 13531.23
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1700.            
            self.Tc0  = 5764.63
            self.V = 5e-6            
            self.etaref = 1e20
            self.cc =  1056.72      
            self.cm = 1215.065        
            self.ccr = 1000.0        
            self.alpha = 1.78e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3            
            self.Qtidal = 0       
            self.L    = 6e5
            self.epsm = 1.856
            self.epsc = 1.108
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km = 5.925         
            self.kcr = 3.0           
            self.kc = 50           
            
            
        elif body == '1.5_M_Earth':
            
            self.name = '1.5_M_Earth'
            self.Rp   = 7155e3
            self.Rc   = 3587e3  
            self.g    = 12.12   
            self.rhom = 4697.73
            self.rhocr = 2900.0      
            self.rhoc = 13746.70
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1700.            
            self.Tc0  = 5938.71
            self.V = 5e-6           
            self.etaref = 1e20
            self.cc =  1056.54         
            self.cm = 1214.10        
            self.ccr = 1000.0      
            self.alpha = 1.73e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3         
            self.Qtidal = 0      
            self.L    = 6e5
            self.epsm = 1.840
            self.epsc = 1.110
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km = 6.158          
            self.kcr = 3.0     
            self.kc = 50          

            
        elif body == '1.6_M_Earth':

            self.name = '1.6_M_Earth'
            self.Rp   = 7280e3
            self.Rc   = 3646e3   
            self.g    = 12.50
            self.rhom = 4756.15
            self.rhocr = 2900.0      
            self.rhoc = 13954.55
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1700.0            
            self.Tc0  = 6110.08
            self.V = 5e-6             
            self.etaref = 1e20
            self.cc =  1056.33       
            self.cm = 1216.0        
            self.ccr = 1000.0      
            self.alpha = 1.72e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3        
            self.Qtidal = 0     
            self.L    = 6e5
            self.epsm = 1.828
            self.epsc = 1.111
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km = 6.399         
            self.kcr = 3.0         
            self.kc = 50           
            
        elif body == '1.7_M_Earth':

            self.name = '1.7_M_Earth'
            self.Rp   = 7429e3  
            self.Rc   = 3730e3   
            self.g    = 12.48
            self.rhom = 4760.97
            self.rhocr = 2900.0      
            self.rhoc = 13846.09
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1700.0        
            self.Tc0  = 5934.05
            self.V = 5e-6            
            self.etaref = 1e20
            self.cc =  1050.81     
            self.cm = 1216.08       
            self.ccr = 1000.0       
            self.alpha = 1.72e-05 
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3        
            self.Qtidal = 0      
            self.L    = 6e5
            self.epsm = 1.803
            self.epsc = 1.117
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km = 6.423        
            self.kcr = 3.0        
            self.kc = 50          

            
        elif body == '1.8_M_Earth':

            self.name = '1.8_M_Earth'
            self.Rp   = 7534e3
            self.Rc   = 3766e3   
            self.g    = 13.064
            self.rhom = 4822.95
            self.rhocr = 2900.0      
            self.rhoc = 14244.51
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1700.0            
            self.Tc0  = 6362.23
            self.V = 5e-6            
            self.etaref = 1e20
            self.cc =  1061.165     
            self.cm = 1215.389      
            self.ccr = 1000.0 
            self.alpha = 1.67e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3     
            self.Qtidal = 0     
            self.L    = 6e5
            self.epsm = 1.801
            self.epsc = 1.121
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km = 6.775          
            self.kcr = 3.0           
            self.kc = 50             
            
        elif body == '1.9_M_Earth':

            self.name = '1.9_M_Earth'
            self.Rp   = 7648e3 
            self.Rc   = 3822e3  
            self.g    = 13.09
            self.rhom = 4867.82
            self.rhocr = 2900.0      
            self.rhoc = 14244.51
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1700.0    
            self.Tc0  = 6578.2
            self.V = 5e-6           
            self.etaref = 1e20
            self.cc =  1060.226 
            self.cm = 1215.926       
            self.ccr = 1000.0         
            self.alpha = 1.644e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6       
            self.lam  = 1 #3          
            self.Qtidal = 0      
            self.L    = 6e5
            self.epsm = 1.796
            self.epsc = 1.104
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km = 7.02 
            self.kcr = 3.0          
            self.kc = 50             

            
        elif body == '2.0_M_Earth':

            self.name = '2.0_M_Earth'
            self.Rp   = 7758e3
            self.Rc   = 3871e3   
            self.g    = 13.404
            self.rhom = 4903.61
            self.rhocr = 2900.0      
            self.rhoc = 14572.8
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  =1750.0      
            self.Tc0  = 6733.53
            self.V =  5e-6          
            self.etaref = 1e20
            self.cc =  1059.454    
            self.cm = 1215.298     
            self.ccr = 1000.0    
            self.alpha = 1.61e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6
            self.lam  = 1 #3          
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.106
            self.epsc = 1.121
            self.Pcrossover =  12e+9   
            self.dIW = 4 
            self.km = 7.237       
            self.kcr = 3.0          
            self.kc = 50            
            
            
        elif body == '2.1_M_Earth':

            self.name = '2.1_M_Earth'
            self.Rp   = 7864e3
            self.Rc   = 3919e3
            self.g    = 13.713 
            self.rhom = 4941.34
            self.rhocr = 2900.0      
            self.rhoc = 14755.4
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1750.0            
            self.Tc0  = 6887.12 
            self.V =  5e-6            
            self.etaref = 1e20
            self.cc =  1058.604    
            self.cm = 1215.798      
            self.ccr = 1000.0       
            self.alpha = 1.59e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6           
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3          
            self.Qtidal = 0         
            self.L    = 6e5
            self.epsm = 1.779
            self.epsc = 1.107 
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km = 7.451           
            self.kcr = 3.0            
            self.kc = 50              

            
        elif body == '2.2_M_Earth':

            self.name = '2.2_M_Earth'
            self.Rp   = 7967e3
            self.Rc   = 3963e3   
            self.g    = 13.404
            self.rhom = 4976.16
            self.rhocr = 2900.0      
            self.rhoc = 14934.1
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1750.0            
            self.Tc0  = 7037.085
            self.V = 2.5e-6 
            self.etaref = 1e20
            self.cc =  1057.65      
            self.cm = 1215.191     
            self.ccr = 1000.0        
            self.alpha = 1.57e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3            
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.769
            self.epsc = 1.108
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km = 7.667 
            self.kcr = 3.0           
            self.kc = 50             
            
        elif body == '2.3_M_Earth':

            self.name = '2.3_M_Earth'
            self.Rp   = 8066e3
            self.Rc   = 4007e3 
            self.g    = 14.306
            self.rhom = 5010.44
            self.rhocr = 2900.0      
            self.rhoc = 15109.58
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1750.0         
            self.Tc0  = 7184.66
            self.V =  5e-6         
            self.etaref = 1e20
            self.cc =  1056.642   
            self.cm = 1214.610    
            self.ccr = 1000.0        
            self.alpha = 1.539e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6     
            self.X0_CO2 = 50e-6
            self.lam  = 1 #3             
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.759
            self.epsc = 1.109
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km = 7.882          
            self.kcr = 3.0            
            self.kc = 50            
  
        elif body == '2.4_M_Earth':

            self.name = '2.4_M_Earth'
            self.Rp   = 8144e3
            self.Rc   = 4050e3  
            self.g    = 14.64
            self.rhom = 5080.54
            self.rhocr = 2900.0      
            self.rhoc = 15274.78
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1750.0            
            self.Tc0  = 7328.69
            self.V = 5e-6             
            self.etaref = 1e20
            self.cc =  1057.65     
            self.cm = 1215.191     
            self.ccr = 1000.0       
            self.alpha = 1.547e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6
            self.lam  = 1 #3            
            self.Qtidal = 0         
            self.L    = 6e5
            self.epsm = 1.75
            self.epsc = 1.109
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km = 8.092
            self.kcr = 3.0           
            self.kc = 50              
            
        elif body == '2.5_M_Earth':

            self.name = '2.5_M_Earth'
            self.Rp   = 8235e3
            self.Rc   = 4090e3 
            self.g    = 14.930 
            self.rhom = 5116.19 
            self.rhocr = 2900.0      
            self.rhoc = 15444.82
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1750.0            
            self.Tc0  = 7472.6 
            self.V =  5e-6             
            self.etaref = 1e20
            self.cc =  1054.667     
            self.cm = 1219.024       
            self.ccr = 1000.0       
            self.alpha = 1.54e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6      
            self.X0_CO2 = 50e-6    # set to very high value, maybe beneath this (be careful with magma formation in the lower mantle)
            self.lam  = 1 #3           
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.746
            self.epsc = 1.110
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km = 8.298           
            self.kcr = 3.0             
            self.kc = 50               
            
            
        elif body == '2.6_M_Earth':

            self.name = '2.6_M_Earth'
            self.Rp   = 8324e3
            self.Rc   = 4129e3  
            self.g    = 15.210
            self.rhom = 5148.91
            self.rhocr = 2900.0      
            self.rhoc = 15611.72
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1750.0            
            self.Tc0  = 7613.56
            self.V = 5e-6            
            self.etaref = 1e20
            self.cc =  1053.481   
            self.cm = 1218.546   
            self.ccr = 1000.0        
            self.alpha = 1.51e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6       
            self.lam  = 1 #3            
            self.Qtidal = 0           
            self.L    = 6e5
            self.epsm = 1.738
            self.epsc = 1.111
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km = 8.509 
            self.kcr = 3.0            
            self.kc = 50               
            
        elif body == '2.7_M_Earth':

            self.name = '2.7_M_Earth'
            self.Rp   = 8411e3 
            self.Rc   = 4167e3
            self.g    = 15.486
            self.rhom = 5181.14 
            self.rhocr = 2900.0      
            self.rhoc = 15776.14
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1750.0           
            self.Tc0  = 7752.63 
            self.V =  5e-6              
            self.etaref = 1e20
            self.cc =  1052.258      
            self.cm = 1218.092        
            self.ccr = 1000.0         
            self.alpha = 1.49e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6      
            self.X0_CO2 = 50e-6  
            self.lam  = 1 #3            
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.730
            self.epsc = 1.112
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km = 8.720           
            self.kcr = 3.0            
            self.kc = 50             

            
        elif body == '2.8_M_Earth':

            self.name = '2.8_M_Earth'
            self.Rp   = 8494e3
            self.Rc   = 4203e3  
            self.g    = 15.757
            self.rhom = 5212.80
            self.rhocr = 2900.0      
            self.rhoc = 15938.31 
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1750.0            
            self.Tc0  = 7889.802
            self.V = 5e-6             
            self.etaref = 1e20
            self.cc =  1050.986       
            self.cm = 1217.666    
            self.ccr = 1000.0         
            self.alpha = 1.47e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3            
            self.Qtidal = 0         
            self.L    = 6e5
            self.epsm = 1.722
            self.epsc = 1.113
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km = 8.929  
            self.kcr = 3.0            
            self.kc = 50              
            
        elif body == '2.9_M_Earth':

            self.name = '2.9_M_Earth'
            self.Rp   = 8575e3
            self.Rc   = 4239e3
            self.g    = 16.028 
            self.rhom = 5246.71
            self.rhocr = 2900.0      
            self.rhoc = 16098.29
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1750.0    
            self.Tc0  = 8026.00
            self.V =  5e-6            
            self.etaref = 1e20
            self.cc =  1049.717    
            self.cm = 1218.649       
            self.ccr = 1000.0        
            self.alpha = 1.46e-05 
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6      
            self.X0_CO2 = 50e-6  
            self.lam  = 1 #3              
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.721
            self.epsc = 1.114
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km = 9.130        
            self.kcr = 3.0            
            self.kc = 50               

            
        elif body == '3.0_M_Earth':

            self.name = '3.0_M_Earth'
            self.Rp   = 8654e3
            self.Rc   = 4273e3  
            self.g    = 16.29
            self.rhom = 5277.84
            self.rhocr = 2900.0      
            self.rhoc = 16255.87 
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1800          
            self.Tc0  = 8159.92
            self.V = 5e-6             
            self.etaref = 1e20
            self.cc =  1048.42     
            self.cm = 1218.23      
            self.ccr = 1000.0        
            self.alpha = 1.44e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6        
            self.lam  = 1 #3             
            self.Qtidal = 0      
            self.L    = 6e5
            self.epsm = 1.714
            self.epsc = 1.115
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km =  9.339   
            self.kcr = 3.0       
            self.kc = 50              
            
        elif body == '3.1_M_Earth':

            self.name = '3.1_M_Earth'
            self.Rp   = 8731e3 
            self.Rc   = 4306e3 
            self.g    = 16.553 
            self.rhom = 5308.60
            self.rhocr = 2900.0      
            self.rhoc = 16411.47
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1800.0          
            self.Tc0  = 8292.23
            self.V =  5e-6             
            self.etaref = 1e20
            self.cc =  1047.089 
            self.cm = 1217.816        
            self.ccr = 1000.0         
            self.alpha = 1.42e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6     
            self.X0_CO2 = 50e-6  
            self.lam  = 1 #3           
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.707
            self.epsc = 1.115
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km = 9.547           
            self.kcr = 3.0         
            self.kc = 50              
            
            
        elif body == '3.2_M_Earth':

            self.name = '3.2_M_Earth'
            self.Rp   = 8805e3 
            self.Rc   = 4339e3  
            self.g    = 16.811
            self.rhom = 5338.96
            self.rhocr = 2900.0      
            self.rhoc = 16565.0
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1800.0             
            self.Tc0  =  8423.081
            self.V = 5e-6             
            self.etaref = 1e20
            self.cc =  1045.76    
            self.cm = 1217.43     
            self.ccr = 1000.0        
            self.alpha = 1.40e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6  
            self.X0_CO2 = 50e-6  
            self.lam  = 1 #3            
            self.Qtidal = 0         
            self.L    = 6e5
            self.epsm = 1.701
            self.epsc = 1.116
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km =  9.755   
            self.kcr = 3.0      
            self.kc = 50             
            
        elif body == '3.3_M_Earth':

            self.name = '3.3_M_Earth'
            self.Rp   = 8878e3 
            self.Rc   = 4370e3 
            self.g    = 17.066 
            self.rhom = 5369.20
            self.rhocr = 2900.0      
            self.rhoc = 16716.94
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1800.0            
            self.Tc0  = 8552.52
            self.V =  5e-6             
            self.etaref = 1e20
            self.cc =  1044.390   
            self.cm =  1217.036      
            self.ccr = 1000.0       
            self.alpha = 1.38e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6    
            self.X0_CO2 = 50e-6  
            self.lam  = 1 #3         
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.695 
            self.epsc = 1.117
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km = 9.963           
            self.kcr = 3.0            
            self.kc = 50             
            
            
        elif body == '3.4_M_Earth':

            self.name = '3.4_M_Earth'
            self.Rp   = 8948e3
            self.Rc   = 4400e3  
            self.g    = 17.321
            self.rhom = 5401.0
            self.rhocr = 2900.0      
            self.rhoc = 16867.0
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1800.0            
            self.Tc0  = 8680.98
            self.V = 5e-6 
            self.etaref = 1e20
            self.cc =  1043.02        
            self.cm = 1218.26   
            self.ccr = 1000.0        
            self.alpha = 1.38e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6  
            self.X0_CO2 = 50e-6  
            self.lam  = 1 #3              
            self.Qtidal = 0      
            self.L    = 6e5
            self.epsm = 1.696
            self.epsc = 1.118
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km =  10.158  
            self.kcr = 3.0        
            self.kc = 50              
            
        elif body == '3.5_M_Earth':

            self.name = '3.5_M_Earth'
            self.Rp   = 9018e3
            self.Rc   = 4431e3
            self.g    = 17.570
            self.rhom = 5430.80
            self.rhocr = 2900.0      
            self.rhoc = 17014.97
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1800.0        
            self.Tc0  = 8807.66
            self.V =  5e-6              
            self.etaref = 1e20
            self.cc =  1041.671       
            self.cm = 1217.890        
            self.ccr = 1000.0         
            self.alpha = 1.36e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6      
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3           
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.690
            self.epsc = 1.118
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km = 10.364           
            self.kcr = 3.0            
            self.kc = 50               

            
        elif body == '3.6_M_Earth':

            self.name = '3.6_M_Earth'
            self.Rp   = 9085e3
            self.Rc   = 4460e3  
            self.g    = 17.816
            self.rhom = 5460.14
            self.rhocr = 2900.0      
            self.rhoc = 17162.11
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1800.0            
            self.Tc0  = 8933.34
            self.V = 5e-6             
            self.etaref = 1e20
            self.cc =  1040.26          
            self.cm = 1217.52          
            self.ccr = 1000.0         
            self.alpha = 1.35e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6       
            self.X0_CO2 = 50e-6         
            self.lam  = 1 #3              
            self.Qtidal = 0           
            self.L    = 6e5
            self.epsm = 1.684
            self.epsc = 1.119
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km =  10.571  
            self.kcr = 3.0       
            self.kc = 50              
            
        elif body == '3.7_M_Earth':

            self.name = '3.7_M_Earth'
            self.Rp   = 9151e3
            self.Rc   = 4488e3
            self.g    = 18.060
            self.rhom = 5489.04
            self.rhocr = 2900.0      
            self.rhoc = 17307.49
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1800.0         
            self.Tc0  = 9057.76
            self.V =  5e-6            
            self.etaref = 1e20
            self.cc =  1038.855     
            self.cm = 1217.175       
            self.ccr = 1000.0         
            self.alpha = 1.33e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6     
            self.X0_CO2 = 50e-6  
            self.lam  = 1 #3             
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.679
            self.epsc = 1.120
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km = 10.777         
            self.kcr = 3.0           
            self.kc = 50             

            
        elif body == '3.8_M_Earth':

            self.name = '3.8_M_Earth'
            self.Rp   = 9216e3
            self.Rc   = 4515e3  
            self.g    = 18.30
            self.rhom = 5517.23
            self.rhocr = 2900.0      
            self.rhoc = 17451.47
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1800.0            
            self.Tc0  = 9181.06
            self.V = 5e-6            
            self.etaref = 1e20
            self.cc =  1037.44         
            self.cm = 1216.87         
            self.ccr = 1000.0          
            self.alpha = 6.65e-06
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6  
            self.X0_CO2 = 50e-6  
            self.lam  = 1 #3             
            self.Qtidal = 0           
            self.L    = 6e5
            self.epsm = 1.673
            self.epsc = 1.120
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km =  10.982         
            self.kcr = 3.0         
            self.kc = 50              
            
            
        elif body == '3.9_M_Earth':

            self.name = '3.9_M_Earth'
            self.Rp   = 9270e3
            self.Rc   = 4543e3
            self.g    = 18.567
            self.rhom = 5564.67
            self.rhocr = 2900.0      
            self.rhoc = 17587.54
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1800.0    
            self.Tc0  = 9300.94
            self.V =  5e-6            
            self.etaref = 1e20
            self.cc =  1036.194       
            self.cm = 1218.532       
            self.ccr = 1000.0         
            self.alpha = 1.32e-05 
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6      
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3             
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.667
            self.epsc = 1.120
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km = 11.203        
            self.kcr = 3.0          
            self.kc = 50              

            
        elif body == '4.0_M_Earth':

            self.name = '4.0_M_Earth'
            self.Rp   = 9332e3
            self.Rc   = 4569e3  
            self.g    = 18.80
            self.rhom = 5592.63
            self.rhocr = 2900.0      
            self.rhoc = 17728.54
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1800.0            
            self.Tc0  = 9421.95
            self.V = 5e-6            
            self.etaref = 1e20
            self.cc =  1034.78  
            self.cm = 1218.24  
            self.ccr = 1000.0        
            self.alpha = 1.30e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6  
            self.X0_CO2 = 50e-6  
            self.lam  = 1 #3         
            self.Qtidal = 0           
            self.L    = 6e5
            self.epsm = 1.662
            self.epsc = 1.121
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km =  11.408         
            self.kcr = 3.0        
            self.kc = 50          

            
        elif body == '4.2_M_Earth':

            self.name = '4.2_M_Earth'
            self.Rp   = 9451e3
            self.Rc   = 4620e3  
            self.g    = 19.274
            self.rhom = 5649.96
            self.rhocr = 2900.0      
            self.rhoc = 18005.80
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1800.0            
            self.Tc0  = 9660.67
            self.V = 5e-6             
            self.etaref = 1e20
            self.cc =  1031.97  
            self.cm = 1219.52         
            self.ccr = 1000.0        
            self.alpha = 1.29e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6  
            self.X0_CO2 = 50e-6  
            self.lam  = 1 #3            
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.661
            self.epsc = 1.122
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km =  11.796      
            self.kcr = 3.0         
            self.kc = 50             

            
        elif body == '4.4_M_Earth':

            self.name = '4.4_M_Earth'
            self.Rp   = 9451e3
            self.Rc   = 4669e3  
            self.g    = 19.733
            self.rhom = 5704.37
            self.rhocr = 2900.0      
            self.rhoc = 18278.68
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1800.0            
            self.Tc0  = 9895.76
            self.V = 5e-6           
            self.etaref = 1e20
            self.cc =  1029.15        
            self.cm = 1218.96      
            self.ccr = 1000.0       
            self.alpha = 1.26e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6  
            self.lam  = 1 #3     
            self.Qtidal = 0        
            self.L    = 6e5
            self.epsm = 1.652
            self.epsc = 1.123
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km =  12.204      
            self.kcr = 3.0           
            self.kc = 50          
            
            
        elif body == '4.5_M_Earth':

            self.name = '4.5_M_Earth'
            self.Rp   = 9622e3 
            self.Rc   = 4692e3   
            self.g    = 19.959
            self.rhom = 5731.17
            self.rhocr = 2900.0      
            self.rhoc = 18413.44
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1850.0           
            self.Tc0  = 10011.89
            self.V = 5e-6            
            self.etaref = 1e20
            self.cc =  1027.73      
            self.cm = 1218.69        
            self.ccr = 1000.0          
            self.alpha = 1.25e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3            
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.647
            self.epsc = 1.124
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km =  12.408      
            self.kcr = 3.0            
            self.kc = 50              

            
        elif body == '4.6_M_Earth':

            self.name = '4.6_M_Earth'
            self.Rp   = 9677e3
            self.Rc   = 4716e3  
            self.g    = 20.18
            self.rhom = 5757.76
            self.rhocr = 2900.0      
            self.rhoc = 18547.09
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1850.0            
            self.Tc0  = 10127.18
            self.V = 5e-6             
            self.etaref = 1e20
            self.cc =  1026.33       
            self.cm = 1218.43         
            self.ccr = 1000.0          
            self.alpha = 1.24e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6  
            self.lam  = 1 #3           
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.643
            self.epsc = 1.124
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km =  12.612        
            self.kcr = 3.0          
            self.kc = 50             
            
        elif body == '4.7_M_Earth':

            self.name = '4.7_M_Earth'
            self.Rp   = 9730e3
            self.Rc   = 4738e3
            self.g    = 20.41
            self.rhom = 5784.19
            self.rhocr = 2900.0      
            self.rhoc = 18679.72
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1850.0            
            self.Tc0  = 10241.58
            self.V = 5e-6             
            self.etaref = 1e20
            self.cc =  1024.91     
            self.cm = 1218.18       
            self.ccr = 1000.0       
            self.alpha = 1.23e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3             
            self.Qtidal = 0           
            self.L    = 6e5
            self.epsm = 1.639
            self.epsc = 1.125
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km =  12.815        
            self.kcr = 3.0          
            self.kc = 50              


        elif body == '4.8_M_Earth':

            self.name = '4.8_M_Earth'
            self.Rp   = 9783e3
            self.Rc   = 4760e3  
            self.g    = 20.63
            self.rhom = 5810.52
            self.rhocr = 2900.0      
            self.rhoc = 18811.16
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1850.0            
            self.Tc0  = 10355.15
            self.V = 5e-6           
            self.etaref = 1e20
            self.cc =  1023.51  
            self.cm = 1217.92      
            self.ccr = 1000.0        
            self.alpha = 1.22e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6  
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3            
            self.Qtidal = 0      
            self.L    = 6e5
            self.epsm = 1.634
            self.epsc = 1.125
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km =  13.019         
            self.kcr = 3.0        
            self.kc = 50              
            
            
        elif body == '4.9_M_Earth':

            self.name = '4.9_M_Earth'
            self.Rp   = 9835e3
            self.Rc   = 4782e3
            self.g    = 20.85
            self.rhom = 5836.45
            self.rhocr = 2900.0      
            self.rhoc =  18941.63
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1850.0            
            self.Tc0  = 10467.87
            self.V = 5e-6           
            self.etaref = 1e20
            self.cc = 1022.11          
            self.cm = 1217.69          
            self.ccr = 1000.0        
            self.alpha = 1.205e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6  
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3          
            self.Qtidal = 0         
            self.L    = 6e5
            self.epsm = 1.630
            self.epsc = 1.126
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km =  13.223       
            self.kcr = 3.0            
            self.kc = 50               
            
        elif body == '5.0_M_Earth':

            self.name = '5.0_M_Earth'
            self.Rp   = 9886e3
            self.Rc   = 4804e3  
            self.g    = 21.066
            self.rhom = 5810.52
            self.rhocr = 2900.0      
            self.rhoc = 19071.14
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1850.          
            self.Tc0  = 10579.71
            self.V = 5e-6             
            self.etaref = 1e20
            self.cc =  1020.70     
            self.cm = 1217.46      
            self.ccr = 1000.0          
            self.alpha = 1.19e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6  
            self.X0_CO2 = 50e-6 
            self.lam  = 1 #3           
            self.Qtidal = 0         
            self.L    = 6e5
            self.epsm = 1.626
            self.epsc = 1.126
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km =  13.43        
            self.kcr = 3.0            
            self.kc = 50               

                            
        else:
            
            #Read input from default input file (Mars)
            with open(inpfile, 'r') as inp:
                data = inp.read()
            inpvar = json.loads(data)

            self.name         = inpvar['name']
            self.tectonics    = inpvar['tectonics']
            self.core_cooling = inpvar['core_cooling']
            self.var_alpha    = inpvar['var_alpha']
            self.Rp           = inpvar['Rp']
            self.Rc           = inpvar['Rc']
            self.g            = inpvar['g']
            self.rhom         = inpvar['rhom']
            self.rhoc         = inpvar['rhoc']
            self.Ts           = inpvar['Ts'] 
            self.Tm0          = inpvar['Tm0']
            self.Tc0          = inpvar['Tc0'] 
            self.etaref       = inpvar['etaref'] 
            self.Q0           = inpvar['Q0']
            self.lam          = inpvar['lam']            
            self.X_U          = inpvar['X_U']
            self.X_Th         = inpvar['X_Th']
            self.X_K          = inpvar['X_K']
            self.Qtidal       = inpvar['Qtidal']            
        
        return
