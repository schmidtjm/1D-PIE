import json
import numpy as np

#####################################################
"""
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
        km:     effective* mantle thermal conductivity (W/(mK)) 
        kcr:    Crustal thermal conductivity (W/(mK)) 
        kc:     Core thermal conductivity (W/(mK))

        """
#####################################################

            
        if body == 'Mercury':

            self.name = 'Mercury'
            self.Rp   = 2440e3        # Archinal, B.A. et al. 2018. "Report of the IAU/IAG Working Group on cartographic coordinates and rotational elements: 2015" Celestial Mech. Dyn. Astr. 130:22. 
            self.Rc   = 2024e3        # Margot et al. (2019). In: Mercury - The view after MESSENGER. Ch. 4, 85-113, CUP. Based on mean values of interior structure model
            self.g    = 3.70          # https://ssd.jpl.nasa.gov/planets/phys_par.html#refs        
            self.rhom = 3295.0        # Margot et al. (2019). In: Mercury - The view after MESSENGER. Ch. 4, 85-113, CUP.
            self.rhocr = 2903.0       # Margot et al. (2019). In: Mercury - The view after MESSENGER. Ch. 4, 85-113, CUP.
            self.rhoc = 7034.0        # Margot et al. (2019). In: Mercury - The view after MESSENGER. Ch. 4, 85-113, CUP.
            self.rho_H2O = 1000.0
            self.Ts   = 440.0         # Breuer et al 2007: Interior Evolution of Mercury; not real value since Ts = 93K (night)-703K (day)
            self.Tm0  = 1680 #1435.   # Tm =60%Tsol+40%Tliq #1600 #1716. #1    # Tsol at Rl
            self.Tc0  = 1930.  # Tsol at CMB
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
            self.L    = 6e5           # Grott, Breuer & Laneuville 2011
            self.epsm = 1.0274        # Ziercke (Master thesis), Spohn 1990: 1.0 
            self.epsc = 1.0807        # Ziercke (Master thesis), Spohn 1990: 1.1 
            self.Pcrossover = 6.8e+9  # Vander Kaaden and McCubbin 2015
            self.dIW = -7 
            self.km_u = 1.5791          # Mantle thermal conductivity (W/(mK)) (Ziercke (Master thesis, based on Tosi et al.)), Morschhauser 2011: 4.0
            self.km_b = 1.5791
            self.kcr = 3.0            # Crustal thermal conductivity (W/(mK)), Morschhauser 2011
            self.kc = 50              # Core thermal conductivity, Ziercke (Master thesis)
    
        elif body == 'Venus':
            
            self.name = 'Venus'
            self.Rp   = 6051e3       #  Archinal, B.A. et al. 2018. "Report of the IAU/IAG Working Group on cartographic coordinates and rotational elements: 2015" Celestial Mech. Dyn. Astr. 130:22. 
            self.Rc   = 3110e3       # Spohn 1991: Mantle Differentiation and Thermal Evolution of Mars, Mercury,and Venus
            self.g    = 8.87         # Noack 2012
            self.rhom = 3551         # O'Rourke & Korenaga 2015, Icarus; formerly: 4400
            self.rhoc = 12500        # O'Rourke & Korenaga 2015 (10100.0 formerly)
            self.rhocr = 2900.0      # basaltic crust density
            self.rho_H2O = 1000.0
            self.Ts   = 730.0         # O'Rourke & Korenaga 2015
            self.Tm0  = 1705. #1470.  # Tm =60%Tsol+40%Tliq      # Tsol at Rl
            self.Tc0  = 4673.         # T sol at CMB
            self.cc = 850.0           # Noack, L., Breuer, D., Spohn, T., 2012. Coupling the atmosphere with interior dynamics
            self.cm = 1200.0          # Spohn 1991: Mantle Differentiation and Thermal Evolution of Mars, Mercury,and Venus
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
            self.L    = 6e5            #  Gillmann & Tackley 2014: Atmosphere/mantle coupling and feedbacks on Venus
            self.epsm = 1.2144         # Ziercke (Master thesis) 
            self.epsc = 1.1184         # Ziercke (Master thesis) 
            self.Pcrossover = 12e+9    # Ohtani et al. 1995
            self.dIW = 4               # buffer
            self.km_u = 4.6              # Ziercke (Master thesis), Morschhauser 2011: 4.0
            self.km_b = 4.6
            self.kcr = 3.0             # Crustal thermal conductivity (W/(mK)) Morschhauser 2011
            self.kc = 50               # Core thermal conductivity, Ziercke (Master thesis)
            
            # rhoc might be a little lower with Rc = 3110 km? Amorim & Gudkova 2024, Fig. 1
            # kc might be on the lower end: O’Rourke et al. (2018) performed numerical simulations of the evolution of the coupled atmosphere-surface-mantle-core system to study the state of Venus’s core and showed that if the thermal conductivity of the core is relatively low (∼40–50 W·m−1K−1), the present-day absence of a global magnetic field on Venus requires a completely solid core. Xiao 2021: Possible Deep Structure and Composition of Venus With Respect to the Current Knowledge From Geodetic Data

        elif body == 'Earth':

            self.name = 'Earth'
            self.Rp   = 6371e3       # Dziewonski & Anderson 1981: Preliminary reference Earth model
            self.Rc   = 3480e3       # Dziewonski & Anderson 1981: Preliminary reference Earth model
            self.g    = 9.81         # Turcotte & Schubert 2002         
            self.rhom =  4460.0      # Stamekovic et al. 2012     
            self.rhoc = 11080.        # Stamekovic et al. 2012 
            #self.rhoCMB= 5500.0      # density post-perovskite
            self.rhocr = 2900.0      # Dziewonski & Anderson 1981: Preliminary reference Earth model, lower end
            self.rho_H2O = 1000.0
            self.Ts   = 288.0         # David G. Andrews An Introduction to Atmoshperic Physics Cambridge University Press 2000 (p. 5)
            self.Tm0  = 1701.5 #1465. # Tm =60%Tsol+40%Tliq  # Tsol at Rl          
            self.Tc0  = 5520.         # Tsol at CMB
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
            self.L    = 5.6e5            # Long et al 2019: Mantle Melting and Intraplate Volcanism Due to Self‐Buoyant Hydrous Upwellings From the Stagnant Slab That Are Conveyed by Small‐Scale Convection
            self.epsm = 1.2393         # Ziercke (Master thesis)
            self.epsc = 1.137          # Ziercke (Master thesis)
            self.Pcrossover = 12e+9    # Ohtani et al. 1995
            self.dIW = 4               # buffer
            self.km_u = 5.1              # Ziercke (Master thesis), Mantle thermal conductivity (W/(mK)) Morschhauser 2011: 4.0 , 5.1 -> Sonja
            self.km_b = 5.1
            self.kcr = 3.0             # Crustal thermal conductivity (W/(mK)) Morschhauser 2011
            self.kc = 50               # Core thermal conductivity, Ziercke (Master thesis)
            
        elif body == 'Moon':

            self.name = 'Moon'
            self.Rp   = 1737e3         # R.F. Garcia et al. / Physics of the Earth and Planetary Interiors 188 (2011) 96–113 
            self.Rc   = 330e3          # Weber, R.C.; Lin, P.-Y.; Garnero, E.J.; Williams, Q.; Lognonne, P. (January 21, 2011). "Seismic Detection of the Lunar Core" (PDF). Science. 331 (6015): 309–312. 
            self.g    = 1.62            # C. Hirt; W. E. Featherstone (2012). "A 1.5 km-resolution gravity field model of the Moon". Earth and Planetary Science Letters. 329–330: 22–30. doi:10.1016/j.epsl.2012.02.012. Retrieved 2012-08-21.
            self.rhom = 3400.0         # Zhang, E. M. Parmentier, Yan Liang (2013). A 3-D numerical study of the thermal evolution of the Moon after cumulate mantle overturn: The importance of rheology and core solidification. JGR Planets; R.F. Garcia et al. / Physics of the Earth and Planetary Interiors 188 (2011) 96–113 
            self.rhoc = 7400.0        # R.F. Garcia et al. / Physics of the Earth and Planetary Interiors 188 (2011) 96–113 
            self.rhocr = 2550.0       # Wieczorek, Neumann et al 2012 (Science)
            self.rho_H2O = 1000.0
            self.Ts   = 250.0         # konrad & Spohn (1997), Ah. Space Rcs; still widely used   ; M. Laneuville, M. A. Wieczorek, D. Breuer, N. Tosi 2013
            self.Tm0  = 1666.5 #1440. # Tm =60%Tsol+40%Tliq        # Tsol  
            self.Tc0  = 2100.         # Tsol
            self.cc = 850.0           # Core heat capacity (J/(kg K))    # Laneuville, Taylor, Wieczorek 2018
            self.cm = 1000.0          # Mantle heat capacity (J/(kg K))  # Laneuville, Taylor, Wieczorek 2018: Asymmetric thermal evolution of the Moon
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
            self.km_u = 1.4789          # Ziercke (Master thesis), Laneuville et al 2013: 3.0
            self.km_b = 1.4789
            self.kcr = 1.5            # Crustal thermal conductivity (W/(mK)) Laneuville et al 2013
            self.kc = 50              # Core thermal conductivity # Zierke

            
        elif body == 'Mars':
            
            self.name = 'Mars'    
            self.Rp   = 3390e3        # Morschhauser et al 2011
            self.Rc   = 1850e3        # Core radius from Plesa et al. (2018). Geophys. Res. Lett., 45(22), 12198-12209.  1850e3
            self.g    = 3.7           # Morschhauser et al 2011
            self.rhom = 3500.0        # Morschhauser et al 2011
            #self.rhoCMB= 4400.0
            self.rhoc = 7200.0        # Morschhauser et al 2011
            self.rhocr= 2900.0        # Morschhauser et al 2011
            self.rho_H2O=1000.0
            self.Ts   = 220.0         # Morschhauser et al 2011
            self.Tm0  = 1679.1 #1434.8# Tm =60%Tsol+40%Tliq         #Tsol  
            self.Tc0  = 2469.5        # Tsol #2500.=40% melt
            self.cc = 840.0           # Core heat capacity (J/(kg K)) Morschhauser et al 2011
            self.cm = 1142.0          # Mantle heat capacity (J/(kg K))  Morschhauser et al 2011
            self.ccr = 1000.0         # Crust heat capacity (J/(kg K)) Morschhauser et al 2011
            self.alpha = 2.712e-5     # Ziercke (Master thesis), Morschhauser 2011: 2.5e-5
            # Heat sources from Waenke & Dreibus (1994). Philos. Trans. R. Soc. London, A349, 2134–2137.
            self.X_U  = 16e-9 
            self.X_Th = 56e-9
            self.X_K  = 305e-6 
            self.X0_H2O  = 300e-6     # initial water content [weight fraction]. Wänke & Dreibus: 36e-6 (assumption Grott2011: 100 ppm -> between 800(plume model)-2500(global melt layer model) ppm H2O in melt, Taylor 2013: min. 300 ppm
            self.X0_CO2 = 50e-6       # initial CO2 content [weight frcaction]
            self.lam  = 3             # Morschhauser 2011: 5  
            self.Qtidal = 0
            self.L    = 6e5           # latent heat of melting (J/kg) , Hauck & Phillips 2002; Morschauser et al 2011, Samuel 2019
            self.epsm = 1.0638        # Ziercke (Master thesis), Morschauser 2011: 1.0 
            self.epsc = 1.0514        # Ziercke (Master thesis), Morschauser 2011: 1.1       
           # self.eps  = 0.6           # Katz et al.,2003
           # self.chi_a = 12           # wt%/GPa, Katz et al.,2003
           # self.chi_b = 1            # wt%/GPa, Katz et al.,2003
            self.Pcrossover = 7e+9    # Ohtani et al. 1995
            self.dIW = 1              # for Mars ~0.5-1 # buffer
            self.km_u = 2.3 #4               # Mantle thermal conductivity (W/(mK)) Morschhauser 2011:4.0 ; Ziercke (Master thesis: 2.3033)
            self.km_b = 2.3
            self.kcr = 2.3 #3              # Crustal thermal conductivity (W/(mK)) Morschhauser 2011:3, Plesa et al 2015: 2.3
            self.kc = 50              # Core thermal conductivity # Zierke

            
            
            ######################################
            ####### Exoplanet Paper ##############
            ######################################
       
        elif body == '0.1_M_Earth':

            self.name = '0.1_M_Earth'
            self.Rp   = 3220e3 #3226e3 #3224e3
            self.Rc   = 1790e3 #1790e3 #1651e3   
            self.g    = 3.85 #3.88        
            self.rhom = 3461.4 #3435.8 #3456.0  
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 8175.1 #8175.0 #9378.0     
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  =  1955.6 #1948.7 #1600.0   
            self.Tc0  = 2518.9 #2518.9 #2636.       
            self.cc = 1006.4 #1006.4 #1008.0          
            self.cm = 1216.2 #1215.4 #1191.0         
            self.ccr = 1000.0     
            self.alpha = 2.78397e-05 #2.73203e-05 #2.594E-05   
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6  
            self.X0_CO2 = 50e-6
            self.lam  = 3             
            self.Qtidal = 0          
            self.L    = 6e5 
            self.epsm = 1.04111 #1.04149 #1.219436693
            self.epsc = 1.05643 #1.05641  #1.043560597
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km_u = 2.1241 #2.1116 #2.24276    
            self.km_b = 2.1241 #2.1116 #2.24276 
            self.kcr = 2.0          
            self.kc = 50 #49.7614 #50                      
 
        elif body == '0.2_M_Earth':

            self.name = '0.2_M_Earth'
            self.Rp   = 3976e3 #3983e3 #3980e3
            self.Rc   = 2212e3 #2046e3 #2044e3   
            self.g    = 5.0526 #5.0883 #5.095        
            self.rhom = 3677.6 #3671.39 #3677.6
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 8658.4 #9870.4 #9894.0
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2028.3 #1915.8 #1600.            
            self.Tc0  = 3020.2 #3321.328 #3245.0            
            self.cc = 1071.3 #1105.0230 #1082.432857          
            self.cm = 1225.5 #1213.6793 #1205.617274      
            self.ccr = 1000.0        
            self.alpha = 2.60904e-05 #2.548530E-05 #2.48E-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 3         
            self.Qtidal = 0        
            self.L    = 6e5
            self.epsm = 1.06593 #1.07080 #1.060
            self.epsc = 1.07784 #1.05829 #1.058
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km_u = 2.5662 #2.5987 #2.6402      
            self.km_b = 2.5662 
            self.kcr = 2.0            
            self.kc = 49.7658 #50              

            
        elif body == '0.3_M_Earth':

            self.name = '0.3_M_Earth'
            self.Rp   = 4485e3 #4494e3 #4493e3
            self.Rc   = 2497e3 #2311e3 #2308e3   
            self.g    = 5.9559 #6.01 #6.009        
            self.rhom = 3843.6 #3837.32 #3836.92
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 9035.5 #10279.11 #10305.67
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2075.9 #1600.          
            self.Tc0  = 3445.7 #3820.535 #3729.4           
            self.cc = 1119.7 #1155.1474 #1129.729    
            self.cm = 1231.7 #1209.852 #1207.351
            self.ccr = 1000.0        
            self.alpha = 2.51135e-05 #2.82533e-05 #2.35e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 3            
            self.Qtidal = 0         
            self.L    = 6e5
            self.epsm = 1.08783 #6.2825 #1.347   
            self.epsc =  1.09192  #1.0677 #1.068
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km_u = 2.8969 #3.012     
            self.km_b = 2.8969 #3.012     
            self.kcr = 2.0           
            self.kc = 50  #50  
            
            
        elif body == '0.4_M_Earth':

            self.name = '0.4_M_Earth'
            self.Rp   = 4884e3 #4892e3 #4894e3
            self.Rc   = 2717e3 #2514e3 #2511e3   
            self.g    = 6.6986 #6.7694 #6.76817      
            self.rhom = 3967.9 #3964.023 #3955.77
            self.rhocr = 3000.0 #2900.0      
            self.rhoc =  9356.5 #10642.729 #10676.58
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2109.5 #1600.            
            self.Tc0  = 3811.3 #4103.45          
            self.cc =   1154.9 #1151.62         
            self.cm = 1235.0 #1212.05       
            self.ccr = 1000.0        
            self.alpha = 2.42329e-05 #2.28e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 3          
            self.Qtidal = 0    
            self.L    = 6e5
            self.epsm = 1.10747 #1.283
            self.epsc = 1.10246  #1.075
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km_u = 3.1902 #3.311        
            self.km_b = 3.1902 #3.311 
            self.kcr = 2.0        
            self.kc = 50       

            
        elif body == '0.5_M_Earth':

            self.name = '0.5_M_Earth'
            self.Rp   = 5215e3 #5227e3
            self.Rc   = 2897e3 #2674e3   
            self.g    = 7.3435  #7.4376      
            self.rhom = 4069.6 #4054.49
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 9646.3 #11050.03
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2137.0 #1650.0 
            self.Tc0  = 4124.7  #4310.0            
            self.cc =  1178.0 #1135.41      
            self.cm = 1236.8 #1213.87        
            self.ccr = 1000.0     
            self.alpha = 2.34409e-05 #2.20e-5
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6
            self.lam  = 3           
            self.Qtidal = 0       
            self.L    = 6e5
            self.epsm = 1.12518 #1.236
            self.epsc = 1.11082 #1.081
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km_u = 3.4626 #3.596 
            self.km_b = 3.4626 #3.596 
            self.kcr = 2.0       
            self.kc = 50 #50            

            
        elif body == '0.6_M_Earth':

            self.name = '0.6_M_Earth'
            self.Rp   = 5500e3 #5507e3
            self.Rc   = 3050e3 #2810e3   
            self.g    = 7.9231 #8.06     
            self.rhom = 4158.4 #4153.23
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 9922.3 #11431.64
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2160.1 #1650.            
            self.Tc0  = 4371.0 #4356.35            
            self.cc =  1185.3 #1090.07      
            self.cm = 1237.6 #1216.0       
            self.ccr = 1000.0       
            self.alpha = 2.27430e-05 #2.14e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 3        
            self.Qtidal = 0         
            self.L    = 6e5
            self.epsm = 1.14134 #1.201
            self.epsc = 1.11770  #1.086
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km_u = 3.7241 #3.87
            self.km_b = 3.7241 
            self.kcr = 2.0      
            self.kc = 50        
            
            
        elif body == '0.7_M_Earth':

            self.name = '0.7_M_Earth'
            self.Rp   = 5752e3 #5758e3
            self.Rc   = 3181e3 #2930e3  
            self.g    = 8.4518 #8.62    
            self.rhom = 4234.7  #4235.70
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 10199.3 #11765.24
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2176.1 #1650.          
            self.Tc0  = 4522.0 #4464.28
            self.cc = 1171.2 # 1066.26    
            self.cm = 1237.8 #1217.27      
            self.ccr = 1000.0        
            self.alpha = 2.20696e-05 #2.08e-5
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6
            self.X0_CO2 = 50e-6     
            self.lam  = 3          
            self.Qtidal = 0         
            self.L    = 6e5
            self.epsm = 1.15679 #1.172
            self.epsc = 1.12349 #1.090
            self.Pcrossover =12e+9   
            self.dIW = 4 
            self.km_u = 3.9771 #4.13  
            self.km_b = 3.9771 #4.13  
            self.kcr = 2.0           
            self.kc = 50             
            
            
        elif body == '0.8_M_Earth':

            self.name = '0.8_M_Earth'
            self.Rp   = 5978e3 #5992e3
            self.Rc   = 3297e3 #3038e3   
            self.g    = 8.9436 #9.12     
            self.rhom = 4304.4 #4286.1
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 10470.1 #12060.71
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2191.7 #1650.            
            self.Tc0  = 4616.5  #4631.7
            self.cc =  1147.7 #1058.4     
            self.cm = 1237.9 #1216.0      
            self.ccr = 1000.0         
            self.alpha = 2.14710e-05 #2.01e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 3         
            self.Qtidal = 0       
            self.L    = 6e5
            self.epsm = 1.17116 #1.144
            self.epsc = 1.12849 #1.093
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km_u = 4.2236 #4.375 
            self.km_b = 4.2236
            self.kcr = 2.0          
            self.kc = 50            
            
            
        elif body == '0.9_M_Earth':

            self.name = '0.9_M_Earth'
            self.Rp   = 6184e3 #6193e3
            self.Rc   = 3402e3 #3136e3   
            self.g    = 9.4024 # 9.62    
            self.rhom = 4368.0 #4367.14
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 10717.5 #12337.25
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2203.5 #1650.      
            self.Tc0  = 4738.9 #4825.54
            self.cc =  1134.5 #1056.2       
            self.cm = 1237.6 #1210.7      
            self.ccr = 1000.0        
            self.alpha = 2.09081e-05 #1.92e-5
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6      
            self.lam  = 3           
            self.Qtidal = 0         
            self.L    = 6e5
            self.epsm = 1.18471  #1.932
            self.epsc = 1.13313 #1.097
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km_u = 4.4643 #4.714 
            self.km_b = 4.4643
            self.kcr = 2.0          
            self.kc = 50              

           
        elif body == '1.0_M_Earth':

            self.name = '1.0_M_Earth'
            self.Rp   = 6373e3 #6377e3 
            self.Rc   = 3499e3 #3225e3   
            self.g    = 9.8369 #10.0    
            self.rhom = 4428.7 #4440.86
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 10946.2 #12595.47
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2215.2 #1702.5            
            self.Tc0  = 4891.7 #5834.28 #5020.19
            self.cc =  1130.0 #1055.93  
            self.cm = 1237.7 #1213.84         
            self.ccr = 1000.0         
            self.alpha = 2.04471e-05 #1.92e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 3            
            self.Qtidal = 0        
            self.L    = 6e5
            self.epsm = 1.19806 #1.92
            self.epsc = 1.13748 #1.099
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km_u = 4.6993 #2.2320  #4.6993  #4.966   
            self.km_b = 4.6993 #9.3862
            self.kcr = 2.0           
            self.kc = 50             
            
            
        elif body == '1.1_M_Earth':

            self.name = '1.1_M_Earth'
            self.Rp   = 6549e3 #6558e3
            self.Rc   = 3589e3 #3308e3   
            self.g    = 10.2490 #10.5   
            self.rhom = 4485.7 #4487.8
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 11159.7 #12843.58
            self.rho_H2O = 1000.0
            self.Ts   = 300 # 288.0       
            self.Tm0  = 2233.0 #1700.        
            self.Tc0  = 5057.6 #5211.7
            self.cc = 1129.1  # 1056.05        
            self.cm = 1238.3  #1213.15         
            self.ccr = 1000.0        
            self.alpha = 2.00682e-05 #1.87e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 #1e-3       
            self.lam  = 3            
            self.Qtidal = 0           
            self.L    = 6e5
            self.epsm = 1.21032 #1.897
            self.epsc = 1.14126 #1.102
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km_u = 4.9263 #5.211 
            self.km_b = 4.9263
            self.kcr = 2.0           
            self.kc = 50            

            
        elif body == '1.2_M_Earth':

            self.name = '1.2_M_Earth'
            self.Rp   = 6712e3 #6725e3
            self.Rc   = 3673e3 #3385e3   
            self.g    = 10.6426 #10.94   
            self.rhom = 4539.8 #4533.97
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 11361.9 #13081.88
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2249.1 #1700.            
            self.Tc0  = 5226.5 #5399.70
            self.cc =  1129.6 #1056.29      
            self.cm = 1238.7 #1212.56       
            self.ccr = 1000.0     
            self.alpha = 1.97040e-05 #1.82e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6       
            self.lam  = 3            
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.22212 #1.877
            self.epsc = 1.14461 #1.104
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km_u = 5.1492 #5.452 
            self.km_b = 5.1492
            self.kcr = 2.0           
            self.kc = 50             
            
            
        elif body == '1.25_M_Earth':

            self.name = '1.25_M_Earth'
            self.Rp   = 6806e3
            self.Rc   = 3421e3 
            self.g    = 11.13  
            self.rhom = 4555.29
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 13197.22
            self.rho_H2O = 1000.0
            self.Ts   = 288.0       
            self.Tm0  = 1700.       
            self.Tc0  = 5491.34
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
            self.lam  = 3             
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
            self.Rp   = 6866e3 #6883e3
            self.Rc   = 3751e3 #3456e3   
            self.g    = 11.0199 #11.33  
            self.rhom = 4591.4 #4578.45
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 11555.1 #13312.10
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2263.7 #1700.          
            self.Tc0  = 5394.4 #5583.56
            self.cc = 1130.6 # 1056.46          
            self.cm = 1239.0 #1212.19       
            self.ccr = 1000.0        
            self.alpha = 1.93680e-05 #1.78e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 3             
            self.Qtidal = 0     
            self.L    = 6e5
            self.epsm = 1.23378 #1.859
            self.epsc = 1.14758 #1.107
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km_u = 5.3693 #5.691  
            self.km_b = 5.3693
            self.kcr = 2.0           
            self.kc = 50             
      
            
        elif body == '1.4_M_Earth':

            self.name = '1.4_M_Earth'
            self.Rp   = 7010e3 #7015e3
            self.Rc   = 3824e3 #3523e3   
            self.g    = 11.3833 #11.75 
            self.rhom = 4640.9 #4657.083
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 11740.7 #13531.23
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2277.0 #1700.            
            self.Tc0  = 5559.7 #5764.63
            self.cc =  1131.7 #1056.72      
            self.cm = 1239.3 #1215.065        
            self.ccr = 1000.0        
            self.alpha = 1.90575e-05 #1.78e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 3            
            self.Qtidal = 0       
            self.L    = 6e5
            self.epsm = 1.24542 #1.856
            self.epsc = 1.15029 #1.108
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km_u = 5.5872 #5.925   
            self.km_b = 5.5872
            self.kcr = 2.0           
            self.kc = 50           
            
            
        elif body == '1.5_M_Earth':
            
            self.name = '1.5_M_Earth'
            self.Rp   = 7147e3 #7155e3
            self.Rc   = 3894e3 #3587e3  
            self.g    = 11.7344 #12.12   
            self.rhom = 4688.6  #4697.73
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 11920.1 #13746.70
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2288.9 #1700.            
            self.Tc0  = 5721.9 #5938.71
            self.cc =  1132.7 #1056.54         
            self.cm = 1239.4 #1214.10        
            self.ccr = 1000.0      
            self.alpha = 1.87542e-05 #1.73e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 3         
            self.Qtidal = 0      
            self.L    = 6e5
            self.epsm = 1.25662 #1.840
            self.epsc = 1.15275 #1.110
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km_u = 5.8031 #6.158    
            self.km_b = 5.8031
            self.kcr = 2.0     
            self.kc = 49.7994 #50          

            
        elif body == '1.6_M_Earth':

            self.name = '1.6_M_Earth'
            self.Rp   = 7277e3 #7280e3
            self.Rc   = 3959e3 #3646e3   
            self.g    = 12.0740 #12.50
            self.rhom =  4734.5 #4756.15
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 12093.7 #13954.55
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2299.7 #1700.0            
            self.Tc0  = 5880.6 #6110.08
            self.cc =  1133.6 #1056.33       
            self.cm = 1239.5 #1216.0        
            self.ccr = 1000.0      
            self.alpha = 1.84757e-05 #1.72e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 3        
            self.Qtidal = 0     
            self.L    = 6e5
            self.epsm = 1.26792 #1.828
            self.epsc = 1.15503 #1.111
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km_u = 6.0171  #6.399  
            self.km_b = 6.0171
            self.kcr = 2.0         
            self.kc = 50           
            
        elif body == '1.7_M_Earth':

            self.name = '1.7_M_Earth'
            self.Rp   = 7400e3 #7429e3  
            self.Rc   = 4021e3 #3730e3   
            self.g    = 12.4044  #12.48
            self.rhom = 4779.4 #4760.97
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 12262.5 #13846.09
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2309.2 #1700.0        
            self.Tc0  = 6036.1 #5934.05
            self.cc =  1134.3 #1050.81     
            self.cm = 1239.5 #1216.08       
            self.ccr = 1000.0       
            self.alpha = 1.8191e-05 #1.72e-05 
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 3        
            self.Qtidal = 0      
            self.L    = 6e5
            self.epsm = 1.27860 #1.803
            self.epsc = 1.15714 #1.117
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km_u = 6.2300 #6.423
            self.km_b = 6.2300
            self.kcr = 2.0        
            self.kc = 50          

            
        elif body == '1.8_M_Earth':

            self.name = '1.8_M_Earth'
            self.Rp   = 7519e3 #7534e3
            self.Rc   = 4081e3 #3766e3   
            self.g    = 12.7247 #13.064
            self.rhom = 4822.3 #4822.95
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 12426.5 #14244.51
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2317.6 #1700.0            
            self.Tc0  = 6188.2 #6362.23
            self.cc =  1134.8 #1061.165     
            self.cm = 1239.5 #1215.389      
            self.ccr = 1000.0 
            self.alpha = 1.79371e-05 #1.67e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6
            self.X0_CO2 = 50e-6 
            self.lam  = 3     
            self.Qtidal = 0     
            self.L    = 6e5
            self.epsm = 1.28962 #1.801
            self.epsc = 1.15907 #1.121
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km_u = 6.4417 #6.775  
            self.km_b = 6.4417
            self.kcr = 2.0           
            self.kc = 50             
            
        elif body == '1.9_M_Earth':

            self.name = '1.9_M_Earth'
            self.Rp   = 7631e3 #7648e3 
            self.Rc   = 4137e3 #3822e3  
            self.g    = 13.0371 #13.09
            self.rhom = 4864.2 #4867.82
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 12586.4 #14244.51
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2336.1 #1700.0    
            self.Tc0  = 6337.1 #6578.2
            self.cc =  1135.1 #1060.226 
            self.cm = 1239.8 #1215.926       
            self.ccr = 1000.0         
            self.alpha = 1.77019e-05 #1.644e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6       
            self.lam  = 3          
            self.Qtidal = 0      
            self.L    = 6e5
            self.epsm = 1.29917 #1.796
            self.epsc = 1.16092 #1.104
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km_u = 6.6451 #7.02 
            self.km_b = 6.6451
            self.kcr = 2.0          
            self.kc = 50             

            
        elif body == '2.0_M_Earth':

            self.name = '2.0_M_Earth'
            self.Rp   = 7741e3 #7758e3
            self.Rc   = 4191e3 #3871e3   
            self.g    = 13.3398  #13.404
            self.rhom = 4903.7 #4903.61
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 12742.4 #14572.8
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2342.6 #1710.69      
            self.Tc0  = 6483.1 #8055.10 #6733.53
            self.cc =  1135.2 #1059.454    
            self.cm = 1239.5 #1215.298     
            self.ccr = 1000.0    
            self.alpha = 1.74484e-05 #1.61e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6
            self.lam  = 3          
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.30933 #1.106
            self.epsc = 1.16262 #1.121
            self.Pcrossover =  12e+9   
            self.dIW = 4 
            self.km_u = 6.8538 #7.237       
            self.km_b = 6.8538 #7.237 
            self.kcr = 2.0          
            self.kc = 50            
            
            
        elif body == '2.1_M_Earth':

            self.name = '2.1_M_Earth'
            self.Rp   = 7845e3 #7864e3
            self.Rc   = 4243e3 #3919e3
            self.g    = 13.6375 #13.713 
            self.rhom = 4943.5 #4941.34
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 12895.1 #14755.4
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2348.1 #1750.0            
            self.Tc0  = 6626.5 #6887.12 
            self.cc =  1135.2 #1058.604    
            self.cm = 1239.4 #1215.798      
            self.ccr = 1000.0       
            self.alpha = 1.72165e-05 #1.59e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6           
            self.X0_CO2 = 50e-6 
            self.lam  = 3          
            self.Qtidal = 0         
            self.L    = 6e5
            self.epsm = 1.31964 #1.779
            self.epsc = 1.16421 #1.107 
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km_u = 7.0629 #7.451   
            self.km_b = 7.0629
            self.kcr = 2.0            
            self.kc = 50              

            
        elif body == '2.2_M_Earth':

            self.name = '2.2_M_Earth'
            self.Rp   = 7945e3 #7967e3
            self.Rc   = 4293e3 #3963e3   
            self.g    = 13.9289 #13.404
            self.rhom = 4982.4 #4976.16
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 13044.3 #14934.1
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2367.2 #1750.0            
            self.Tc0  =  6766.9 #7037.085
            self.cc =  1135.1 #1057.65      
            self.cm = 1239.7 #1215.191     
            self.ccr = 1000.0        
            self.alpha = 1.70242e-05 #1.57e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 3            
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.32904 #1.769
            self.epsc = 1.16574 #1.108
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km_u = 7.2604 #7.667 
            self.km_b = 7.2604
            self.kcr = 2.0           
            self.kc = 50          
            
            
        elif body == '2.3_M_Earth':

            self.name = '2.3_M_Earth'
            self.Rp   = 8041e3 #8066e3
            self.Rc   = 4341e3 #4007e3 
            self.g    = 14.2150 #14.306
            self.rhom = 5021.0 #5010.44
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 13190.7 #15109.58
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2371.5 #1750.0         
            self.Tc0  = 6905.2 #7184.66
            self.cc =  1134.8 #1056.642   
            self.cm = 1239.6 #1214.610    
            self.ccr = 1000.0        
            self.alpha = 1.68223e-05 #1.539e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6     
            self.X0_CO2 = 50e-6
            self.lam  = 3             
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.33947 #1.759
            self.epsc = 1.16717 #1.109
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km_u = 7.4675 #7.882   
            self.km_b = 7.4675
            self.kcr = 2.0            
            self.kc = 50            
  
        elif body == '2.4_M_Earth':

            self.name = '2.4_M_Earth'
            self.Rp   = 8136e3 #8144e3
            self.Rc   = 4387e3 #4050e3  
            self.g    = 14.4918  #14.64
            self.rhom = 5056.4 #5080.54
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 13334.0 #15274.78
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2389.8 #1750.0            
            self.Tc0  = 7040.6 #7328.69
            self.cc =  1134.5 #1057.65     
            self.cm = 1239.6 #1215.191     
            self.ccr = 1000.0       
            self.alpha = 1.65964e-05 #1.547e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6
            self.lam  = 3            
            self.Qtidal = 0         
            self.L    = 6e5
            self.epsm = 1.34713 #1.75
            self.epsc = 1.16855 #1.109
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km_u = 7.6628 #8.092
            self.km_b = 7.6628
            self.kcr = 2.0           
            self.kc = 50   
            
        elif body == '2.5_M_Earth':

            self.name = '2.5_M_Earth'
            self.Rp   =  8226e3 #8235e3
            self.Rc   = 4432e3 #4090e3 
            self.g    = 14.7652  #14.930 
            self.rhom = 5091.8 #5116.19 
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 13474.6 #15444.82
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2408.6 #1750.0            
            self.Tc0  = 7174.0 #7472.6 
            self.cc =  1134.0 #1054.667     
            self.cm = 1240.1 #1219.024       
            self.ccr = 1000.0       
            self.alpha = 1.64250e-05 #1.54e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6      
            self.X0_CO2 = 50e-6    # set to very high value, maybe beneath this (be careful with magma formation in the lower mantle)
            self.lam  = 3           
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.35751 #1.746
            self.epsc = 1.16982 #1.110
            self.Pcrossover = 12e+9 
            self.dIW = 4 
            self.km_u = 7.8511 #8.298    
            self.km_b = 7.8511
            self.kcr = 2.0             
            self.kc = 50               
            
            
        elif body == '2.6_M_Earth':

            self.name = '2.6_M_Earth'
            self.Rp   = 8314e3 #8324e3
            self.Rc   = 4475e3 #4129e3  
            self.g    = 15.0327 #15.210
            self.rhom = 5126.8 #5148.91
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 13613.0 #15611.72
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2410.5 #1750.0            
            self.Tc0  = 7305.7 #7613.56
            self.cc =  1133.4 #1053.481   
            self.cm = 1239.5 #1218.546   
            self.ccr = 1000.0        
            self.alpha = 1.62264e-05 #1.51e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6       
            self.lam  = 3            
            self.Qtidal = 0           
            self.L    = 6e5
            self.epsm = 1.36559 #1.738
            self.epsc = 1.17106 #1.111
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km_u = 8.0608 #8.509 
            self.km_b = 8.0608 
            self.kcr = 2.0            
            self.kc = 50               
            
        elif body == '2.7_M_Earth':

            self.name = '2.7_M_Earth'
            self.Rp   = 8399e3 #8411e3 
            self.Rc   = 4516e3 #4167e3
            self.g    = 15.2976 #15.486
            self.rhom = 5161.9 #5181.14 
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 13748.6 #15776.14
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2428.7 #1750.0           
            self.Tc0  = 7435.0 #7752.63 
            self.cc =  1132.8 #1052.258      
            self.cm = 1239.8 #1218.092        
            self.ccr = 1000.0         
            self.alpha = 1.60562e-05 #1.49e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6      
            self.X0_CO2 = 50e-6  
            self.lam  = 3            
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.37353 #1.730
            self.epsc = 1.17222 #1.112
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km_u = 8.2532 #8.720    
            self.km_b = 8.2532 
            self.kcr = 2.0            
            self.kc = 50             

            
        elif body == '2.8_M_Earth':

            self.name = '2.8_M_Earth'
            self.Rp   = 8481e3 #8494e3
            self.Rc   = 4557e3 #4203e3  
            self.g    = 15.5574 #15.757
            self.rhom = 5196.0 #5212.80
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 13882.6 #15938.31 
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2429.0 #1750.0            
            self.Tc0  = 7563.0 #7889.802
            self.cc =  1132.1 #1050.986       
            self.cm = 1239.4 #1217.666    
            self.ccr = 1000.0         
            self.alpha = 1.58728e-05 #1.47e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6 
            self.lam  = 3            
            self.Qtidal = 0         
            self.L    = 6e5
            self.epsm = 1.38302 #1.722
            self.epsc = 1.17333 #1.113
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km_u = 8.4584 #8.929  
            self.km_b = 8.4584
            self.kcr = 2.0            
            self.kc = 50              
            
        elif body == '2.9_M_Earth':

            self.name = '2.9_M_Earth'
            self.Rp   = 8561e3 #8575e3
            self.Rc   =  4596e3 #4239e3
            self.g    = 15.8146 #16.028 
            self.rhom = 5230.3 #5246.71
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 14013.9 #16098.29
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2446.5 #1750.0    
            self.Tc0  = 7688.7 #8026.00
            self.cc =   1131.3 #1049.717    
            self.cm = 1239.7 #1218.649       
            self.ccr = 1000.0        
            self.alpha = 1.57228e-05 #1.46e-05 
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6      
            self.X0_CO2 = 50e-6  
            self.lam  = 3              
            self.Qtidal = 0          
            self.L    = 6e5
            self.epsm = 1.39093 #1.721
            self.epsc = 1.17440 #1.114
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km_u = 8.6495 #9.130   
            self.km_b = 8.6495
            self.kcr = 2.0            
            self.kc = 50               

            
        elif body == '3.0_M_Earth':

            self.name = '3.0_M_Earth'
            self.Rp   = 8639e3 #8654e3
            self.Rc   = 4634e3 #4273e3  
            self.g    = 16.0652 #16.29
            self.rhom = 5262.5 #5277.84
            self.rhocr = 3000.0 #2900.0      
            self.rhoc = 14143.5 #16255.87 
            self.rho_H2O = 1000.0
            self.Ts   = 300 #288.0       
            self.Tm0  = 2463.9 #1715.45 #1800          
            self.Tc0  = 7812.7 #9841.39 #8159.92
            self.cc =  1130.5 #1048.42     
            self.cm = 1239.8 #1218.23      
            self.ccr = 1000.0        
            self.alpha = 1.55622e-05 #1.44e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6 
            self.X0_CO2 = 50e-6        
            self.lam  = 3             
            self.Qtidal = 0      
            self.L    = 6e5
            self.epsm = 1.39819 #1.714
            self.epsc = 1.17544 #1.115
            self.Pcrossover = 12e+9  
            self.dIW = 4 
            self.km_u =  8.8393 #9.339  
            self.km_b =  8.8393 #
            self.kcr = 2.0       
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
            self.lam  = 3           
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
            
        elif body == '1.0_M_Earth_Fe30':

            self.name = '1.0_M_Earth_Fe30'
            self.Rp   = 6463e3
            self.Rc   = 3026e3  
            self.g    = 9.70
            self.rhom = 4458.6
            self.rhocr = 2900.0      
            self.rhoc = 12471
            self.rho_H2O = 1000.0
            self.Ts   = 288.      
            self.Tm0  = 1600. #1315.86        
            self.Tc0  = 5077
            self.cc =  1065.377     
            self.cm = 1215.41     
            self.ccr = 1000.0          
            self.alpha = 1.19229e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6  
            self.X0_CO2 = 100e-6 
            self.lam  = 1 #3           
            self.Qtidal = 0         
            self.L    = 6e5
            self.epsm = 1.911
            self.epsc = 1.080
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km =  5.0149        
            self.kcr = 3.0            
            self.kc = 50 
            
            
        elif body == '2.0_M_Earth_Fe30':

            self.name = '2.0_M_Earth_Fe30'
            self.Rp   = 7861e3
            self.Rc   = 3621e3  
            self.g    = 13.263
            self.rhom = 4927.75
            self.rhocr = 2900.0      
            self.rhoc = 14560.
            self.rho_H2O = 1000.0
            self.Ts   = 288.      
            self.Tm0  = 1700.    #1714.76      
            self.Tc0  = 6838.014
            self.cc =  1064.376    
            self.cm = 1215.815  
            self.ccr = 1000.0          
            self.alpha = 1.604899e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6  
            self.X0_CO2 = 100e-6 
            self.lam  = 1 #3           
            self.Qtidal = 0         
            self.L    = 6e5
            self.epsm = 1.776
            self.epsc = 1.0951
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km =  7.3662       
            self.kcr = 3.0            
            self.kc = 50 
            
            
        elif body == '3.0_M_Earth_Fe30':

            self.name = '3.0_M_Earth_Fe30'
            self.Rp   = 8769e3
            self.Rc   = 3997e3  
            self.g    = 16.1113
            self.rhom = 5307.13
            self.rhocr = 2900.0      
            self.rhoc = 16242.
            self.rho_H2O = 1000.0
            self.Ts   = 288.      
            self.Tm0  = 1700 #1800.    #2098.318     
            self.Tc0  = 8292.574
            self.cc =  1053.022   
            self.cm = 1218.5595 
            self.ccr = 1000.0          
            self.alpha = 1.431051e-05
            # Heat sources from McDonough & Sun (1995), Chem. Geol., 223-253.
            self.X_U  = 20e-9
            self.X_Th = 80e-9 
            self.X_K  = 240e-6 
            self.X0_H2O  = 100e-6  
            self.X0_CO2 = 100e-6 
            self.lam  = 1 #3           
            self.Qtidal = 0         
            self.L    = 6e5
            self.epsm = 1.7059
            self.epsc = 1.1032
            self.Pcrossover = 12e+9   
            self.dIW = 4 
            self.km =  9.5218       
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
