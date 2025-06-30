#####################################################
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker, cm, colors
import scipy as scp

import interior_evolution
import support_functions as suppf
#####################################################

#####################################################
def plot_evolution(s): 
    """Plot time evolution of main quantities"""
#####################################################

    yr = 365.0*24.0*60.0*60.0   # 1 year in seconds
        
    fig = plt.figure(figsize=(28,65)) #(figsize=(12,12))    (16,20)  (23,40)
    plt.tight_layout()
    small_size = 10
    medium_size = 15
    bigger_size = 20
   

    lw = 2.5 

    mcolor = 'tab:blue'
    bcolor = 'tab:orange'
    ccolor = 'tab:red'
    lcolor = 'tab:green'
    crcolor = 'tab:orange'
    
    plt.rc('font', size=bigger_size)          # controls default text sizes
    plt.rc('axes', titlesize=medium_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=bigger_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=bigger_size)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=bigger_size)   # fontsize of the tick labels
    plt.rc('legend', fontsize=bigger_size)   # legend fontsize_
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

    nx_panels = 13
    ny_panels = 2
    n = 0
    
    

    ############################################
    # Heat production
    ############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[:-1]/yr/1e6,  s.Q_U238[:-1]*1e12, label='$^{238}$U', lw=lw)
    ax.plot(s.t[:-1]/yr/1e6,  s.Q_U235[:-1]*1e12, label='$^{235}$U', lw=lw)
    ax.plot(s.t[:-1]/yr/1e6, s.Q_Th232[:-1]*1e12, label='$^{232}$Th', lw=lw)
    ax.plot(s.t[:-1]/yr/1e6,   s.Q_K40[:-1]*1e12, label='$^{40}$K', lw=lw)
    ax.plot(s.t[:-1]/yr/1e6,   s.Q_tot[:-1]*1e12, label='Total', lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Heat production [pW/kg]')
    ax.grid()
    ax.legend(loc=1)

    ############################################
    # Heating rate with HPE redistribution
    ############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[:-1]/yr/1e6, s.Qm[:-1]*1e12, label='Qm', lw=lw)
    ax.plot(s.t[:-1]/yr/1e6, s.Qcr[:-1]*1e12, label='Qcr', lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Heating rate [pW/kg]')
    ax.grid()
    ax.legend(loc=1)

    ############################################
    # HPE mass in mantle
    ############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[:-1]/yr/1e6, s.Xm_U238[:-1]*1e6, '--', label='$U_{238}m$', color=mcolor, lw=lw)
    ax.plot(s.t[:-1]/yr/1e6, s.Xm_U235[:-1]*1e6, '--', label='$U_{235}m$', color=lcolor, lw=lw)
    ax.plot(s.t[:-1]/yr/1e6, s.Xm_Th232[:-1]*1e6, '--', label='$Th_{232}m$', color=bcolor, lw=lw)
    ax.plot(s.t[:-1]/yr/1e6, s.Xm_K40[:-1]*1e6, '--', label='$K_{40}m$', color=ccolor, lw=lw) 
    ax.plot(s.t[:-1]/yr/1e6, s.Xm_HPE[:-1]*1e6, '-', label='$HPE_m$', color=ccolor, lw=lw) 
    
    #ax.plot(s.t[:-1]/yr/1e6, s.Mscr_tot[:-1]/s.Mcr[:-1], label='$M_{tot}$', lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('HPE in mantle [ppm]')
    ax.grid()
    ax.legend(loc=1)
    
    ############################################
    # HPE mass in  crust
    ############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[:-1]/yr/1e6, s.Mcr_U238[:-1]/s.Mcr[:-1]*1e6, label='$U_{238}$', color=mcolor, lw=lw)
    ax.plot(s.t[:-1]/yr/1e6, s.Mcr_U235[:-1]/s.Mcr[:-1]*1e6, label='$U_{235}$', color=lcolor, lw=lw)
    ax.plot(s.t[:-1]/yr/1e6, s.Mcr_Th232[:-1]/s.Mcr[:-1]*1e6, label='$Th_{232}$', color=bcolor, lw=lw)
    ax.plot(s.t[:-1]/yr/1e6, s.Mcr_K40[:-1]/s.Mcr[:-1]*1e6, label='$K_{40}$', color=ccolor, lw=lw)

    #ax.plot(s.t[:-1]/yr/1e6, s.Mscr_tot[:-1]/s.Mcr[:-1], label='$M_{tot}$', lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('HPE in crust [ppm]')
    ax.grid()
    ax.legend(loc=1)
    
    ############################################
    # K mass in  crust
    ############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
   # ax.plot(s.t[:-1]/yr/1e6, s.Xcrust_K[:-1]*1e6, color=crcolor, lw=lw)
    ax.plot(s.t[:-1]/yr/1e6, s.Mcr_K[:-1]/s.Mcr[:-1]*1e6, color=crcolor, lw=lw)
    
    #ax.plot(s.t[:-1]/yr/1e6, s.Mscr_tot[:-1]/s.Mcr[:-1], label='$M_{tot}$', lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('K in crust [ppm]')
    ax.grid()
    #ax.legend(loc=1)
    
    ############################################
    # HPE mass in liquid
    ############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[:-1]/yr/1e6, s.X_U238_liq[:-1]*1e6, '--', label='$U_{238}m$', color=mcolor, lw=lw)
    ax.plot(s.t[:-1]/yr/1e6, s.X_U235_liq[:-1]*1e6, '--', label='$U_{235}m$', color=lcolor, lw=lw)
    ax.plot(s.t[:-1]/yr/1e6, s.X_Th232_liq[:-1]*1e6, '--', label='$Th_{232}m$', color=bcolor, lw=lw)
    ax.plot(s.t[:-1]/yr/1e6, s.X_K40_liq[:-1]*1e6, '--', label='$K_{40}m$', color=ccolor, lw=lw) 
    
    #ax.plot(s.t[:-1]/yr/1e6, s.Mscr_tot[:-1]/s.Mcr[:-1], label='$M_{tot}$', lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('HPE in liquid [ppm]')
    ax.grid()
    
    
    """
    ############################################
    # Precent of redistributed HPE
    ############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[:-1]/yr/1e6, s.Xredist_HPE[:-1]*100, label='$HPE$', lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Redistributed HPE $[\%]$')
    ax.grid()
    ax.legend(loc=1)
    """

    """
    ############################################
    # HPE mass in liquid
    ############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[:-1]/yr/1e6, s.Mcr_U238[:-1]/s.Mcr[1:]*1e6, label='$U_{238}$', lw=lw)
    ax.plot(s.t[:-1]/yr/1e6, s.Mcr_U235[:-1]/s.Mcr[1:]*1e6, label='$U_{235}$', lw=lw)
    ax.plot(s.t[:-1]/yr/1e6, s.Mcr_Th232[:-1]/s.Mcr[1:]*1e6, label='$Th_{232}$', lw=lw)
    ax.plot(s.t[:-1]/yr/1e6, s.Mcr_K40[:-1]/s.Mcr[1:]*1e6, label='$K_{40}$', lw=lw)
    
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[:-1]/yr/1e6, s.Xm_U238[:-1]*1e6, '--', label='$U_{238}m$', lw=lw)
    ax.plot(s.t[:-1]/yr/1e6, s.Xm_U235[:-1]*1e6, '--', label='$U_{235}m$', lw=lw)
    ax.plot(s.t[:-1]/yr/1e6, s.Xm_Th232[:-1]*1e6, '--', label='$Th_{232}m$', lw=lw)
    ax.plot(s.t[:-1]/yr/1e6, s.Xm_K40[:-1]*1e6, '--', label='$K_{40}m$', lw=lw) 
    
    #ax.plot(s.t[:-1]/yr/1e6, s.Mscr_tot[:-1]/s.Mcr[:-1], label='$M_{tot}$', lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('HPE in sec. crust [ppm]')
    ax.grid()
    ax.legend(loc=1)
    """
    ############################################
    # K amount in liquid melt
    ############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[1:-1]/yr/1e6, s.X_K_liq[1:-1]*100, label='$X_K^{melt}$', lw=lw)
    
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('K in melt [%]')
    ax.grid()
    
    ############################################
    # Urey ratio & depletion
    ############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    #ax.plot(s.t[1:-1]/yr/1e6,  s.Ur[1:-1], label='', lw=lw)
    ax.plot(s.t[1:-1]/yr/1e6,  s.depl[1:-1]*100, label='', lw=lw)
    ax.set_xlabel('Time [Myr]')
    #ax.set_ylabel('Urey ratio')
    ax.set_ylabel('Depletion $[\%]$')
    ax.grid()

    ##################################################################################
    # Mantle temperature, CMB temperature and temperature at the top of the lower TBL
    ##################################################################################    
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ##############################################################################
  #  T_belowsol, T_abovesol = suppf.melting_range(s)
    ax.plot(s.t[:-1]/yr/1e6, s.Tm[:-1], '-', color=mcolor, label='$T_m$', lw=lw)
  #  ax.plot(s.t[:-1]/yr/1e6, T_abovesol, '-', color=mcolor, label='$T_m > T_{sol}$', lw=lw)
  #  ax.plot(s.t[:-1]/yr/1e6, T_belowsol, '--', color=mcolor, label='$T_m < T_{sol}$', lw=lw)
    ##############################################################################    
    ax.plot(s.t[:-1]/yr/1e6, s.Tc[:-1], label='$T_c$', color=ccolor, lw=lw)
    ax.plot(s.t[:-1]/yr/1e6, s.Tl[:-1], label='$T_l$', color=lcolor, lw=lw)
    ax.plot(s.t[:-1]/yr/1e6, s.Tcr[:-1], label='$T_{cr}$', color=crcolor, lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Temperature [K]')
    ax.grid()
    ax.legend(loc=1)

    ############################################
    # Mantle and CMB viscosities
    ############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[1:-1]/yr/1e6, s.etam[1:-1], label='$\eta_m$', color=mcolor, lw=lw)
    ax.plot(s.t[1:-1]/yr/1e6, s.etac[1:-1], label='$\eta_c$', color=ccolor, lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Viscosity [Pa s]')
    ax.set_yscale('log')
    ax.grid()
    ax.legend(loc=1)

    ############################################
    # Surface and CMB heat flux
    ############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[1:-1]/yr/1e6, s.qs[1:-1]*1e3, label='$q_s$', color=mcolor, lw=lw)
    ax.plot(s.t[1:-1]/yr/1e6, s.qc[1:-1]*1e3, label='$q_c$', color=ccolor, lw=lw)
    ax.plot(s.t[1:-1]/yr/1e6, s.ql[1:-1]*1e3, label='$q_l$', color=lcolor, lw=lw)
    
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Heat flux [mW/m$^2$]')
    ax.grid()
    ax.legend(loc=1)

    ############################################
    # Top and bottom boundary layer thicknesses
    ############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[1:-1]/yr/1e6, s.delta_s[1:-1]/1e3, label='$\delta_s$', color=mcolor, lw=lw)
    ax.plot(s.t[1:-1]/yr/1e6, s.delta_c[1:-1]/1e3, label='$\delta_c$', color=ccolor, lw=lw)
    ax.plot(s.t[1:-1]/yr/1e6, s.Dl[1:-1]/1e3, label='$Dl$', color=lcolor, lw=lw)
    ax.plot(s.t[1:-1]/yr/1e6, s.Dcr[1:-1]/1e3, label='$Dcr$', color=crcolor, lw=lw, linestyle=':')
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Boundary layer thickness [km]')
    #ax.set_ylim([0.0, 100.0])
    ax.grid()
    ax.legend(loc=1)
    
    ############################################
    # Crust production rate
    ############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
   # ax.plot(s.t[:-1]/yr/1e6, s.Vcr_prod[:-1]/1e9, color=mcolor, lw=lw) 
    ax.plot(s.t[:-1]/yr/1e6, s.Dcr_prod[:-1]*s.yrs/1e9, color=mcolor, lw=lw) 
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Crust Production Rate [$km^3/yr$]')
    ax.grid()
    
    #############################################
    # H2O in liquid
    ############################################# 
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[1:-1]/yr/1e6, s.X_H2O_liq[1:-1]*1e2, color=mcolor, lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('$X_{H2O}^{melt}$')
    ax.grid()    
    
    #############################################
    # H2O in mantle
    #############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[1:-1]/yr/1e6, s.Xm_H2O[1:-1]*1e6, color=mcolor, lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('$X_{H2O}^m in ppm$')
    ax.grid()

    #############################################
    # H2O in crust
    #############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[:-1]/yr/1e6, s.Mcr_H2O[:-1]*1e6, color=ccolor, lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Mass $H_2O$ in crust [kg]')
    ax.grid()

    ''' 
    #############################################
    # X_H2O in crust and mantle
    #############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[1:-1]/yr/1e6, s.Mcr_H2O[1:-1]/s.Mcr[1:-1], color=ccolor, lw=lw)
    ax.plot(s.t[1:-1]/yr/1e6, s.Mm_H2O[1:-1]/s.Mcr[1:-1], color=ccolor, lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('$X_{H_2O}$')
    ax.grid()
    ax.legend(loc=1)

    '''
    #############################################
    # H2O equivalent global layer thickness
    #############################################   
    
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[1:-1]/yr/1e6, s.EGL[1:-1], color=ccolor, lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('EGL [m]')
    ax.grid()

    #############################################
    # Mass of outgassed H2O
    #############################################
    
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[1:-1]/yr/1e6, s.M_H2O_gas[1:-1], color=ccolor, lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('$M^{gas}_{H2O} [kg]$')
    ax.grid()

    """
    #############################################
    # Concentration of Carbonate in melt over time
    #############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[1:-1]/yr/1e6, s.X_carbonate_melt[1:-1], color=ccolor, lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('$X_{CO_3^{2-}}^{melt}$')
    ax.grid()
    """
    
    #############################################
    # Concentration of water in mantle over time
    #############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[1:-1]/yr/1e6, s.Xm_H2O[1:-1]*1e6, color=ccolor, lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('$X_{H2O}^{m} [ppm]$')
    ax.grid()
    
    #############################################
    # average melt fraction
    #############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[:-1]/yr/1e6, s.F_av[:-1], color=ccolor, lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('$F$')
    ax.grid()

    #############################################
    # H2O 
    #############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[1:-1]/yr/1e6, s.Xm_H2O[1:-1]*s.Mm_evol[1:-1]+s.Xcrust_H2O[1:-1]*s.Mcr[1:-1], color=ccolor, lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('$X_{H2O}$')
    ax.grid()

    #############################################
    # Mass of H2O in secondary crust
    #############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[:-1]/yr/1e6, s.Mscr_H2O[:-1], color=ccolor, lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('$X_{H2O}^{crust}$')
    ax.grid()

    
    """
    #############################################
    # Rayleigh number
    #############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[:-1]/yr/1e6, s.Ra[:-1], color=ccolor, lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('$Rayleigh number$')
    ax.grid()
    """

#####################################################
def plot_evolution_2(s): 
    """Plot time evolution of main quantities"""
#####################################################

    yr = 365.0*24.0*60.0*60.0   # 1 year in seconds
        
    fig = plt.figure(figsize=(12,10)) #(figsize=(12,12))    (16,20)
    small_size = 10
    medium_size = 15
    bigger_size = 20
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    lw = 2.5 

    mcolor = 'tab:blue'
    bcolor = 'tab:orange'
    ccolor = 'tab:red'
    lcolor = 'tab:green'
    crcolor = 'tab:orange'
    
    plt.rc('font', size=small_size)          # controls default text sizes
    plt.rc('axes', titlesize=small_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=medium_size)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=medium_size)   # fontsize of the tick labels
    plt.rc('legend', fontsize=medium_size)   # legend fontsize
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title
    

    nx_panels = 1
    ny_panels = 1
    n = 0

    ############################################
    # HPE mass (isotopes in ppm) in secondary crust
    ############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[:-1]/yr/1e6, (s.Mcr_U238[:-1]/s.Mcr[1:])*1e6, linewidth = 5, label='$U_{238}^{cr}$', color=ccolor)
    ax.plot(s.t[:-1]/yr/1e6, (s.Mcr_U235[:-1]/s.Mcr[1:])*1e6, linewidth = 5, label='$U_{235}^{cr}$', color=bcolor)
    ax.plot(s.t[:-1]/yr/1e6, (s.Mcr_Th232[:-1]/s.Mcr[1:])*1e6, linewidth = 5, label='$Th_{232}^{cr}$', color=lcolor)
    ax.plot(s.t[:-1]/yr/1e6, (s.Mcr_K40[:-1]/s.Mcr[1:])*1e6, linewidth = 5, label='$K_{40}^{cr}$', color=mcolor)
    
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[:-1]/yr/1e6, s.Xm_U238[:-1]*1e6, '--', linewidth = 5 ,label='$U_{238}^m$', color=ccolor)
    ax.plot(s.t[:-1]/yr/1e6, s.Xm_U235[:-1]*1e6, '--', linewidth = 5, label='$U_{235}^m$', color=bcolor)
    ax.plot(s.t[:-1]/yr/1e6, s.Xm_Th232[:-1]*1e6, '--', linewidth = 5, label='$Th_{232}^m$', color=lcolor)
    ax.plot(s.t[:-1]/yr/1e6, s.Xm_K40[:-1]*1e6, '--', linewidth = 5, label='$K_{40}^m$', color=mcolor) 
    
    ax.set_xlabel('Time [Myr]',fontsize=40)
    ax.set_ylabel('HPE [ppm]',fontsize=40)
    ax.grid()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(30)
    plt.ylim(ymax = 9.5, ymin = -0.2)
    ax.legend(loc=1, prop={'size':30})
    
    plt.tight_layout()

#####################################################
def plot_H2O(s): 
    """Plot time evolution of main quantities"""
#####################################################
    yr = 365.0*24.0*60.0*60.0   # 1 year in seconds
    
    nx_panels = 1
    ny_panels = 1
    n = 0
       
    fig = plt.figure(figsize=(12,10)) #(figsize=(12,12))    (16,20)
    small_size = 10
    medium_size = 15
    bigger_size = 20
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    lw = 2.5 

    mcolor = 'tab:blue'
    bcolor = 'tab:orange'
    ccolor = 'tab:red'
    lcolor = 'tab:green'
    crcolor = 'tab:orange'
 
    #############################################
    # X_H2O in crust and mantle
    #############################################
    n = n + 1
    
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    #ax.plot(s.t[1:-1]/yr/1e6, s.X_H2O_liq[1:-1]*100, color=ccolor, lw=lw)
   # ax.plot(s.t[1:-1]/yr/1e6, s.X_H2O_liq[1:-1]/s.X0_H2O*s.Mcr[1:-1], color=ccolor, lw=lw)
    ax.plot(s.t[1:-1]/yr/1e6, s.Xm_H2O[1:-1]/s.X0_H2O, color=ccolor, lw=lw)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('$X_{H_2O}$')
    ax.grid()
    ax.legend(loc=1)
  #  print(s.Mcr_H2O[1:-1])
  #  print(s.Mcr[1:-1])

    plt.tight_layout()
    

#####################################################
def plot_potassium_mass(s): 
    """Plot time evolution of main quantities"""
#####################################################   
    # K mass in secondary crust
    
    yr = 365.0*24.0*60.0*60.0   # 1 year in seconds
        
    fig = plt.figure(figsize=(12,10)) #(figsize=(12,12))    (16,20)
    
    nx_panels = 1
    ny_panels = 1
    n = 0
    
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
   # ax.plot(s.t[2:-1]/yr/1e6, s.Mcr_K[2:-1]/s.Mcr[3:]*1e6, linewidth = 7, label='$K^{cr}$', c="midnightblue")
    #ax.plot(s.t[:-1]/yr/1e6, s.Xm_K[:-1]*1e6, '--', linewidth = 5, label='$K^m$', color=mcolor) 
    
    X_K_eta19 = np.genfromtxt(fname='K_Mcr_Tm01650_part_off_nomin_eta19.dat')
    X_K_eta20 = np.genfromtxt(fname='K_Mcr_Tm01650_part_off_nomin_eta20.dat')
    X_K_eta21 = np.genfromtxt(fname='K_Mcr_Tm01650_part_off_nomin_eta21.dat')
    #print(np.size(X_K_1600K[:-1]))
    #print(np.size(s.t[:-1]/yr/1e6))
    ax.plot(s.t[2:-1]/yr/1e6, X_K_eta19[2:-1]/s.Mcr[3:]*1e6, label='$\eta_{ref}=19$ Pa s', linewidth=7, c="coral")
    ax.plot(s.t[2:-1]/yr/1e6, X_K_eta20[2:-1]/s.Mcr[3:]*1e6, label='$\eta_{ref}=20$ Pa s',linewidth=7, c="crimson")
    ax.plot(s.t[2:-1]/yr/1e6, X_K_eta21[2:-1]/s.Mcr[3:]*1e6,  label='$\eta_{ref}=21$ Pa s', linewidth=7, c="maroon")
    
    ax.set_xlabel('Time [Myr]',fontsize=40)
    ax.set_ylabel('K in crust [ppm]',fontsize=40)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(30)
    ax.grid()
    #plt.ylim(ymax = 2800, ymin = 650)
    ax.legend(loc=1, prop={'size': 30})
    
    plt.tight_layout()

#####################################################
def plot_heat_fluxes(s): 
    """Plot time evolution of main quantities"""
#####################################################    
    yr = 365.0*24.0*60.0*60.0   # 1 year in seconds
        
    fig = plt.figure(figsize=(12,10)) #(figsize=(12,12))    (16,20)
    
    nx_panels = 1
    ny_panels = 1
    n = 0
    
    # Surface and CMB heat flux

    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[:-1]/yr/1e6, s.qs[:-1]*1e3, label='$q_s$', linewidth=7, c="midnightblue")
    ax.plot(s.t[:-1]/yr/1e6, s.qc[:-1]*1e3, label='$q_c$', linewidth=7, c="crimson")
    ax.plot(s.t[:-1]/yr/1e6, s.ql[:-1]*1e3, label='$q_l$', linewidth=7, c="orchid")
    
    ax.set_xlabel('Time [Myr]', fontsize=40)
    ax.set_ylabel('Heat flux [mW/m$^2$]',fontsize=40)
    # Set the tick labels font
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(30)
    ax.grid()
    plt.ylim(ymax = 90, ymin = 0)
    ax.legend(loc=1, prop={'size': 30})
    
    plt.tight_layout()
    
#####################################################
def plot_heating_rate(s): 
    """Plot time evolution of main quantities"""
#####################################################    
    yr = 365.0*24.0*60.0*60.0   # 1 year in seconds
        
    fig = plt.figure(figsize=(12,10)) #(figsize=(12,12))    (16,20)
    
    nx_panels = 1
    ny_panels = 1
    n = 0
    
    # Surface and CMB heat flux

    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    ax.plot(s.t[:-1]/yr/1e6, s.Qcr[:-1]*1e12, label='$Q_{cr}$', linewidth=7, c="midnightblue")
    ax.plot(s.t[:-1]/yr/1e6, s.Qm[:-1]*1e12, label='$Q_m$', linewidth=7, c="crimson")
    
    ax.set_xlabel('Time [Myr]', fontsize=40)
    ax.set_ylabel('Heating rate [pW/kg]',fontsize=40)
    # Set the tick labels font
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(30)
    ax.grid()
    plt.ylim(ymax = 250, ymin = 0)
    ax.legend(loc=1, prop={'size': 30})
    
    plt.tight_layout()

    
####################################################################
def plot_profiles(s, time, plot_solidus=0, plot_liquidus=0):
    """Plot temperature and viscosity profiles at a given time"""
####################################################################
    
    yr = 365.0*24.0*60.0*60.0
    etamax = 1e26
    Pmax = 10e9
    # Index of the time array closest to input time
    i = np.abs(s.t/yr/1e6 - time).argmin()
    
    fig = plt.figure(figsize=(15,5))    
    small_size = 10
    medium_size = 15
    bigger_size = 20

    Tcolor = 'tab:blue'
    Tsolcolor = 'tab:red'
    Tliqcolor = 'tab:red'
    etacolor = 'tab:blue'

    lw = 2.5 
        
    plt.rc('font', size=small_size)          # controls default text sizes
    plt.rc('axes', titlesize=small_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=medium_size)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=medium_size)   # fontsize of the tick labels
    plt.rc('legend', fontsize=medium_size)   # legend fontsize
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title
 
    nx_panels = 2
    ny_panels = 3
    n = 0
    
    nptb = 20
    npta = 150
    npts = 20
    #npt = nptb + npta + npts
    
    nptl = 20
    nptc = 20
    npt = nptb + npta + npts + nptl + nptc
    
    # Pressure at the two TBLs
    Pm = s.rhom*s.g*s.delta_s[i]
    Pb = s.rhom*s.g*(s.Rp - (s.Rc + s.delta_c[i]))

    rmin = s.Rc/1e3 - 400e3
    rmax = s.Rp/1e3 + 0.05*s.Rc/1e3 

    ########################
    # Temperature
    ########################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)

    # Adiabatic mantle
    ra = np.linspace(s.Rc + s.delta_c[i], s.Rp  - s.Dl[i], npta) #20.4. 
    Ta = suppf.calculate_adiabat(s, npta, s.Tm[i], Pm, Pb)
    Ta = np.flipud(Ta)
    Pa = s.rhom*s.g*(s.Rp - ra) 
    etaa = suppf.calculate_viscosity(s, Ta, Pa)

    # Lower boundary layer
    rb = np.linspace(s.Rc, s.Rc + s.delta_c[i], nptb)
    Trb = np.linspace(s.Tc[i], Ta[0], nptb)
    Prb = s.rhom*s.g*(s.Rp - rb) # früher: - (s.Rc + s.delta_c[i])
    etab = suppf.calculate_viscosity(s, Trb, Prb)

    # Upper boundary layer
    rs = np.linspace(s.Rp - s.Dl[i] - s.delta_s[i], s.Rp - s.Dl[i], npts)
    Trs = np.linspace(Ta[npts-1], s.Tl[i], npts)
    Prs = s.rhom*s.g*(s.Rp - rs)
    etas = suppf.calculate_viscosity(s, Trs, Prs) 

    # Lithosphere
    rl = np.linspace(s.Rp - s.Dl[i], s.Rp-s.Dcr[i], nptl)    
    Trl = np.linspace(s.Tl[i],s.Tcr[i], nptl)        
    Prl = s.rhom*s.g*(s.Rp - rl)
    etal = suppf.calculate_viscosity(s, Trl, Prl) 

    # Crust
    rcr = np.linspace(s.Rp - s.Dcr[i], s.Rp, nptc)   
    Trcr = np.linspace(s.Rp -s.Dcr[i], s.Rp, nptc)   
    Trcr = np.linspace(s.Tcr[i], s.Ts, nptc)
    Prcr = s.rhom*s.g*(s.Rp - rcr) 
    etacr = suppf.calculate_viscosity(s, Trcr, Prcr)


    Ttemp = np.concatenate((Trb, Ta, Trs, Trl, Trcr)) # concatenate: joining a sequence of arrays
    rtemp = np.concatenate((rb, ra, rs, rl, rcr))                     
    Ptemp = np.concatenate((Prb, Pa, Prs, Prl, Prcr))                    

    Tprof[:,i] = np.interp(r, rtemp, Ttemp) # interpolating temperature
    Pprof = np.interp(r, rtemp, Ptemp) # interpolating pressure
    
    # Solidus
    if plot_solidus == 'yes':
        Psol = np.linspace(Pmax,0,npt)
        rsol = s.Rp - Psol/(s.rhom*s.g)
        Tsol = suppf.calculate_dry_solidus(Psol/1e9)
        ax.plot(Tsol, rsol/1e3, '--', color=Tsolcolor, lw=lw)    
        
    # Liquidus
    if plot_liquidus == 'yes':
        Pliq = np.linspace(Pmax,0,npt)
        rliq = s.Rp - Pliq/(s.rhom*s.g)
        Tliq = suppf.calculate_dry_liquidus(Pliq/1e9)
        ax.plot(Tliq, rsol/1e3, '--', color=Tliqcolor, lw=lw)    
        
    ax.set_ylim(rmin,rmax)        
             
    ax.grid()    
    ax.set_ylabel('Radius [km]')
    ax.set_xlabel('Temperautre [K]')        

    ax.text(0.05, 0.1, 'Time =' + str(time) + ' Myr', horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes, fontsize = medium_size)
    
    ########################
    #  Viscosity 
    ########################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)

    # Lower boundary layer
    Prb = s.rhom*s.g*(s.Rp - rb)
    etab = suppf.calculate_viscosity(s, Trb, Prb)
    ax.plot(etab, rb/1e3, '-', color=etacolor, lw=lw)

    # Adiabatic mantle  
    Pa = s.rhom*s.g*(s.Rp  - ra)             
    etaa = suppf.calculate_viscosity(s, Ta, Pa)
    ax.plot(etaa, ra/1e3, '-', color=etacolor, lw=lw)
        
    # Upper boundary layer
    Prs = s.rhom*s.g*(s.Rp - rs)   
    etas = suppf.calculate_viscosity(s, Trs, Prs)
    ax.plot(etas, rs/1e3, '-', color=etacolor, lw=lw)
    
    ax.set_ylim(rmin,rmax)  
     
    ax.grid()    
    ax.set_xscale('log')
    ax.set_xlim(right=etamax)
    ax.set_ylabel('Radius [km]')
    ax.set_xlabel('Viscosity [Pa s]')
    
    ########################
    #  Expansivity
    ########################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)

    c = 1e5
    if (s.var_alpha == 'yes'):
        # Lower boundary layer
        Prb = s.rhom*s.g*(s.Rp - rb)
        alphab = suppf.calculate_thermal_expansivity(Trb, Prb/1e9)
        ax.plot(alphab*c, rb/1e3, '-', color=etacolor, lw=lw)

        # Adiabatic mantle  
        Pa = s.rhom*s.g*(s.Rp  - ra)             
        alphaa = suppf.calculate_thermal_expansivity(Ta, Pa/1e9)
        ax.plot(alphaa*c, ra/1e3, '-', color=etacolor, lw=lw)
    
        # Upper boundary layer
        Prs = s.rhom*s.g*(s.Rp - rs)   
        alphas = suppf.calculate_thermal_expansivity(Trs, Prs/1e9)
        ax.plot(alphas*c, rs/1e3, '-', color=etacolor, lw=lw)
    
    else:
        alphaa = np.ones(npt)*s.alpha
        r = np.linspace(s.Rc, s.Rp, npt)
        ax.plot(alphaa*c, r/1e3, '-', color=etacolor, lw=lw)
    
    ax.set_ylim(rmin,rmax)  
     
    ax.grid()    
    ax.set_ylabel('Radius [km]')
    ax.set_xlabel('Thermal expansivity [$10^{-5}$ 1/K]')
           
    
    #########################
    #  H2O concentration in melt
    #########################
    
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    
    F = np.linspace(0,0.5,100)
    Xm_H2O = [10e-6, 100e-6, 1000e-6]
    
    for X in Xm_H2O:
        X_H2O_liq = X/F*(1-(1-F)**(1/s.D_H2O))*100
    
        ax.plot(F, X_H2O_liq)
        ax.set_xlabel('melt fraction F')
        ax.set_ylabel('$X_{H2O}^{liq}$')
        ax.set_ylim(0,1)
        ax.grid()
        ax.legend(loc=1)
    
    plt.tight_layout() 


####################################################################
def plot_profiles_evolution(s):
    """Plot evolution of the temperature and viscosity profiles"""
####################################################################

    yr = 365.0*24.0*60.0*60.0   # 1 year in seconds
    etamax = 1e25
    colormap_1 = plt.cm.plasma
    colormap_2 = plt.cm.plasma_r

    fig=plt.figure(figsize=(20,15))
    n = 0
    nx_panels = 1
    ny_panels = 1 # change back to 2 for adding viscosity plot
    lw = 1
    
    small_size = 10
    medium_size = 15
    bigger_size = 20
    plt.rc('font', size=small_size)          # controls default text sizes
    plt.rc('axes', titlesize=small_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=medium_size)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=medium_size)   # fontsize of the tick labels
    plt.rc('legend', fontsize=medium_size)   # legend fontsize
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

    nt = np.size(s.t)-1
    npta = 500 #300 #3000
    nptb = 20 #15
    npts = 20 #15 
    nptl = 40 #30
    nptc = 200 #20                     
    npt = nptb + npta + npts + nptl + nptc
    Tprof = np.zeros((npt,nt))
    etaprof = np.zeros((npt,nt))
    meltzone_top = np.zeros((nt))
    meltzone_bot = np.zeros((nt))
    
    
    r = np.linspace(s.Rc, s.Rp, npt)
    Pr = np.linspace(s.rhom*s.g*(s.Rp - s.Rc)/1e9, 0., npt)  
    Dref = 0.2/3*((s.Rp**3-s.Rc**3)/s.Rp**2)   # Morschhauser 2011
    
    for i in np.arange(0, nt):
        r = np.linspace(s.Rc, s.Rp, npt)
        Pr = np.linspace(s.rhom*s.g*(s.Rp - s.Rc)/1e9, 0., npt)  
        delta_T_sol = 43*np.minimum(s.Xm_H2O[i]*1e-4*(Pr*0+1),12*Pr**0.6+Pr)**0.75 # Katz et al 2003
        Tsolr_ini = suppf.calculate_dry_solidus(Pr) - delta_T_sol
        Tliqr = suppf.calculate_dry_liquidus(Pr)  
        Tsolr = Tsolr_ini + s.depl[i]*(Tliqr - Tsolr_ini)
        
        if(i == 0):
            if s.tectonics == 'ML':
                Pm = s.rhom*s.g*(s.delta_s0)    # Pressure at the base of the lithosphere
            else:
                Pm = s.rhom*s.g*(s.Dl0+s.delta_s0)    # Pressure at the base of the lithosphere
        else:   
            Pm = s.rhom*s.g*(s.Dl[i]+s.delta_s[i-1])  # Pressure at the base of the lithosphere from previous time step
        s.etam[i] = suppf.calculate_viscosity(s, s.Tm[i], Pm)  
        
        if(i == 0):
            dc = s.delta_c0
        else: 
            dc = s.delta_c[i-1] 
        
        zb = s.Rp - (s.Rc + dc)           # depth of the lower TBL
        Pb = s.rhom*zb*s.g + s.rhocr*s.Dcr[i]*s.g

        # Adiabatic mantle
        ra = np.linspace(s.Rc + s.delta_c[i], s.Rp - s.Dl[i], npta) 
        Ta = suppf.calculate_adiabat(s, npta, s.Tm[i], Pm, Pb)
        Ta = np.flipud(Ta)
        Pa = s.rhom*s.g*(s.Rp - ra) 
        etaa = suppf.calculate_viscosity(s, Ta, Pa)

        # Lower boundary layer
        rb = np.linspace(s.Rc, s.Rc + s.delta_c[i], nptb)
        Trb = np.linspace(s.Tc[i], Ta[0], nptb)
        Prb = s.rhom*s.g*(s.Rp - rb) 
        etab = suppf.calculate_viscosity(s, Trb, Prb)

        # Upper boundary layer
        rs = np.linspace(s.Rp - s.delta_s[i] - s.Dl[i], s.Rp - s.Dl[i], npts) 
        Trs = np.linspace(Ta[npts-1], s.Tl[i], npts)
        Prs = s.rhom*s.g*(s.Rp - rs)
        etas = suppf.calculate_viscosity(s, Trs, Prs) 

        # Lithosphere
        rl = np.linspace(s.Rp - s.Dl[i], s.Rp-s.Dcr[i], nptl)    
        Trl = np.linspace(s.Tl[i],s.Tcr[i], nptl)          
        Prl = s.rhom*s.g*(s.Rp - rl)
        etal = suppf.calculate_viscosity(s, Trl, Prl) 

        # Crust
        rcr = np.linspace(s.Rp - s.Dcr[i], s.Rp, nptc)    
        Trcr = np.linspace(s.Tcr[i], s.Ts, nptc)
        Prcr = s.rhom*s.g*(s.Rp - rcr) 
        etacr = suppf.calculate_viscosity(s, Trcr, Prcr)

        Ttemp = np.concatenate((Trb, Ta, Trs, Trl, Trcr)) 
        etatemp = np.concatenate((etab, etaa, etas, etal, etacr))
        rtemp = np.concatenate((rb, ra, rs, rl, rcr))                     
        Ptemp = np.concatenate((Prb, Pa, Prs, Prl, Prcr))         

        Tprof[:,i] = np.interp(r, rtemp, Ttemp) # interpolating temperature
        Pprof = np.interp(r, rtemp, Ptemp) # interpolating pressure
        etaprof[:,i] = np.interp(r, rtemp, etatemp)

        idx_meltzone = np.argwhere(Tprof[:,i] > Tsolr).flatten()
        
        # Partial melt zone
        idx_meltzone = np.argwhere(np.diff(np.sign(Tprof[:,i] - Tsolr))).flatten()
        if (np.size(idx_meltzone) >= 2) :
            meltzone_top[i] = r[idx_meltzone[-1]]
            meltzone_bot[i] = r[idx_meltzone[-2]]
        else:    
            meltzone_top[i] = None
            meltzone_bot[i] = None

    ############################################
    # Temperature and melt zone
    ############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)  
    D = s.Rc #km depth
    rmin = D
    Pmax = s.rhom*s.g*(s.Rp-rmin) #max(Pprof)
    Pmin = 0 # pressure at surface

    rmin = rmin/1e3
    rmax = s.Rp/1e3 
    
    Pmin= Pmin/1e9
    Pmax=Pmax/1e9
    
    Tmin =  s.Ts 
    Tmax = np.max(Tprof) #3700 #4000
    levsT_cont = np.arange(Tmin, Tmax, 10) #10
    levsT_disc = np.arange(Tmin, Tmax, 500) #300
    
    
    cf = ax.contourf(s.t[:-1]/yr/1e6, r/1e3, Tprof, levsT_cont, cmap=colormap_1)
    cb = plt.colorbar(cf, extend='both', ticks = levsT_disc, orientation = 'horizontal', pad=0.13)#pad=0.15
    
    #### boundary layers ####
    ax.plot(s.t[:-1]/yr/1e6, (s.Rp-s.Dl[:-1])/1e3,  color='mediumblue', linestyle="dashed", linewidth=7)
    ax.plot(s.t[:-1]/yr/1e6, (s.Rp-s.Dcr[:-1])/1e3,  color='cyan', linewidth=7)

    

    ax.set_ylim(rmin, rmax) 
    
    cb.ax.set_xlabel('Temperature [K]').set(fontsize=50)
    cb.ax.tick_params(labelsize=45)
    ax.set_xlabel('Time [Myr]').set(fontsize=50)
    ax.set_ylabel('Radius [km]').set(fontsize=50)
    plt.yticks(fontsize=45)
    plt.xticks(fontsize=45)
    ax.grid()
    
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylim(Pmax, Pmin)
    
    ax2.set_ylabel('P [GPa]').set(fontsize=50)
    plt.yticks(fontsize=45)
    plt.xticks(fontsize=45)
    
    """
    ############################################
    # Viscosity
    ############################################
    n = n + 1
    ax = fig.add_subplot(nx_panels,ny_panels,n)
    
    # Continous colorscale for contour plot
    levse_exp_cont = np.arange(np.floor(np.log10(etaprof.min())-1), np.ceil(np.log10(etamax)+1), 0.02)
    levse_cont = np.power(10, levse_exp_cont)
    # Discrete intervals for colorbar ticks
    levse_exp_disc = np.arange(np.floor(np.log10(etaprof.min())-1), np.ceil(np.log10(etamax)+1), 2)
    levse_disc = np.power(10, levse_exp_disc)
    
    cf = ax.contourf(s.t[:-1]/yr/1e6, r/1e3, etaprof, levse_cont, norm=colors.LogNorm(vmin = etaprof.min(), vmax = etamax), extend='max', cmap=colormap_2) 
    cb = plt.colorbar(cf, extend='max', ticks=levse_disc)
    
    ax.set_ylim(rmin,rmax)
    
    cb.ax.set_ylabel('Viscosity [Pa s]')
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Radius [km]')
    ax.grid()
    
    """
    plt.tight_layout()
    
    ##########################################
def plot_profiles_evolution_zoom(s):
    """higher resolution for meltzone"""
    ##########################################   
    yr = 365.0*24.0*60.0*60.0   # 1 year in seconds
    etamax = 1e25
    colormap_1 = plt.cm.plasma
    colormap_2 = plt.cm.plasma_r
    
    
    fig = plt.figure(figsize=(20,13))
    n = 0
    nx_panels = 1
    ny_panels = 1  ## 13.9.
    lw = 1
    
    small_size = 10
    medium_size = 15
    bigger_size = 20
    plt.rc('font', size=small_size)          # controls default text sizes
    plt.rc('axes', titlesize=small_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=medium_size)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=medium_size)   # fontsize of the tick labels
    plt.rc('legend', fontsize=medium_size)   # legend fontsize
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title
    
    nt = np.size(s.t)-1
    npta = 3000#1800 #300 #3000
    nptb = 20 #15
    npts = 2500#20 #15 
    nptl = 300#200 #30
    nptc = 200 #20                     
    npt = npta + npts + nptl + nptc
    Tprof = np.zeros((npt,nt))
    etaprof = np.zeros((npt,nt))
    meltzone_top = np.zeros((nt))
    meltzone_bot = np.zeros((nt))
    
    
    for i in np.arange(0, nt):
        r = np.linspace(s.Rc, s.Rp, npt)
        Pr = np.linspace(s.rhom*s.g*(s.Rp - s.Rc)/1e9, 0., npt)  
        delta_T_sol = 43*np.minimum(s.Xm_H2O[i]*1e-4*(Pr*0+1),12*Pr**0.6+Pr)**0.75 # Katz et al 2003
        Tsolr_ini = suppf.calculate_dry_solidus(Pr) - delta_T_sol
        Tliqr = suppf.calculate_dry_liquidus(Pr)  
        Tsolr = Tsolr_ini + s.depl[i]*(Tliqr - Tsolr_ini)
        
        if(i == 0):
            if s.tectonics == 'ML':
                Pm = s.rhom*s.g*(s.delta_s0)    # Pressure at the base of the lithosphere
            else:
                Pm = s.rhom*s.g*(s.Dl0+s.delta_s0)    # Pressure at the base of the lithosphere
        else:   
            Pm = s.rhom*s.g*(s.Dl[i]+s.delta_s[i-1])  # Pressure at the base of the lithosphere from previous time step
        s.etam[i] = suppf.calculate_viscosity(s, s.Tm[i], Pm)  
        
        if(i == 0):
            dc = s.delta_c0
        else: 
            dc = s.delta_c[i-1] 
        
        zb = s.Rp - (s.Rc + dc)           # depth of the lower TBL
        Pb = s.rhom*zb*s.g + s.rhocr*s.Dcr[i]*s.g
        
        # Adiabatic mantle
        ra = np.linspace(s.Rc + s.delta_c[i], s.Rp - s.Dl[i], npta)
        Ta = suppf.calculate_adiabat(s, npta, s.Tm[i], Pm, Pb)
        Ta = np.flipud(Ta)
        Pa = s.rhom*s.g*(s.Rp - ra) 
        etaa = suppf.calculate_viscosity(s, Ta, Pa)

        # Lower boundary layer
        rb = np.linspace(s.Rc, s.Rc + s.delta_c[i], nptb)
        Trb = np.linspace(s.Tc[i], Ta[0], nptb)
        Prb = s.rhom*s.g*(s.Rp - rb) 
        etab = suppf.calculate_viscosity(s, Trb, Prb)

        # Upper boundary layer
        rs = np.linspace(s.Rp - s.delta_s[i] - s.Dl[i], s.Rp - s.Dl[i], npts) 
        Trs = np.linspace(Ta[npts-1], s.Tl[i], npts)
        Prs = s.rhom*s.g*(s.Rp - rs)
        etas = suppf.calculate_viscosity(s, Trs, Prs) 

        # Lithosphere
        rl = np.linspace(s.Rp - s.Dl[i], s.Rp-s.Dcr[i], nptl)    
        Trl = np.linspace(s.Tl[i],s.Tcr[i], nptl)          
        Prl = s.rhom*s.g*(s.Rp - rl)
        etal = suppf.calculate_viscosity(s, Trl, Prl) 

        # Crust
        rcr = np.linspace(s.Rp - s.Dcr[i], s.Rp, nptc)   
        Trcr = np.linspace(s.Tcr[i], s.Ts, nptc)
        Prcr = s.rhom*s.g*(s.Rp - rcr) 
        etacr = suppf.calculate_viscosity(s, Trcr, Prcr)

        Ttemp = np.concatenate((Ta, Trs, Trl, Trcr)) 
        etatemp = np.concatenate((etaa, etas, etal, etacr))
        rtemp = np.concatenate((ra, rs, rl, rcr))                     
        Ptemp = np.concatenate((Pa, Prs, Prl, Prcr))         

        Tprof[:,i] = np.interp(r, rtemp, Ttemp) # interpolating temperature
        Pprof = np.interp(r, rtemp, Ptemp) # interpolating pressure
        etaprof[:,i] = np.interp(r, rtemp, etatemp)
  
        idx_meltzone = np.argwhere(Tprof[:,i] > Tsolr).flatten()
        
        # Partial melt zone
        idx_meltzone = np.argwhere(np.diff(np.sign(Tprof[:,i] - Tsolr))).flatten()
        if (np.size(idx_meltzone) >= 2) :
            meltzone_top[i] = r[idx_meltzone[-1]]
            meltzone_bot[i] = r[idx_meltzone[-2]]
        else:    
            meltzone_top[i] = None
            meltzone_bot[i] = None
            
    ############################################
    # Temperature and melt zone
    ############################################
    n = n + 1             
    ax = fig.add_subplot(nx_panels,ny_panels,n) 
     
    #Pmax = 24e9 
    #dmin = Pmax/s.rhom/s.g
   # rmin = s.Rp-s.Rc# (s.Rp - dmin)
    D =400e3 #km depth
    rmin = s.Rp-D
    Pmax = s.rhom*s.g*(s.Rp-rmin) #np.max(Pprof)
    Pmin = 0 # pressure at surface
    
    rmin = rmin/1e3
    rmax = s.Rp/1e3 
    
    Pmin= Pmin/1e9
    Pmax=Pmax/1e9
    
    Tmin = s.Ts
    Tmax = 2400  #np.max(Tprof)
    
    levsT_cont = np.arange(Tmin, Tmax, 10)
    levsT_disc = np.arange(Tmin, Tmax, 300)
    
    cf = ax.contourf(s.t[:-1]/yr/1e6, r/1e3, Tprof, levsT_cont, cmap=colormap_1)
    cb = plt.colorbar(cf, extend='both', ticks = levsT_disc, orientation = 'horizontal', pad=0.15) 

    
    ax.plot(s.t[:-1]/yr/1e6, (s.Rp-s.Dcr[:-1])/1e3,  color='cyan', linewidth=6)
    ax.plot(s.t[:-1]/yr/1e6, (s.Rp-s.Dl[:-1])/1e3,  color='mediumblue', linestyle="dashed", linewidth=6)

    ax.plot(s.t[:-1]/yr/1e6, meltzone_top/1e3, '-', color='black', lw=lw)
    ax.plot(s.t[:-1]/yr/1e6, meltzone_bot/1e3, '-', color='black', lw=lw)
    plt.fill_between(s.t[:-1]/yr/1e6,meltzone_top/1e3, meltzone_bot/1e3, hatch='\\',color='red', alpha = 0.8)

    ax.set_ylim(rmin, rmax) 
   
    cb.ax.set_title('Temperature [K]').set(fontsize=50, rotation="horizontal")
    cb.ax.tick_params(labelsize=45)

    ax.set_ylabel('Radius [km]').set(fontsize=50)
    plt.yticks(fontsize=45)
    plt.xticks(fontsize=45)
    
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylim(Pmax, Pmin)
    
    ax2.set_ylabel('P [GPa]').set(fontsize=50)
    plt.yticks(fontsize=45)
    plt.xticks(fontsize=45)



    