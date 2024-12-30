# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:58:06 2022

@author: herve.guillou
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import optimize
from scipy import interpolate
from scipy import integrate
plt.close('all')


def Load_data_NA_ITC(fn):
    """
    Load data from txt file exported from the Nanoanalyze software
    T (°C), Power, Time
    you eventually need to check the encoding and the decimal separator
    """
    df = pd.read_csv(fn, sep = '\t', skiprows = 2, names=['t', 'Power'], decimal =',')
    # data are in the form of T, Power, t separated by tabs
    # data are exported from a french system so decimal separator is ',' change eventually to '.'
    # first 2 rows are for informations
    t = np.array(df['t'])
    P = np.array(df['Power'])
    return t, P


def split_injection(P, ts, ti, N):
    """ take the time trace (t and P) of an experiment and split it in a serie of N injections 
    ts = starting time
    ti = duration of injection
    N = number of injection
    """
    a = np.zeros((N,ti))
    for i in range(N):
        a[i] = P[(ts+i*ti):(ts+i*ti+ti)]
    return a

def flat_injection(t,P):
    """ remove a linear background from the injection"""
    a = P - (t * (P[-1]-P[0])/(t[-1]-t[0]) + P[0])
    return a

def P_flat(ti, P):
    """
    remove from each injection the flat_background
    """
    Pflat = P
    T1s = np.arange(ti)
    for index, Pin in enumerate(P):
        Pflat[index] = flat_injection(T1s,Pin)
    return Pflat

def plot_Raw_ITC(ax, xdata,ydata,param_dict):
    """
    A helper function to make a graph always the same
    """
    ax.set_xlabel("time (s)", size= 18)
    ax.set_ylabel("$ P \; \mu \mathrm{W}$", size= 18)
    out = ax.plot(xdata,ydata,**param_dict)
    return out

def plot_DH_ITC(ax, xdata,ydata,param_dict):
    """
    A helper function to make a graph always the same
    """
    ax.set_xlabel("molar ratio", size= 18)
    ax.set_ylabel("$ H \; \mu \mathrm{J}$", size= 18)
    out = ax.plot(xdata,ydata,**param_dict)
    return out


file1 = "D:/OneDrive/Documents/PostCPGE/PHELMA 2020-2023/BIOMED 2022-2023/TP Bio/TP DSC/SHARE/ITC/29-11-2022/stp2loop16-63um-in-loop1pts-10um-25deg-250rpm.txt"

t1,P1 = Load_data_NA_ITC(file1)
t1s = 9000
t1i = 600
N1 = 23

cmap = cm.get_cmap('plasma')
colors1 = []
for i in np.linspace(0, 1, N1):
    colors1.append(cmap(i))

P1split = split_injection(P1, t1s,t1i,N1)
P1flat = P1split
time1s = np.arange(t1i)
for index, Pin in enumerate(P1split):
    P1flat[index] = flat_injection(time1s,Pin)

Q1 = np.arange(N1)
for index, P in enumerate(P1flat):
    Q1[index] = integrate.trapz(P,time1s)



fig1, ax1 = plt.subplots(1,1,figsize=(10,10/1.5))
plot_Raw_ITC(ax1,t1,P1,{'linestyle' : '-', 'color' : 'blue', 'label':'STP2LOOP4 56µM in POOL1PTS'} )
ax1.legend()
fig1.tight_layout()
plt.title("ITC Raw Data - 29/11/2022")
fig1.savefig("ITC_exemple_raw_data.png",dpi=600)


fig2, ax2 = plt.subplots(1,1,figsize=(10,10/1.5))
for index, P in enumerate(P1flat):
    plot_Raw_ITC(ax2,time1s,P,{'linestyle' : '-', 'label': 'inj %d'%(index)})
ax2.legend()
fig2.tight_layout()
plt.title("ITC Injections - 29/11/2022")
fig2.savefig("ITC_individual_injection_flat.png",dpi=600)


###############################################################################
#### Model A+B <-> AB #########################################################
###############################################################################

V0=1e-3
Vinj = 10e-6

R = 8.314
T = 25+273.15
DH = -730000 #(200 kcal/mol) # initial guess
DS = -2290 # very sensitive to the value change by small values

C0 = 56e-6 # can be changed eventually within 5% to account for dilution and measurement of concentration error

CB0 = C0 # multiply by a facot 0.95 or 1.05 to account for 5% error

Nstoe = 10.20 # this is the guess of the injection at which stoechiometry is achieved

x = np.arange(1,23+1)/Nstoe # define molar ratio a posteriori

n0 = Nstoe*Vinj*CB0
CA0 = n0/V0
print("the initail concentration of material in the reaction volume is C0 = %f"%(CA0))



def V(i, V0, Vinj):
    """define the volume after i injections"""
    return V0+i*Vinj

def CAi(i, V0, Vinj, CA0):
    """retunr the concentration of A after i injection if no A reacts"""
    return CA0*V0 / V(i,V0,Vinj)

def CBi(i, V0, Vinj, CB0):
    """retunr the concentration of A after i injection if no A reacts"""
    return CB0*i*Vinj / V(i,V0,Vinj)

def Keq(T,DH,DS):
    """return the equilibrium constant
    T : Float
     Absolute Temperature in K
    DH : Float
     Reaction enthalpy, should be negative (J)
    DS : Float
     Reaction entropy, should be negative (J/K)
    returns
    ---------
    Binding Constant (Molar^-1)

    """
    return np.exp(-(DH-T*DS)/(R*T))

def Qmod(N,C0,Nsto, DH, DS, T, Vinj, V0):
    """
    Parameters
    ----------
    N : Integer
        Number of injections 
    C0 : Float
        Concentration of titrant in the syringe, molar
    Nsto : Float
        Injection number for which stoechiometrie is achieved
    DH : Float
        Enthalpy of the reaction (J)
    DS : Float
        Entropy of the reaction  (J/K)
    T : Float
        Absolute temperature (K)
    Vinj : float
        Injected volume (L)
    V0 : float
        initial sample volume (L)

    Returns
    -------
    Array of Heat (µJ)

    """
    i = 0
    ABmodel = []
    CB0= C0
    n0 = Nsto*Vinj*CB0
    CA0 = n0/V0
    for i in range(N+1):
        b = -(CAi(i,V0,Vinj,CA0) + CBi(i,V0,Vinj,CB0) + 1.0/Keq(T,DH,DS))
        c = CAi(i,V0,Vinj,CA0) * CBi(i,V0,Vinj,CB0)
        Delta = b**2 - 4.0*c
        ab1 = (-b-np.sqrt(Delta))/2.0
        ABmodel.append(ab1*V(i,V0,Vinj))
        ab2 = (-b+np.sqrt(Delta))/2.0 # it is another solution not physical ?
        #print("i = %d, Delta = %e, ab1 = %e, ab2 = %e"%(i,Delta,ab1,ab2) )
        #print("concentration finale en A si rien n'avait réagit CAi(25) = %e"%(CAi(25,V0,Vinj,CA0)))
        #ABmodel
    return np.diff(np.array(ABmodel))*(-DH)*1e6

fig3, ax3 = plt.subplots(1,1,figsize=(10,10/1.5))
plot_DH_ITC(ax3,x,Q1+25, {'linestyle':'None', 'marker' : 'o', 'label':'data' })
DH0 = -730000
DS0 = -2280
Qmod0 = Qmod(23,56e-6,10.2,DH0, DS0,35+273,10e-6,1e-3)

DH1 = -730000
DS1 = -2290
Qmod1 = Qmod(23,56e-6,10.2,DH1, DS1,25+273,10e-6,1e-3)
DH2 = -730000
DS2 = -2300
Qmod2 = Qmod(23,56e-6,10.2,DH2, DS2,25+273,10e-6,1e-3)
DH3 = -730000
DS3 = -2310
Qmod3 = Qmod(23,56e-6,10.2,DH3, DS3,25+273,10e-6,1e-3)
DH4 = -730000
DS4 = -2320
Qmod4 = Qmod(23,56e-6,10.2,DH4, DS4,25+273,10e-6,1e-3)

plot_DH_ITC(ax3,x,Qmod0, {'linestyle':'-', 'label':'model DH = %.2e kJ/mol ; DS =%.2e kJ/K/mol ; Keq = %.3e'%(DH0/1000.0,DS0/1000.0, Keq(35+273,DH0,DS0)) })
plot_DH_ITC(ax3,x,Qmod1, {'linestyle':'-', 'label':'model DH = %.2e kJ/mol ; DS =%.2e kJ/K/mol ; Keq = %.3e'%(DH1/1000.0,DS1/1000.0, Keq(25+273,DH1,DS1)) })
plot_DH_ITC(ax3,x,Qmod2, {'linestyle':'-', 'label':'model DH = %.2e kJ/mol ; DS =%.2e kJ/K/mol ; Keq = %.3e'%(DH2/1000.0,DS2/1000.0, Keq(25+273,DH2,DS2)) })
#plot_DH_ITC(ax3,x,Qmod3, {'linestyle':'-', 'label':'model DH = %.2e kJ/mol ; DS =%.2e kJ/K/mol ; Keq = %.3e'%(DH3/1000.0,DS3/1000.0, Keq(25+273,DH3,DS3)) })
#plot_DH_ITC(ax3,x,Qmod4, {'linestyle':'-', 'label':'model DH = %.2e kJ/mol ; DS =%.2e kJ/K/mol ; Keq = %.3e'%(DH4/1000.0,DS4/1000.0, Keq(25+273,DH4,DS4)) })
ax3.legend()
plt.title("Models - ITC - 29/11/2022")
fig3.savefig("ITC_savefig_heat.png",dpi=600)