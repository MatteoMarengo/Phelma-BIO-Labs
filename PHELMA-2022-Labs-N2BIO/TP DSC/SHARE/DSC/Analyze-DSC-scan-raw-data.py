# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:55:54 2022

@author: herve.guillou
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import interpolate


def Load_data_NA(fn):
    """
    Load data from txt file exported from the Nanoanalyze software
    T (°C), Power, Time
    you eventually need to check the encoding and the decimal separator
    """
    df = pd.read_csv(fn, sep = '\t', skiprows = 2, names=['T', 'Power', 'Time'], decimal =',')
    # data are in the form of T, Power, t separated by tabs
    # data are exported from a french system so decimal separator is ',' change eventually to '.'
    # first 2 rows are for informations
    t = np.array(df['Time'])
    P = np.array(df['Power'])
    T = np.array(df['T'])
    return t, T, P


def LT_backgrd(p,T):
    """ low temperature polynamial to fit the data"""
    return p[0] + p[1] * T + p[2]*T**2 + p[3]*T**3 #+ p[4]*T**4 

def LT_error(p,T,Power):
    """Error function to fit bckgrd at HT"""
    return Power - LT_backgrd(p,T)

def HT_backgrd(p,T):
    """ high temperature polynamial to fit the data"""
    return p[0] + p[1] * T + p[2]*T**2 + p[3]*T**3 #+ p[4]*T**4 

def HT_error(p,T,Power):
    """Error function to fit bckgrd at HT"""
    return Power - HT_backgrd(p,T)

def get_BG_NDSCIII(T,P,Range, kind):
    """
    get the background as defined by the functions LT_backgrd and HT_backgrd and the range 
    """
    T0,T1,T2,T3=Range
    #selection of the LT data
    LT_select = np.where((T>T0)&(T<T1))
    LT_T = T[LT_select]
    LT_P = P[LT_select]
    # fit the LT data
    LT_p0 = np.array([5,-0.010,0.0,0.0,0.0]) 
    LT_p1 = optimize.leastsq(LT_error, LT_p0, args=(LT_T,LT_P))
    # selection of the HT data
    HT_select =np.where((T>T2)&(T<T3)) 
    HT_T = T[HT_select]
    HT_P = P[HT_select]
    #fit the HT data
    HT_p0 = np.array([5,-0.010,0.0,0.0,0.0]) 
    HT_p1 = optimize.leastsq(HT_error, HT_p0, args=(HT_T,HT_P))
    # selection of temperature range where background needs to be interpolated
    Interp_select = np.where((T>=T1)&(T<=T2))
    Interp_T = T[Interp_select]
    # tricky part
    T_background = np.append(LT_T,HT_T)
    P_background = np.append(LT_backgrd(LT_p1[0],LT_T),HT_backgrd(HT_p1[0],HT_T) )
    background_data =  interpolate.interp1d(T_background,P_background, kind=kind, fill_value="extrapolate")
    return background_data(T), LT_T, background_data(LT_T), HT_T, background_data(HT_T), Interp_T, background_data(Interp_T)


def plot_Raw_DSC(ax, xdata,ydata,param_dict):
    """
    A helper function to make a graph always the same
    """
    ax.set_xlabel("Temperature (°C)", size= 18)
    ax.set_ylabel("$\Delta P_{\mathrm{comp}} \; \mu \mathrm{W}$", size= 18)
    out = ax.plot(xdata,ydata,**param_dict)
    return out



plt.close('all')

### File name, check the path ....

# Scan Up 01
file1 = "D:/OneDrive/Documents/PostCPGE/PHELMA 2020-2023/BIOMED 2022-2023/TP Bio/TP DSC/SHARE/DSC/2022-11-28/1PTS-LOOP16-24uM-PBbuffer-ScanUp05.txt"
# Scan Up 02
file2 = "D:/OneDrive/Documents/PostCPGE/PHELMA 2020-2023/BIOMED 2022-2023/TP Bio/TP DSC/SHARE/DSC/2022-11-28/1PTS-LOOP16-24uM-PBbuffer-ScanUp06.txt"

# Scan Down 01
file3 = "D:/OneDrive/Documents/PostCPGE/PHELMA 2020-2023/BIOMED 2022-2023/TP Bio/TP DSC/SHARE/DSC/2022-11-28/1PTS-LOOP16-24uM-PBbuffer-ScanDown05.txt"
# Scan Down 02
file4 = "D:/OneDrive/Documents/PostCPGE/PHELMA 2020-2023/BIOMED 2022-2023/TP Bio/TP DSC/SHARE/DSC/2022-11-28/1PTS-LOOP16-24uM-PBbuffer-ScanDown06.txt"

t, T1, P1 = Load_data_NA(file1)
Pback1_tot, LT_T1, Pback1_LT, HT_T1, Pback1_HT, Interp_T1, Pback1_Int = get_BG_NDSCIII(T1, P1, [30,50,80,95], 'cubic')

t, T2, P2 = Load_data_NA(file2)
Pback2_tot, LT_T2, Pback2_LT, HT_T2, Pback2_HT, Interp_T2, Pback2_Int = get_BG_NDSCIII(T2, P2, [30,50,80,95], 'cubic')
t, T3, P3 = Load_data_NA(file3)
Pback3_tot, LT_T3, Pback3_LT, HT_T3, Pback3_HT, Interp_T3, Pback3_Int = get_BG_NDSCIII(T3, P3, [20,55,80,90], 'cubic') 
t, T4, P4 = Load_data_NA(file4)
Pback4_tot  = get_BG_NDSCIII(T4, P4, [20,55,80,90], 'cubic') [0] # just get the first response



fig1, ax1 = plt.subplots(1,figsize=(10,10/1.5))
plot_Raw_DSC(ax1, T1, P1, {'linestyle' : '-','color': 'r', 'label':'first scan up'})
plot_Raw_DSC(ax1, LT_T1, Pback1_LT, {'linestyle' : '--','color': 'blueviolet', 'label' : 'Lower Temperature Background'})
plot_Raw_DSC(ax1, HT_T1, Pback1_HT, {'linestyle' : '--','color': 'darkred', 'label' : 'Higher Temperature Background'})
plot_Raw_DSC(ax1, Interp_T1, Pback1_Int, {'linestyle' : '--','color': 'red', 'label' :'Interpolated background'})
plot_Raw_DSC(ax1, T2, P2, {'linestyle' : '-','color': 'r', 'label':'second scan up'})
plot_Raw_DSC(ax1, T2, Pback2_tot, {'linestyle' : '--','color': 'darkred', 'label' : 'Background'})

plot_Raw_DSC(ax1, T3, P3, {'linestyle' : '-','color': 'blue', 'label' :'First scan down'})
plot_Raw_DSC(ax1, T3, Pback3_tot, {'linestyle' : '--','color': 'darkblue', 'label' : 'Background'})

plot_Raw_DSC(ax1, T4, P4, {'linestyle' : '-','color': 'blue', 'label' :'First scan down'})
plot_Raw_DSC(ax1, T4, Pback4_tot, {'linestyle' : '--','color': 'darkblue', 'label' : 'Background'})


ax1.set_xlim(xmin = 20, xmax =95)
ax1.set_ylim(ymin = -50, ymax =50)
ax1.legend()
plt.title("Raw Data DSC 28/11/2022 - Scan 05 & Scan 06")
fig1.tight_layout()
#fig1.savefig("Raw-data-exemple.png",dpi=600)