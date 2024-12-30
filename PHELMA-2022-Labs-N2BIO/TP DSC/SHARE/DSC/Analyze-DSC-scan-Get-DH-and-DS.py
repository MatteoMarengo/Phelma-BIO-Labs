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
from scipy import integrate

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


def plot_DC_DSC(ax, xdata,ydata,param_dict):
    """
    A helper function to make a graph always the same
    """
    ax.set_xlabel("Temperature (°C)", size= 18)
    ax.set_ylabel("$\Delta C_{p} \; \mathrm{J}\mathrm{K}^{-1}\mathrm{mol}^{-1}$", size= 18)
    out = ax.plot(xdata,ydata,**param_dict)
    return out

def plot_DH_DSC(ax, xdata,ydata,param_dict):
    """
    A helper function to make a graph always the same
    """
    ax.set_xlabel("Temperature (°C)", size= 18)
    ax.set_ylabel("$\Delta H \; \mathrm{J}\mathrm{mol}^{-1}$", size= 18)
    out = ax.plot(xdata,ydata,**param_dict)
    return out

def plot_DS_DSC(ax, xdata,ydata,param_dict):
    """
    A helper function to make a graph always the same
    """
    ax.set_xlabel("Temperature (°C)", size= 18)
    ax.set_ylabel("$\Delta S \; \mathrm{J}\mathrm{K}^{-1}\mathrm{mol}^{-1}$", size= 18)
    out = ax.plot(xdata,ydata,**param_dict)
    return out

plt.close('all')

### File name, check the path ....
V0=300.0e-6 # sample volume
C0=50e-6 # concentration
gamma = 1.0/60 # scan rate

# Scan Up 01
file1 = "D:/OneDrive/Documents/PostCPGE/PHELMA 2020-2023/BIOMED 2022-2023/TP Bio/TP DSC/SHARE/DSC/2022-11-28/1PTS-LOOP16-24uM-PBbuffer-ScanUp01.txt"
# Scan Up 02
file2 = "D:/OneDrive/Documents/PostCPGE/PHELMA 2020-2023/BIOMED 2022-2023/TP Bio/TP DSC/SHARE/DSC/2022-11-28/1PTS-LOOP16-24uM-PBbuffer-ScanUp02.txt"

# Scan Down 01
file3 = "D:/OneDrive/Documents/PostCPGE/PHELMA 2020-2023/BIOMED 2022-2023/TP Bio/TP DSC/SHARE/DSC/2022-11-28/1PTS-LOOP16-24uM-PBbuffer-ScanDown01.txt"
# Scan Down 02
file4 = "D:/OneDrive/Documents/PostCPGE/PHELMA 2020-2023/BIOMED 2022-2023/TP Bio/TP DSC/SHARE/DSC/2022-11-28/1PTS-LOOP16-24uM-PBbuffer-ScanDown02.txt"

t, T1, P1 = Load_data_NA(file1)
Pback1_tot, LT_T1, Pback1_LT, HT_T1, Pback1_HT, Interp_T1, Pback1_Int = get_BG_NDSCIII(T1, P1, [30,50,80,95], 'cubic')
DC1 = (P1-Pback1_tot)/gamma/(V0*C0) * 1e-6

# use valid range of temperature to analyse
select1 = np.where((T1>30)&(T1<95))[0]
DH1 = [0]
DS1 = [0]
DH1 = np.append(DH1,integrate.cumtrapz(DC1[select1], T1[select1]))
DS1 = np.append(DS1, integrate.cumtrapz(DC1[select1]/(T1[select1]+273), T1[select1]))

t, T2, P2 = Load_data_NA(file2)
Pback2_tot, LT_T2, Pback2_LT, HT_T2, Pback2_HT, Interp_T2, Pback2_Int = get_BG_NDSCIII(T2, P2, [30,50,80,95], 'cubic')
DC2 = (P2-Pback2_tot)/gamma/(V0*C0) * 1e-6

# use valid range of temperature to analyse
select2 = np.where((T2>30)&(T2<95))[0]
DH2 = [0]
DS2 = [0]
DH2 = np.append(DH2, integrate.cumtrapz(DC2[select2], T2[select2]))
DS2 = np.append(DS2, integrate.cumtrapz(DC2[select2]/(T2[select2]+273), T2[select2]))


t, T3, P3 = Load_data_NA(file3)
Pback3_tot  = get_BG_NDSCIII(T3, P3, [20,55,80,90], 'cubic') [0]
DC3 = (P3-Pback3_tot)/(-gamma)/(V0*C0) * 1e-6
# need to reverse the order for the downs scans
DC3= DC3[::-1]
T3 = T3[::-1]

# use valid range of temperature to analyse
select3 = np.where((T3>20)&(T3<90))[0]
DH3 = [0]
DS3 = [0]
DH3 = np.append(DH3, integrate.cumtrapz(DC3[select3], T3[select3]))  
DS3 = np.append(DS3, integrate.cumtrapz(DC3[select3]/(T3[select3]+273), T3[select3]))



t, T4, P4 = Load_data_NA(file4)
Pback4_tot  = get_BG_NDSCIII(T4, P4, [20,55,80,90], 'cubic') [0]
DC4 = (P4-Pback4_tot)/(-gamma)/(V0*C0) * 1e-6
# need to reverse the order for the downs scans
DC4= DC4[::-1]
T4 = T4[::-1]
select4 = np.where((T4>20)&(T4<90))[0]
DH4 = [0]
DS4 = [0]
DH4 = np.append(DH4, integrate.cumtrapz(DC4[select4], T4[select4]))
DS4 = np.append(DS4, integrate.cumtrapz(DC4[select4]/(T4[select4]+273), T4[select4]))


fig1, ax1 = plt.subplots(1,figsize=(10,10/1.5))
plot_DH_DSC(ax1, T1[select1],DH1, {'color' :'red', 'linestyle':'-', 'label' : 'scan up'})
plot_DH_DSC(ax1, T2[select2],DH2, {'color' :'orange', 'linestyle':'-.', 'label' : 'scan up'})
plot_DH_DSC(ax1, T3[select3],DH3, {'color' :'deepskyblue', 'linestyle':'--', 'label' : 'scan down'})
plot_DH_DSC(ax1, T4[select4],DH4, {'color' :'navy', 'linestyle':':', 'label' : 'scan down'})
ax1.set_xlim(xmin=20, xmax=95)

ax1.legend()
plt.title("DH over the temperature - Scan 01 & 02 - 28/11/2022")
fig1.tight_layout()
fig1.savefig("DH_exemple.png", dpi = 600)


fig2, ax2 = plt.subplots(1,figsize=(10,10/1.5))
plot_DS_DSC(ax2, T1[select1],DS1, {'color' :'red', 'linestyle':'-', 'label' : 'scan up'})
plot_DS_DSC(ax2, T2[select2],DS2, {'color' :'orange', 'linestyle':'-.', 'label' : 'scan up'})
plot_DS_DSC(ax2, T3[select3],DS3, {'color' :'deepskyblue', 'linestyle':'--', 'label' : 'scan down'})
plot_DS_DSC(ax2, T4[select4],DS4, {'color' :'navy', 'linestyle':':', 'label' : 'scan down'})
ax2.set_xlim(xmin=20, xmax=95)

ax2.legend()
plt.title("DS over the temperature - Scan 01 & 02 - 28/11/2022")
fig2.tight_layout()
fig2.savefig("DS_exemple.png", dpi = 600)
