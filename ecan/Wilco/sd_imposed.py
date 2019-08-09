
import matplotlib.pyplot as plt
from scipy.special import erfc
import numpy as np
from math import sqrt
import pandas as pd

# #-Transmissivity [m2/d]
# T = 1500
# #-Storage coefficient [-]
# S = 0.1
# #-Distance [m]
# L = 500
# #-time-steps
# ts = 30
# 
# #-Calculate the SDF
# def sdf(l,s,t):
#     y = (l**2)*s/t
#     return y
# 
# #-SD for on/off pumping at various rates
# def SD(sdf, t, Q, avg=True):
#     ###-Calculate sd for average pumping rate throughout the entire period t (only if avg=True)
#     print Q
#     if avg:
#         Qavg = np.average(Q)
#         print Qavg
#         y = np.sqrt(sdf/(4*t))
#         y = erfc(y)
#         sd_avg = y*Qavg
#         
#     ###-Calculate SD pumping going on and off and variable pumping rates
#     dQ = np.zeros(len(t))
#     sd_matrix = np.zeros([len(t), len(t)])
#     for i in t:
#         ix = np.argwhere(t==i)[0][0]
#         if ix==0:
#             dQ[ix] = Q[ix]
#         else:
#             dQ[ix] = Q[ix] - Q[ix-1]
#             
#         for j in t:
#             jx = np.argwhere(t==j)[0][0]
#             if j>=i:
#                 y = erfc(sqrt(sdf/(4*(j-i+1))))
#                 y = y * dQ[ix]
#                 sd_matrix[ix,jx] = y
#     #-super position the individual curves
#     sd_matrix = np.sum(sd_matrix, axis=0)
#     
#     #-make a figure of the scenario with average pumping for the entire period, and on/off pumping at different rates
#     plt.figure(facecolor='#FFFFFF')
#     lines = plt.plot(t,sd_avg, t, sd_matrix)
#     plt.xlabel('Time')
#     plt.ylabel('Stream depletion [l/s]')
#     plt.grid(True)
#     plt.legend(['Average continuous', 'On/Off'])
#     plt.show()
# 
#         
# 
# 
# 
# 
# #-Calculate the sdf
# SDF = sdf(L,S,T)
# #-time vector
# t = np.arange(1,ts+1,1)
# 
# 
# #-pumping rate vector
# Q_pump = np.random.randint(low=0, high=100, size=ts)
# Q_pump[4:8]=0
# Q_pump[13:20]=0
# 
# SD(SDF, t, Q_pump)
# 


#-Transmissivity [m2/d]
T = 1500
#-Storage coefficient [-]
S = 0.1
#-Distance [m]
L = 500
#-CSV file with puming rates for each date
Qcsv = r'C:\Active\Projects\Rakaia\Ref_data\Groundwater\SD_calculations\dummy_pumping.csv'
# #-time-steps
# ts = 30

#-Calculate the SDF
def sdf(l,s,t):
    y = (l**2)*s/t
    return y

#-SD for on/off pumping at various rates
def SD(sdf, Q, avg=True):
    t = np.arange(1,len(Q)+1,1)
    ###-Calculate sd for average pumping rate throughout the entire period t (only if avg=True)
    if avg:
        Qavg = np.average(Q)
        print Qavg
        y = np.sqrt(sdf/(4*t))
        y = erfc(y)
        sd_avg = y*Qavg
    ###-Calculate SD pumping going on and off and variable pumping rates
    dQ = np.zeros(len(t))
    sd_matrix = np.zeros([len(t), len(t)])
    for i in t:
        ix = np.argwhere(t==i)[0][0]
        if ix==0:
            dQ[ix] = Q[ix]
        else:
            dQ[ix] = Q[ix] - Q[ix-1]
             
        for j in t:
            jx = np.argwhere(t==j)[0][0]
            if j>=i:
                y = erfc(sqrt(sdf/(4*(j-i+1))))
                y = y * dQ[ix]
                sd_matrix[ix,jx] = y
    #-super position the individual curves
    sd_matrix = np.sum(sd_matrix, axis=0)
    
    if avg:
        df = sd_avg, sd_matrix
    else:
        df = sd_matrix
    return df
    


        

Qpump = pd.read_csv(Qcsv, parse_dates=True, dayfirst=True, index_col=0)


#-Calculate the sdf
SDF = sdf(L,S,T)

# 
sd = SD(SDF, Qpump['c'].values)

#-make a figure of the scenario with average pumping for the entire period, and on/off pumping at different rates
plt.figure(facecolor='#FFFFFF')
t = np.arange(1,len(sd[0])+1,1)
lines = plt.plot(t,sd[0], t, sd[1])
plt.xlabel('Time')
plt.ylabel('Stream depletion [l/s]')
plt.grid(True)
plt.legend(['Average continuous', 'On/Off'])
plt.show()



