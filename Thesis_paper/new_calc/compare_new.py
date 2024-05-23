#importing python libs

# import sympy as sym
# sym.init_printing()

import numpy as np
import sys
sys.path.append('/home/peter/model_upconversion')
sys.path.append('Linear1/build')
from math import pi
# import math
import matplotlib.pyplot as plt
# from sympy import I, Matrix, symbols
# from sympy.physics.quantum import TensorProduct, Dagger
import scipy.optimize
# import scipy.integrate
# import scipy.constants as const

#import qutip
import scipy.io as sio
from matplotlib.colors import Normalize as Norm

import time
import c_linear_ground5 as c_linear_ground5
from output_calcs.c_funs_test3 import steady_rho_single_c, gauss_fun_1d

from Thesis_figs.Ground_params2 import p, sd_delao_from_B, deltaao_from_B
P_pump = 10*np.log10(10) #in dBm, 1.74 mW are going into the resonator
# P_mu = -30 # in dBm
T=50e-3
delaovals=np.linspace(-2e9,2e9,50)*1000
delamvals=np.linspace(-2e9,2e9,50)*1000


# delaovals=np.linspace(-2e9,0,30)*100
# delamvals=np.linspace(-2e9,0,30)*100
P_mu=10*np.log10(5000e6*1.05e-34/1e-6*1000)
#P_mu=-60
print("P_mu = " + str(P_mu))
Q=1e8
# Q=2*pi*p['freq_pump']/p['gammaoi']

p['gammaoi']=2*pi*p['freq_pump']/Q
print('gammaoi = ' + str(p['gammaoi']))

def Omega_from_PdBm(PdBm,p):
    mu0=4*np.pi*1e-7
    c=3e8
    hbar=1.05457e-34; # in J*s
    P=1e-3*10**(PdBm/10)
    Abeam=pi*p['Wbeam']**2/4
    Efield=np.sqrt(2*mu0*c*P/Abeam)
    Omega=p['d23']*Efield/hbar
    return Omega
def Omega_cavity_from_PdBm_resonance(PdBm,p):
    mu0=4*pi*1e-7
    epsilon0=8.854187817e-12
    c=3e8
    hbar=1.05457e-34; # in J*s
    P=1e-3*10**(PdBm/10)
    pflux=P/(2*pi*hbar*p['freq_pump']) #photons/sec
    n_cavity=pflux*p['gammaoc']/((p['gammaoc']*2+p['gammaoi'])**2/4)
    A_beam = pi*p['Wbeam']**2/4 #cross section of beam
    V_beam=A_beam*(p['Lcavity_vac']+p['Lsample']*p['nYSO']**3)/2 # why /2??
    Efield=np.sqrt(n_cavity*hbar*2*pi*p['freq_pump']/2/epsilon0/V_beam)
    Omega=p['d23']*Efield/hbar
    return Omega
def nbath_from_T(T,p,deltamacval=0):
    omega=np.abs(2*pi*p['freqmu']+deltamacval)
    nbath = 1/(np.exp(1.0545718e-34*omega/1.38064852e-23/T)-1)
    return nbath
p['Omega']= Omega_cavity_from_PdBm_resonance(P_pump,p)
p['nbath']=nbath_from_T(T,p)

def bin_from_PdBm(Pdbm,p,deltamval=0):
    P=1e-3*10**(Pdbm/10)
    hbar=1.05457e-34; # in J*s
    omega=2*np.pi*(p['freqmu'])+deltamval
    bin=np.sqrt(P/hbar/omega)
    return bin

def steady_rho_single(delao,delam,aval,bval,delo,delm,p):
    S_out_full=steady_rho_single_c(delao,delam,aval,bval,delo,delm,p)
    rho1=[[S_out_full[0],S_out_full[3]+1j*S_out_full[4],S_out_full[5]+1j*S_out_full[6]],[S_out_full[3]-1j*S_out_full[4],S_out_full[1],S_out_full[7]+1j*S_out_full[8]],[S_out_full[5]-1j*S_out_full[6],S_out_full[7]-1j*S_out_full[8],S_out_full[2]]]

    return np.array(rho1)#*gauss_fun_1d(delam,p['mean_delam'],p['sd_delam'])*gauss_fun_1d(delao,p['mean_delao'],p['sd_delao'])*1e12
def steady_rho(delaovals,delamvals,aval,bval,delo,delm,p):
    rhovals=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
    for ii, delao in enumerate(delaovals):
        for jj, delam in enumerate(delamvals):
            rhovals[:,:,ii,jj]=steady_rho_single(delao,delam,aval,bval,delo,delm,p)
    return rhovals


bval=bin_from_PdBm(P_mu,p)/np.sqrt(p['gammamc'])
aval=bval*0.2
delo=.2e8*0
delm=-0.4e8*0

rhovals=steady_rho(delaovals,delamvals,aval,bval,delo,delm,p)


rho0=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhoa=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhoac=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhob=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhobc=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)

t1=time.time()
for ii, deloval in enumerate(delaovals):
    #print(ii)
    for jj, delmval in enumerate(delamvals):
        S_out_full=c_linear_ground5.steady_rhos(deloval,delmval, delo,delm,p)
        #print(S_out_full)
        S_out_fullpy=np.zeros(45)
        for ll,newind in enumerate([ 0,  1,  2,  7,  8, 12, 13, 14, 15, 21, 22, 23, 24, 30, 31, 32, 33,39, 40, 41, 42]):
            S_out_fullpy[newind]=S_out_full[ll]
        #print(S_out_fullpy)
        CtoRinv=np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, -1j, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1j, 0, 0],
        [0, 0, 0, 1, 1j, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, -1j],
        [0, 0, 0, 0, 0, 1, 1j, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1j],
        [0, 0, 1, 0, 0, 0, 0, 0, 0]])/2

        rho0[:,:,ii,jj]=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[0:9])),(3,3))
        rhoar=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[9:18])),(3,3))
        rhoai=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[18:27])),(3,3))
        rhobr=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[27:36])),(3,3))
        rhobi=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[36:45])),(3,3))
        rhoa[:,:,ii,jj]=rhoar + 1j*rhoai
        rhoac[:,:,ii,jj]=rhoar - 1j*rhoai
        rhob[:,:,ii,jj]=rhobr + 1j*rhobi
        rhobc[:,:,ii,jj]=rhobr - 1j*rhobi

def find_difference(aa,bb):
    return np.max(np.sqrt((np.abs(aa.real)-np.abs(bb.real))**2+(np.abs(aa.imag)-np.abs(bb.imag))**2)/np.abs(bb))

# print(rhovals[2,0,:,:])
# # print(rhob[0,2,:,:]*bval)
# # print(rhoa[0,2,:,:]*aval)
# print(rhoa[2,0,:,:]*aval)
# print(rhob[2,0,:,:]*bval)
# print((rhoa[2,0,:,:]*aval+rhob[2,0,:,:]*bval)/rhovals[2,0,:,:])
# print('==========')
#
# print(rhovals[1,0,:,:])
# # print(rhoa[0,1,:,:]*aval)
# # print(rhob[0,1,:,:]*bval)
# print(rhoa[1,0,:,:]*aval)
# print(rhob[1,0,:,:]*bval)
# print((rhoa[1,0,:,:]*aval+rhob[1,0,:,:]*bval)/rhovals[1,0,:,:])
print(np.max((np.abs(rhoa[2,0,:,:]*aval+rhob[2,0,:,:]*bval)-np.abs(rhovals[2,0,:,:]))/np.abs(rhovals[2,0,:,:])))
print(np.max((np.abs(rhoa[1,0,:,:]*aval+rhob[1,0,:,:]*bval)-np.abs(rhovals[1,0,:,:]))/np.abs(rhovals[1,0,:,:])))
print(find_difference(rhoa[2,0,:,:]*aval+rhob[2,0,:,:]*bval,rhovals[2,0,:,:]))
print(find_difference(rhoa[1,0,:,:]*aval+rhob[1,0,:,:]*bval,rhovals[1,0,:,:]))
