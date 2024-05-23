#importing python libs

import sympy as sym
sym.init_printing()

import numpy as np
import sys

from math import pi
import math
import matplotlib.pyplot as plt
from sympy import I, Matrix, symbols
from sympy.physics.quantum import TensorProduct, Dagger

import scipy.constants as const

#import qutip

from matplotlib.colors import Normalize as Norm

import time


filename='Thesis_figs/sims1/test_crossing_excited_out4_testS21'
#B_vals=np.linspace(0.23,0.237,6)
B_vals=np.linspace(0.225,0.24,51)
deltamvals=np.linspace(-25e6,25e6,51)*2*np.pi
#deltamvals=np.linspace(-25e6,25e6,5)
B_vals=np.linspace(0.226,0.238,21)
deltamvals=np.linspace(-25e6,25e6,11)*2*np.pi
# B_vals=np.linspace(0.231,0.236,31)
# deltamvals=np.linspace(-25e6,25e6,11)*2*np.pi
P_pump = 10*np.log10(1.74/1) #in dBm, 1.74 mW are going into the resonator
P_mu = -30.1 # in dBm
#P_mu = -30.1-20 # in dBm

T=150e-3
def deltamac_from_I(B_mag,p):
    #B_mag=(0.027684*I_mag*1e3-0.056331)*1e-3
    deltamacvals=(p['Ge'])*(B_mag)-p['freqmu']
    return deltamacvals*2*np.pi
def omegaao1_from_I(B_mag,p):
    return (p['f0_no_B']+(-p['Gg']-p['Ge'])/2*B_mag)*2*np.pi
def omegaao2_from_I(B_mag,p):
    return (p['f0_no_B']+(-p['Gg']+p['Ge'])/2*B_mag)*2*np.pi
def omegaao3_from_I(B_mag,p):
    return (p['f0_no_B']+(+p['Gg']-p['Ge'])/2*B_mag)*2*np.pi
def omegaao4_from_I(B_mag,p):
    return (p['f0_no_B']+(+p['Gg']+p['Ge'])/2*B_mag)*2*np.pi

def sd_delao1_from_I(B_mag,p):
    return 1e9*(0.148588212918272+0.308180352441052*B_mag)*np.pi*2
def sd_delao2_from_I(B_mag,p):
    #return 1e9*(0.096660692060221-0.336895927831593*B_mag)*np.pi*2
    return 1e9*(0.096660692060221+0.336895927831593*B_mag)*np.pi*2
def sd_delao3_from_I(B_mag,p):
    return 1e9*(0.391366413926165+0.137444967951503*B_mag)*np.pi*2
def sd_delao4_from_I(B_mag,p):
    return 1e9*(0.279915072366011+0.000852680155581*B_mag)*np.pi*2

p={}
p['freqmu']=4732e6 #this is the microwave cavity frequency
p['freq_pump'] = 195117.044e9 #9.38e9#pump frequency
#p['freqo']=p['freqmu']+p['freq_pump']
p['Gg']=0.024085156244027e12
p['Ge']=0.017976119414574e12
p['Ge']=0.020275e12
p['Gg']=0.0241886e12
p['f0_no_B']=195.1167943776907e12
p['d13'] = 2e-32*math.sqrt(2/3)
p['d12'] = 2e-32*math.sqrt(1/3)
p['gamma13'] = p['d13']**2/(p['d13']**2+p['d12']**2)*1/11e-3#*10
p['gamma12'] = p['d12']**2/(p['d13']**2+p['d12']**2)*1/11e-3#*10
p['gamma2d'] = 1e6
p['gamma3d'] = 1e6
#p['nbath'] = 20
p['gamma23'] = 1/11#1/(p['nbath']+1) * 1e3
p['sd_delam']=2*pi*2e6
#p['sd_delam']=2*pi*25e6/2.355/2
#p['sd_delam']=2*pi*14e6/2.355/2

p['gammaoc']=2*pi*1.7e6
p['gammaoi']=2*pi*7.95e6
p['gammamc']=2*pi*0.1522e6
p['gammami']=2*pi*3.255e6

p['gammamc']=2*pi*1.525e6
p['gammami']=2*pi*2.163e6

p['gammamc']=2*pi*1.24e6 #from linear fit
p['gammami']=2*pi*0.9399e6

#p['gammamc']=2*pi*1.289e6 #from dbB fit
#p['gammami']=2*pi*1.171e6

muBohr=927.4009994e-26; # Bohr magneton in J/T in J* T^-1
p['mu12'] = 4.3803*muBohr # transition dipole moment for microwave cavity (J T^-1)


p['go'] = 51.9  #optical coupling

p['No_total'] = 2.2e15 # number of atoms in the optical mode

p['Nm_total'] = 6e16  #toal number of atoms
#p['Nm_total'] = 2e16  #toal number of atoms

p['gm'] = 1.04/10 #coupling between atoms and microwave field
p['T']=T
p['Wbeam']=0.6e-3

p['Lsample']=12e-3 # the length of the sample, in m
p['Lcavity_vac'] = 49.5e-3 # length of the vacuum part of the optical Fabry Perot (m)
p['nYSO'] = 1.76 #refractive index of YSO

#functions to define simulation parameters which change as we change the frequencies

# def gammamu_from_T(T):
#     T_zeeman=[2.34,0.66,0.3]#,0.18]
#     gammamu=1/(5.63e-6)
#     for Ti in T_zeeman:

def Omega_from_PdBm(PdBm,p):
    mu0=4*pi*1e-7
    c=3e8
    hbar=1.05457e-34; # in J*s
    P=1e-3*10**(PdBm/10)
    Abeam=pi*p['Wbeam']**2/4
    Efield=np.sqrt(2*mu0*c*P/Abeam)
    Omega=p['d12']*Efield/hbar
    return Omega
def bin_from_PdBm(Pdbm,p,deltamval=0):
    P=1e-3*10**(Pdbm/10)
    hbar=1.05457e-34; # in J*s
    omega=2*np.pi*(p['freqmu'])+deltamval
    bin=np.sqrt(P/hbar/omega)
    return bin
def nbath_from_T(T,p,deltamacval=0):
    omega=np.abs(2*pi*p['freqmu']-deltamacval)
    nbath = 1/(np.exp(1.0545718e-34*omega/1.38064852e-23/T)-1)
    return nbath
def deltamac_from_I(B_mag,p):
    #B_mag=(0.027684*I_mag*1e3-0.056331)*1e-3
    deltamacvals=(p['Ge'])*(B_mag)-p['freqmu']
    return deltamacvals*2*np.pi
def omegaao_from_I(B_mag,p): #transition 2
    return (p['f0_no_B']+(-p['Gg']-p['Ge'])/2*B_mag)*2*np.pi
def sd_delao_from_I(B_mag,p):
    return 1e9*(0.148588212918272+0.308180352441052*B_mag)*np.pi*2
#    return 1e9*(0.391366413926165+0.137444967951503*B_mag)*np.pi*2



def deltaao_from_I(deltamval,B_mag,p):
    return 2*pi*(p['freqmu']+p['freq_pump'])+deltamval-omegaao2_from_I(B_mag,p)

def deltaao_from_I(deltamval,B_mag,p):
    return 2*pi*(-p['freqmu']+p['freq_pump'])-deltamval-omegaao2_from_I(B_mag,p)

def deltaao_from_I(deltamval,B_mag,p):
    return -deltamval
B_val=0.228
print(omegaao2_from_I(B_val,p)*1e-9)
print(2*pi*p['freqmu']*1e-9)
print((p['freq_pump']-0.5e9)*2*pi-2*pi*p['freqmu']-omegaao1_from_I(B_val,p)-deltamvals)
