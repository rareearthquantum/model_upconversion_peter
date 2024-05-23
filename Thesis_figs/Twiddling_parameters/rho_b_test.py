import sympy as sym
sym.init_printing()

import numpy as np
import sys
sys.path.append('/home/peter/model_upconversion')
from math import pi
import math
import matplotlib.pyplot as plt
from sympy import I, Matrix, symbols
from sympy.physics.quantum import TensorProduct, Dagger
import scipy.optimize
import scipy.integrate
import scipy.constants as const

#import qutip

from matplotlib.colors import Normalize as Norm

import time
from c_funs_ds3 import rho_broad_full

filename='Thesis_figs/Double_cavity/Ground_crossing2'
B_vals=np.linspace(0.204,0.212,21)
deltamvals=np.linspace(-70e6,70e6,21)*2*np.pi
# B_vals=np.linspace(0.206,0.210,21)
# deltamvals=np.linspace(-70e6,70e6,21)*2*np.pi
P_pump = 10*np.log10(1.74) #in dBm, 1.74 mW are going into the resonator
P_mu = -39.9 # in dBm
# P_mu = -30.9 # in dBm
# P_mu = 3-30 # in dBm

P_mu = -47.9 # in dBm
T=150e-3
p={}
p['freqmu']=5015e6#4733e6 #this is the microwave cavity frequency
#p['freqmu']=5021e6#4733e6 #this is the microwave cavity frequency

p['freq_pump'] = 1951170.00e9 #pump frequency
#p['freqo']=p['freqmu']+p['freq_pump']

p['Gg']=0.024085156244027e12
p['Ge']=0.017976119414574e12
#p['Ge']=0.020275e12
#p['Gg']=0.0241886e12
p['f0_no_B']=195.1167943776907e12
p['d13'] = 2e-32*math.sqrt(1/3)
p['d23'] = 2e-32*math.sqrt(2/3)
p['gamma13'] = p['d13']**2/(p['d13']**2+p['d23']**2)*1/11e-3
p['gamma23'] = p['d23']**2/(p['d13']**2+p['d23']**2)*1/11e-3
p['gamma2d'] = 1e6
p['gamma3d'] = 1e6
#p['nbath'] = 20
p['gammamu'] = 1/11*1#/2*pi
p['sd_delam']=2*pi*25e6/2.355
p['sd_delam']=2*pi*14e6/2.355
p['sd_delam']=2*pi*2e6

p['gammaoc']=2*pi*1.7e6
p['gammaoi']=2*pi*7.95e6#*1e-9
p['gammamc']=2*pi*0.0622e6
p['gammami']=2*pi*5.69e6

p['gammamc']=2*pi*2.199e6 #from dB fit
p['gammami']=2*pi*5.083e6

p['gammamc']=2*pi*1.452e6 #from linear fit
p['gammami']=2*pi*1.17e6


muBohr=927.4009994e-26; # Bohr magneton in J/T in J* T^-1
p['mu12'] = 4.3803*muBohr # transition dipole moment for microwave cavity (J T^-1)


p['go'] = 51.9  #optical coupling

p['No'] = 2.2e15 # number of atoms in the optical mode
p['Nm'] = 6e16  #toal number of atoms
p['Nm'] = 6e16*0.8  #toal number of atoms
#p['Nm'] = 2e16  #toal number of atoms
#p['No'] = p['Nm'] # number of atoms in the optical mode

#p['No'] = 1.3e15 # number of atoms in the optical mode
#p['Nm'] = 2e16  #toal number of atoms

p['gm'] = 1.04 #coupling between atoms and microwave field

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
    Omega=p['d13']*Efield/hbar
    return Omega
def bin_from_PdBm(Pdbm,p,deltamval=0):
    P=1e-3*10**(Pdbm/10)
    hbar=1.05457e-34; # in J*s
    omega=2*np.pi*(p['freqmu'])+deltamval
    bin=np.sqrt(P/hbar/omega)
    return bin
def nbath_from_T(T,p,deltamacval=0):
    omega=np.abs(2*pi*p['freqmu']+deltamacval)
    nbath = 1/(np.exp(1.0545718e-34*omega/1.38064852e-23/T)-1)
    return nbath
def deltamac_from_I(B_mag,p):
    #B_mag=(0.027684*I_mag*1e3-0.056331)*1e-3
    deltamacvals=(p['Gg'])*(B_mag)-p['freqmu']
    return deltamacvals*2*np.pi
def omegaao_from_I(B_mag,p): #transition 2
    return (p['f0_no_B']+(-p['Gg']+p['Ge'])/2*B_mag)*2*np.pi
def sd_delao_from_I(B_mag,p):
    return 1e9*(0.096660692060221+0.336895927831593*B_mag)*np.pi*2
def deltaao_from_I(B_mag,p):
    return -(2*pi*(p['freqmu']+p['freq_pump'])-omegaao_from_I(B_mag,p))

def Omega_cavity_from_PdBm(PdBm,p):
    mu0=4*pi*1e-7
    epsilon0=8.854187817e-12
    c=3e8
    hbar=1.05457e-34; # in J*s
    P=1e-3*10**(PdBm/10)
    pflux=P/(2*pi*hbar*p['freq_pump']) #photons/sec
    n_cavity=pflux*p['gammaoc']/((2*pi*(p['freq_pump']-p['freq_pump_cavity']))**2+(p['gammaoc']*2+p['gammaoi'])**2/4)
    A_beam = pi*p['Wbeam']**2/4 #cross section of beam
    V_beam=A_beam*(p['Lcavity_vac']+p['Lsample']*p['nYSO']**3)/2 # why /2??
    Efield=np.sqrt(n_cavity*hbar*2*pi*p['freq_pump']/2/epsilon0/V_beam)
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

p['Omega']= Omega_cavity_from_PdBm_resonance(P_pump,p)

print('Omega = ' + str(p['Omega']))
print('bin   = ' + str(bin_from_PdBm(P_mu,p)))
binvals=bin_from_PdBm(-45.9,p)#*1e7#bin_from_PdBm(P_mu,p)
b_phase=np.linspace(0,2*np.pi,200)
bvals= binvals/np.sqrt(p['gammamc'])*np.exp(1j*b_phase)

rho_out=np.zeros((3,3,len(bvals)),dtype=np.complex_)
TT=500e-3
B_val=0.205
for ii, bval in enumerate(bvals):
    aval=0#bval*1e-3
    #bval= binval/np.sqrt(p['gammamc'])
    p['mean_delam']=deltamac_from_I(B_val,p)
    p['nbath']=nbath_from_T(TT,p,deltamacval=p['mean_delam'])
    p['mean_delao']=deltaao_from_I(B_val,p)
    p['sd_delao']=sd_delao_from_I(B_val,p)
    p['Omega']= Omega_cavity_from_PdBm_resonance(P_pump,p)
    deltaoval=1e8
    deltamval=2.9
    rho_out[:,:,ii]=rho_broad_full(aval,bval,deltaoval,deltamval,p)

fig=plt.figure(figsize=(12,12))

fig.clf()
pltnum=1
for ii in range(3):
    for jj in range(3):
        ax=fig.add_subplot(3,3,pltnum)

        plt.plot(b_phase,abs(rho_out[ii,jj,:]))

        plt.title('$\\rho_{' +str(ii+1)+',' +str(jj+1)+'}$')
        #plt.margins(0.1)
        pltnum=pltnum+1
#fig.tight_layout()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

fig=plt.figure(figsize=(12,12))

fig.clf()
pltnum=1
for ii in range(3):
    for jj in range(3):
        ax=fig.add_subplot(3,3,pltnum)

        plt.plot(b_phase,np.angle(rho_out[ii,jj,:]))

        plt.title('$\\rho_{' +str(ii+1)+',' +str(jj+1)+'}$')
        #plt.margins(0.1)
        pltnum=pltnum+1
#fig.tight_layout()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.figure()
plt.plot(b_phase,rho_out[0,0,:]-rho_out[1,1,:])

plt.show()
