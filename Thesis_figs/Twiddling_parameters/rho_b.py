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
from Thesis_figs.Ground_params2 import p, sd_delao_from_B, deltaao_from_B


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
binvals=np.linspace(bin_from_PdBm(-45.9,p),bin_from_PdBm(-30,p),201)#*1e7#bin_from_PdBm(P_mu,p)
binvals=np.linspace(0,bin_from_PdBm(-52,p),201)#*1e7#bin_from_PdBm(P_mu,p)
bvals= binvals/np.sqrt(p['gammamc'])

rho_out=np.zeros((3,3,len(binvals)),dtype=np.complex_)
frac_diff=np.zeros((len(binvals)),dtype=np.complex_)
TT=100e-3
B_val=0.2115
filename='Thesis_figs/Twiddling_parameters/rho_b4'
deltamval=0
deltaoval=0
for ii, bval in enumerate(bvals):
    aval=0
    #bval= binval/np.sqrt(p['gammamc'])
    p['mean_delam']=0#deltamac_from_I(B_val,p)
    p['nbath']=nbath_from_T(TT,p,deltamacval=p['mean_delam'])
    p['mean_delao']=deltaao_from_I(B_val,p)
    p['sd_delao']=sd_delao_from_I(B_val,p)
    p['Omega']= Omega_cavity_from_PdBm_resonance(P_pump,p)*0
    #frac_diff[ii]=p['gamma12']/2*(p['gamma12']*(2*p['nbath']+1)+p['gamma2d'])/(4*abs(p['gm']*bval)**2+p['gamma12']/2*(2*p['nbath']+1)*(p['gamma12']*(2*p['nbath']+1)+p['gamma2d']))
    deltaoval=0e8
    deltamval=0e8
    rho_out[:,:,ii]=rho_broad_full(aval,bval,deltaoval,deltamval,p)
np.savez(filename,rho_out=rho_out,TT=TT,aval=aval,p=p,B_val=B_val,deltamval=deltamval,deltaoval=deltaoval,
        bvals=bvals,P_pump=P_pump,binvals=binvals)
fig=plt.figure(figsize=(12,12))

fig.clf()
pltnum=1
for ii in range(3):
    for jj in range(3):
        ax=fig.add_subplot(3,3,pltnum)

        plt.plot(bvals,abs(rho_out[ii,jj,:]))

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

        plt.plot(bvals,np.angle(rho_out[ii,jj,:]))

        plt.title('$\\rho_{' +str(ii+1)+',' +str(jj+1)+'}$')
        #plt.margins(0.1)
        pltnum=pltnum+1
#fig.tight_layout()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

fig=plt.figure('rho/b',figsize=(12,12))

fig.clf()
pltnum=1
for ii in range(3):
    for jj in range(3):
        ax=fig.add_subplot(3,3,pltnum)

        plt.plot(bvals,abs(rho_out[ii,jj,:])/bvals)

        plt.title('$\\rho_{' +str(ii+1)+',' +str(jj+1)+'}$')
        #plt.margins(0.1)
        pltnum=pltnum+1
#fig.tight_layout()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.suptitle('rho/b')

plt.figure()
plt.plot(bvals,rho_out[0,0,:]-rho_out[1,1,:])

plt.show()
