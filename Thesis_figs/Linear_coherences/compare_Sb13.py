#importing python libs

# import sympy as sym
# sym.init_printing()

import numpy as np
import sys
sys.path.append('/home/peter/model_upconversion')
from math import pi
# import math
import matplotlib.pyplot as plt
# from sympy import I, Matrix, symbols
# from sympy.physics.quantum import TensorProduct, Dagger
import scipy.optimize
# import scipy.integrate
# import scipy.constants as const
from math import sqrt
#import qutip
import scipy.io as sio
from matplotlib.colors import Normalize as Norm

import time
import Linear1.c_linear_ground5 as c_linear_ground5

from Thesis_figs.Ground_params2 import p, sd_delao_from_B, deltaao_from_B
# p['d13'] = 2e-32*sqrt(2/3) #same spin
# p['d12'] = 2e-32*sqrt(1/3) #different spin
# p['d23'] = 2e-32*sqrt(1/3) #different spin
# # p['gammamu'] = p['d23']**2/(p['d13']**2+p['d23']**2)*1/11e-3
# # p['gamma12'] = p['d23']**2/(p['d13']**2+p['d23']**2)*1/11e-3#*10
# # p['gamma23'] = 1/11#1/(p['nbath']+1) * 1e3
# # p['gamma23'] = 1/11#1/(p['nbath']+1) * 1e3
# #
# # p['gammamu'] = 1/11
# # p['gamma12'] = 1/11
# # p['gamma23'] = p['d23']**2/(p['d13']**2+p['d23']**2)*1/11e-3#*10
#
# p['gammamu'] =  p['d23']**2/(p['d13']**2+p['d23']**2)*1/11e-3
# p['gamma12'] =  p['d23']**2/(p['d13']**2+p['d23']**2)*1/11e-3
# p['gamma23'] =  p['d23']**2/(p['d13']**2+p['d23']**2)*1/11e-3
P_pump = 10*np.log10(1.74*1) #in dBm, 1.74 mW are going into the resonator
P_mu = -30 # in dBm
T=150e-3
delaovals=np.linspace(-1e9,1e9,11)*10
delamvals=np.linspace(-1e9,1e9,3)/1
# delaovals=np.linspace(-1e9,1e9,11)*10
# delamvals=np.linspace(-2e9,2e9,21)/1

# delaovals=np.linspace(-1e9,1e9,2)*10
# delamvals=np.linspace(-1e9,1e9,2)/1
delo=.2e8*0
delm=-0.4e8*0
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

p['mean_delao']=0#deltaao_from_B(0,p)
p['sd_delao']=sd_delao_from_B(0.2,p)
p['mean_delam']=0#deltamac_from_B(I_val,p)
N=1e16
N1=2.2e15
p['No']=N
p['Nm']=N
rho0=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhoa=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhoac=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhob=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhobc=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)

delo=.2e8*0
delm=-0.4e8*0
t1=time.time()
Sb13_ground=np.zeros((len(delaovals),len(delamvals)),dtype=np.complex_)

t1=time.time()
for ii, deloval in enumerate(delaovals):
    #print(ii)
    for jj, delmval in enumerate(delamvals):
        rho0[:,:,ii,jj],rhoa[:,:,ii,jj],rhoac[:,:,ii,jj],rhob[:,:,ii,jj],rhobc[:,:,ii,jj]=c_linear_ground5.rho_broad_full(deloval,delmval, 0,p)

        Sb13_ground[ii,jj]=rhob[0,2,ii,jj]*p['Nm']*p['go']
        t1tot=time.time()-t1
        #print(t1tot)

rho0ground=rho0
rhoaground=rhoa
rhoacground=rhoac
rhobground=rhob
rhobcground=rhobc

import Linear_excited1.c_linear_excited1 as c_linear_excited1

from Thesis_figs.Excited_params import p, sd_delao_from_B, deltaao_from_B
# p['d13'] = 2e-32*sqrt(2/3) #same spin
# p['d12'] = 2e-32*sqrt(1/3) #different spin
# p['d23'] = 2e-32*sqrt(1/3) #different spin
# # p['gammamu'] = p['d23']**2/(p['d13']**2+p['d23']**2)*1/11e-3
# # p['gamma12'] = p['d23']**2/(p['d13']**2+p['d23']**2)*1/11e-3#*10
# # p['gamma23'] = 1/11#1/(p['nbath']+1) * 1e3
# # p['gamma23'] = 1/11#1/(p['nbath']+1) * 1e3
# #
# # p['gammamu'] = 1/11
# # p['gamma12'] = 1/11
# # p['gamma23'] = p['d23']**2/(p['d13']**2+p['d23']**2)*1/11e-3#*10
#
# p['gammamu'] =  p['d23']**2/(p['d13']**2+p['d23']**2)*1/11e-3
# p['gamma12'] =  p['d23']**2/(p['d13']**2+p['d23']**2)*1/11e-3
# p['gamma23'] =  p['d23']**2/(p['d13']**2+p['d23']**2)*1/11e-3

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
    Omega=p['d13']*Efield/hbar
    return Omega
def nbath_from_T(T,p,deltamacval=0):
    omega=np.abs(2*pi*p['freqmu']+deltamacval)
    nbath = 1/(np.exp(1.0545718e-34*omega/1.38064852e-23/T)-1)
    return nbath
p['Omega']= Omega_cavity_from_PdBm_resonance(P_pump,p)
p['nbath']=nbath_from_T(T,p)
p['mean_delao']=0#deltaao_from_B(0,p)
p['sd_delao']=sd_delao_from_B(0.2,p)
p['mean_delam']=0#deltamac_from_B(I_val,p)
N=1e16
N1=2.2e15
p['No']=N
p['Nm']=N
rho0=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhoa=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhoac=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhob=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhobc=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
Sb13_excited=np.zeros((len(delaovals),len(delamvals)),dtype=np.complex_)

t1=time.time()
for ii, deloval in enumerate(delaovals):
    #print(ii)
    for jj, delmval in enumerate(delamvals):
        rho0[:,:,ii,jj],rhoa[:,:,ii,jj],rhoac[:,:,ii,jj],rhob[:,:,ii,jj],rhobc[:,:,ii,jj]=c_linear_excited1.rho_broad_full(deloval,delmval, 0,p)

        Sb13_excited[ii,jj]=rhob[0,2,ii,jj]*p['Nm']*p['go']
        t1tot=time.time()-t1
        #print(t1tot)
rho0excited=rho0
rhoaexcited=rhoa
rhoacexcited=rhoac
rhobexcited=rhob
rhobcexcited=rhobc


filename='compare_Sb13_nothing'
np.savez('Thesis_figs/Linear_coherences/'+filename,delamvals=delamvals,delaovals=delaovals,p=p,P_pump=P_pump,T=T,
        Sb13_excited=Sb13_excited,Sb13_ground=Sb13_ground,delm=delm,delo=delo,
        rho0excited=rho0excited,rhoaexcited=rhoaexcited,rhoacexcited=rhoacexcited,rhobexcited=rhobexcited,rhobcexcited=rhobcexcited,
        rho0ground=rho0ground,rhoaground=rhoaground,rhoacground=rhoacground,rhobground=rhobground,rhobcground=rhobcground)#,deltaoval=deltaoval)


fig=plt.figure()

ax=fig.add_subplot(1,2,1)
img1=ax.imshow((np.log10(np.abs(Sb13_ground[:,:]))),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
fig.colorbar(img1)
ax=fig.add_subplot(1,2,2)
img1=ax.imshow((np.log10(np.abs(Sb13_excited[:,:]))),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
fig.colorbar(img1)
print(np.max(np.abs(Sb13_ground[:,:])))
print(np.max(np.abs(Sb13_excited[:,:])))
fig=plt.figure()

ax=fig.add_subplot(1,2,1)
img1=ax.imshow(((np.abs(Sb13_ground[:,:]))),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
fig.colorbar(img1)
ax=fig.add_subplot(1,2,2)
img1=ax.imshow(((np.abs(Sb13_excited[:,:]))),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
fig.colorbar(img1)
print(np.max(np.abs(Sb13_ground[:,:])))
print(np.max(np.abs(Sb13_excited[:,:])))


plt.show()
