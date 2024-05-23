#importing python libs

# import sympy as sym
# sym.init_printing()

import numpy as np
import sys
sys.path.append('/home/peter/model_upconversion')
from math import pi
import math
import matplotlib.pyplot as plt
# from sympy import I, Matrix, symbols
# from sympy.physics.quantum import TensorProduct, Dagger
import scipy.optimize
# import scipy.integrate
# import scipy.constants as const

#import qutip

from matplotlib.colors import Normalize as Norm

import time
from Linear_excited1.c_linear_excited1 import rho_broad_full
# from c_linear_excited1 import rho_broad_full

from Thesis_figs.Excited_params import p, sd_delao_from_B, deltaao_from_B

filename='Linear_effic_excited_sameN_lob1'

#I_vals=np.linspace(7.43,7.55,31)
delta2vals=np.linspace(-1000e6,1000e6,31)
delta3vals=np.linspace(-15e10,15e10,31)
delta2vals=np.linspace(-250e6,250e6,21)*10
delta3vals=np.linspace(-1e10,1e10,21)
# delta2vals=np.linspace(-250e6,250e6,11)#[:-1]
# delta3vals=np.linspace(-45e9,45e9,11)#[:-1]
# delta2vals=np.linspace(-10e7,-2.5e7,20)
# delta3vals=np.linspace(-2.8e10,-0.6e10,20)

#delta2vals=np.linspace(-300e6,000e6,11)[:-1]/1000
#delta3vals=np.linspace(-45e9,0e10,11)[:-1]/1000
# delta2vals=np.linspace(-1000e6,000e6,11)[:-1]/10*3
# delta3vals=np.linspace(-4e10,0e10,11)[:-1]
# delta2vals=np.linspace(-200e6,000e6,31)[:-1]
# delta3vals=np.linspace(-25e9,0e10,31)[:-1]
delta2vals=np.linspace(-1e9,0,21)[:-1]*3*10*2
delta3vals=np.linspace(-1e10,0,6)[:-1]
delta2vals=np.linspace(-1e9,0,11)[:-1]*4
delta3vals=np.linspace(-1e10,0,5)[:-1]
# delta2vals=np.linspace(-2,-0.5,21)*1e9
# delta3vals=np.linspace(-9,-3,21)*1e9
# delta2vals=np.linspace(-1.2e8,-0.5e8,40)#[:-1]?
# delta3vals=np.linspace(-1.7e10,-1.45e10,10)#[:-1]
P_pump_mW=100.74
P_pump_mW=10

P_pump = 10*np.log10(P_pump_mW) #in dBm, 1.74 mW are going into the resonator
T=100e-3
Q=1e9
p['gammaoi']=2*pi*p['freq_pump']/Q
print('gammaoi = ' + str(p['gammaoi']))
# p['Nm']=p['No']
# p['No']=p['Nm']
N=1e16
N=2.2e15
p['Nm']=N
p['No']=N
p['gammami']=650e3*2*pi*1e0

filename=filename+'T='+str(T)+'_Q='+str(Q)+'_Ppump='+str(P_pump_mW)


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
def deltamac_from_B(B_mag,p):
    #B_mag=(0.027684*I_mag*1e3-0.056331)*1e-3
    deltamacvals=(p['Ge'])*(B_mag)-p['freqmu']
    return deltamacvals*2*np.pi

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
    Omega=p['d13']*Efield/hbar
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

p['Omega']= Omega_cavity_from_PdBm_resonance(P_pump,p)
print(p['Omega'])
# p['Omega']= Omega_from_PdBm(P_pump,p)*10
print(p['Omega'])

def calc_efficiency(deloval,delmval,delocval,delmucval,p):
    #aout=Caa*ain+Cba*bin
    #bout=Cab*ain+Cbb*bin
    rho0,rhoa,rhoac,rhob,rhobc=rho_broad_full(deloval,delmval, 0,p)

    Sa13=rhoa[0,2]*p['No']*p['go']
    Sb13=rhob[0,2]*p['Nm']*p['go']

    Sa23=rhoa[1,2]*p['No']*p['gm']
    Sb23=rhob[1,2]*p['Nm']*p['gm']



    Caa=p['gammaoc']*(1j*Sb23-1j*(delmval-delmucval)+(p['gammami']+p['gammamc']*2)/2)/(Sa23*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc']*2)/2)*(1j*Sb23-1j*(delmval-delmucval)+(p['gammami']+p['gammamc']*2)/2))
    Cba=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa23*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc']*2)/2)*(1j*Sb23-1j*(delmval-delmucval)+(p['gammami']+p['gammamc']*2)/2))
    Cab=           -1j*np.sqrt(p['gammaoc'])*np.sqrt(p['gammamc'])*Sa23/(Sa23*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc']*2)/2)*(1j*Sb23-1j*(delmval-delmucval)+(p['gammami']+p['gammamc']*2)/2))
    Cbb=p['gammamc']*(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc']*2)/2)/(Sa23*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc']*2)/2)*(1j*Sb23-1j*(delmval-delmucval)+(p['gammami']+p['gammamc']*2)/2))

    return Caa, Cba, Cab, Cbb
Caa=np.zeros((len(delta2vals),len(delta3vals)),dtype=np.complex_)
Cba=np.zeros((len(delta2vals),len(delta3vals)),dtype=np.complex_)
Cab=np.zeros((len(delta2vals),len(delta3vals)),dtype=np.complex_)
Cbb=np.zeros((len(delta2vals),len(delta3vals)),dtype=np.complex_)
N_frac=np.zeros((len(delta2vals),len(delta3vals)),dtype=np.complex_)

calc_time=np.zeros((len(delta2vals),len(delta3vals)))

start_time=time.time()
deltaoval=0#10.e9

start_time=time.time()
deltaocval=0e9
deltamucval=0e8

p['mean_delao']=0#deltaao_from_B(0,p)
p['sd_delao']=sd_delao_from_B(0.2,p)
for ii,delta2val in enumerate(delta2vals):
    p['mean_delam']=delta2val#deltamac_from_B(I_val,p)
    #deltamucval=p['Nm']*p['gm']**2/delta2val
    p['nbath']=nbath_from_T(T,p)
    for jj, delta3val in enumerate(delta3vals):
        p['mean_delao']=delta3val
        deltaocval=p['No']*p['go']**2/delta3val

        #time1=time.time()
        Caa[ii,jj], Cba[ii,jj], Cab[ii,jj], Cbb[ii,jj]=calc_efficiency(0,0,deltaocval,deltamucval,p)

    elapsed_time=time.time()-start_time
    print('    ' + str(ii) +', Time: ' + time.ctime() +', Elapsed: '+ str(elapsed_time))
    #np.savez(filename,binvals=binvals,delta2vals=delta2vals,p=p,bvals=bvals,deltamucvals=deltamucvals,rho_out=rho_out,P_pump=P_pump,P_mu=P_mu,T=T,elapsed_time=elapsed_time,avals=avals,deltaocval=deltaocval,calc_time=calc_time)
    #np.savez(filename,p=p,P_mu=P_mu,P_pump=P_pump,T=T,elapsed_time=elapsed_time,calc_time=calc_time,binvals=binvals,ainval=ainval,bvals=bvals,avals=avals,delta2vals=delta2vals,delta3vals=delta3vals)
    np.savez(filename,Caa=Caa, Cba=Cba,Cab=Cab,Cbb=Cbb,p=p,P_pump=P_pump,T=T,elapsed_time=elapsed_time,calc_time=calc_time,delta2vals=delta2vals,delta3vals=delta3vals)

print('    ===========================Complete===========================')


fig=plt.figure(filename+'_Caa')

fig.clf()

ax=fig.add_subplot(3,1,1)
img1=ax.imshow(10*np.log10(np.abs(Caa)**2),extent=(np.min(delta3vals),np.max(delta3vals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title(' |C|^2 (dB scale)')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,2)
img1=ax.imshow((np.abs(Caa)**2),extent=(np.min(delta3vals),np.max(delta3vals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('|C|^2 ')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,1,3)
img1=ax.imshow((np.angle(Caa)),extent=(np.min(delta3vals),np.max(delta3vals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' C phase')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1)

fig.suptitle('Caa')

fig=plt.figure(filename+'_Cba')

fig.clf()

ax=fig.add_subplot(3,1,1)
img1=ax.imshow(10*np.log10(np.abs(Cba)**2),extent=(np.min(delta3vals),np.max(delta3vals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title(' |C|^2 (dB scale)')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,2)
img1=ax.imshow((np.abs(Cba)**2),extent=(np.min(delta3vals),np.max(delta3vals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('|C|^2')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,1,3)
img1=ax.imshow((np.angle(Cba)),extent=(np.min(delta3vals),np.max(delta3vals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' C phase')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1)

fig.suptitle('Cba')

fig=plt.figure(filename+'_Cab')

fig.clf()

ax=fig.add_subplot(3,1,1)
img1=ax.imshow(10*np.log10(np.abs(Cab)**2),extent=(np.min(delta3vals),np.max(delta3vals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title(' |C|^2 (dB scale)')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,2)
img1=ax.imshow((np.abs(Cab**2)),extent=(np.min(delta3vals),np.max(delta3vals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('|C|^2')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,1,3)
img1=ax.imshow((np.angle(Cab)),extent=(np.min(delta3vals),np.max(delta3vals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' C phase')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1)

fig.suptitle('Cab')

fig=plt.figure(filename+'_Cbb')

fig.clf()

ax=fig.add_subplot(3,1,1)
img1=ax.imshow(10*np.log10(np.abs(Cbb)**2),extent=(np.min(delta3vals),np.max(delta3vals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title(' |C|^2 (dB scale)')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,2)
img1=ax.imshow((np.abs(Cbb)**2),extent=(np.min(delta3vals),np.max(delta3vals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('|C|')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,1,3)
img1=ax.imshow((np.angle(Cbb)),extent=(np.min(delta3vals),np.max(delta3vals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title('C phase')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1)

fig.suptitle('Cbb')

# fig=plt.figure(filename+'_Cbb')
#
# fig.clf()
#
# ax=fig.add_subplot(2,1,1)
# img1=ax.imshow(N_frac.real,extent=(np.min(delta3vals),np.max(delta3vals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# plt.title('N_frac real')
# plt.xlabel('delta_3')
# plt.ylabel('delta_2')
# fig.colorbar(img1)
#
# ax=fig.add_subplot(2,1,2)
# img1=ax.imshow(N_frac.imag,extent=(np.min(delta3vals),np.max(delta3vals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# plt.title('N_frac imag should be zero')
# plt.xlabel('delta_3')
# plt.ylabel('delta_2')
# fig.colorbar(img1)


plt.show()
