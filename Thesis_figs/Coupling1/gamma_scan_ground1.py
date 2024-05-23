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

#import qutip

from matplotlib.colors import Normalize as Norm

import time
from Linear1.c_linear_ground5 import rho_broad_full

from Thesis_figs.Ground_params2 import p, sd_delao_from_B, deltaao_from_B
filename='Thesis_figs/Cavity_shifts/microwave_linear_shift1'
#I_vals=np.linspace(7.43,7.55,31)
delta2vals=np.linspace(-2000e6,2000e6,25)
deltamucvals=np.linspace(-250e6,250e6,25)


def Omega_from_PdBm(PdBm,p):
    mu0=4*np.pi*1e-7
    c=3e8
    hbar=1.05457e-34; # in J*s
    P=1e-3*10**(PdBm/10)
    Abeam=pi*p['Wbeam']**2/4
    Efield=np.sqrt(2*mu0*c*P/Abeam)
    Omega=p['d23']*Efield/hbar
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
    deltamacvals=(p['Gg'])*(B_mag)-p['freqmu']
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


def calc_efficiency(deloval,delmval,delocval,delmucval,p):

    rho0,rhoa,rhoac,rhob,rhobc=rho_broad_full(deloval,delmval, 0,p)
    rho0=-rho0

    if p['Nm']==p['No']:
        # rho0_NoPump=0
        # rhoa_NoPump=0
        # rhoac_NoPump=0
        # rhob_NoPump=0
        # rhobc_NoPump=0
        Sa13=rhoa[0,2]*p['No']*p['go']#+rhoa_NoPump[0,2]*(p['Nm']-p['No'])*p['go']
        #Sb13=rhob[0,2]*p['Nm']*p['go']
        Sb13=rhob[0,2]*p['No']*p['go']#+rhob_NoPump[0,2]*(p['Nm']-p['No'])*p['go']

        Sa12=rhoa[0,1]*p['No']*p['gm']#+rhoa_NoPump[0,1]*(p['Nm']-p['No'])*p['gm']
        #Sb12=rhob[0,1]*p['Nm']*p['gm']
        Sb12=rhob[0,1]*p['No']*p['gm']#+rhob_NoPump[0,1]*(p['Nm']-p['No'])*p['gm']
    else:
        Omega=p['Omega']
        p['Omega']=0
        rho0_NoPump,rhoa_NoPump,rhoac_NoPump,rhob_NoPump,rhobc_NoPump=rho_broad_full(deloval,delmval,0,p)
        p['Omega']=Omega
        rho0_NoPump=-rho0_NoPump
    # Nm_eff=np.real(p['No']*(rho0[0,0]-rho0[1,1])+(p['Nm']-p['No'])*(rho0_NoPump[0,0]-rho0_NoPump[1,1]))
    # print([rho0[0,0],rho0_NoPump[0,0]])
    # print([rho0[1,1],rho0_NoPump[1,1]])
    # print([rho0[2,2],rho0_NoPump[2,2]])
    # print('===')
    # delocval=p['No']*p['go']**2/p['mean_delao']
    # delmucval=Nm_eff*p['gm']**2/p['mean_delam']

        Sa13=rhoa[0,2]*p['No']*p['go']#+rhoa_NoPump[0,2]*(p['Nm']-p['No'])*p['go']
        #Sb13=rhob[0,2]*p['Nm']*p['go']
        Sb13=rhob[0,2]*p['No']*p['go']+rhob_NoPump[0,2]*(p['Nm']-p['No'])*p['go']

        Sa12=rhoa[0,1]*p['No']*p['gm']#+rhoa_NoPump[0,1]*(p['Nm']-p['No'])*p['gm']
        #Sb12=rhob[0,1]*p['Nm']*p['gm']
        Sb12=rhob[0,1]*p['No']*p['gm']+rhob_NoPump[0,1]*(p['Nm']-p['No'])*p['gm']


    Caa=p['gammaoc']*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2)/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    Cab=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    Cba=           -1j*np.sqrt(p['gammaoc'])*np.sqrt(p['gammamc'])*Sa12/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    Cbb=p['gammamc']*(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    return Caa, Cba, Cab, Cbb#, N_frac

filename='Thesis_figs/Coupling1/gamma_scan_ground3'

P_pump = 10*np.log10(10.74) #in dBm, 1.74 mW are going into the resonator
P_mu = -30 # in dBm
T=100e-3
ainval=0
start_time=time.time()
deltaocval=-10e6
deltamucval=-10e6
deloval=0
delmval=0
p['mean_delao']=-75e9#deltaao_from_B(0.212,0,p)
p['sd_delao']=sd_delao_from_B(0.212,p)
p['mean_delam']=-50e6#deltamac_from_B(0.212,p)
p['nbath']=nbath_from_T(T,p,deltamacval=p['mean_delam'])
p['Omega']= Omega_cavity_from_PdBm_resonance(P_pump,p)
Q=1e9
p['gammaoi']=2*pi*p['freq_pump']/Q
print('gammaoi = ' + str(p['gammaoi']))
N=1e16
# N=2.2e15
p['Nm']=N
p['No']=N
p['gammami']=650e3*2*pi*1e0

# deltaocval=p['go']**2*p['No']/p['mean_delao']
# deltamucval=p['gm']**2*p['Nm']/p['mean_delam']/(2*p['nbath']+1)
print(filename + ' Starting!')
gammamc_vals=np.linspace(0,1,11)*2*pi*10e6*3
gammaoc_vals=np.linspace(0,1,11)*2*pi*10e6*3

gammamc_vals=np.linspace(0,1,151)*2*pi*10e6*1*2 #N=2.2e15
gammaoc_vals=np.linspace(0,1,151)*2*pi*10e6

gammamc_vals=np.linspace(0,1,151)*2*pi*10e6*10 #N=1e16
gammaoc_vals=np.linspace(0,1,151)*2*pi*10e6*10/2

# gammamc_vals=np.linspace(0,1,10)*2*pi*10e15
# gammaoc_vals=np.linspace(0,1,10)*2*pi*10e6


bvals=np.zeros(((len(gammamc_vals),len(gammaoc_vals))),dtype=np.complex_)
avals=np.zeros(((len(gammamc_vals),len(gammaoc_vals))),dtype=np.complex_)
rho_out=np.zeros((3,3,len(gammamc_vals),len(gammaoc_vals)),dtype=np.complex_)
avals=np.zeros(((len(gammamc_vals),len(gammaoc_vals))),dtype=np.complex_)
calc_time=np.zeros(((len(gammamc_vals),len(gammaoc_vals))))
binvals=np.zeros(((len(gammamc_vals),len(gammaoc_vals))),dtype=np.complex_)
Caa=np.zeros(((len(gammamc_vals),len(gammaoc_vals))),dtype=np.complex_)
Cba=np.zeros(((len(gammamc_vals),len(gammaoc_vals))),dtype=np.complex_)
Cab=np.zeros(((len(gammamc_vals),len(gammaoc_vals))),dtype=np.complex_)
Cbb=np.zeros(((len(gammamc_vals),len(gammaoc_vals))),dtype=np.complex_)
N_frac=np.zeros(((len(gammamc_vals),len(gammaoc_vals))),dtype=np.complex_)
filename=filename+'N='+str(N/1e15)
for ii,gammamc_val in enumerate(gammamc_vals):
    p['gammamc']=gammamc_val
    for jj, gammaoc_val in enumerate(gammaoc_vals):
        p['gammaoc']=gammaoc_val
        p['Omega']= Omega_cavity_from_PdBm_resonance(P_pump,p)

        time1=time.time()


        Caa[ii,jj], Cba[ii,jj], Cab[ii,jj], Cbb[ii,jj]=calc_efficiency(deloval,delmval,deltaocval,deltamucval,p)

    elapsed_time=time.time()-start_time
    print('    ' +filename+ ': '+str(ii) +', Time: ' + time.ctime() +', Elapsed: '+ str(elapsed_time))
np.savez(filename,Caa=Caa,Cab=Cab,Cbb=Cbb,Cba=Cba,gammamc_vals=gammamc_vals,gammaoc_vals=gammaoc_vals,
            p=p,deloval=deloval,delmval=delmval,deltaocval=deltaocval,deltamucval=deltamucval,T=T,P_pump=P_pump)
print('    ===========================Complete===========================')

boutvals=bvals*np.sqrt(p['gammamc'])
aoutvals=avals*np.sqrt(p['gammaoc'])
fig=plt.figure(filename+'Cab')

fig.clf()

ax=fig.add_subplot(3,1,1)
img1=ax.imshow(10*np.log10(np.abs(Cab)**2),extent=(np.min(gammaoc_vals),np.max(gammaoc_vals),np.min(gammamc_vals),np.max(gammamc_vals)),aspect='auto',origin='lower')
#plt.plot(deltamucvals,np.sqrt(p['Nm'])*p['gm']/deltamucvals)
plt.xlabel('gammaoc')
plt.ylabel('gammamc')
fig.colorbar(img1)


ax=fig.add_subplot(3,1,2)
img1=ax.imshow((np.abs(Cab)**2),extent=(np.min(gammaoc_vals),np.max(gammaoc_vals),np.min(gammamc_vals),np.max(gammamc_vals)),aspect='auto',origin='lower')
# plt.title('|bout/bin|^2')
plt.xlabel('gammaoc')
plt.ylabel('gammamc')
fig.colorbar(img1)


ax=fig.add_subplot(3,1,3)
img1=ax.imshow((np.angle(Cab)),extent=(np.min(gammaoc_vals),np.max(gammaoc_vals),np.min(gammamc_vals),np.max(gammamc_vals)),aspect='auto',origin='lower',cmap='hsv')
# plt.ylabel('atom delta')
plt.xlabel('gammaoc')
plt.ylabel('gammamc')
fig.colorbar(img1)
#
# fig=plt.figure(filename+'Cba')
#
# fig.clf()
#
# ax=fig.add_subplot(3,1,1)
# img1=ax.imshow(10*np.log10(np.abs(Cba)**2),extent=(np.min(gammaoc_vals),np.max(gammaoc_vals),np.min(gammamc_vals),np.max(gammamc_vals)),aspect='auto',origin='lower')
# #plt.plot(deltamucvals,np.sqrt(p['Nm'])*p['gm']/deltamucvals)
# plt.xlabel('gammaoc')
# plt.ylabel('gammamc')
# fig.colorbar(img1)
#
#
# ax=fig.add_subplot(3,1,2)
# img1=ax.imshow((np.abs(Cba)**2),extent=(np.min(gammaoc_vals),np.max(gammaoc_vals),np.min(gammamc_vals),np.max(gammamc_vals)),aspect='auto',origin='lower')
# # plt.title('|bout/bin|^2')
# plt.xlabel('gammaoc')
# plt.ylabel('gammamc')
# fig.colorbar(img1)
#
#
# ax=fig.add_subplot(3,1,3)
# img1=ax.imshow((np.angle(Cba)),extent=(np.min(gammaoc_vals),np.max(gammaoc_vals),np.min(gammamc_vals),np.max(gammamc_vals)),aspect='auto',origin='lower',cmap='hsv')
# # plt.ylabel('atom delta')
# plt.xlabel('gammaoc')
# plt.ylabel('gammamc')
# fig.colorbar(img1)
#
# fig=plt.figure(filename+'Caa')
#
# fig.clf()
#
# ax=fig.add_subplot(3,1,1)
# img1=ax.imshow(10*np.log10(np.abs(Caa)**2),extent=(np.min(gammaoc_vals),np.max(gammaoc_vals),np.min(gammamc_vals),np.max(gammamc_vals)),aspect='auto',origin='lower')
# #plt.plot(deltamucvals,np.sqrt(p['Nm'])*p['gm']/deltamucvals)
# plt.xlabel('gammaoc')
# plt.ylabel('gammamc')
# fig.colorbar(img1)
#
#
# ax=fig.add_subplot(3,1,2)
# img1=ax.imshow((np.abs(Caa)**2),extent=(np.min(gammaoc_vals),np.max(gammaoc_vals),np.min(gammamc_vals),np.max(gammamc_vals)),aspect='auto',origin='lower')
# # plt.title('|bout/bin|^2')
# plt.xlabel('gammaoc')
# plt.ylabel('gammamc')
# fig.colorbar(img1)
#
#
# ax=fig.add_subplot(3,1,3)
# img1=ax.imshow((np.angle(Caa)),extent=(np.min(gammaoc_vals),np.max(gammaoc_vals),np.min(gammamc_vals),np.max(gammamc_vals)),aspect='auto',origin='lower',cmap='hsv')
# # plt.ylabel('atom delta')
# plt.xlabel('gammaoc')
# plt.ylabel('gammamc')
# fig.colorbar(img1)
#
#
# fig=plt.figure(filename+'Cbb')
#
# fig.clf()
#
# ax=fig.add_subplot(3,1,1)
# img1=ax.imshow(10*np.log10(np.abs(Cbb)**2),extent=(np.min(gammaoc_vals),np.max(gammaoc_vals),np.min(gammamc_vals),np.max(gammamc_vals)),aspect='auto',origin='lower')
# #plt.plot(deltamucvals,np.sqrt(p['Nm'])*p['gm']/deltamucvals)
# plt.xlabel('gammaoc')
# plt.ylabel('gammamc')
# fig.colorbar(img1)
#
#
# ax=fig.add_subplot(3,1,2)
# img1=ax.imshow((np.abs(Cbb)**2),extent=(np.min(gammaoc_vals),np.max(gammaoc_vals),np.min(gammamc_vals),np.max(gammamc_vals)),aspect='auto',origin='lower')
# # plt.title('|bout/bin|^2')
# plt.xlabel('gammaoc')
# plt.ylabel('gammamc')
# fig.colorbar(img1)
#
#
# ax=fig.add_subplot(3,1,3)
# img1=ax.imshow((np.angle(Cbb)),extent=(np.min(gammaoc_vals),np.max(gammaoc_vals),np.min(gammamc_vals),np.max(gammamc_vals)),aspect='auto',origin='lower',cmap='hsv')
# # plt.ylabel('atom delta')
# plt.xlabel('gammaoc')
# plt.ylabel('gammamc')
# fig.colorbar(img1)

plt.show()
