#importing python libs

# import sympy as sym
# sym.init_printing()

import numpy as np

import sys
sys.path.append('/home/peter/model_upconversion')
from math import pi, sqrt
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
# Q=2*pi*p['freq_pump']/p['gammaoi']
# print(Q)
#filename='Linear_figs1/data_temp1'

#I_vals=np.linspace(7.43,7.55,31)
delta2vals=np.linspace(-1000e6,1000e6,301)
delta3vals=np.linspace(-15e10,15e10,301)
# delta2vals=np.linspace(-250e6,000e6,11)[:-1]
# delta3vals=np.linspace(-45e9,0e10,11)[:-1]
# delta2vals=np.linspace(-10e7,-2.5e7,20)
# delta3vals=np.linspace(-2.8e10,-0.6e10,20)

#delta2vals=np.linspace(-300e6,000e6,11)[:-1]/1000
#delta3vals=np.linspace(-45e9,0e10,11)[:-1]/1000
# delta2vals=np.linspace(-1000e6,000e6,11)[:-1]/10*3
# delta3vals=np.linspace(-4e10,0e10,11)[:-1]
# delta2vals=np.linspace(-200e6,000e6,11)[:-1]*4
# delta3vals=np.linspace(-25e9,0e10,11)[:-1]*4

delta2vals=np.linspace(-300e6,300e6,301)
delta3vals=np.linspace(-5e10,5e10,301)

delta2vals=np.linspace(-300e6,300e6,301)
delta3vals=np.linspace(-2e10,2e10,301)


delta2vals=np.linspace(0,0.1e9,11)
delta3vals=np.linspace(0e10,9e9,11)

fmvals=np.linspace(5.16,5.26,11)*1e9
fovals=np.linspace(0,0.9,11)*1e9+195113e9
# delta2vals=np.linspace(-0.6e8,0,10)#[:-1]?
# delta3vals=np.linspace(-1.5e10,-1.0e10,10)#[:-1]
P_pump_mW=6.48
# P_pump_mW=1

P_pump = 10*np.log10(P_pump_mW) #in dBm, 1.74 mW are going into the resonator
T=4000e-3

p={}
p['freqmu']=5017e6#4733e6 #this is the microwave cavity frequency
laser_temp=38.6
p['freq_pump']=195113.30e9
# p['freq_pump'] = 195117.044e9 #pump frequency
#p['freqo']=p['freqmu']+p['freq_pump']
print(p['freqmu']+p['freq_pump'])

print(p['freq_pump'])

# p['Gg']=0.024085156244027e12
# p['Ge']=0.017976119414574e12
p['Gg']=0.023721e12
p['Ge']=0.017976119414574e12
p['f0_no_B']=195.1167943776907e12

p['d13'] = 2e-32*sqrt(1/3)
p['d23'] = 2e-32*sqrt(2/3)
p['gamma13'] = p['d13']**2/(p['d13']**2+p['d23']**2)*1/11e-3
p['gamma23'] = p['d23']**2/(p['d13']**2+p['d23']**2)*1/11e-3
p['gamma2d'] = 1e6
p['gamma3d'] = 1e6
p['gammamu'] = 1/11#*2*pi
# p['sd_delam']=2*pi*25e6/2.355
# p['sd_delam']=2*pi*14e6/2.355
# p['sd_delam']=2*pi*2e6
p['sd_delam']=2*pi*5e6
p['sd_delam']=2*pi*3e6

p['gammaoc']=2*pi*7.95e6
p['gammaoi']=2*pi*1.7e6#*1e-9
# p['gammamc']=2*pi*0.0622e6
# p['gammami']=2*pi*5.69e6

# p['gammamc']=2*pi*2.063e6 #from linear fit and a super good cavity
# p['gammami']=2*pi*0.01141e6

p['gammamc']=2*pi*70e3 #from linear fit
p['gammami']=2*pi*650e3


muBohr=927.4009994e-26; # Bohr magneton in J/T in J* T^-1
p['mu12'] = 4.3803*muBohr # transition dipole moment for microwave cavity (J T^-1)


p['go'] = 51.9  #optical coupling

p['No'] = 1.28e15 # number of atoms in the optical mode
p['Nm'] = 2.2e16  #toal number of atoms


#p['No'] = 1.3e15 # number of atoms in the optical mode
#p['Nm'] = 2e16  #toal number of atoms

p['gm'] = 1.04 #coupling between atoms and microwave field

p['Wbeam']=0.6e-3
p['Lsample']=12e-3 # the length of the sample, in m
p['Lcavity_vac'] = 49.5e-3 # length of the vacuum part of the optical Fabry Perot (m)
p['nYSO'] = 1.76 #refractive index of YSO

# we are dealing with transition 3
# From |g,-> to |e,->

def omegaao1_from_B(B_mag,p):
    return (p['f0_no_B']+(-p['Gg1']-p['Ge1'])/2*B_mag)*2*pi
def omegaao2_from_B(B_mag,p):
    return (p['f0_no_B']+(-p['Gg']+p['Ge'])/2*B_mag)*2*pi
def omegaao3_from_B(B_mag,p):
    return (p['f0_no_B']+(+p['Gg']-p['Ge'])/2*B_mag)*2*pi
def omegaao4_from_B(B_mag,p):
    return (p['f0_no_B']+(+p['Gg']+p['Ge'])/2*B_mag)*2*pi

def sd_delao1_from_B(B_mag,p):
    return 1e9*(0.148588212918272+0.308180352441052*B_mag)*pi*2
def sd_delao2_from_B(B_mag,p):
    #return 1e9*(0.096660692060221-0.336895927831593*B_mag)*pi*2
    return 1e9*(0.096660692060221+0.336895927831593*B_mag)*pi*2
def sd_delao3_from_B(B_mag,p):
    return 1e9*(0.391366413926165+0.137444967951503*B_mag)*pi*2
def sd_delao4_from_B(B_mag,p):
    return 1e9*(0.279915072366011+0.000852680155581*B_mag)*pi*2


def omegaao_from_B(B_mag,p): #transition 4, which is the transition for the output optical photons
    return (p['f0_no_B']+(+p['Gg']+p['Ge'])/2*B_mag)*2*pi

def sd_delao_from_B(B_mag,p):
    return sd_delao3_from_B(B_mag,p)
def deltaao_from_B(B_mag,deltamval,p):
    #return -(2*pi*(p['freqmu']+p['freq_pump'])-omegaa3o_from_B(B_mag,p))
    return omegaao4_from_B(B_mag,p)-(2*pi*p['freq_pump']+2*pi*p['freqmu']+deltamval)
# p['gammaoc']=0.35e8
# p['gammamc']=0.2e9
# delta2vals=np.linspace(-60e6,000e6,11)[:-1]
# delta3vals=np.linspace(-75e8,0e10,11)[:-1]
# p['gammamc']=0.8e9
# p['gammaoc']=0.7e10
filename='Thesis_paper/effic_scan'
filename=filename+'xavi_test1'
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

p['Omega']= Omega_cavity_from_PdBm_resonance(P_pump,p)

def calc_efficiency(deloval,delmval,delocval,delmucval,p):

    rho0,rhoa,rhoac,rhob,rhobc=rho_broad_full(deloval,delmval, 0,p)
    if p['Nm']==p['No']:
        rho0_NoPump=np.zeros((3,3))
        rhoa_NoPump=0
        rhoac_NoPump=0
        rhob_NoPump=np.zeros((3,3))
        rhobc_NoPump=0
    else:
        Omega=p['Omega']
        p['Omega']=0
        rho0_NoPump,rhoa_NoPump,rhoac_NoPump,rhob_NoPump,rhobc_NoPump=rho_broad_full(deloval,delmval,0,p)
        p['Omega']=Omega
    rho0=-rho0
    rho0_NoPump=-rho0_NoPump
    Nm_eff=np.real(p['No']*(rho0[0,0]-rho0[1,1])+(p['Nm']-p['No'])*(rho0_NoPump[0,0]-rho0_NoPump[1,1]))
    # print([rho0[0,0],rho0_NoPump[0,0]])
    # print([rho0[1,1],rho0_NoPump[1,1]])
    # print([rho0[2,2],rho0_NoPump[2,2]])
    # print('===')
    # delocval=p['No']*p['go']**2/p['mean_delao']
    # delmucval=Nm_eff*p['gm']**2/p['mean_delam']
    N_frac=rho0[0,0]-rho0[1,1]
    delmucval=delmucval*N_frac

    Sa13=rhoa[0,2]*p['No']*p['go']#+rhoa_NoPump[0,2]*(p['Nm']-p['No'])*p['go']
    #Sb13=rhob[0,2]*p['Nm']*p['go']
    Sb13=rhob[0,2]*p['No']*p['go']+rhob_NoPump[0,2]*(p['Nm']-p['No'])*p['go']

    Sa12=rhoa[0,1]*p['No']*p['gm']#+rhoa_NoPump[0,1]*(p['Nm']-p['No'])*p['gm']
    #Sb12=rhob[0,1]*p['Nm']*p['gm']
    Sb12=rhob[0,1]*p['No']*p['gm']+rhob_NoPump[0,1]*(p['Nm']-p['No'])*p['gm']
    # Sb13=np.conj(Sb13)
    # Sa13=np.conj(Sa13)
    # Sb12=np.conj(Sb12)
    # Sa12=np.conj(Sa12)

    Caa=p['gammaoc']*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+p['gammamc'])/2)/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+p['gammamc'])/2))
    Cab=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+p['gammamc'])/2))
    Cba=           -1j*np.sqrt(p['gammaoc'])*np.sqrt(p['gammamc'])*Sa12/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+p['gammamc'])/2))
    Cbb=p['gammamc']*(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc'])/2)/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+p['gammamc'])/2))
    return Caa, Cba, Cab, Cbb, N_frac



start_time=time.time()
deltaoval=0#10.e9
deltaocval=0
deltamval=0#10.e9
deltamcval=0
start_time=time.time()

# delta2val_start=-0.65e8
# delta3val_start=-1.15e10
# deltamuc_start=p['Nm']*p['gm']**2/delta2val_start
# deltaoc_start=p['No']*p['go']**2/delta3val_start
# x_init=[deltaoc_start*1e-10,deltamuc_start*1e-8,delta3val_start*1e-10,delta2val_start*1e-8,p['gammaoc']*1e-6,p['gammamc']*1e-6]
# deloval=0
# delmval=0
# delocval=x_init[0]*1e10
# delmucval=x_init[1]*1e8
# p['mean_delao']=x_init[2]*1e10
# p['mean_delam']=x_init[3]*1e8
# p['gammaoc']=x_init[4]*1e6
# p['nbath']=nbath_from_T(T,p)
# p['gammamc']=x_init[5]*1e6
#
#
#
# p['mean_delao']=delta3val_start
# p['mean_delam']=delta2val_start
# p['sd_delao']=sd_delao_from_B(0.2,p)
# p['nbath']=nbath_from_T(T,p)
#
# # Caa_test,Cba_test,Cab_test,Cbb_test,N_frac_test=calc_efficiency(deloval,delmval,delocval,delmucval,p)
# # print(calc_efficiency(deloval,delmval,delocval,delmucval,p))
# # print(np.abs(Cab_test)**2)
f12=5.186e9
f13=195112e9+6.8e9
# f13=s
fmvals=np.linspace(5.12,5.26,41)*1e9
fpumpvals=np.linspace(0,0.9,41)*1e9+195113e9
p['mean_delao']=deltaao_from_B(0.2,0,p)
p['sd_delao']=sd_delao_from_B(0.2,p)

Caa=np.zeros((len(fmvals),len(fpumpvals)),dtype=np.complex_)
Cba=np.zeros((len(fmvals),len(fpumpvals)),dtype=np.complex_)
Cab=np.zeros((len(fmvals),len(fpumpvals)),dtype=np.complex_)
Cbb=np.zeros((len(fmvals),len(fpumpvals)),dtype=np.complex_)
N_frac=np.zeros((len(fmvals),len(fpumpvals)),dtype=np.complex_)

calc_time=np.zeros((len(fmvals),len(fpumpvals)))


for ii,fm in enumerate(fmvals):
    p['mean_delam']=2*pi*f12-2*pi*fm
    deltamucval=0#p['Nm']*p['gm']**2/p['mean_delam']
    p['nbath']=nbath_from_T(T,p)
    for jj, fpump in enumerate(fpumpvals):
        p['freq_pump']=fpump

        p['mean_delao']=2*pi*f13-2*pi*(fm+fpump)
        deltaocval=2*pi*(fpump+fm-f13)#p['No']*p['go']**2/p['mean_delao']

        #time1=time.time()
        Caa[ii,jj], Cba[ii,jj], Cab[ii,jj], Cbb[ii,jj],N_frac[ii,jj]=calc_efficiency(deltaoval,deltamval,deltaocval,deltamcval,p)

    elapsed_time=time.time()-start_time
    print('    ' + str(ii) +', Time: ' + time.ctime() +', Elapsed: '+ str(elapsed_time))
    #np.savez(filename,binvals=binvals,delta2vals=delta2vals,p=p,bvals=bvals,deltamucvals=deltamucvals,rho_out=rho_out,P_pump=P_pump,P_mu=P_mu,T=T,elapsed_time=elapsed_time,avals=avals,deltaocval=deltaocval,calc_time=calc_time)
    #np.savez(filename,p=p,P_mu=P_mu,P_pump=P_pump,T=T,elapsed_time=elapsed_time,calc_time=calc_time,binvals=binvals,ainval=ainval,bvals=bvals,avals=avals,delta2vals=delta2vals,delta3vals=delta3vals)
    np.savez(filename,Caa=Caa, Cba=Cba,Cab=Cab,Cbb=Cbb,p=p,P_pump=P_pump,T=T,elapsed_time=elapsed_time,calc_time=calc_time,delta2vals=delta2vals,delta3vals=delta3vals)

print('    ===========================Complete===========================')


fig=plt.figure(filename+'_Caa')

fig.clf()

ax=fig.add_subplot(3,1,1)
img1=ax.imshow(10*np.log10(np.abs(Caa)**2),extent=(np.min(fpumpvals),np.max(fpumpvals),np.min(fmvals),np.max(fmvals)),aspect='auto',origin='lower')
plt.title(' |C|^2 (dB scale)')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,2)
img1=ax.imshow((np.abs(Caa)**2),extent=(np.min(fpumpvals),np.max(fpumpvals),np.min(fmvals),np.max(fmvals)),aspect='auto',origin='lower')
plt.title('|C|^2 ')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,1,3)
img1=ax.imshow((np.angle(Caa)),extent=(np.min(fpumpvals),np.max(fpumpvals),np.min(fmvals),np.max(fmvals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' C phase')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1)

fig.suptitle('Caa')
#
# fig=plt.figure(filename+'_Cba')
#
# fig.clf()
#
# ax=fig.add_subplot(3,1,1)
# img1=ax.imshow(10*np.log10(np.abs(Cba)**2),extent=(np.min(fpumpvals),np.max(fpumpvals),np.min(fmvals),np.max(fmvals)),aspect='auto',origin='lower')
# plt.title(' |C|^2 (dB scale)')
# plt.xlabel('delta_3')
# plt.ylabel('delta_2')
# fig.colorbar(img1)
#
# ax=fig.add_subplot(3,1,2)
# img1=ax.imshow((np.abs(Cba)**2),extent=(np.min(fpumpvals),np.max(fpumpvals),np.min(fmvals),np.max(fmvals)),aspect='auto',origin='lower')
# plt.title('|C|^2')
# plt.xlabel('delta_3')
# plt.ylabel('delta_2')
# fig.colorbar(img1, format='%.0e')
#
# ax=fig.add_subplot(3,1,3)
# img1=ax.imshow((np.angle(Cba)),extent=(np.min(fpumpvals),np.max(fpumpvals),np.min(fmvals),np.max(fmvals)),aspect='auto',origin='lower',cmap='hsv')
# plt.title(' C phase')
# plt.xlabel('delta_3')
# plt.ylabel('delta_2')
# fig.colorbar(img1)
#
# fig.suptitle('Cba')

fig=plt.figure(filename+'_Cab')

fig.clf()

ax=fig.add_subplot(3,1,1)
img1=ax.imshow(10*np.log10(np.abs(Cab)**2),extent=(np.min(fpumpvals),np.max(fpumpvals),np.min(fmvals),np.max(fmvals)),aspect='auto',origin='lower')
plt.title(' |C|^2 (dB scale)')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,2)
img1=ax.imshow((np.abs(Cab**2)),extent=(np.min(fpumpvals),np.max(fpumpvals),np.min(fmvals),np.max(fmvals)),aspect='auto',origin='lower')
plt.title('|C|^2')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,1,3)
img1=ax.imshow((np.angle(Cab)),extent=(np.min(fpumpvals),np.max(fpumpvals),np.min(fmvals),np.max(fmvals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' C phase')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1)

fig.suptitle('Cab')

fig=plt.figure(filename+'_Cbb')

fig.clf()

ax=fig.add_subplot(3,1,1)
img1=ax.imshow(10*np.log10(np.abs(Cbb)**2),extent=(np.min(fpumpvals),np.max(fpumpvals),np.min(fmvals),np.max(fmvals)),aspect='auto',origin='lower')
plt.title(' |C|^2 (dB scale)')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,2)
img1=ax.imshow((np.abs(Cbb)**2),extent=(np.min(fpumpvals),np.max(fpumpvals),np.min(fmvals),np.max(fmvals)),aspect='auto',origin='lower')
plt.title('|C|')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,1,3)
img1=ax.imshow((np.angle(Cbb)),extent=(np.min(fpumpvals),np.max(fpumpvals),np.min(fmvals),np.max(fmvals)),aspect='auto',origin='lower',cmap='hsv')
plt.title('C phase')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1)

fig.suptitle('Cbb')

fig=plt.figure(filename+'_Cbb')
#
# fig.clf()
#
# ax=fig.add_subplot(2,1,1)
# img1=ax.imshow(N_frac.real,extent=(np.min(fpumpvals),np.max(fpumpvals),np.min(fmvals),np.max(fmvals)),aspect='auto',origin='lower')
# plt.title('N_frac real')
# plt.xlabel('delta_3')
# plt.ylabel('delta_2')
# fig.colorbar(img1)
#
# ax=fig.add_subplot(2,1,2)
# img1=ax.imshow(N_frac.imag,extent=(np.min(fpumpvals),np.max(fpumpvals),np.min(fmvals),np.max(fmvals)),aspect='auto',origin='lower')
# plt.title('N_frac imag should be zero')
# plt.xlabel('delta_3')
# plt.ylabel('delta_2')
# fig.colorbar(img1)


plt.show()
