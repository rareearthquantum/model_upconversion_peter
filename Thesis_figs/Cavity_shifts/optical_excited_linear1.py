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
from Linear_excited1.c_linear_excited1 import rho_broad_full
from Thesis_figs.Excited_params import p, sd_delao_from_B, deltaao_from_B

deltamvals=np.linspace(-25e6,25e6,11)*2*np.pi
filename='Thesis_figs/Cavity_shifts/optical_linear_excited_shift2'
#I_vals=np.linspace(7.43,7.55,31)
delta3vals=np.linspace(-390e8,390e8,15)*2*np.pi
deltaocvals=np.linspace(-5e8,5e8,15)
delta3vals=np.linspace(-1,1,125)*2.5e11
deltaocvals=np.linspace(-5e8,5e8,125)
P_pump = 10*np.log10(1.74) #in dBm, 1.74 mW are going into the resonator
P_o = -105-30 # in dBm
P_pump = 10*np.log10(1.74) #in dBm, 1.74 mW are going into the resonator
P_mu = -30 # in dBm
T=150e-3
p['freqo']=p['freqmu']+p['freq_pump']
p['No']=p['No_total']
p['Nm']=p['No']

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
def ain_from_PdBm(Pdbm,p,deltaoval=0):
    P=1e-3*10**(Pdbm/10)
    hbar=1.05457e-34; # in J*s
    omega=2*np.pi*(p['freqo'])+deltaoval
    ain=np.sqrt(P/hbar/omega)
    return ain
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

print('Omega = ' + str(p['Omega']))
print('bin   = ' + str(bin_from_PdBm(P_mu,p)))

def calc_efficiency(deloval,delmval,delocval,delmucval,p):

    rho0,rhoa,rhoac,rhob,rhobc=rho_broad_full(deloval,delmval, 0,p)
    if p['Nm']==p['No']:
        rho0_NoPump=np.zeros((3,3))
        rhoa_NoPump=np.zeros((3,3))
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

    Sa13=rhoa[0,2]*p['No']*p['go']#+rhoa_NoPump[0,2]*(p['Nm']-p['No'])*p['go']
    #Sb13=rhob[0,2]*p['Nm']*p['go']
    Sb13=rhob[0,2]*p['No']*p['go']+rhob_NoPump[0,2]*(p['Nm']-p['No'])*p['go']

    Sa12=rhoa[1,2]*p['No']*p['gm']#+rhoa_NoPump[1,2]*(p['Nm']-p['No'])*p['gm']
    #Sb12=rhob[1,2]*p['Nm']*p['gm']
    Sb12=rhob[1,2]*p['No']*p['gm']+rhob_NoPump[1,2]*(p['Nm']-p['No'])*p['gm']


    Caa=p['gammaoc']*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2)/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    Cba=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    Cab=           -1j*np.sqrt(p['gammaoc'])*np.sqrt(p['gammamc'])*Sa12/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    Cbb=p['gammamc']*(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    return Caa, Cba, Cab, Cbb#, N_frac


bvals=np.zeros((len(delta3vals),len(deltaocvals)),dtype=np.complex_)
avals=np.zeros((len(delta3vals),len(deltaocvals)),dtype=np.complex_)
rho_out=np.zeros((3,3,len(delta3vals),len(deltaocvals)),dtype=np.complex_)
avals=np.zeros((len(delta3vals),len(deltaocvals)),dtype=np.complex_)
calc_time=np.zeros((len(delta3vals),len(deltaocvals)))
ainvals=np.zeros((len(delta3vals),len(deltaocvals)),dtype=np.complex_)
Caa=np.zeros((len(delta3vals),len(deltaocvals)),dtype=np.complex_)
Cba=np.zeros((len(delta3vals),len(deltaocvals)),dtype=np.complex_)
Cab=np.zeros((len(delta3vals),len(deltaocvals)),dtype=np.complex_)
Cbb=np.zeros((len(delta3vals),len(deltaocvals)),dtype=np.complex_)
N_frac=np.zeros((len(delta3vals),len(deltaocvals)),dtype=np.complex_)
binval=0
start_time=time.time()
deltaocval=0
deloval=0
delmval=0
deltamucval=0
p['mean_delam']=0#deltaao_from_I(0,p)

p['mean_delao']=deltaao_from_B(0.212,0,p)
p['sd_delao']=sd_delao_from_B(0.212,p)
print(filename + ' Starting!')
for ii,delta3val in enumerate(delta3vals):
    # p['No']=p['No_total']#/(np.exp((p['freqmu']+delta2val/(2*pi))*6.63e-34/(1.38e-23*T))+1)
    # p['Nm']=p['Nm_total']#/(np.exp((p['freqmu']+delta2val/(2*pi))*6.63e-34/(1.38e-23*T))+1)
    p['mean_delao']=delta3val#deltamac_from_I(I_val,p)
    p['nbath']=nbath_from_T(T,p)
    p['Omega']= 0#Omega_cavity_from_PdBm_resonance(P_pump,p)

    for jj, deltaocval in enumerate(deltaocvals):
        time1=time.time()
        ainval=ain_from_PdBm(P_o,p,0)
        ainvals[ii,jj]=ainval
        start_guess_vec=[0,0,0,0]

        Caa[ii,jj], Cba[ii,jj], Cab[ii,jj], Cbb[ii,jj]=calc_efficiency(deloval,delmval,deltaocval,deltamucval,p)
    elapsed_time=time.time()-start_time
    print('    ' +filename+': '+ str(ii) +', Time: ' + time.ctime() +', Elapsed: '+ str(elapsed_time))
    np.savez(filename,binval=binval,ainvals=ainvals,delta3vals=delta3vals,p=p,bvals=bvals,deltamucval=deltamucval,rho_out=rho_out,P_pump=P_pump,P_o=P_o,T=T,elapsed_time=elapsed_time,avals=avals,
    deltaocvals=deltaocvals,calc_time=calc_time,deloval=deloval,delmval=delmval,ii=ii,jj=jj,Caa=Caa, Cba=Cba,Cab=Cab,Cbb=Cbb)
print('    ===========================Complete===========================')

boutvals=bvals*np.sqrt(p['gammamc'])
aoutvals=avals*np.sqrt(p['gammaoc'])
delta3vals2=delta3vals

fig=plt.figure(filename)

fig.clf()

ax=fig.add_subplot(3,2,1)
img1=ax.imshow(10*np.log10(np.abs(Caa)**2),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower')
plt.plot(p['No']*p['go']**2/delta3vals2,delta3vals2)

plt.title(' |aout/bin| (dB scale)')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,2)
img1=ax.imshow(10*np.log10(np.abs(Cab)**2),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower')
plt.plot(p['No']*p['go']**2/delta3vals2,delta3vals2)
plt.title('|bout/bin|^2 (dB)')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,3)
img1=ax.imshow((np.abs(Caa)),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower')
plt.plot(p['No']*p['go']**2/delta3vals2,delta3vals2)
plt.title('|aout/bin|')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,2,4)
img1=ax.imshow((np.abs(Cab)**2),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower')
plt.plot(p['No']*p['go']**2/delta3vals2,delta3vals2)
plt.title('|bout/bin|^2')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

#matplotlib.cm.register_cmap(name='twilight',cmap=twilight)


ax=fig.add_subplot(3,2,5)
img1=ax.imshow((np.angle(Caa)),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' rho13 phase')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)
ax=fig.add_subplot(3,2,6)
img1=ax.imshow((np.angle(Cab)),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title('bout phase')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)
fig.suptitle('P_o = ' + str(P_o)+ ' dBm')
plt.show()
