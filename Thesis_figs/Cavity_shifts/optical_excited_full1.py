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
#from c_funs_excited_ds2 import rho_broad_full
from c_funs_lobatto_excited1small import rho_broad_full
from Thesis_figs.Excited_params import p, sd_delao_from_B, deltaao_from_B


deltamvals=np.linspace(-25e6,25e6,11)*2*np.pi
filename='Thesis_figs/Cavity_shifts/optical_full_excited_shift1'
#I_vals=np.linspace(7.43,7.55,31)
delta3vals=np.linspace(-390e8,390e8,25)*2*np.pi
deltaocvals=np.linspace(-5e8,5e8,25)
delta3vals=np.linspace(-1,1,125)*2.5e11
deltaocvals=np.linspace(-5e8,5e8,125)
P_pump = 10*np.log10(1.74) #in dBm, 1.74 mW are going into the resonator
P_o = -105-30 # in dBm
P_pump = 10*np.log10(1.74) #in dBm, 1.74 mW are going into the resonator
P_mu = -30 # in dBm
T=150e-3
p['freqo']=p['freqmu']+p['freq_pump']

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

def fields_vec_fun(fields_vec,ainval,binval,deloval,delmval,delocval,delmucval,p):
    aval=fields_vec[0]+1j*fields_vec[1]
    bval=fields_vec[2]+1j*fields_vec[3]
    rho=np.array(rho_broad_full(aval,bval,deloval,delmval,p))
    if p['Nm']==p['No']:
        S21_onlyb_val=0
    else:
        Omega=p['Omega']
        p['Omega']=0
        rho_only_b=np.array(rho_broad_full(0,bval,0,delmval,p))
        p['Omega']=Omega
        S21_onlyb_val=rho_only_b[1,2]*(p['Nm']-p['No'])*p['gm']
    S12val=rho[1,2]*p['No']*p['gm']
    S13val=rho[0,2]*p['No']*p['go']
    bval1=(-1j*S12val-1j*S21_onlyb_val+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*(delmval-delmucval))
    #bval1=(-1j*S12val+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*(delmval-delmucval))
    aval1=(-1j*S13val+np.sqrt(p['gammaoc'])*ainval)/((2*p['gammaoc']+p['gammaoi'])/2-1j*(deloval-delocval))
    return [aval1.real,aval1.imag,bval1.real,bval1.imag]-fields_vec

# def fields_vec_fun_2(fields_vec,ainval,binval,deloval,delmval,delocval,delmucval,p):
#     aval=fields_vec[0]+1j*fields_vec[1]
#     bval=fields_vec[2]+1j*fields_vec[3]
#     rho=np.array(rho_broad_full_big(aval,bval,deloval,delmval,p))
#     if p['Nm']==p['No']:
#         S21_onlyb_val=0
#     else:
#         Omega=p['Omega']
#         p['Omega']=0
#         rho_only_b=np.array(rho_broad_full_big(0,bval,0,delmval,p))
#         p['Omega']=Omega
#         S21_onlyb_val=rho_only_b[1,2]*(p['Nm']-p['No'])*p['gm']
#     S12val=rho[1,2]*p['No']*p['gm']
#     S13val=rho[0,2]*p['No']*p['go']
#     bval1=(-1j*S12val-1j*S21_onlyb_val+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*(delmval-delmucval))
#     #bval1=(-1j*S12val+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*(delmval-delmucval))
#     aval1=(-1j*S13val+np.sqrt(p['gammaoc'])*ainval)/((2*p['gammaoc']+p['gammaoi'])/2-1j*(deloval-delocval))
#     return [aval1.real,aval1.imag,bval1.real,bval1.imag]-fields_vec
#
#
# def fields_vec_fun_3(fields_vec,ainval,binval,deloval,delmval,delocval,delmucval,p):
#     aval=fields_vec[0]+1j*fields_vec[1]
#     bval=fields_vec[2]+1j*fields_vec[3]
#     rho=np.array(rho_broad_full_bigds(aval,bval,deloval,delmval,p))
#     if p['Nm']==p['No']:
#         S21_onlyb_val=0
#     else:
#         Omega=p['Omega']
#         p['Omega']=0
#         rho_only_b=np.array(rho_broad_full_bigds(0,bval,0,delmval,p))
#         p['Omega']=Omega
#         S21_onlyb_val=rho_only_b[1,2]*(p['Nm']-p['No'])*p['gm']
#     S12val=rho[1,2]*p['No']*p['gm']
#     S13val=rho[0,2]*p['No']*p['go']
#     bval1=(-1j*S12val-1j*S21_onlyb_val+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*(delmval-delmucval))
#     #bval1=(-1j*S12val+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*(delmval-delmucval))
#     aval1=(-1j*S13val+np.sqrt(p['gammaoc'])*ainval)/((2*p['gammaoc']+p['gammaoi'])/2-1j*(deloval-delocval))
#     return [aval1.real,aval1.imag,bval1.real,bval1.imag]-fields_vec
#
# def fields_vec_fun_4(fields_vec,ainval,binval,deloval,delmval,delocval,delmucval,p):
#     aval=fields_vec[0]+1j*fields_vec[1]
#     bval=fields_vec[2]+1j*fields_vec[3]
#     rho=np.array(rho_broad_full_bigds2(aval,bval,deloval,delmval,p))
#     if p['Nm']==p['No']:
#         S21_onlyb_val=0
#     else:
#         Omega=p['Omega']
#         p['Omega']=0
#         rho_only_b=np.array(rho_broad_full_bigds2(0,bval,0,delmval,p))
#         p['Omega']=Omega
#         S21_onlyb_val=rho_only_b[1,2]*(p['Nm']-p['No'])*p['gm']
#     S12val=rho[1,2]*p['No']*p['gm']
#     S13val=rho[0,2]*p['No']*p['go']
#     bval1=(-1j*S12val-1j*S21_onlyb_val+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*(delmval-delmucval))
#     #bval1=(-1j*S12val+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*(delmval-delmucval))
#     aval1=(-1j*S13val+np.sqrt(p['gammaoc'])*ainval)/((2*p['gammaoc']+p['gammaoi'])/2-1j*(deloval-delocval))
#     return [aval1.real,aval1.imag,bval1.real,bval1.imag]-fields_vec
#
# def find_fields(ainval,binval,deloval,delmval,delocval,delmucval,start_guess_vec,p):
#
#     fields_found=scipy.optimize.root(fields_vec_fun,start_guess_vec,args=(ainval,binval,deloval,delmval,delocval,delmucval,p),method='hybr')
#
#     if fields_found.success:
#         rho_out=rho_broad_full_big(fields_found.x[0]+1j*fields_found.x[1], fields_found.x[2]+1j*fields_found.x[3],deloval,delmval,p)
#
#     if not fields_found.success:
#         print(filename)
#         print(fields_found.message)
#         print('Attempt 2')
#         fields_found=scipy.optimize.root(fields_vec_fun_2,start_guess_vec,args=(ainval,binval,deloval,delmval,delocval,delmucval,p),method='hybr')
#         if fields_found.success:
#             rho_out=rho_broad_full_big(fields_found.x[0]+1j*fields_found.x[1], fields_found.x[2]+1j*fields_found.x[3],deloval,delmval,p)
#
#     if not fields_found.success:
#         print(filename)
#         print(fields_found.message)
#         print('Attempt 3')
#         start_guess_vec=fields_found.x
#         fields_found=scipy.optimize.root(fields_vec_fun_3,start_guess_vec,args=(ainval,binval,deloval,delmval,delocval,delmucval,p),method='hybr')
#         if fields_found.success:
#             rho_out=rho_broad_full_bigds(fields_found.x[0]+1j*fields_found.x[1], fields_found.x[2]+1j*fields_found.x[3],deloval,delmval,p)
#
#     if not fields_found.success:
#         print(filename)
#         print(fields_found.message)
#         print('Attempt 4')
#         start_guess_vec=fields_found.x
#         fields_found=scipy.optimize.root(fields_vec_fun_4,start_guess_vec,args=(ainval,binval,deloval,delmval,delocval,delmucval,p),method='hybr',options={'maxfev':1000})
#         rho_out=rho_broad_full_bigds2(fields_found.x[0]+1j*fields_found.x[1], fields_found.x[2]+1j*fields_found.x[3],deloval,delmval,p)
#         if fields_found.success:
#             print(str(fields_found.nfev) +' evalulations of the big boy')
#     if not fields_found.success:
#         print(fields_found.message)
#         print('Oh well')
#
#     fields_found=fields_found.x
#     return fields_found[0]+1j*fields_found[1], fields_found[2]+1j*fields_found[3], rho_out

def find_fields_small(ainval,binval,deloval,delmval,delocval,delmucval,start_guess_vec,p):

    fields_found=scipy.optimize.root(fields_vec_fun,start_guess_vec,args=(ainval,binval,deloval,delmval,delocval,delmucval,p),method='hybr')
    rho_out=rho_broad_full(fields_found.x[0]+1j*fields_found.x[1], fields_found.x[2]+1j*fields_found.x[3],deloval,delmval,p)

    fields_found=fields_found.x
    return fields_found[0]+1j*fields_found[1], fields_found[2]+1j*fields_found[3], rho_out

bvals=np.zeros((len(delta3vals),len(deltaocvals)),dtype=np.complex_)
avals=np.zeros((len(delta3vals),len(deltaocvals)),dtype=np.complex_)
rho_out=np.zeros((3,3,len(delta3vals),len(deltaocvals)),dtype=np.complex_)
avals=np.zeros((len(delta3vals),len(deltaocvals)),dtype=np.complex_)
calc_time=np.zeros((len(delta3vals),len(deltaocvals)))
ainvals=np.zeros((len(delta3vals),len(deltaocvals)),dtype=np.complex_)
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
    p['No']=p['No_total']#/(np.exp(B_val*p['Gg']*6.63e-34/(1.38e-23*T))+1)
    p['Nm']=p['Nm_total']#/(np.exp(B_val*p['Gg']*6.63e-34/(1.38e-23*T))+1)
    p['mean_delao']=delta3val#deltamac_from_I(I_val,p)
    p['nbath']=nbath_from_T(T,p)
    p['Omega']= 0#Omega_cavity_from_PdBm_resonance(P_pump,p)

    for jj, deltaocval in enumerate(deltaocvals):
        time1=time.time()
        ainval=ain_from_PdBm(P_o,p,0)
        ainvals[ii,jj]=ainval
        start_guess_vec=[0,0,0,0]

        avals[ii,jj], bvals[ii,jj],rho_out[:,:,ii,jj]= find_fields_small(ainval,binval,deloval,delmval,deltaocval,deltamucval,start_guess_vec,p)
    elapsed_time=time.time()-start_time
    print('    ' +filename+': '+ str(ii) +', Time: ' + time.ctime() +', Elapsed: '+ str(elapsed_time))
    np.savez(filename,binval=binval,ainvals=ainvals,delta3vals=delta3vals,p=p,bvals=bvals,deltamucval=deltamucval,rho_out=rho_out,P_pump=P_pump,P_o=P_o,T=T,elapsed_time=elapsed_time,avals=avals,
    deltaocvals=deltaocvals,calc_time=calc_time,deloval=deloval,delmval=delmval,ii=ii,jj=jj)
print('    ===========================Complete===========================')

boutvals=bvals*np.sqrt(p['gammamc'])
aoutvals=avals*np.sqrt(p['gammaoc'])
delta3vals2=delta3vals

fig=plt.figure(filename)

fig.clf()

ax=fig.add_subplot(3,2,1)
img1=ax.imshow(10*np.log10(np.abs(aoutvals/ainvals)**2),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower')
plt.plot(p['No']*p['go']**2/delta3vals2,delta3vals2)

plt.title(' |aout/bin| (dB scale)')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,2)
img1=ax.imshow(10*np.log10(np.abs(boutvals/ainvals)**2),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower')
plt.plot(p['No']*p['go']**2/delta3vals2,delta3vals2)
plt.title('|bout/bin|^2 (dB)')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,3)
img1=ax.imshow((np.abs(aoutvals/ainvals)**2),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower')
plt.plot(p['No']*p['go']**2/delta3vals2,delta3vals2)
plt.title('|aout/bin|')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,2,4)
img1=ax.imshow((np.abs(boutvals/ainvals)**2),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower')
plt.plot(p['No']*p['go']**2/delta3vals2,delta3vals2)
plt.title('|bout/bin|^2')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

#matplotlib.cm.register_cmap(name='twilight',cmap=twilight)


ax=fig.add_subplot(3,2,5)
img1=ax.imshow((np.angle(aoutvals)),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' rho13 phase')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)
ax=fig.add_subplot(3,2,6)
img1=ax.imshow((np.angle(boutvals[:,:])),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title('bout phase')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)
fig.suptitle('P_o = ' + str(P_o)+ ' dBm')
plt.show()
