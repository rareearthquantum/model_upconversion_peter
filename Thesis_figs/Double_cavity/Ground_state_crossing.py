#importing python libs

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
from c_funs_ds4 import rho_broad_full

filename='Thesis_figs/Double_cavity/Ground_crossing2_big'
B_vals=np.linspace(0.204,0.212,81)
deltamvals=np.linspace(-70e6,70e6,81)*2*np.pi
# B_vals=np.linspace(0.207,0.209,15)
# deltamvals=np.linspace(-30e6,30e6,11)*2*np.pi
P_pump = 10*np.log10(1.74) #in dBm, 1.74 mW are going into the resonator
P_mu = -30.9 # in dBm
# P_mu = -39.9 # in dBm
# P_mu = 3-30 # in dBm

# P_mu = -47.9 # in dBm
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
# p['sd_delam']=2*pi*25e6/2.355
# p['sd_delam']=2*pi*14e6/2.355
# p['sd_delam']=2*pi*2e6
p['sd_delam']=2*pi*3e6

p['gammaoc']=2*pi*1.7e6
p['gammaoi']=2*pi*7.95e6#*1e-9
p['gammamc']=2*pi*0.0622e6
p['gammami']=2*pi*5.69e6

# p['gammamc']=2*pi*2.199e6 #from dB fit
# p['gammami']=2*pi*5.083e6

p['gammamc']=2*pi*2.063e6 #from linear fit and a super good cavity
p['gammami']=2*pi*0.01141e6

# p['gammamc']=2*pi*1.495e6 #from linear fit
# p['gammami']=2*pi*1.149e6


muBohr=927.4009994e-26; # Bohr magneton in J/T in J* T^-1
p['mu12'] = 4.3803*muBohr # transition dipole moment for microwave cavity (J T^-1)


p['go'] = 51.9  #optical coupling

p['No'] = 2.2e15 # number of atoms in the optical mode
p['Nm'] = 6e16  #toal number of atoms
p['Nm'] = 6e16*0.8  #toal number of atoms
# p['Nm'] = 2e16  #toal number of atoms
#p['No'] = p['Nm'] # number of atoms in the optical mode

#p['No'] = 1.3e15 # number of atoms in the optical mode
#p['Nm'] = 2e16  #toal number of atoms

p['gm'] = 1.04 #coupling between atoms and microwave field
# p['gm'] = 0.025 #coupling between atoms and microwave field

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

def fields_vec_fun(fields_vec,ainval,binval,deloval,delmval,p):
    aval=fields_vec[0]+1j*fields_vec[1]
    bval=fields_vec[2]+1j*fields_vec[3]
    Omega=p['Omega']
    rho=np.array(rho_broad_full(aval,bval,deloval,delmval,p))
    p['Omega']=0
    rho_only_b=np.array(rho_broad_full(0,bval,0,delmval,p))
    p['Omega']=Omega
    S12val=rho[1,0]*p['No']*p['gm']
    S13val=rho[2,0]*p['No']*p['go']
    S21_onlyb_val=rho_only_b[1,0]*(p['Nm']-p['No'])*p['gm']
    bval1=(-1j*S12val-1j*S21_onlyb_val+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*delmval)
    #bval1=(-1j*S12val+np.sqrt(p['gammamc'])*binval)/((p['gammamc']+p['gammami'])/2-1j*delmval)
    aval1=(-1j*S13val+np.sqrt(p['gammaoc'])*ainval)/((2*p['gammaoc']+p['gammaoi'])/2-1j*deloval)
    return [aval1.real,aval1.imag,bval1.real,bval1.imag]-fields_vec
def fields_abs_fun(fields_vec,ainval,binval,deloval,delmval,p):
    aval=fields_vec[0]+1j*fields_vec[1]
    bval=fields_vec[2]+1j*fields_vec[3]
    Omega=p['Omega']
    rho=np.array(rho_broad_full(aval,bval,deloval,delmval,p))
    p['Omega']=0
    rho_only_b=np.array(rho_broad_full(0,bval,0,delmval,p))
    p['Omega']=Omega
    S12val=rho[1,0]*p['No']*p['gm']
    S13val=rho[2,0]*p['No']*p['go']
    S21_onlyb_val=rho_only_b[1,0]*(p['Nm']-p['No'])*p['gm']
    bval1=(-1j*S12val-1j*S21_onlyb_val+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*delmval)
    #bval1=(-1j*S12val+np.sqrt(p['gammamc'])*binval)/((p['gammamc']+p['gammami'])/2-1j*delmval)
    aval1=(-1j*S13val+np.sqrt(p['gammaoc'])*ainval)/((2*p['gammaoc']+p['gammaoi'])/2-1j*deloval)
    out_vec=np.array([aval1.real,aval1.imag,bval1.real,bval1.imag]-fields_vec)
    out_abs=np.linalg.norm(out_vec)
    return out_abs

def find_fields(ainval,binval,deloval,delmval,start_guess_vec,p):
    #fields_found=scipy.optimize.root(fields_vec_fun,start_guess_vec,args=(ainval,binval,deloval,delmval,p),method='hybr',options={'xtol':1.49012e-08,'maxfev':5000})
    #fields_found=scipy.optimize.root(fields_vec_fun,start_guess_vec,args=(ainval,binval,deloval,delmval,p),method='broyden1',options={'maxiter':10000})
    #fields_found=scipy.optimize.root(fields_vec_fun,start_guess_vec,args=(ainval,binval,deloval,delmval,p),method='lm',options={})
    fields_found=scipy.optimize.root(fields_vec_fun,start_guess_vec,args=(ainval,binval,deloval,delmval,p),method='hybr')
    #minimize_bounds=scipy.optimize.Bounds([-binval,-binval,-binval,-binval],[binval,binval,binval,binval])
    #fields_found=scipy.optimize.minimize(fields_abs_fun,start_guess_vec,args=(ainval,binval,deloval,delmval,p),bounds=minimize_bounds)
    # if not fields_found.success:
    #     print(fields_found.message)
    #     minimize_bounds=scipy.optimize.Bounds([-binval/np.sqrt(p['gammamc']),-binval/np.sqrt(p['gammamc']),-binval/np.sqrt(p['gammamc']),-binval/np.sqrt(p['gammamc'])],[binval/np.sqrt(p['gammamc']),binval/np.sqrt(p['gammamc']),binval/np.sqrt(p['gammamc']),binval/np.sqrt(p['gammamc'])])
    #     fields_found=scipy.optimize.minimize(fields_abs_fun,start_guess_vec,args=(ainval,binval,deloval,delmval,p),bounds=minimize_bounds)
    if not fields_found.success:
        print(fields_found.message)
    fields_found=fields_found.x
    return fields_found[0]+1j*fields_found[1], fields_found[2]+1j*fields_found[3]

bvals=np.zeros((len(B_vals),len(deltamvals)),dtype=np.complex_)
avals=np.zeros((len(B_vals),len(deltamvals)),dtype=np.complex_)
rho_out=np.zeros((3,3,len(B_vals),len(deltamvals)),dtype=np.complex_)
avals=np.zeros((len(B_vals),len(deltamvals)),dtype=np.complex_)
calc_time=np.zeros((len(B_vals),len(deltamvals)))
binvals=np.zeros((len(B_vals),len(deltamvals)),dtype=np.complex_)
ainval=0
start_time=time.time()
deltaoval=0e8

for ii, B_val in enumerate(B_vals):
    p['mean_delam']=deltamac_from_I(B_val,p)
    p['nbath']=nbath_from_T(T,p,deltamacval=p['mean_delam'])
    p['mean_delao']=deltaao_from_I(B_val,p)*0
    p['sd_delao']=sd_delao_from_I(B_val,p)
    p['Omega']= Omega_cavity_from_PdBm_resonance(P_pump,p)

    for jj, deltamval in enumerate(deltamvals):
        time1=time.time()
        binval=bin_from_PdBm(P_mu,p,deltamval)
        binvals[ii,jj]=binval
        if jj==0 & ii==0:
            #start_guess_vec_bout=[binval.real,binval.imag]
            start_guess_complex_b=(np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*deltamval)
            start_guess_vec=[0,0,start_guess_complex_b.real,start_guess_complex_b.imag]
            #start_guess_vec_b2=[binval.real/p['gammamc'],binval.imag/p['gammamc'],0,0]

        elif ii==0:
            #start_guess_vec_bout=[boutvals[ii,jj-1].real,boutvals[ii,jj-1].imag]
            start_guess_vec=[avals[0,jj-1].real,avals[0,jj-1].imag,bvals[0,jj-1].real,bvals[0,jj-1].imag]
            #start_guess_vec_b2=[bvals2[ii,jj-1].real,bvals2[ii,jj-1].imag,avals[ii,jj-1].real,avals[ii,jj-1].imag]
        elif jj==0:
            start_guess_vec=[avals[ii-1,0].real,avals[ii-1,0].imag,bvals[ii-1,0].real,bvals[ii-1,0].imag]
        else:
            start_guess_complex_b=(bvals[ii,jj-1]+bvals[ii-1,jj])/2

            start_guess_complex_a=(avals[ii,jj-1]+avals[ii-1,jj])/2
            start_guess_vec=[start_guess_complex_a.real,start_guess_complex_a.imag,start_guess_complex_b.real,start_guess_complex_b.imag]

        start_guess_vec=[0,0,0,0]
        avals[ii,jj], bvals[ii,jj] = find_fields(ainval,binval,deltaoval,deltamval,start_guess_vec,p)
        rho_out[:,:,ii,jj]=rho_broad_full(avals[ii,jj],bvals[ii,jj],deltaoval,deltamval,p)
        calc_time[ii,jj]=time.time()-time1
    elapsed_time=time.time()-start_time
    print('    ' + str(ii) +', Time: ' + time.ctime() +', Elapsed: '+ str(elapsed_time))
    np.savez(filename,binvals=binvals,B_vals=B_vals,deltamvals=deltamvals,p=p,bvals=bvals,rho_out=rho_out,P_pump=P_pump,P_mu=P_mu,T=T,elapsed_time=elapsed_time,avals=avals,deltaoval=deltaoval,calc_time=calc_time)
print('    ===========================Complete===========================')

#binvals=1
boutvals=bvals*np.sqrt(p['gammamc'])
aoutvals=avals*np.sqrt(p['gammaoc'])

freqmu_vals=deltamvals/(2*np.pi)+p['freqmu']


fig=plt.figure(filename)

fig.clf()


ax=fig.add_subplot(3,2,1)
img1=ax.imshow(10*np.log10(np.abs(aoutvals/binvals)**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower')
plt.title(' |aout/bin| (dB scale)')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,2)
img1=ax.imshow(10*np.log10(np.abs(boutvals/binvals)**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower')
plt.title('|bout/bin|^2 (dB)')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,3)
img1=ax.imshow((np.abs(aoutvals/binvals)**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower')
plt.title('|aout/bin|')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,2,4)
img1=ax.imshow((np.abs(boutvals/binvals)**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower')
plt.title('|bout/bin|^2')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,5)
img1=ax.imshow((np.angle(aoutvals)),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' rho13 phase')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)
ax=fig.add_subplot(3,2,6)
img1=ax.imshow((np.angle(boutvals[:,:])),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title('bout phase')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)
fig.suptitle('P_mu = ' + str(P_mu)+ ' dBm')
#plt.show()

fig=plt.figure(figsize=(12,12))

fig.clf()
pltnum=1
for ii in range(3):
    for jj in range(3):
        ax=fig.add_subplot(3,3,pltnum)

        img1=ax.imshow((np.abs(rho_out[ii,jj,:,:])),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower')
        #ax.plot(mlines[:,0],delaovals1,alpha=0.5,color='red')
        #ax.set_xlim(min(delamvals),max(delamvals))
        plt.xlabel('delta_mu')
        plt.ylabel('B')
        #plt.title(str(ii)+', ' +str(jj))
        plt.title('$\\rho_{' +str(ii+1)+',' +str(jj+1)+'}$')
        #plt.margins(0.1)
        plt.colorbar(img1)
        pltnum=pltnum+1
#fig.tight_layout()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()
