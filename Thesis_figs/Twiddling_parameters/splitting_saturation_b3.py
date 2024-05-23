#importing python libs

import sympy as sym
sym.init_printing()

import numpy as np
import sys
sys.path.append('/home/peter/model_upconversion')
from math import pi
import matplotlib.pyplot as plt
# from sympy import I, Matrix, symbols
# from sympy.physics.quantum import TensorProduct, Dagger
import scipy.optimize
# import scipy.integrate
# import scipy.constants as const

#import qutip

from matplotlib.colors import Normalize as Norm

import time
from c_funs_ds3 import rho_broad_full
from c_funs_ds3_big import rho_broad_full as rho_broad_full_big
from c_funs_ds3_bigds import rho_broad_full as rho_broad_full_bigds
from c_funs_ds4 import rho_broad_full as rho_broad_full_bigds2

from Thesis_figs.Ground_params2 import p, sd_delao_from_B, deltaao_from_B

def bin_from_PdBm(Pdbm,p,deltamval=0):
    P=1e-3*10**(Pdbm/10)
    hbar=1.05457e-34; # in J*s
    omega=2*np.pi*(p['freqmu'])+deltamval
    bin=np.sqrt(P/hbar/omega)
    return bin
# print(bin_from_PdBm(-55,p))
# print(bin_from_PdBm(-50,p))
# print(bin_from_PdBm(-45,p))
# print(bin_from_PdBm(-40,p))
# print(bin_from_PdBm(-35,p))

deltamvals=np.linspace(-50e6,50e6,13)*2*np.pi
deltamvals=np.linspace(-40e6,40e6,41)*2*np.pi

deltamvals=np.linspace(-40e6,40e6,181)*2*np.pi
deltamvals=np.linspace(-90e6,90e6,21)*2*np.pi
deltamvals=np.linspace(-35e6,35e6,21)*2*np.pi

P_pump = 10*np.log10(1.74) #in dBm, 1.74 mW are going into the resonator

T=670e-3
T=50e-3

# binvals=np.linspace(bin_from_PdBm(-55.9,p),bin_from_PdBm(-35.9,p),21)#*1e7#bin_from_PdBm(P_mu,p)
# binvals=np.linspace(0,1,15)*7.5e8#*1e7#bin_from_PdBm(P_mu,p)
# binvals=np.linspace(0,1,15)*2.5e8#*1e7#bin_from_PdBm(P_mu,p)
# binvals=np.linspace(0,2,11)*1e8#*1e7#bin_from_PdBm(P_mu,p)
# binvals=binvals[0:]
# binvals=np.linspace(bin_from_PdBm(-55.9,p),bin_from_PdBm(-35.9,p),21)#*1e7#bin_from_PdBm(P_mu,p)
# binvals=np.linspace(bin_from_PdBm(-55.9,p),bin_from_PdBm(-30.9,p),15)#*1e7#bin_from_PdBm(P_mu,p)
#
# # binvals=np.linspace(0.4,0.65,5)*1e8#*1e7#bin_from_PdBm(P_mu,p)
# binvals=np.linspace(0,1.5,11)*1e8#*1e7#bin_from_PdBm(P_mu,p)
# binvals=binvals[1:]
# binvals=np.linspace(bin_from_PdBm(-47.9,p),bin_from_PdBm(-39.9,p),3)#*1e7#bin_from_PdBm(P_mu,p)
deltamvals=np.linspace(-90e6,90e6,81)*2*np.pi
binvals=np.linspace(bin_from_PdBm(-47.9,p),bin_from_PdBm(-30.9,p),21)#*1e7#bin_from_PdBm(P_mu,p)
binvals=np.linspace(bin_from_PdBm(-60,p),bin_from_PdBm(-45,p),21)#*1e7#bin_from_PdBm(P_mu,p)

deltaoval=1e8*0
deltamvals=np.linspace(-90e6,90e6,41)*2*np.pi
deltamvals=np.linspace(-15e6,15e6,41)*2*np.pi
deltamvals=np.linspace(-30e6,30e6,101)*2*np.pi
deltamvals=np.linspace(-100e6,100e6,101)*2*np.pi

binvals=np.linspace(bin_from_PdBm(-60,p),bin_from_PdBm(-25.9,p),21)#*1e7#bin_from_PdBm(P_mu,p)
# binvals=np.linspace(bin_from_PdBm(-60,p),bin_from_PdBm(-45,p),21)#*1e7#bin_from_PdBm(P_mu,p)



B_val=0.2115
B_val=0.21165

ainval=0
# p['Nm']=p['No']

filename='Thesis_figs/Twiddling_parameters/splitting_saturation_b3_3_B='+str(B_val)+',T='+str(T)+'diffN'

def Omega_from_PdBm(PdBm,p):
    mu0=4*pi*1e-7
    c=3e8
    hbar=1.05457e-34; # in J*s
    P=1e-3*10**(PdBm/10)
    Abeam=pi*p['Wbeam']**2/4
    Efield=np.sqrt(2*mu0*c*P/Abeam)
    Omega=p['d23']*Efield/hbar
    return Omega

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


print('Omega = ' + str(p['Omega']))



def fields_vec_fun(fields_vec,ainval,binval,deloval,delmval,p):
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
        S21_onlyb_val=rho_only_b[1,0]*(p['Nm']-p['No'])*p['gm']
    S12val=rho[1,0]*p['No']*p['gm']
    S13val=rho[2,0]*p['No']*p['go']
    bval1=(-1j*S12val-1j*S21_onlyb_val+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*delmval)
    #bval1=(-1j*S12val+np.sqrt(p['gammamc'])*binval)/((p['gammamc']+p['gammami'])/2-1j*delmval)
    aval1=(-1j*S13val+np.sqrt(p['gammaoc'])*ainval)/((2*p['gammaoc']+p['gammaoi'])/2-1j*deloval)
    return [aval1.real,aval1.imag,bval1.real,bval1.imag]-fields_vec

def fields_vec_fun_2(fields_vec,ainval,binval,deloval,delmval,p):
    aval=fields_vec[0]+1j*fields_vec[1]
    bval=fields_vec[2]+1j*fields_vec[3]
    rho=np.array(rho_broad_full_big(aval,bval,deloval,delmval,p))
    if p['Nm']==p['No']:
        S21_onlyb_val=0
    else:
        Omega=p['Omega']
        p['Omega']=0
        rho_only_b=np.array(rho_broad_full_big(0,bval,0,delmval,p))
        p['Omega']=Omega
        S21_onlyb_val=rho_only_b[1,0]*(p['Nm']-p['No'])*p['gm']
    S12val=rho[1,0]*p['No']*p['gm']
    S13val=rho[2,0]*p['No']*p['go']
    bval1=(-1j*S12val-1j*S21_onlyb_val+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*delmval)
    #bval1=(-1j*S12val+np.sqrt(p['gammamc'])*binval)/((p['gammamc']+p['gammami'])/2-1j*delmval)
    aval1=(-1j*S13val+np.sqrt(p['gammaoc'])*ainval)/((2*p['gammaoc']+p['gammaoi'])/2-1j*deloval)
    return [aval1.real,aval1.imag,bval1.real,bval1.imag]-fields_vec


def fields_vec_fun_3(fields_vec,ainval,binval,deloval,delmval,p):
    aval=fields_vec[0]+1j*fields_vec[1]
    bval=fields_vec[2]+1j*fields_vec[3]
    rho=np.array(rho_broad_full_bigds(aval,bval,deloval,delmval,p))
    if p['Nm']==p['No']:
        S21_onlyb_val=0
    else:
        Omega=p['Omega']
        p['Omega']=0
        rho_only_b=np.array(rho_broad_full_bigds(0,bval,0,delmval,p))
        p['Omega']=Omega
        S21_onlyb_val=rho_only_b[1,0]*(p['Nm']-p['No'])*p['gm']
    S12val=rho[1,0]*p['No']*p['gm']
    S13val=rho[2,0]*p['No']*p['go']
    bval1=(-1j*S12val-1j*S21_onlyb_val+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*delmval)
    #bval1=(-1j*S12val+np.sqrt(p['gammamc'])*binval)/((p['gammamc']+p['gammami'])/2-1j*delmval)
    aval1=(-1j*S13val+np.sqrt(p['gammaoc'])*ainval)/((2*p['gammaoc']+p['gammaoi'])/2-1j*deloval)
    return [aval1.real,aval1.imag,bval1.real,bval1.imag]-fields_vec

def fields_vec_fun_4(fields_vec,ainval,binval,deloval,delmval,p):
    aval=fields_vec[0]+1j*fields_vec[1]
    bval=fields_vec[2]+1j*fields_vec[3]
    rho=np.array(rho_broad_full_bigds2(aval,bval,deloval,delmval,p))
    if p['Nm']==p['No']:
        S21_onlyb_val=0
    else:
        Omega=p['Omega']
        p['Omega']=0
        rho_only_b=np.array(rho_broad_full_bigds2(0,bval,0,delmval,p))
        p['Omega']=Omega
        S21_onlyb_val=rho_only_b[1,0]*(p['Nm']-p['No'])*p['gm']
    S12val=rho[1,0]*p['No']*p['gm']
    S13val=rho[2,0]*p['No']*p['go']
    bval1=(-1j*S12val-1j*S21_onlyb_val+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*delmval)
    #bval1=(-1j*S12val+np.sqrt(p['gammamc'])*binval)/((p['gammamc']+p['gammami'])/2-1j*delmval)
    aval1=(-1j*S13val+np.sqrt(p['gammaoc'])*ainval)/((2*p['gammaoc']+p['gammaoi'])/2-1j*deloval)
    return [aval1.real,aval1.imag,bval1.real,bval1.imag]-fields_vec

def find_fields(ainval,binval,deloval,delmval,start_guess_vec,p):

    fields_found=scipy.optimize.root(fields_vec_fun,start_guess_vec,args=(ainval,binval,deloval,delmval,p),method='hybr')

    if fields_found.success:
        rho_out=rho_broad_full_big(fields_found.x[0]+1j*fields_found.x[1], fields_found.x[2]+1j*fields_found.x[3],deloval,delmval,p)

    if not fields_found.success:
        print(filename)
        print(fields_found.message)
        print('Attempt 2')
        fields_found=scipy.optimize.root(fields_vec_fun_2,start_guess_vec,args=(ainval,binval,deloval,delmval,p),method='hybr')
        if fields_found.success:
            rho_out=rho_broad_full_big(fields_found.x[0]+1j*fields_found.x[1], fields_found.x[2]+1j*fields_found.x[3],deloval,delmval,p)

    if not fields_found.success:
        print(filename)
        print(fields_found.message)
        print('Attempt 3')
        start_guess_vec=fields_found.x
        fields_found=scipy.optimize.root(fields_vec_fun_3,start_guess_vec,args=(ainval,binval,deloval,delmval,p),method='hybr')
        if fields_found.success:
            rho_out=rho_broad_full_bigds(fields_found.x[0]+1j*fields_found.x[1], fields_found.x[2]+1j*fields_found.x[3],deloval,delmval,p)

    if not fields_found.success:
        print(filename)
        print(fields_found.message)
        print('Attempt 4')
        start_guess_vec=fields_found.x
        fields_found=scipy.optimize.root(fields_vec_fun_4,start_guess_vec,args=(ainval,binval,deloval,delmval,p),method='hybr',options={'maxfev':500})
        rho_out=rho_broad_full_bigds2(fields_found.x[0]+1j*fields_found.x[1], fields_found.x[2]+1j*fields_found.x[3],deloval,delmval,p)
        if fields_found.success:
            print(str(fields_found.nfev) +' evalulations of the big boy')
    if not fields_found.success:
        print(fields_found.message)
        print('Oh well')

    fields_found=fields_found.x
    return fields_found[0]+1j*fields_found[1], fields_found[2]+1j*fields_found[3], rho_out

def find_fields_small(ainval,binval,deloval,delmval,start_guess_vec,p):

    fields_found=scipy.optimize.root(fields_vec_fun,start_guess_vec,args=(ainval,binval,deloval,delmval,p),method='hybr')
    rho_out=rho_broad_full_big(fields_found.x[0]+1j*fields_found.x[1], fields_found.x[2]+1j*fields_found.x[3],deloval,delmval,p)

    fields_found=fields_found.x
    return fields_found[0]+1j*fields_found[1], fields_found[2]+1j*fields_found[3], rho_out



bvals=np.zeros((len(binvals),len(deltamvals)),dtype=np.complex_)
avals=np.zeros((len(binvals),len(deltamvals)),dtype=np.complex_)
rho_out=np.zeros((3,3,len(binvals),len(deltamvals)),dtype=np.complex_)
avals=np.zeros((len(binvals),len(deltamvals)),dtype=np.complex_)
binvals_save=np.zeros((len(binvals),len(deltamvals)),dtype=np.complex_)
calc_time=np.zeros((len(binvals),len(deltamvals)))

start_time=time.time()


for ii, binval in enumerate(binvals):
    p['mean_delam']=deltamac_from_B(B_val,p)
    p['nbath']=nbath_from_T(T,p,deltamacval=p['mean_delam'])
    # p['mean_delao']=deltaao_from_B(B_val,p)*0
    p['sd_delao']=sd_delao_from_B(B_val,p)
    p['Omega']= Omega_cavity_from_PdBm_resonance(P_pump,p)



    for jj, deltamval in enumerate(deltamvals):

        p['mean_delao']=deltaao_from_B(B_val,deltamval,p)
        time1=time.time()
        binvals_save[ii,jj]=binval
        # if jj==0 & ii==0:
        #     #start_guess_vec_bout=[binval.real,binval.imag]
        #     start_guess_complex_b=(np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*deltamval)
        #     start_guess_vec=[0,0,start_guess_complex_b.real,start_guess_complex_b.imag]
        #     #start_guess_vec_b2=[binval.real/p['gammamc'],binval.imag/p['gammamc'],0,0]
        #
        # elif ii==0:
        #     #start_guess_vec_bout=[boutvals[ii,jj-1].real,boutvals[ii,jj-1].imag]
        #     start_guess_vec=[avals[0,jj-1].real,avals[0,jj-1].imag,bvals[0,jj-1].real,bvals[0,jj-1].imag]
        #     #start_guess_vec_b2=[bvals2[ii,jj-1].real,bvals2[ii,jj-1].imag,avals[ii,jj-1].real,avals[ii,jj-1].imag]
        # elif jj==0:
        #     start_guess_vec=[avals[ii-1,0].real,avals[ii-1,0].imag,bvals[ii-1,0].real,bvals[ii-1,0].imag]
        # else:
        #     start_guess_complex_b=(bvals[ii,jj-1]+bvals[ii-1,jj])/2
        #
        #     start_guess_complex_a=(avals[ii,jj-1]+avals[ii-1,jj])/2
        #     start_guess_vec=[start_guess_complex_a.real,start_guess_complex_a.imag,start_guess_complex_b.real,start_guess_complex_b.imag]

        start_guess_vec=[0,0,0,0]
        start_guess_vec=[0,0,0,0]
        # if jj==0:
        #     start_guess_vec=[0,0,0,0]
        # else:
        #     start_guess_vec=[avals[ii,jj-1].real,avals[ii,jj-1].imag,bvals[ii,jj-1].real,bvals[ii,jj-1].imag]
        if ii==0:
            start_guess_vec=[0,0,0,0]
        else:
            start_guess_vec=[avals[ii-1,jj].real,avals[ii-1,jj].imag,bvals[ii-1,jj].real,bvals[ii-1,jj].imag]

        start_guess_vec=[0,0,0,0]

        avals[ii,jj], bvals[ii,jj],rho_out[:,:,ii,jj]= find_fields(ainval,binval,deltaoval,deltamval,start_guess_vec,p)
        #rho_out[:,:,ii,jj]=rho_broad_full(avals[ii,jj],bvals[ii,jj],deltaoval,deltamval,p)
        calc_time[ii,jj]=time.time()-time1
    elapsed_time=time.time()-start_time
    print(' ' +filename +': '+ str(ii) +', Time: ' + time.ctime() +', Elapsed: '+ str(elapsed_time))
    np.savez(filename,binvals_save=binvals_save,binvals=binvals,B_val=B_val,deltamvals=deltamvals,p=p,bvals=bvals,rho_out=rho_out,P_pump=P_pump,T=T,elapsed_time=elapsed_time,avals=avals,deltaoval=deltaoval,calc_time=calc_time)
print('    ===========================Complete===========================')

#binvals=1
boutvals=bvals*np.sqrt(p['gammamc'])
aoutvals=avals*np.sqrt(p['gammaoc'])
binvals1=binvals
binvals=binvals_save
freqmu_vals=deltamvals/(2*np.pi)#+p['freqmu']

# print(avals)
fig=plt.figure(filename)

fig.clf()


ax=fig.add_subplot(3,2,1)
img1=ax.imshow(10*np.log10(np.abs(aoutvals/binvals)**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(binvals),np.max(binvals)),aspect='auto',origin='lower')
plt.title(' |aout/bin| (dB scale)')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,2)
img1=ax.imshow(10*np.log10(np.abs(boutvals/binvals)**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(binvals),np.max(binvals)),aspect='auto',origin='lower')
plt.title('|bout/bin|^2 (dB)')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,3)
img1=ax.imshow((np.abs(aoutvals/binvals)**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(binvals),np.max(binvals)),aspect='auto',origin='lower')
plt.title('|aout/bin|')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,2,4)
img1=ax.imshow((np.abs(boutvals/binvals)**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(binvals),np.max(binvals)),aspect='auto',origin='lower')
plt.title('|bout/bin|^2')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,5)
img1=ax.imshow((np.angle(aoutvals)),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(binvals),np.max(binvals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' rho13 phase')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)
ax=fig.add_subplot(3,2,6)
img1=ax.imshow((np.angle(boutvals[:,:])),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(binvals),np.max(binvals)),aspect='auto',origin='lower',cmap='hsv')
plt.title('bout phase')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

#plt.show()

fig=plt.figure(filename+ '_rho',figsize=(12,12))

fig.clf()
pltnum=1
for ii in range(3):
    for jj in range(3):
        ax=fig.add_subplot(3,3,pltnum)

        img1=ax.imshow((np.abs(rho_out[ii,jj,:,:])),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(binvals),np.max(binvals)),aspect='auto',origin='lower')
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


fig=plt.figure()

fig.clf()


ax=fig.add_subplot(1,2,1)
plt.plot(deltamvals,10*np.log10(np.abs(aoutvals/binvals)**2).T)
plt.title(' |aout/bin| (dB scale)')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(1,2,2)
plt.plot(deltamvals,10*np.log10(np.abs(boutvals/binvals)**2).T)
plt.title('|bout/bin|^2 (dB)')
plt.xlabel('delta_mu')


fig=plt.figure()

fig.clf()

ax=fig.add_subplot(1,2,1)
plt.plot(deltamvals,(np.abs(aoutvals/binvals)**2).T)
plt.title(' |aout/bin|')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(1,2,2)
plt.plot(deltamvals,(np.abs(boutvals/binvals)**2).T)
plt.title('|bout/bin|^2')
plt.xlabel('delta_mu')

plt.show()
