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
#from c_funs_excited_ds2 import rho_broad_full
from c_funs_lobatto_excited1small import rho_broad_full as rho_broad_fullsmall
from c_funs_lobatto_excited1big import rho_broad_full
# from c_funs_lobatto_excited1 import rho_broad_full

from Thesis_figs.Excited_params import p, sd_delao_from_B, deltaao_from_B
# p['sd_delam']=2*pi*3e6
filename='Thesis_figs/sims1/test_crossing_excited4_sd2_Bigpump2'
#B_vals=np.linspace(0.23,0.237,6)
B_vals=np.linspace(0.225,0.24,61)
deltamvals=np.linspace(-25e6,25e6,21)*2*np.pi
#deltamvals=np.linspace(-25e6,25e6,5)
# B_vals=np.linspace(0.225,0.24,11)+0.030
# deltamvals=np.linspace(-25e6,25e6,11)*2*np.pi


B_vals=np.linspace(0.223,0.241,201)
deltamvals=np.linspace(-29e6,29e6,201)*2*np.pi
# B_vals=np.linspace(0.223,0.241,61)
# deltamvals=np.linspace(-29e6,29e6,31)*2*np.pi


# filename='Thesis_figs/sims1/test_crossing_excited4_small_test'
# B_vals=np.linspace(0.223,0.241,51)
# deltamvals=np.linspace(-29e6,29e6,21)*2*np.pi
# B_vals=np.linspace(0.232,0.234,21)
# deltamvals=np.linspace(-25e6,25e6,7)*2*np.pi
# B_vals=np.linspace(0.227,0.238,61)
# deltamvals=np.linspace(-25e6,25e6,21)*2*np.pi
P_pump = 10*np.log10(1.74/1) #in dBm, 1.74 mW are going into the resonator
P_pump = 10*np.log10(20.1/1) #in dBm, 1.74 mW are going into the resonator

P_mu = -30.1-0 # in dBm
# P_mu = -30.1-20 # in dBm

T=650e-3
T=250e-3
filename=filename+'T='+str(T)+'Pmu='+str(P_mu)
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

p['Omega']= Omega_from_PdBm(P_pump,p)

print('Omega = ' + str(p['Omega']))
print('bin   = ' + str(bin_from_PdBm(P_mu,p)))

#
def b_vec_fun_no_a(b_vec,binval,delmval,p):
    bval=b_vec[0]+1j*b_vec[1]
    Omega=p['Omega']
    rho=np.array(rho_broad_full(0,bval, 0,delmval,p))
    p['Omega']=0
    rho_no_pump=np.array(rho_broad_full(0,bval, 0,delmval,p))
    p['Omega']=Omega
    #S12val=rho[1,0]*p['Nm']*p['gm']
    S12val=rho[1,2]*p['No']*p['gm']+rho_no_pump[1,2]*(p['Nm']-p['No'])*p['gm']
    bval1=(-1j*(S12val)+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*delmval)
    # bval1=(-1j*np.conj(S12val)+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*delmval)
    return [bval1.real, bval1.imag]-b_vec

def find_b_no_a(binval,delmval,p,start_guess_vec):
    #b_zero = lambda x: b_vec_fun_no_a(x,binval,delmval,p)-x
    #b_found=scipy.optimize.fsolve(b_zero,start_guess_vec)
    #b_found=scipy.optimize.fsolve(b_vec_fun_no_a,start_guess_vec,args=(binval,delmval,p))
    b_found=scipy.optimize.root(b_vec_fun_no_a,start_guess_vec,args=(binval,delmval,p),method='hybr')
    if not b_found.success:
        print(b_found.message)
    b_found=b_found.x

    return b_found[0]+1j*b_found[1]

bvals=np.zeros((len(B_vals),len(deltamvals)),dtype=np.complex_)
avals=np.zeros((len(B_vals),len(deltamvals)),dtype=np.complex_)
rho_out=np.zeros((3,3,len(B_vals),len(deltamvals)),dtype=np.complex_)
avals=np.zeros((len(B_vals),len(deltamvals)),dtype=np.complex_)
calc_time=np.zeros((len(B_vals),len(deltamvals)))
binvals=np.zeros((len(B_vals),len(deltamvals)),dtype=np.complex_)
#start_guess_complex_b=np.zeros((len(B_vals),len(deltamvals)),dtype=np.complex_)

ainval=0
start_time=time.time()
print('Starting!')
for ii, B_val in enumerate(B_vals):
    p['No']=p['No_total']/(np.exp(B_val*p['Gg']*6.63e-34/(1.38e-23*T))+1)
    p['Nm']=p['Nm_total']/(np.exp(B_val*p['Gg']*6.63e-34/(1.38e-23*T))+1)
    p['mean_delam']=deltamac_from_B(B_val,p)
    p['nbath']=nbath_from_T(T,p,deltamacval=p['mean_delam'])
    #p['mean_delao']=deltaao_from_B(deltamval,B_val,p)#1e6
    p['sd_delao']=sd_delao_from_B(B_val,p)
    for jj, deltamval in enumerate(deltamvals):
        p['mean_delao']=deltaao_from_B(B_val,deltamval,p)#+1000e6
        #p['mean_delao']=deltaao_from_B(B_val,0,p)

        #print(p['mean_delao']*1e-9)
        time1=time.time()
        binval=bin_from_PdBm(P_mu,p,deltamval=deltamval)
        binvals[ii,jj]=binval
        if jj==0 & ii==0:
            start_guess_complex_b=(np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*deltamval)
            start_guess_vec_b=[start_guess_complex_b.real,start_guess_complex_b.imag]

        elif ii==0:
            #start_guess_vec_bout=[boutvals[ii,jj-1].real,boutvals[ii,jj-1].imag]
            start_guess_vec_b=[bvals[0,jj-1].real,bvals[0,jj-1].imag]
            #start_guess_vec_b2=[bvals2[ii,jj-1].real,bvals2[ii,jj-1].imag,avals[ii,jj-1].real,avals[ii,jj-1].imag]
        elif jj==0:
            start_guess_vec_b=[bvals[ii-1,0].real,bvals[ii-1,0].imag]
        else:
            start_guess_complex_b=(bvals[ii,jj-1]+bvals[ii-1,jj])/2
            start_guess_vec_b=[start_guess_complex_b.real,start_guess_complex_b.imag]
    #    for pkey in p:
    #        print(pkey +' = '+str(type(p[pkey])))
        #print(type(binval))
        #print(type(deltamval))
        start_guess_vec_b=[0,0]

        start_guess_complex_b=(np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*deltamval)
        start_guess_vec_b=[start_guess_complex_b.real,start_guess_complex_b.imag]
        bvals[ii,jj] = find_b_no_a(binval,deltamval,p,start_guess_vec_b)
        # bvals[ii,jj]=start_guess_complex_b

        rho_out[:,:,ii,jj]=rho_broad_full(0,bvals[ii,jj],0,deltamval,p)
        # bvals[ii,jj]=bvals[ii,jj]-start_guess_complex_b
        calc_time[ii,jj]=time.time()-time1
    elapsed_time=time.time()-start_time
    print('    ' + filename + ': '+ str(ii) +', Time: ' + time.ctime() +', Elapsed: '+ str(elapsed_time))
    np.savez(filename,binvals=binvals,B_vals=B_vals,deltamvals=deltamvals,p=p,bvals=bvals,rho_out=rho_out,P_pump=P_pump,P_mu=P_mu,T=T,elapsed_time=elapsed_time,avals=avals,calc_time=calc_time)#,deltaoval=deltaoval)
print('    ===========================Complete===========================')


boutvals=bvals*np.sqrt(p['gammamc'])
aoutvals=avals*np.sqrt(p['gammaoc'])

rho13_background=10.0**(-22)

rho13=rho_out[0,2,:,:]
rho_abs2=np.abs(rho13**2)
rho_abs2[rho_abs2<rho13_background]=rho13_background
freqmu_vals=deltamvals/(2*np.pi)+p['freqmu']


fig=plt.figure(filename)

fig.clf()


ax=fig.add_subplot(3,2,1)
img1=ax.imshow(np.log10(rho_abs2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower')#,cmap='viridis_r')
plt.title(' |rho13|^2 (log scale)')
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
img1=ax.imshow((rho_abs2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower')
plt.title('|rho13|^2 ')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,2,4)
img1=ax.imshow((np.abs(boutvals/binvals)**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower')
plt.title('|bout/bin|^2')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

#matplotlib.cm.register_cmap(name='twilight',cmap=twilight)


ax=fig.add_subplot(3,2,5)
img1=ax.imshow((np.angle(rho13)),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower',cmap='hsv')
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
plt.show()
