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
from output_calcs.c_funs_test3 import rho_broad_full
from c_funs_ds3 import rho_broad_full
from c_funs_ds3_big import rho_broad_full as rho_broad_full_big
from c_funs_ds3_bigds import rho_broad_full as rho_broad_full_bigds
from c_funs_ds4 import rho_broad_full as rho_broad_full_bigds2



from Thesis_figs.Ground_params2 import p, sd_delao_from_B, deltaao_from_B
filename='Thesis_figs/sims1/test_crossing_ground_test'
B_vals=np.linspace(0.2,0.215,31)
deltamvals=np.linspace(-100e6,100e6,31)*2*np.pi
#B_vals=np.linspace(0.206,0.210,21)
#deltamvals=np.linspace(-40e6,40e6,21)*2*np.pi
B_vals=np.linspace(0.206,0.210,5)
deltamvals=np.linspace(-40e6,40e6,5)*2*np.pi
B_vals=np.linspace(0.204,0.212,21)
deltamvals=np.linspace(-70e6,70e6,21)*2*np.pi
#B_vals=np.linspace(0.201,0.211,21)
#deltamvals=np.linspace(-100e6,100e6,21)*2*np.pi
B_vals=np.linspace(0.202,0.212,41)
deltamvals=np.linspace(-100e6,100e6,41)*2*np.pi
B_vals=np.linspace(0.202,0.214,41)
deltamvals=np.linspace(-100e6,100e6,41)*2*np.pi


B_vals=np.linspace(0.206,0.218,61)
deltamvals=np.linspace(-100e6,100e6,61)*2*np.pi
#
# B_vals=np.linspace(0.206,0.210,21)
# deltamvals=np.linspace(-70e6,70e6,31)*2*np.pi
# B_vals=np.linspace(0.209,0.213,21)
# deltamvals=np.linspace(-70e6,70e6,31)*2*np.pi
#
# B_vals=np.linspace(0.211,0.213,11)
# deltamvals=np.linspace(-50e6,50e6,21)*2*np.pi


P_pump = 10*np.log10(1.74) #in dBm, 1.74 mW are going into the resonator

P_mu = -30.9 # in dBm
P_mu = -39.9 # in dBm
# P_mu = -35.9 # in dBm

# P_mu = 3-30 # in dBm

# P_mu = 3-32 # in dBm
P_mu = -39.9 # in dBm
P_mu = 3-32 # in dBm

# P_mu = -47.9 # in dBm
T=450e-3
T=670e-3
filename='Thesis_figs/sims1/test_crossing_ground2_1_sdmu = '+str(p['sd_delam']/(2*pi*1e6))+ 'Pmu=' +str(P_mu)+ 'T='+str(T)


def Omega_from_PdBm(PdBm,p):
    mu0=4*pi*1e-7
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

p['Omega']= Omega_from_PdBm(P_pump,p)

print('Omega = ' + str(p['Omega']))
print('bin   = ' + str(bin_from_PdBm(P_mu,p)))

#=====================================================
#define some s pre/post operators
#binval=binval/p['gammamc']
# def spre(m):
#     return TensorProduct(sym.eye(m.shape[0]),m)
#
# def spost(m):
#     return TensorProduct(m.T, sym.eye(m.shape[0]))
#
# def collapse(c):
#     tmp = Dagger(c)*c/2
#     return spre(c)*spost(Dagger(c))-spre(tmp)-spost(tmp)
#
#
# s13=Matrix([[0,0,1],[0,0,0],[0,0,0]])
# s23=Matrix([[0,0,0],[0,0,1],[0,0,0]])
# s12=Matrix([[0,1,0],[0,0,0],[0,0,0]])
#
# s31=s13.T
# s32=s23.T
# s21=s12.T
#
# s11 = s12*s21
# s22 = s21*s12
# s33 = s31*s13
#
# delo,delm=sym.symbols('delta_o delta_mu', real=True) #detunings between input and cavity
# delta3,delta2=sym.symbols('delta_3 delta_2', real=True)
#
# delao, delam =sym.symbols('delta_a_o delta_a_mu') #detunings between atom and cavity
# gamma13,gamma23,gamma2d,gamma3d,nbath,gammamu=sym.symbols('gamma_13 gamma_23 gamma_2d gamma_3d n_b gamma_mu', real=True, negative=False) #energy decay for atom levels
# Omega=sym.symbols('Omega', real=False, negative=False) #pump Rabi frequency
# rho11, rho12, rho13, rho21, rho22, rho23, rho31, rho32, rho33=sym.symbols('rho_11 rho_12 rho_13 rho_21 rho_22 rho_23 rho_31 rho_32 rho_33') #Density matrix elements
# a, b = sym.symbols('a b') #classical amplitudes of the optical and microwave fields
# #ar,ai=sym.symbols('a_r a_i', real=True)
# go, gm=sym.symbols('g_o, g_mu',real=False, negative=False) #coupling strengths for optical and microwave fields
# lam=sym.symbols('lambda')
#
# H_sys=Omega*s32+gm*s21*b+go*s31*a
# H_sys=H_sys+Dagger(H_sys)
# H_sys=H_sys+(delao -delo)*s33+(delam-delm)*s22
#
# LH=-I*spre(H_sys)+I*spost(H_sys)
# L21 = gammamu*(nbath+1)*collapse(s12)
# L12 = gammamu*nbath*collapse(s21)
# L32 = gamma23*collapse(s23)
# L31 = gamma13*collapse(s13)
# L22 = gamma2d*collapse(s22)
# L33 = gamma3d*collapse(s33)
#
# L=LH + L21 + L12 + L32 + L31 + L22 + L33
#
# L = L.row_insert(0,Matrix([[1,0,0,0,1,0,0,0,1]]))
# L.row_del(1)
#
# #define the density matrix in square and row form
# #the row form is so the Liovillian in matrix form can be acted on it
# rho = Matrix([[rho11,rho21,rho31],[rho12,rho22,rho32],[rho13,rho23,rho33]])
# rho = 1*rho.T #because we are using "fortran" style matrix flatteneing
# rho[:]
# rhoflat = 1*rho.T
# rhoflat = rhoflat[:]
#
# Lfunc = sym.lambdify((a,b,delo, delm,delao, delam, gamma13, gamma23, gamma2d, gamma3d, nbath,gammamu,Omega,go,gm),L)
# #change of variables to make things real to make it a bit faster maybe
# CtoR = Matrix([[2,0,0,0,0,0,0,0,0],
#                [0,0,0,0,2,0,0,0,0],
#                [0,0,0,0,0,0,0,0,2],
#                [0,1,0,1,0,0,0,0,0],
#                [0,I,0,-I,0,0,0,0,0],
#                [0,0,1,0,0,0,1,0,0],
#                [0,0,I,0,0,0,-I,0,0],
#                [0,0,0,0,0,1,0,1,0],
#                [0,0,0,0,0,I,0,-I,0]
#               ])
# CtoR=CtoR/2
# Lreal = sym.simplify(CtoR*L*CtoR.inv())
# #ar,ai,br,bi,gmr,gmi,gor,goi=sym.symbols('a_r a_i b_r b_i g_mu_r g_mu_i g_o_r g_o_i')
# #agor,agoi, bgmr, bgmi,Wr,Wi=sym.symbols('ag_or ag_oi bg_mu_r bg_mu_i Omega_r Omega_i')
# ar,ai,br,bi=sym.symbols('a_r a_i b_r b_i ')
# #Lreal.subs({agor:(a*go+sym.conjugate(a)*sym.conjugate(go))/2,I*(sym.conjugate(a)*sym.conjugate(go)-a*go)/2:agoi})
# Lreal=Lreal.subs({(a+sym.conjugate(a)):2*ar,(sym.conjugate(a)-a):2*I*ai,(b+sym.conjugate(b)):2*br,(sym.conjugate(b)-b):2*I*bi})
# #Lreal = Lreal.subs(a,ar+I*ai)
#
# Lrealfunc = sym.lambdify((ar,ai,br,bi,delo, delm,delao, delam, gamma13, gamma23, gamma2d, gamma3d, nbath,gammamu,Omega,go,gm),Lreal)
#
#
# H_disc_diff=sym.diff(sym.discriminant(sym.det(H_sys.subs({a:0,delao-delo:delta3,delam-delm:delta2})-lam*sym.eye(3)),lam),delta3)
# H_disc_diff_symfun=sym.lambdify((delta2,delta3,b,gm,Omega),H_disc_diff)
# H_disc_diff_fun=lambda delta2,delta3,b,gm,Omega: np.array(H_disc_diff_symfun(delta2,delta3,b,gm,Omega).real,dtype=float)
# def find_dressed_states_m(delaoval, deloval,delmval,bval,p):
#     try:
#         deltam_ds_val=scipy.optimize.fsolve((H_disc_diff_fun),np.array(delmval),args=(delaoval-deloval,bval,p['gm'],p['Omega']))+delmval
#     except ValueError:
#         deltam_ds_val=np.nan
#     except TypeError:
#         deltam_ds_val=np.nan
#     #except NameError:
#     #    deltam_ds_val=np.nan
#
#     return deltam_ds_val


def b_vec_fun_no_a(b_vec,binval,delmval,p):
    bval=b_vec[0]+1j*b_vec[1]
    Omega=p['Omega']
    rho=np.array(rho_broad_full(0,bval, 0,delmval,p))
    p['Omega']=0
    rho_no_pump=np.array(rho_broad_full(0,bval, 0,delmval,p))
    p['Omega']=Omega
    #S12val=rho[1,0]*p['Nm']*p['gm']
    S12val=rho[1,0]*p['No']*p['gm']+rho_no_pump[1,0]*(p['Nm']-p['No'])*p['gm']
    bval1=(-1j*S12val+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*delmval)
    return [bval1.real, bval1.imag]-b_vec

def b_vec_fun_no_a_2(b_vec,binval,delmval,p):
    bval=b_vec[0]+1j*b_vec[1]
    Omega=p['Omega']
    rho=np.array(rho_broad_full_big(0,bval, 0,delmval,p))
    p['Omega']=0
    rho_no_pump=np.array(rho_broad_full_big(0,bval, 0,delmval,p))
    p['Omega']=Omega
    #S12val=rho[1,0]*p['Nm']*p['gm']
    S12val=rho[1,0]*p['No']*p['gm']+rho_no_pump[1,0]*(p['Nm']-p['No'])*p['gm']
    bval1=(-1j*S12val+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*delmval)
    return [bval1.real, bval1.imag]-b_vec

def b_vec_fun_no_a_3(b_vec,binval,delmval,p):
    bval=b_vec[0]+1j*b_vec[1]
    Omega=p['Omega']
    rho=np.array(rho_broad_full_bigds(0,bval, 0,delmval,p))
    p['Omega']=0
    rho_no_pump=np.array(rho_broad_full_bigds(0,bval, 0,delmval,p))
    p['Omega']=Omega
    #S12val=rho[1,0]*p['Nm']*p['gm']
    S12val=rho[1,0]*p['No']*p['gm']+rho_no_pump[1,0]*(p['Nm']-p['No'])*p['gm']
    bval1=(-1j*S12val+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*delmval)
    return [bval1.real, bval1.imag]-b_vec

def b_vec_fun_no_a_4(b_vec,binval,delmval,p):
    bval=b_vec[0]+1j*b_vec[1]
    Omega=p['Omega']
    rho=np.array(rho_broad_full_bigds2(0,bval, 0,delmval,p))
    p['Omega']=0
    rho_no_pump=np.array(rho_broad_full_bigds2(0,bval, 0,delmval,p))
    p['Omega']=Omega
    #S12val=rho[1,0]*p['Nm']*p['gm']
    S12val=rho[1,0]*p['No']*p['gm']+rho_no_pump[1,0]*(p['Nm']-p['No'])*p['gm']
    bval1=(-1j*S12val+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*delmval)
    return [bval1.real, bval1.imag]-b_vec

def find_b_no_a(binval,delmval,p,start_guess_vec):
    #b_zero = lambda x: b_vec_fun_no_a(x,binval,delmval,p)-x
    #b_found=scipy.optimize.fsolve(b_zero,start_guess_vec)
    #b_found=scipy.optimize.fsolve(b_vec_fun_no_a,start_guess_vec,args=(binval,delmval,p))
#    b_found=scipy.optimize.root(b_vec_fun_no_a,start_guess_vec,args=(binval,delmval,p),method='broyden1')
    b_found=scipy.optimize.root(b_vec_fun_no_a,start_guess_vec,args=(binval,delmval,p),method='hybr')

    if not b_found.success:
        print(b_found.message)
        print('Attempt 2')
        b_found=scipy.optimize.root(b_vec_fun_no_a_2,start_guess_vec,args=(binval,delmval,p),method='hybr')
    if not b_found.success:
        start_guess_vec=b_found.x
        print(b_found.message)
        print('Attempt 3')
        b_found=scipy.optimize.root(b_vec_fun_no_a_3,start_guess_vec,args=(binval,delmval,p),method='hybr')
    if not b_found.success:
        start_guess_vec=b_found.x
        print(b_found.message)
        print('Attempt 4')
        b_found=scipy.optimize.root(b_vec_fun_no_a_4,start_guess_vec,args=(binval,delmval,p),method='hybr',maxfev=1000)
        if b_found.success:
            print(str(b_found.nit) +' evalulations of the big boy')
    if not b_found.success:
        print(b_found.message)
        print('Oh well')
    b_found=b_found.x

    return b_found[0]+1j*b_found[1]

bvals=np.zeros((len(B_vals),len(deltamvals)),dtype=np.complex_)
avals=np.zeros((len(B_vals),len(deltamvals)),dtype=np.complex_)
rho_out=np.zeros((3,3,len(B_vals),len(deltamvals)),dtype=np.complex_)
avals=np.zeros((len(B_vals),len(deltamvals)),dtype=np.complex_)
calc_time=np.zeros((len(B_vals),len(deltamvals)))
binvals=np.zeros((len(B_vals),len(deltamvals)),dtype=np.complex_)
ainval=0
start_time=time.time()
print('Starting!')
for ii, B_val in enumerate(B_vals):
    p['mean_delam']=deltamac_from_B(B_val,p)
    p['nbath']=nbath_from_T(T,p,deltamacval=p['mean_delam'])
    p['sd_delao']=sd_delao_from_B(B_val,p)
    # print('sd_o= '+str(p['sd_delao']))
    for jj, deltamval in enumerate(deltamvals):
        p['mean_delao']=deltaao_from_B(B_val,deltamval,p)
        # p['mean_delao']=-deltaao_from_B(B_val,0,p)

        # print(p['mean_delao'])
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
        if ii==0:
            start_guess_vec=[0,0]
        else:
            start_guess_vec=[bvals[ii-1,jj].real,bvals[ii-1,jj].imag]
        # start_guess_vec_b=[binval/np.sqrt(p['gammamc']),binval/np.sqrt(p['gammamc'])]
        bvals[ii,jj] = find_b_no_a(binval,deltamval,p,start_guess_vec_b)
        rho_out[:,:,ii,jj]=rho_broad_full_bigds2(0,bvals[ii,jj],0,deltamval,p)
        calc_time[ii,jj]=time.time()-time1
    elapsed_time=time.time()-start_time
    print('    ' + filename + ': '+str(ii) +', Time: ' + time.ctime() +', Elapsed: '+ str(elapsed_time))
    np.savez(filename,binvals=binvals,B_vals=B_vals,deltamvals=deltamvals,p=p,bvals=bvals,rho_out=rho_out,P_pump=P_pump,P_mu=P_mu,T=T,elapsed_time=elapsed_time,avals=avals,calc_time=calc_time,ii=ii,jj=jj)#,deltaoval=deltaoval)
print('    ===========================Complete===========================')


boutvals=bvals*np.sqrt(p['gammamc'])
aoutvals=avals*np.sqrt(p['gammaoc'])
rho13=rho_out[0,2,:,:]

freqmu_vals=deltamvals/(2*np.pi)+p['freqmu']


fig=plt.figure(filename)

fig.clf()
fig.patch.set_facecolor('none')

ax=fig.add_subplot(2,2,1)
img1=ax.imshow(np.log10(np.abs(rho13)**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower',cmap='viridis')
plt.title(' |rho13|^2 (log scale)')
plt.xlabel('delta_mu')
plt.ylabel('B')
fig.colorbar(img1)

ax=fig.add_subplot(2,2,2)
img1=ax.imshow(10*np.log10(np.abs(boutvals/binvals)**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower',cmap='viridis')
plt.title('|bout/bin|^2 (dB)')
plt.xlabel('delta_mu')
plt.ylabel('B')
fig.colorbar(img1)

ax=fig.add_subplot(2,2,3)
img1=ax.imshow((np.abs(rho13)**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower')
plt.title('|rho13|^2 ')
plt.xlabel('delta_mu')
plt.ylabel('B')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(2,2,4)
img1=ax.imshow((np.abs(boutvals/binvals)**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower')
plt.title('|bout/bin|^2')
plt.xlabel('delta_mu')
plt.ylabel('B')
fig.colorbar(img1)


plt.show()
