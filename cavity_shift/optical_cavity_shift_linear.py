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
# from c_funs_ds import rho_broad_full as rho_broad_full_ds
# from c_funs_test4 import rho_broad_full

from Linear1.c_linear_ground5 import rho_broad_full,rho_broad_full_real_imag

filename='optical_cavity_linear_sim_T4000'
#I_vals=np.linspace(7.43,7.55,31)
delta3vals=np.linspace(-2e11,2e11,21)*2*np.pi
deltaocvals=np.linspace(-5e7,5e7,21)
P_pump = 10*np.log10(1.74) #in dBm, 1.74 mW are going into the resonator
P_mu = -105-30 # in dBm
T=4000e-3

p={}
p['freqmu']=5015e6#4733e6 #this is the microwave cavity frequency
p['freq_pump'] = 195116.71e9 #pump frequency
#p['freqo']=p['freqmu']+p['freq_pump']

#p['nbath']=nbath_from_T(T,p)
p['d13'] = 2e-32*math.sqrt(1/3)
p['d23'] = 2e-32*math.sqrt(2/3)
p['gamma13'] = p['d13']**2/(p['d13']**2+p['d23']**2)*1/11e-3
p['gamma23'] = p['d23']**2/(p['d13']**2+p['d23']**2)*1/11e-3
p['gamma2d'] = 1e6
p['gamma3d'] = 1e6
#p['nbath'] = 20
p['gammamu'] = 1/11#1/(p['nbath']+1) * 1e3
p['sd_delam']=2*pi*2e6
p['gammaoc']=2*pi*1.7e6
p['gammaoi']=2*pi*7.95e6
p['gammamc']=2*pi*0.0622e6
p['gammami']=2*pi*5.69e6


muBohr=927.4009994e-26; # Bohr magneton in J/T in J* T^-1
p['mu12'] = 4.3803*muBohr # transition dipole moment for microwave cavity (J T^-1)


p['go'] = 51.9  #optical coupling

p['No'] = 2.2e15 # number of atoms in the optical mode


p['Nm'] = 2e16  #toal number of atoms
p['gm'] = 1.04 #coupling between atoms and microwave field

p['Wbeam']=0.6e-3
p['Lsample']=12e-3 # the length of the sample, in m
p['Lcavity_vac'] = 49.5e-3 # length of the vacuum part of the optical Fabry Perot (m)
p['nYSO'] = 1.76 #refractive index of YSO

#functions to define simulation parameters which change as we change the frequencies
def Omega_from_PdBm(PdBm,p):
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
def Omega_from_PdBm_resonance(PdBm,p):
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

def bin_from_PdBm(Pdbm,p,deltamval=0):
    P=1e-3*10**(Pdbm/10)
    hbar=1.05457e-34; # in J*s
    omega=2*np.pi*(p['freqmu'])+deltamval
    bin=np.sqrt(P/hbar/omega)
    return bin
def nbath_from_T(T,p,deltamacval=0):
    omega=np.abs(2*pi*p['freqmu']-deltamacval)
    nbath = 1/(np.exp(1.0545718e-34*omega/1.38064852e-23/T)-1)
    return nbath
def deltamac_from_I(I_mag,p):
    B_mag=(0.027684*I_mag*1e3-0.056331)*1e-3
    deltamacvals=(24.1886*1e9)*(B_mag)-p['freqmu']
    return deltamacvals*2*np.pi

def omegaao_from_I(I_mag,p):
    return (195.1166+1.0227e-4*I_mag)*1e12*2*np.pi
def sd_delao_from_I(I_mag,p):
    return 1e9*(0.55293-0.015897*I_mag)*np.pi*2
def deltaao_from_I(I_mag,p):
    return 2*pi*(p['freqmu']+p['freq_pump'])-omegaao_from_I(I_mag,p)

p['Omega']= 0#Omega_from_PdBm_resonance(P_pump,p)

print('Omega = ' + str(p['Omega']))
print('bin   = ' + str(bin_from_PdBm(P_mu,p)))

#=====================================================
#define some s pre/post operators
#binval=binval/p['gammamc']
def spre(m):
    return TensorProduct(sym.eye(m.shape[0]),m)

def spost(m):
    return TensorProduct(m.T, sym.eye(m.shape[0]))

def collapse(c):
    tmp = Dagger(c)*c/2
    return spre(c)*spost(Dagger(c))-spre(tmp)-spost(tmp)


s13=Matrix([[0,0,1],[0,0,0],[0,0,0]])
s23=Matrix([[0,0,0],[0,0,1],[0,0,0]])
s12=Matrix([[0,1,0],[0,0,0],[0,0,0]])

s31=s13.T
s32=s23.T
s21=s12.T

s11 = s12*s21
s22 = s21*s12
s33 = s31*s13

delo,delm=sym.symbols('delta_o delta_mu', real=True) #detunings between input and cavity
delta3,delta2=sym.symbols('delta_3 delta_2', real=True)

delao, delam =sym.symbols('delta_a_o delta_a_mu') #detunings between atom and cavity
gamma13,gamma23,gamma2d,gamma3d,nbath,gammamu=sym.symbols('gamma_13 gamma_23 gamma_2d gamma_3d n_b gamma_mu', real=True, negative=False) #energy decay for atom levels
Omega=sym.symbols('Omega', real=False, negative=False) #pump Rabi frequency
rho11, rho12, rho13, rho21, rho22, rho23, rho31, rho32, rho33=sym.symbols('rho_11 rho_12 rho_13 rho_21 rho_22 rho_23 rho_31 rho_32 rho_33') #Density matrix elements
a, b = sym.symbols('a b') #classical amplitudes of the optical and microwave fields
#ar,ai=sym.symbols('a_r a_i', real=True)
go, gm=sym.symbols('g_o, g_mu',real=False, negative=False) #coupling strengths for optical and microwave fields
lam=sym.symbols('lambda')

H_sys=Omega*s32+gm*s21*b+go*s31*a
H_sys=H_sys+Dagger(H_sys)
H_sys=H_sys+(delao -delo)*s33+(delam-delm)*s22

LH=-I*spre(H_sys)+I*spost(H_sys)
L21 = gammamu*(nbath+1)*collapse(s12)
L12 = gammamu*nbath*collapse(s21)
L32 = gamma23*collapse(s23)
L31 = gamma13*collapse(s13)
L22 = gamma2d*collapse(s22)
L33 = gamma3d*collapse(s33)

L=LH + L21 + L12 + L32 + L31 + L22 + L33

L = L.row_insert(0,Matrix([[1,0,0,0,1,0,0,0,1]]))
L.row_del(1)

#define the density matrix in square and row form
#the row form is so the Liovillian in matrix form can be acted on it
rho = Matrix([[rho11,rho21,rho31],[rho12,rho22,rho32],[rho13,rho23,rho33]])
rho = 1*rho.T #because we are using "fortran" style matrix flatteneing
rho[:]
rhoflat = 1*rho.T
rhoflat = rhoflat[:]

Lfunc = sym.lambdify((a,b,delo, delm,delao, delam, gamma13, gamma23, gamma2d, gamma3d, nbath,gammamu,Omega,go,gm),L)
#change of variables to make things real to make it a bit faster maybe
CtoR = Matrix([[2,0,0,0,0,0,0,0,0],
               [0,0,0,0,2,0,0,0,0],
               [0,0,0,0,0,0,0,0,2],
               [0,1,0,1,0,0,0,0,0],
               [0,I,0,-I,0,0,0,0,0],
               [0,0,1,0,0,0,1,0,0],
               [0,0,I,0,0,0,-I,0,0],
               [0,0,0,0,0,1,0,1,0],
               [0,0,0,0,0,I,0,-I,0]
              ])
CtoR=CtoR/2
Lreal = sym.simplify(CtoR*L*CtoR.inv())
#ar,ai,br,bi,gmr,gmi,gor,goi=sym.symbols('a_r a_i b_r b_i g_mu_r g_mu_i g_o_r g_o_i')
#agor,agoi, bgmr, bgmi,Wr,Wi=sym.symbols('ag_or ag_oi bg_mu_r bg_mu_i Omega_r Omega_i')
ar,ai,br,bi=sym.symbols('a_r a_i b_r b_i ')
#Lreal.subs({agor:(a*go+sym.conjugate(a)*sym.conjugate(go))/2,I*(sym.conjugate(a)*sym.conjugate(go)-a*go)/2:agoi})
Lreal=Lreal.subs({(a+sym.conjugate(a)):2*ar,(sym.conjugate(a)-a):2*I*ai,(b+sym.conjugate(b)):2*br,(sym.conjugate(b)-b):2*I*bi})
#Lreal = Lreal.subs(a,ar+I*ai)

Lrealfunc = sym.lambdify((ar,ai,br,bi,delo, delm,delao, delam, gamma13, gamma23, gamma2d, gamma3d, nbath,gammamu,Omega,go,gm),Lreal)


H_disc_diff=sym.diff(sym.discriminant(sym.det(H_sys.subs({a:0,delao-delo:delta3,delam-delm:delta2})-lam*sym.eye(3)),lam),delta3)
H_disc_diff_symfun=sym.lambdify((delta2,delta3,b,gm,Omega),H_disc_diff)
H_disc_diff_fun=lambda delta2,delta3,b,gm,Omega: np.array(H_disc_diff_symfun(delta2,delta3,b,gm,Omega).real,dtype=float)
def find_dressed_states_m(delaoval, deloval,delmval,bval,p):
    try:
        deltam_ds_val=scipy.optimize.fsolve((H_disc_diff_fun),np.array(delmval),args=(delaoval-deloval,bval,p['gm'],p['Omega']))+delmval
    except ValueError:
        deltam_ds_val=np.nan
    except TypeError:
        deltam_ds_val=np.nan
    #except NameError:
    #    deltam_ds_val=np.nan

    return deltam_ds_val


def fields_vec_fun(fields_vec,ainval,binval,deltaocval,deltamucval,p):
    aval=fields_vec[0]+1j*fields_vec[1]
    bval=fields_vec[2]+1j*fields_vec[3]
    rho=np.array(rho_broad_full(aval,bval,0,0,p))
    #rho=np.array(rho_broad_full_ds(aval,bval,0,0,find_dressed_states_m,p))

    #rho_only_b=np.array(rho_broad_full(0,bval,0,delmval,find_dressed_states_m,p))
    S12val=rho[1,0]*p['No']*p['gm']
    S13val=rho[2,0]*p['No']*p['go']
    #S21_onlyb_val=rho_only_b[1,0]*(p['Nm']-p['No'])*p['gm']
    bval1=(-1j*S12val+np.sqrt(p['gammamc'])*binval)/((p['gammamc']+p['gammami'])/2+1j*deltamucval)
    #bval1=(-1j*S12val+np.sqrt(p['gammamc'])*binval)/((p['gammamc']+p['gammami'])/2-1j*delmval)
    aval1=(-1j*S13val+np.sqrt(p['gammaoc'])*ainval)/((p['gammaoc']+p['gammaoi'])/2+1j*deltaocval)
    return [aval1.real,aval1.imag,bval1.real,bval1.imag]-fields_vec
def fields_abs_fun(fields_vec,ainval,binval,deloval,delmval,p):
    aval=fields_vec[0]+1j*fields_vec[1]
    bval=fields_vec[2]+1j*fields_vec[3]
    rho=np.array(rho_broad_full(aval,bval,deloval,delmval,find_dressed_states_m,p))
    rho_only_b=np.array(rho_broad_full(0,bval,0,delmval,find_dressed_states_m,p))
    S12val=rho[1,0]*p['No']*p['gm']
    S13val=rho[2,0]*p['No']*p['go']
    S21_onlyb_val=rho_only_b[1,0]*(p['Nm']-p['No'])*p['gm']
    bval1=(-1j*S12val-1j*S21_onlyb_val+np.sqrt(p['gammamc'])*binval)/((p['gammamc']+p['gammami'])/2-1j*delmval)
    #bval1=(-1j*S12val+np.sqrt(p['gammamc'])*binval)/((p['gammamc']+p['gammami'])/2-1j*delmval)
    aval1=(-1j*S13val+np.sqrt(p['gammaoc'])*ainval)/((p['gammaoc']+p['gammaoi'])/2-1j*deloval)
    out_vec=np.array([aval1.real,aval1.imag,bval1.real,bval1.imag]-fields_vec)
    out_abs=np.linalg.norm(out_vec)
    return out_abs

def find_fields(ainval,binval,deltaocval,deltamucval,start_guess_vec,p):
    #fields_found=scipy.optimize.root(fields_vec_fun,start_guess_vec,args=(ainval,binval,deloval,delmval,p),method='hybr',options={'xtol':1.49012e-08,'maxfev':5000})
    #fields_found=scipy.optimize.root(fields_vec_fun,start_guess_vec,args=(ainval,binval,deloval,delmval,p),method='broyden1',options={'maxiter':10000})
    #fields_found=scipy.optimize.root(fields_vec_fun,start_guess_vec,args=(ainval,binval,deloval,delmval,p),method='lm',options={})
    fields_found=scipy.optimize.root(fields_vec_fun,start_guess_vec,args=(ainval,binval,deltaocval,deltamucval,p),method='hybr')
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

def calc_efficiency(deloval,delmval,delocval,delmucval,p):
    #a=Caa*ain+Cba*bin
    #b=Cab*ain+Cbb*bin
    rho0,rhoa,rhoac,rhob,rhobc=rho_broad_full(0,0, 0,p)

    Sa13=rhoa[0,2]*p['No']*p['go']
    #Sb13=rhob[0,2]*p['Nm']*p['go']
    Sb13=rhob[0,2]*p['No']*p['go']

    Sa12=rhoa[0,1]*p['No']*p['gm']
    #Sb12=rhob[0,1]*p['Nm']*p['gm']
    Sb12=rhob[0,1]*p['No']*p['gm']

    #print(Sa12)
    Caa=p['gammaoc']*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+p['gammamc'])/2)/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+p['gammamc'])/2))
    Cba=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+p['gammamc'])/2))
    Cab=           -1j*np.sqrt(p['gammaoc'])*np.sqrt(p['gammamc'])*Sa12/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+p['gammamc'])/2))
    Cbb=p['gammamc']*(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc'])/2)/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+p['gammamc'])/2))
    return Caa, Cba, Cab, Cbb
Caa=np.zeros((len(delta3vals),len(deltaocvals)),dtype=np.complex_)
Cba=np.zeros((len(delta3vals),len(deltaocvals)),dtype=np.complex_)
Cab=np.zeros((len(delta3vals),len(deltaocvals)),dtype=np.complex_)
Cbb=np.zeros((len(delta3vals),len(deltaocvals)),dtype=np.complex_)
calc_time=np.zeros((len(delta3vals),len(deltaocvals)))

start_time=time.time()
deltamucval=0
p['mean_delam']=deltaao_from_I(0,p)
p['sd_delao']=sd_delao_from_I(0,p)
for ii,delta3val in enumerate(delta3vals):
    p['mean_delao']=delta3val#deltamac_from_I(I_val,p)
    p['nbath']=nbath_from_T(T,p)
    for jj, deltaocval in enumerate(deltaocvals):
        time1=time.time()
        Caa[ii,jj], Cba[ii,jj], Cab[ii,jj], Cbb[ii,jj]=calc_efficiency(0,0,deltaocval,deltamucval,p)

    elapsed_time=time.time()-start_time
    print('    ' + str(ii) +', Time: ' + time.ctime() +', Elapsed: '+ str(elapsed_time))
    np.savez(filename,delta3vals=delta3vals,p=p,deltaocvals=deltaocvals,P_pump=P_pump,T=T,elapsed_time=elapsed_time,deltamucval=deltamucval,calc_time=calc_time,Caa=Caa, Cba=Cba,Cab=Cab,Cbb=Cbb)
print('    ===========================Complete===========================')

delta3vals2=np.linspace(np.min(delta3vals),np.max(delta3vals),51)

fig=plt.figure(filename+'_Caa')

fig.clf()

ax=fig.add_subplot(3,1,1)
img1=ax.imshow(10*np.log10(np.abs(Caa)**2),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower')
#plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)
plt.plot(p['No']*p['go']**2/delta3vals2,delta3vals2)

plt.title(' |C|^2 (dB scale)')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,2)
img1=ax.imshow((np.abs(Caa)**2),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower')
plt.plot(p['No']*p['go']**2/delta3vals2,delta3vals2)

plt.title('|C|^2 ')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,1,3)
img1=ax.imshow((np.angle(Caa)),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' C phase')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

fig.suptitle('Caa')

fig=plt.figure(filename+'_Cba')

fig.clf()

ax=fig.add_subplot(3,1,1)
img1=ax.imshow(10*np.log10(np.abs(Cba)),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower')
plt.title(' |C| (dB scale)')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,2)
img1=ax.imshow((np.abs(Cba)),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower')
plt.title('|C|')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,1,3)
img1=ax.imshow((np.angle(Cba)),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' C phase')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

fig.suptitle('Cba')

fig=plt.figure(filename+'_Cab')

fig.clf()

ax=fig.add_subplot(3,1,1)
img1=ax.imshow(10*np.log10(np.abs(Cab)),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower')
plt.title(' |C| (dB scale)')
plt.xlabel('delta_o')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,2)
img1=ax.imshow((np.abs(Cab)),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower')
plt.title('|C|')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,1,3)
img1=ax.imshow((np.angle(Cab)),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' C phase')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

fig.suptitle('Cab')

fig=plt.figure(filename+'_Cbb')

fig.clf()

ax=fig.add_subplot(3,1,1)
img1=ax.imshow(10*np.log10(np.abs(Cbb)**2),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower')
#plt.plot(p['No']*p['gm']**2/delta3vals2,delta3vals2)
plt.title(' |C|^2 (dB scale)')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,2)
img1=ax.imshow((np.abs(Cbb)**2),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower')
#plt.plot(p['No']*p['gm']**2/delta3vals2,delta3vals2)
plt.title('|C|')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,1,3)
img1=ax.imshow((np.angle(Cbb)),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title('C phase')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

fig.suptitle('Cbb')
plt.show()
