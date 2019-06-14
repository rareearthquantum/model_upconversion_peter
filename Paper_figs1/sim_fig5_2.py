

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

filename='Paper_figs1/sim_fig5_data2'
B_val=200.4e-3#0.4e-3
#B_val=232.4e-3#0.4e-3

deltamvals=np.linspace(-60e6,260e6,11)*2*np.pi
#deltamvals=np.linspace(0,1e6,1)*2*np.pi

freq_pump_vals=np.linspace(195.10,195.124,51)*1e12
freq_pump_vals=1.951e14+np.linspace(0.5,1.8,51)*1e10
freq_pump_vals=np.linspace(195.110,195.124,51)*1e12-4732e6
#freq_pump_vals=np.linspace(195.107,195.1085,2)*1e12
#freq_pump_vals=np.linspace(195.11,195.114,21)*1e12
#freq_pump_vals=np.linspace(195.106,195.108,2)*1e12
#freq_pump_vals=np.linspace(195.105,195.12,101)*1e12

P_pump = 10*np.log10(1.74) #in dBm, 1.74 mW are going into the resonator
P_mu = -15-30 # in dBm
T=190e-3

p={}
p['freqmu']=4732e6#4733e6 #this is the microwave cavity frequency
#p['freq_pump'] = 195117.044e9 #pump frequency
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

p['Nm'] = 6e16  #toal number of atoms

p['gm'] = 1.04/1 #coupling between atoms and microwave field
p['T']=T
p['Wbeam']=0.6e-3

p['Lsample']=12e-3 # the length of the sample, in m
p['Lcavity_vac'] = 49.5e-3 # length of the vacuum part of the optical Fabry Perot (m)
p['nYSO'] = 1.76 #refractive index of YSO

#functions to define simulation parameters which change as we change the frequencies
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
    omega=np.abs(2*pi*p['freqmu']-deltamacval)
    nbath = 1/(np.exp(1.0545718e-34*omega/1.38064852e-23/T)-1)
    return nbath
def I_from_B(B_mag):
    return (B_mag+0.056331e-3)/0.027684

def deltamac_from_I(B_mag,p):
    #B_mag=(0.027684*I_mag*1e3-0.056331)*1e-3
    deltamacvals=(p['Gg'])*(B_mag)-p['freqmu']
    return deltamacvals*2*np.pi
def omegaao1_from_I(B_mag,p):
    return (p['f0_no_B']+(-p['Gg']-p['Ge'])/2*B_mag)*2*np.pi
def omegaao2_from_I(B_mag,p):
    return (p['f0_no_B']+(-p['Gg']+p['Ge'])/2*B_mag)*2*np.pi
def omegaao3_from_I(B_mag,p):
    return (p['f0_no_B']+(+p['Gg']-p['Ge'])/2*B_mag)*2*np.pi
def omegaao4_from_I(B_mag,p):
    return (p['f0_no_B']+(+p['Gg']+p['Ge'])/2*B_mag)*2*np.pi

def sd_delao1_from_I(B_mag,p):
    return 1e9*(0.148588212918272+0.308180352441052*B_mag)*np.pi*2
def sd_delao2_from_I(B_mag,p):
    #return 1e9*(0.096660692060221-0.336895927831593*B_mag)*np.pi*2
    return 1e9*(0.096660692060221+0.336895927831593*B_mag)*np.pi*2
def sd_delao3_from_I(B_mag,p):
    return 1e9*(0.391366413926165+0.137444967951503*B_mag)*np.pi*2
def sd_delao4_from_I(B_mag,p):
    return 1e9*(0.279915072366011+0.000852680155581*B_mag)*np.pi*2

# def omegaao1_from_I(I_mag,p):
#     return 195.112e12*np.pi*2
# def omegaao2_from_I(I_mag,p):
#     return 195.117e12*np.pi*2
# def omegaao3_from_I(I_mag,p):
#     return 195.118e12*np.pi*2
# def omegaao4_from_I(I_mag,p):
#     return 195.12275e12*np.pi*2

def deltaao_from_I(I_mag,omegaao,p):
    return 2*pi*(p['freqmu']+p['freq_pump'])-omegaao

    #return 2*pi*(p['freq_pump'])-omegaao
p['nbath']=nbath_from_T(T,p)

p['Omega']= Omega_from_PdBm(P_pump,p)
I_val=I_from_B(B_val)
#
print('freq1 = ' + str(omegaao1_from_I(B_val,p)/(2*pi)))
print('freq2 = ' + str(omegaao2_from_I(B_val,p)/(2*pi)))
print('freq3 = ' + str(omegaao3_from_I(B_val,p)/(2*pi)))
print('freq4 = ' + str(omegaao4_from_I(B_val,p)/(2*pi)))
print('Omega = ' + str(p['Omega']))
print('bin   = ' + str(bin_from_PdBm(P_mu,p)))
print('I_val = ' + str(I_val))
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

def S12_1(B_val,bval,delmval,p):

    omegaao=omegaao1_from_I(B_val,p)
    p['sd_delao']=sd_delao1_from_I(B_val,p)
    p['mean_delao']=deltaao_from_I(B_val,omegaao,p)
    p['mean_delam']=deltamac_from_I(B_val,p)
    p['nbath']=nbath_from_T(T,p,deltamacval=p['mean_delam'])
    p['Omega']= Omega_from_PdBm(P_pump,p)
    rho=np.array(rho_broad_full(0,bval, 0,delmval,p))
    p['Omega']=0
    rho_no_pump=np.array(rho_broad_full(0,bval, 0,delmval,p))
    return rho[1,0]*p['No']*p['gm']+rho_no_pump[1,0]*(p['Nm']-p['No'])*p['gm']

def S12_2(Bval,bval,delmval,p):

    omegaao=omegaao2_from_I(B_val,p)
    p['sd_delao']=sd_delao2_from_I(B_val,p)
    p['mean_delao']=deltaao_from_I(B_val,omegaao,p)
    p['mean_delam']=deltamac_from_I(B_val,p)
    p['nbath']=nbath_from_T(T,p,p['mean_delam'])
    p['Omega']= Omega_from_PdBm(P_pump,p)
    rho=np.array(rho_broad_full(0,bval, 0,delmval,p))
    p['Omega']=0
    rho_no_pump=np.array(rho_broad_full(0,bval, 0,delmval,p))
    return rho[1,0]*p['No']*p['gm']+rho_no_pump[1,0]*(p['Nm']-p['No'])*p['gm']
def S12_3(Bval,bval,delmval,p):

    omegaao=omegaao3_from_I(B_val,p)
    p['sd_delao']=sd_delao2_from_I(B_val,p)
    p['mean_delao']=deltaao_from_I(B_val,omegaao,p)
    p['mean_delam']=deltamac_from_I(B_val,p)
    p['nbath']=nbath_from_T(T,p,p['mean_delam'])
    p['Omega']= Omega_from_PdBm(P_pump,p)
    rho=np.array(rho_broad_full(0,bval, 0,delmval,p))
    p['Omega']=0
    rho_no_pump=np.array(rho_broad_full(0,bval, 0,delmval,p))
    return rho[1,0]*p['No']*p['gm']+rho_no_pump[1,0]*(p['Nm']-p['No'])*p['gm']
def S12_4(Bval,bval,delmval,p):

    omegaao=omegaao4_from_I(B_val,p)
    p['sd_delao']=sd_delao4_from_I(B_val,p)
    p['mean_delao']=deltaao_from_I(B_val,omegaao,p)
    p['mean_delam']=deltamac_from_I(B_val,p)
    p['nbath']=nbath_from_T(T,p,p['mean_delam'])
    p['Omega']= Omega_from_PdBm(P_pump,p)
    rho=np.array(rho_broad_full(0,bval, 0,delmval,p))
    p['Omega']=0
    rho_no_pump=np.array(rho_broad_full(0,bval, 0,delmval,p))
    return rho[1,0]*p['No']*p['gm']+rho_no_pump[1,0]*(p['Nm']-p['No'])*p['gm']

def b_vec_fun_4_3lvl(b_vec,binval,delmval,Bval,p):
    bval=b_vec[0]+1j*b_vec[1]

    S12val=S12_1(Bval,bval,delmval,p)#+S12_2(Bval,bval,delmval,p)+S12_3(Bval,bval,delmval,p)+S12_4(Bval,bval,delmval,p)
    bval1=(-1j*S12val+np.sqrt(p['gammamc'])*binval)/((p['gammamc']+p['gammami'])/2-1j*delmval)
    return [bval1.real, bval1.imag]-b_vec

def find_b_4_3lvl(binval,delmval,Bval,p,start_guess_vec):
    #b_zero = lambda x: b_vec_fun_no_a(x,binval,delmval,p)-x
    #b_found=scipy.optimize.fsolve(b_zero,start_guess_vec)
    #b_found=scipy.optimize.fsolve(b_vec_fun_no_a,start_guess_vec,args=(binval,delmval,p))
    b_found=scipy.optimize.root(b_vec_fun_4_3lvl,start_guess_vec,args=(binval,delmval,Bval,p),method='broyden1')
    b_found=b_found.x
    return b_found[0]+1j*b_found[1]

def b_vec_fun_no_a(b_vec,binval,delmval,p):
    bval=b_vec[0]+1j*b_vec[1]
    Omega=p['Omega']
    rho=np.array(rho_broad_full(0,bval, 0,delmval,p))
    p['Omega']=0
    rho_no_pump=np.array(rho_broad_full(0,bval, 0,delmval,p))
    p['Omega']=Omega
    #S12val=rho[1,0]*p['Nm']*p['gm']
    S12val=rho[1,0]*p['No']*p['gm']+rho_no_pump[1,0]*(p['Nm']-p['No'])*p['gm']
    bval1=(-1j*S12val+np.sqrt(p['gammamc'])*binval)/((p['gammamc']+p['gammami'])/2-1j*delmval)
    return [bval1.real, bval1.imag]-b_vec

def find_b_no_a(binval,delmval,p,start_guess_vec):
    #b_zero = lambda x: b_vec_fun_no_a(x,binval,delmval,p)-x
    #b_found=scipy.optimize.fsolve(b_zero,start_guess_vec)
    #b_found=scipy.optimize.fsolve(b_vec_fun_no_a,start_guess_vec,args=(binval,delmval,p))
    b_found=scipy.optimize.root(b_vec_fun_no_a,start_guess_vec,args=(binval,delmval,p),method='broyden1')
    b_found=b_found.x
    return b_found[0]+1j*b_found[1]
bvals=np.zeros((len(freq_pump_vals),len(deltamvals)),dtype=np.complex_)
avals=np.zeros((len(freq_pump_vals),len(deltamvals)),dtype=np.complex_)
rho_out_b=np.zeros((3,3,len(freq_pump_vals),len(deltamvals)),dtype=np.complex_)
avals=np.zeros((len(freq_pump_vals),len(deltamvals)),dtype=np.complex_)
calc_time=np.zeros((len(freq_pump_vals),len(deltamvals)))
binvals=np.zeros((len(freq_pump_vals),len(deltamvals)),dtype=np.complex_)
ainval=0
start_time=time.time()

for ii, freq_pump_val in enumerate(freq_pump_vals):
    p['freq_pump']=freq_pump_val
    for jj, deltamval in enumerate(deltamvals):
        binvals[ii,jj]=bin_from_PdBm(P_mu,p,deltamval)
        binval=binvals[ii,jj]
        if jj==0 & ii==0:
            #start_guess_vec_bout=[binval.real,binval.imag]
            start_guess_vec_b=[binval.real/p['gammamc'],binval.imag/p['gammamc']]
            #start_guess_vec_b2=[binval.real/p['gammamc'],binval.imag/p['gammamc'],0,0]

        elif ii==0:
            #start_guess_vec_bout=[boutvals[ii,jj-1].real,boutvals[ii,jj-1].imag]
            start_guess_vec_b=[bvals[0,jj-1].real,bvals[0,jj-1].imag]
            #start_guess_vec_b2=[bvals2[ii,jj-1].real,bvals2[ii,jj-1].imag,avals[ii,jj-1].real,avals[ii,jj-1].imag]
        elif jj==0:
            start_guess_vec_b=[bvals[ii-1,0].real,bvals[ii-1,0].imag]
        else:
            start_guess_complex_b=(bvals[ii,jj-1]+bvals[ii-1,jj])/2
            start_guess_vec_b=[start_guess_complex_b.real,start_guess_complex_b.imag]
        #start_guess_vec_b=[0,0]
        bvals[ii,jj]=find_b_4_3lvl(binval,deltamval,B_val,p,start_guess_vec_b)
        #rho_out_b[:,:,ii,jj]=S12_2(B_val,bvals[ii,jj],deltamval,p)#+S12_2(B_val,bvals[ii,jj],deltamval,p)+S12_3(B_val,bvals[ii,jj],deltamval,p)+S12_4(B_val,bvals[ii,jj],deltamval,p)

        omegaao=omegaao1_from_I(B_val,p)
        p['sd_delao']=sd_delao1_from_I(B_val,p)
        p['mean_delao']=deltaao_from_I(B_val,omegaao,p)
        p['mean_delam']=deltamac_from_I(B_val,p)
        p['nbath']=nbath_from_T(T,p,deltamacval=p['mean_delam'])
        p['Omega']= Omega_from_PdBm(P_pump,p)
        #bvals[ii,jj]=find_b_no_a(binval,deltamval,p,start_guess_vec_b)
        rho_out_b[:,:,ii,jj]=np.array(rho_broad_full(0,(bvals[ii,jj]), 0,deltamval,p))

        omegaao=omegaao2_from_I(B_val,p)
        p['sd_delao']=sd_delao2_from_I(B_val,p)
        p['mean_delao']=deltaao_from_I(B_val,omegaao,p)
        p['mean_delam']=deltamac_from_I(B_val,p)
        p['nbath']=nbath_from_T(T,p,p['mean_delam'])
        p['Omega']= Omega_from_PdBm(P_pump,p)

        rho_out_b[:,:,ii,jj]=rho_out_b[:,:,ii,jj]+np.array(rho_broad_full(0,(bvals[ii,jj]), 0,deltamval,p))


        omegaao=omegaao3_from_I(B_val,p)
        p['sd_delao']=sd_delao3_from_I(B_val,p)
        p['mean_delao']=deltaao_from_I(B_val,omegaao,p)
        p['mean_delam']=deltamac_from_I(B_val,p)
        p['nbath']=nbath_from_T(T,p,p['mean_delam'])
        p['Omega']= Omega_from_PdBm(P_pump,p)

        rho_out_b[:,:,ii,jj]=rho_out_b[:,:,ii,jj]+np.array(rho_broad_full(0,(bvals[ii,jj]), 0,deltamval,p))

        omegaao=omegaao4_from_I(B_val,p)
        p['sd_delao']=sd_delao4_from_I(B_val,p)
        p['mean_delao']=deltaao_from_I(B_val,omegaao,p)
        p['mean_delam']=deltamac_from_I(B_val,p)
        p['nbath']=nbath_from_T(T,p,p['mean_delam'])
        p['Omega']= Omega_from_PdBm(P_pump,p)

        rho_out_b[:,:,ii,jj]=rho_out_b[:,:,ii,jj]+np.array(rho_broad_full(0,(bvals[ii,jj]), 0,deltamval,p))

        elapsed_time=time.time()-start_time
        #print('aout = ' +str(aoutvals[ii,jj])+', bout = ' + str(boutvals[ii,jj]))

    print('    ' + filename + ': '+str(ii) +', Time: ' + time.ctime() +', Elapsed: '+ str(elapsed_time))

    #np.savez(filename,boutvals=boutvals,binval=binval,deltamacvals=deltamacvals,deltamvals=deltamvals,p=p,rho_out_b=rho_out_b, rho_out_bout=rho_out_bout,bvals=bvals,bvals2=bvals2,rho13_out_b2=rho13_out_b2)
    np.savez(filename,binvals=binvals,I_val=I_val,B_val=B_val,deltamvals=deltamvals,p=p,bvals=bvals,rho_out_b=rho_out_b,P_pump=P_pump,P_mu=P_mu,T=T,freq_pump_vals=freq_pump_vals,elapsed_time=elapsed_time)
print('    ===========================Complete===========================')

delfreqmu_vals=deltamvals/(2*np.pi)
rho13=rho_out_b[0,2,:,:]
boutvals=bvals*np.sqrt(p['gammamc'])
freq_pump_vals=freq_pump_vals+p['freqmu']

fig=plt.figure(filename)

fig.clf()
ax=fig.add_subplot(3,2,1)
img1=ax.imshow(np.log10(np.abs(rho13[:,:])**2),extent=(np.min(delfreqmu_vals),np.max(delfreqmu_vals),np.min(freq_pump_vals),np.max(freq_pump_vals)),aspect='auto',origin='lower')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('$\nu$')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,2)
img1=ax.imshow(10*np.log10(np.abs(boutvals[:,:]/binvals)**2),extent=(np.min(delfreqmu_vals),np.max(delfreqmu_vals),np.min(freq_pump_vals),np.max(freq_pump_vals)),aspect='auto',origin='lower')
plt.title('bout')
plt.xlabel('delta_mu')
plt.ylabel('$\nu$')

fig.colorbar(img1)

ax=fig.add_subplot(3,2,3)
img1=ax.imshow((np.abs(rho13[:,:])**2),extent=(np.min(delfreqmu_vals),np.max(delfreqmu_vals),np.min(freq_pump_vals),np.max(freq_pump_vals)),aspect='auto',origin='lower')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('$\nu$')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,4)
img1=ax.imshow((np.abs(boutvals[:,:])**2),extent=(np.min(delfreqmu_vals),np.max(delfreqmu_vals),np.min(freq_pump_vals),np.max(freq_pump_vals)),aspect='auto',origin='lower')
plt.title('bout')
plt.xlabel('delta_mu')
plt.ylabel('$\nu$')
fig.colorbar(img1)

#matplotlib.cm.register_cmap(name='twilight',cmap=twilight)


ax=fig.add_subplot(3,2,5)
img1=ax.imshow((np.angle(rho13[:,:])),extent=(np.min(delfreqmu_vals),np.max(delfreqmu_vals),np.min(freq_pump_vals),np.max(freq_pump_vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('$\nu$')
fig.colorbar(img1)
ax=fig.add_subplot(3,2,6)
img1=ax.imshow((np.angle(boutvals[:,:])),extent=(np.min(delfreqmu_vals),np.max(delfreqmu_vals),np.min(freq_pump_vals),np.max(freq_pump_vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title('bout')
plt.xlabel('delta_mu')
plt.ylabel('$\nu$')
fig.colorbar(img1)

fig=plt.figure(filename+'line')
#plt.plot(np.log10(np.abs(rho13[:,:])**2/10),freq_pump_vals)
plt.plot((np.abs(rho13[:,:])**2/10),freq_pump_vals)
fig=plt.figure(filename+'line2')
#plt.plot(np.log10(np.abs(rho13[:,:])**2/10),freq_pump_vals)
plt.plot(delfreqmu_vals,(np.abs(rho13[:,:].T)**2/10))

plt.show()
