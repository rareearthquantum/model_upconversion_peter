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
#from output_calcs.c_funs_test3 import rho_broad_full
#from c_linear_ground_slow_af import rho_broad_full,rho_broad_full_real_imag
from Linear1.c_linear_ground5 import rho_broad_full,rho_broad_full_real_imag

#filename='Linear_figs1/data_temp1'
filename='Linear_effic3_sim_test_sameN_'

#I_vals=np.linspace(7.43,7.55,31)
delta2vals=np.linspace(-1000e6,1000e6,301)
delta3vals=np.linspace(-15e10,15e10,301)
delta2vals=np.linspace(-250e6,000e6,21)[:-1]
delta3vals=np.linspace(-45e9,0e10,21)[:-1]
# delta2vals=np.linspace(-10e7,-2.5e7,20)
# delta3vals=np.linspace(-2.8e10,-0.6e10,20)

#delta2vals=np.linspace(-300e6,000e6,11)[:-1]/1000
#delta3vals=np.linspace(-45e9,0e10,11)[:-1]/1000
# delta2vals=np.linspace(-1000e6,000e6,11)[:-1]/10*3
# delta3vals=np.linspace(-4e10,0e10,11)[:-1]

#deltamucvals=np.linspace(-250e6,250e6,21)
P_pump = 10*np.log10(1.74) #in dBm, 1.74 mW are going into the resonator
#P_mu = -35-30 # in dBm
#P_pump = 0#10*np.log10(10.74) #in dBm, 1.74 mW are going into the resonator

T=10e-3


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
p['gammamc']=70e3*2*pi
p['gammami']=650e3*2*pi
Q=1e9
p['gammaoi']=2*pi*p['freq_pump']/Q
print('gammaoi = ' + str(p['gammaoi']))
# p['gammaoc']=2*pi*1.7e6
# p['gammaoi']=2*pi*7.95e6
# p['gammamc']=2*pi*0.0622e6
# p['gammami']=2*pi*5.69e6
muBohr=927.4009994e-26; # Bohr magneton in J/T in J* T^-1
p['mu12'] = 4.3803*muBohr # transition dipole moment for microwave cavity (J T^-1)


p['go'] = 51.9  #optical coupling

p['No'] = 1.28e16#*0 # number of atoms in the optical mode
p['No'] = 2.2e15#*0 # number of atoms in the optical mode


p['Nm'] = 2.2e16  #toal number of atoms
p['Nm']=p['No']
#p['No'] = 2e16#*0 # number of atoms in the optical mode
#p['Nm'] = 2e16  #toal number of atoms
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

p['Omega']= Omega_from_PdBm_resonance(P_pump,p)

print('Omega = ' + str(p['Omega']))
#print('bin   = ' + str(bin_from_PdBm(P_mu,p)))

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


def calc_efficiency(deloval,delmval,delocval,delmucval,p):
    #aout=Caa*ain+Cba*bin
    #bout=Cab*ain+Cbb*bin
    rho0,rhoa,rhoac,rhob,rhobc=rho_broad_full(deloval,delmval, 0,p)
    rho0=-rho0

    N_frac = (rho0[0,0]-rho0[1,1]) #effective fraction of atoms due to Temperature

    #print(N_frac)
    assert N_frac<=1
    delmucval=delmucval*N_frac
    Sa13=rhoa[0,2]*p['No']*p['go']
    #Sb13=rhob[0,2]*p['Nm']*p['go']
    Sb13=rhob[0,2]*p['Nm']*p['go']

    Sa12=rhoa[0,1]*p['No']*p['gm']
    #Sb12=rhob[0,1]*p['Nm']*p['gm']
    Sb12=rhob[0,1]*p['Nm']*p['gm']
    #print([Sa13,Sb13,Sa12,Sb12])
    #p['gammaoc']=-(p['gammaoi']+2*np.real(1j*Sa13))
    #p['gammamc']=-(p['gammami']+2*np.real(1j*Sb12))
    #p['gammaoc']=np.max([0,2*np.real(Sa12*Sb13/(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+p['gammamc'])/2))-2*np.real(1j*Sa13)-p['gammaoi']])
    #p['gammaoc']=np.max([2*np.real(Sa12*Sb13/(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+p['gammamc'])/2))-2*np.real(1j*Sa13)-p['gammaoi']])

    # if p['gammaoc']==0:
    #     print('Gammaoc = 0')
    # print(p['gammaoc'])
    # if p['gammaoc']<0:
    #     p['gammaoc']=p['gammaoc']
    Caa=p['gammaoc']*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+p['gammamc'])/2)/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+p['gammamc'])/2))
    Cba=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+p['gammamc'])/2))
    Cab=           -1j*np.sqrt(p['gammaoc'])*np.sqrt(p['gammamc'])*Sa12/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+p['gammamc'])/2))
    Cbb=p['gammamc']*(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc'])/2)/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+p['gammamc'])/2))
    return Caa, Cba, Cab, Cbb, N_frac

def effic_fun(x, p):
#x=[deloval,delmval,delocval,delmucval,mean_delao,mean_delao,gammaoc]
    deloval=x[0]*1e10
    delmval=x[1]*1e8
    delocval=x[2]*1e10
    delmucval=x[3]*1e8
    p['mean_delao']=x[4]*1e10
    p['mean_delam']=x[5]*1e8
    p['gammaoc']=x[6]*1e6
    p['nbath']=nbath_from_T(T,p)

    rho0,rhoa,rhoac,rhob,rhobc=rho_broad_full(deloval,delmval, 0,p)
    rho0=-rho0

    Sa13=rhoa[0,2]*p['No']*p['go']
    #Sb13=rhob[0,2]*p['Nm']*p['go']
    Sb13=rhob[0,2]*p['Nm']*p['go']

    Sa12=rhoa[0,1]*p['No']*p['gm']
    #Sb12=rhob[0,1]*p['Nm']*p['gm']
    Sb12=rhob[0,1]*p['Nm']*p['gm']
    Cba=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+p['gammamc'])/2))
    effic=-np.abs(Cba)**2
    return effic
def effic_fun2(x, p):
#x=[deloval,delmval,delocval,delmucval,mean_delao,mean_delao]
    deloval=x[0]*1e10
    delmval=x[1]*1e8
    delocval=x[2]*1e10
    delmucval=x[3]*1e8
    p['mean_delao']=x[4]*1e10
    p['mean_delam']=x[5]*1e8
    #p['gammaoc']=x[6]*1e6
    p['nbath']=nbath_from_T(T,p)

    rho0,rhoa,rhoac,rhob,rhobc=rho_broad_full(deloval,delmval, 0,p)
    rho0=-rho0

    Sa13=rhoa[0,2]*p['No']*p['go']
    #Sb13=rhob[0,2]*p['Nm']*p['go']
    Sb13=rhob[0,2]*p['Nm']*p['go']

    Sa12=rhoa[0,1]*p['No']*p['gm']
    #Sb12=rhob[0,1]*p['Nm']*p['gm']
    Sb12=rhob[0,1]*p['Nm']*p['gm']
    Cba=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+p['gammamc'])/2))
    effic=-np.abs(Cba)**2
    return effic
def effic_fun3(x, p):
    #x=[mean_delao,mean_delao]
    deloval=0
    delmval=0
    p['mean_delao']=x[0]*1e10
    p['mean_delam']=x[1]*1e8
    p['gammaoc']=x[2]*1e6

    rho0,rhoa,rhoac,rhob,rhobc=rho_broad_full(0,0, 0,p)
    rho0=-rho0
    N_frac = (rho0[0,0]-rho0[1,1])
    delocval=p['No']*p['go']**2/p['mean_delao']
    delmucval=p['No']*p['gm']**2/p['mean_delam']*N_frac
    #print(str(delocval)+', ' + str(delmucval))
    Sa13=rhoa[0,2]*p['No']*p['go']
    #Sb13=rhob[0,2]*p['Nm']*p['go']
    Sb13=rhob[0,2]*p['Nm']*p['go']

    Sa12=rhoa[0,1]*p['No']*p['gm']
    #Sb12=rhob[0,1]*p['Nm']*p['gm']
    Sb12=rhob[0,1]*p['Nm']*p['gm']
    Cba=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+p['gammamc'])/2))
    effic=-np.abs(Cba)**2
    return effic
def effic_fun5(x, p):
#x=[deloval,delmval,delocval,delmucval,mean_delao,mean_delao,gammaoc]
    deloval=x[0]*1e10
    delmval=x[1]*1e8
    delocval=x[2]*1e10
    delmucval=x[3]*1e8
    p['mean_delao']=x[4]*1e10
    p['mean_delam']=x[5]*1e8
    p['gammaoc']=x[6]*1e6
    p['nbath']=nbath_from_T(T,p)
    p['gammamc']=x[7]*1e4
    rho0,rhoa,rhoac,rhob,rhobc=rho_broad_full(deloval,delmval, 0,p)
    rho0=-rho0

    Sa13=rhoa[0,2]*p['No']*p['go']
    #Sb13=rhob[0,2]*p['Nm']*p['go']
    Sb13=rhob[0,2]*p['Nm']*p['go']

    Sa12=rhoa[0,1]*p['No']*p['gm']
    #Sb12=rhob[0,1]*p['Nm']*p['gm']
    Sb12=rhob[0,1]*p['Nm']*p['gm']
    Cba=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+p['gammamc'])/2))
    effic=-np.abs(Cba)**2
    return effic
p['sd_delao']=sd_delao_from_I(0,p)

x_bounds=scipy.optimize.Bounds([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,0],
                               [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
x_bounds=scipy.optimize.Bounds([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,0],
                               [0,0,0,0,0,0, np.inf])
x_bounds5=scipy.optimize.Bounds([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,0,0],
                               [0,0,0,0,0,0, np.inf,np.inf])

x_init= [0,0,p['No']*p['go']**2/-1.9e10*1e-10,p['No']*p['gm']**2/-0.6e8*1e-8,-1.9,-0.6,2*pi*1.7]
x_init2=[0,0,p['No']*p['go']**2/-1.9e10*1e-10,p['No']*p['gm']**2/-0.6e8*1e-8,-1.9,-0.6]
x_init5= [0,0,p['No']*p['go']**2/-1.9e10*1e-10,p['No']*p['gm']**2/-0.6e8*1e-8,-1.9,-0.6,2*pi*1.7,2*pi*7]
x_init5=[-2.45834604e-01, -8.75708881e-02, -2.85655608e-01, -5.88780978e-01,
       -1.80086724e+00, -5.84970326e-01,  2.66296129e+00,  1.48161895e+02]
x_init3=[-1.9,-0.6,2*pi*1.7]
x_init=np.array([-0.00834916, -0.06764364, -0.04772829, -0.54771599, -1.57842699, -0.58349296,  2.88029718])
x_init5=[-0.25404136, -0.08627085, -0.2974926 , -0.59215814, -1.69867529,-0.58171882, 10.58930213, 44.02155404]
x_init5=[-2.57796022e-01, -8.20858620e-02, -2.97300939e-01, -6.68378706e-01,
       -1.84386065e+00, -5.32713554e-01,  2.42334795e+00,  7.85417077e+02]
x_init5=[-2.63976142e-01, -8.23868198e-02, -3.03943300e-01, -6.78784963e-01,
       -1.83386327e+00, -5.26168730e-01,  2.07694512e+00,  9.24601326e+02]
x_init5=[-2.66527766e-01, -8.25525443e-02, -3.06543085e-01, -6.78675850e-01,
       -1.83473206e+00, -5.26265201e-01,  2.15837484e+00,  9.36858605e+02]

effic_result=scipy.optimize.minimize(effic_fun5,x_init5,args=(p),method='L-BFGS-B',bounds=x_bounds5)
print(effic_fun5(x_init5,p))
print(effic_result)
print('=======')
effic_result=scipy.optimize.minimize(effic_fun5,x_init5,args=(p),method='TNC',bounds=x_bounds5)
print(effic_fun5(x_init5,p))
print(effic_result)
print('=======')
effic_result=scipy.optimize.minimize(effic_fun5,effic_result.x,args=(p),method='TNC',bounds=x_bounds5,options = {'maxiter' : 400})
print(effic_fun5(x_init5,p))
print(effic_result)
print('=======')
#effic_result=scipy.optimize.minimize(effic_fun,x_init,args=(p),bounds=x_bounds)
# effic_result=scipy.optimize.minimize(effic_fun,x_init,args=(p),method='TNC',bounds=x_bounds)
# print(effic_fun(x_init,p))
# print(effic_result)
# print('=======')
# effic_result=scipy.optimize.minimize(effic_fun,effic_result.x,args=(p),method='TNC',bounds=x_bounds,options = {'maxiter' : 400})
# print(effic_fun(x_init,p))
# print(effic_result)
# print('=======')
#myfactr = 1e2
#effic_result2=scipy.optimize.minimize(effic_fun2,x_init2,args=(p),method='L-BFGS-B',options={'ftol' : myfactr * np.finfo(float).eps})
# effic_result2=scipy.optimize.minimize(effic_fun2,x_init2,args=(p),method='L-BFGS-B',options={'ftol' : myfactr * np.finfo(float).eps})
# print(effic_fun2(x_init2,p))
# print(effic_result2)
# print('=======')
# effic_result3=scipy.optimize.minimize(effic_fun3,x_init3,args=(p),bounds=((-np.inf,0),(-np.inf,0),(0,np.inf)))
# print(effic_fun3(x_init3,p))
# print(effic_result3)
#,method='Nelder-Mead'
    #np.savez(filename,binvals=binvals,delta2vals=delta2vals,p=p,bvals=bvals,deltamucvals=deltamucvals,rho_out=rho_out,P_pump=P_pump,P_mu=P_mu,T=T,elapsed_time=elapsed_time,avals=avals,deltaocval=deltaocval,calc_time=calc_time)
    #np.savez(filename,p=p,P_mu=P_mu,P_pump=P_pump,T=T,elapsed_time=elapsed_time,calc_time=calc_time,binvals=binvals,ainval=ainval,bvals=bvals,avals=avals,delta2vals=delta2vals,delta3vals=delta3vals)
    #np.savez(filename,Caa=Caa, Cba=Cba,Cab=Cab,Cbb=Cbb,p=p,P_pump=P_pump,T=T,elapsed_time=elapsed_time,calc_time=calc_time,delta2vals=delta2vals,delta3vals=delta3vals)
