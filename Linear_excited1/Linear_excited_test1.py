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


filename='ground_params2_test20'
I_vals=np.linspace(7.43,7.55,31)
deltamvals=np.linspace(-20e6,20e6,31)*2*np.pi
P_pump = 10*np.log10(1.74) #in dBm, 1.74 mW are going into the resonator
P_mu = -15-30 # in dBm
T=500e-3


p={}
p['freqmu']=4732e6 #this is the microwave cavity frequency
p['freq_pump'] = 195117.044e9 #pump frequency
#p['freqo']=p['freqmu']+p['freq_pump']
#p['Gg']=0.024085156244027e12
#p['Ge']=0.017976119414574e12
p['Ge']=0.020275e12
p['Gg']=0.0241886e12
p['f0_no_B']=195.1167943776907e12
p['d13'] = 2e-32*math.sqrt(2/3)
p['d12'] = 2e-32*math.sqrt(1/3)
p['gamma13'] = p['d13']**2/(p['d13']**2+p['d12']**2)*1/11e-3
p['gamma12'] = p['d12']**2/(p['d13']**2+p['d12']**2)*1/11e-3*10
p['gamma2d'] = 1e6
p['gamma3d'] = 1e6
#p['nbath'] = 20
p['gamma23'] = 1/11#1/(p['nbath']+1) * 1e3
p['sd_delam']=2*pi*2e6
p['gammaoc']=2*pi*1.7e6
p['gammaoi']=2*pi*7.95e6
p['gammamc']=2*pi*0.1522e6
p['gammami']=2*pi*3.255e6


muBohr=927.4009994e-26; # Bohr magneton in J/T in J* T^-1
p['mu12'] = 4.3803*muBohr # transition dipole moment for microwave cavity (J T^-1)


p['go'] = 51.9  #optical coupling

p['No_total'] = 2.2e15 # number of atoms in the optical mode

p['Nm_total'] = 6e16  #toal number of atoms

p['gm'] = 1.04/10 #coupling between atoms and microwave field
p['T']=T
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
    Omega=p['d12']*Efield/hbar
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
def deltamac_from_I(B_mag,p):
    #B_mag=(0.027684*I_mag*1e3-0.056331)*1e-3
    deltamacvals=(p['Ge'])*(B_mag)-p['freqmu']
    return deltamacvals*2*np.pi
def omegaao_from_I(B_mag,p): #transition 2
    return (p['f0_no_B']+(-p['Gg']-p['Ge'])/2*B_mag)*2*np.pi
def sd_delao_from_I(B_mag,p):
    return 1e9*(0.148588212918272+0.308180352441052*B_mag)*np.pi*2
#    return 1e9*(0.391366413926165+0.137444967951503*B_mag)*np.pi*2

def deltaao_from_I(B_mag,p):
    return 2*pi*(p['freqmu']+p['freq_pump'])-omegaao_from_I(B_mag,p)

p['Omega']= Omega_from_PdBm(P_pump,p)

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
gamma13,gamma23,gamma2d,gamma3d,nbath,gamma12=sym.symbols('gamma_13 gamma_23 gamma_2d gamma_3d n_b gamma_12', real=True, negative=False) #energy decay for atom levels
rho11, rho12, rho13, rho21, rho22, rho23, rho31, rho32, rho33=sym.symbols('rho_11 rho_12 rho_13 rho_21 rho_22 rho_23 rho_31 rho_32 rho_33') #Density matrix elements
a, b = sym.symbols('a b') #classical amplitudes of the optical and microwave fields
#ar,ai=sym.symbols('a_r a_i', real=True)
go, gm=sym.symbols('g_o, g_mu',real=True, negative=False) #coupling strengths for optical and microwave fields
Omega=sym.symbols('Omega', real=True) #pump Rabi frequency
lam=sym.symbols('lambda')
a, b = sym.symbols('a b') #classical amplitudes of the optical and microwave fields
ar,ai=sym.symbols('a_r a_i', real=True)
br,bi=sym.symbols('b_r b_i', real=True)


H_sys=Omega*s21+gm*s32*b+go*s31*a
H_sys=H_sys+Dagger(H_sys)
H_sys=H_sys+(delao -delo)*s33+(delam-delm)*s22

LH=-I*spre(H_sys)+I*spost(H_sys)
L32 = gamma23*(nbath+1)*collapse(s23)
L23 = gamma23*nbath*collapse(s32)

L21 = gamma12*collapse(s12)
L31 = gamma13*collapse(s13)
L22 = gamma2d*collapse(s22)
L33 = gamma3d*collapse(s33)

L=LH + L21 + L23 + L32 + L31 + L22 + L33
print(sym.latex(L))
L = L.row_insert(0,Matrix([[1,0,0,0,1,0,0,0,1]]))
L.row_del(1)

#define the density matrix in square and row form
#the row form is so the Liovillian in matrix form can be acted on it
rho = Matrix([[rho11,rho21,rho31],[rho12,rho22,rho32],[rho13,rho23,rho33]])
rho = 1*rho.T #because we are using "fortran" style matrix flatteneing
rho[:]
rhoflat = 1*rho.T
rhoflat = rhoflat[:]

L0=L.subs({a:0,b:0})
# La=(L-L0).subs({b:0,sym.conjugate(a):0}).subs({a:1})
# Lac=(L-L0).subs({b:0,sym.conjugate(a):1})-a*La
# Lb=(L-L0).subs({a:0,sym.conjugate(b):0}).subs({b:1})
# Lbc=(L-L0).subs({a:0,sym.conjugate(b):1})-b*Lb
Lb1=(L-L0).subs({a:0,sym.conjugate(b):0}).subs({b:1})
La=sym.simplify((L-L0).subs({b:0,sym.conjugate(a):ar-I*ai}).subs({a:ar+I*ai}))
Lb=sym.simplify((L-L0).subs({a:0,sym.conjugate(b):br-I*bi}).subs({b:br+I*bi}))
#Laa=sym.simplify((L-L0).subs({b:0,sym.conjugate(a):0}).subs({a:ar+I*ai}))
#Lbb=sym.simplify((L-L0).subs({a:0,sym.conjugate(b):0}).subs({b:br+I*bi}))

Lar=La.subs({ai:0,ar:1})
Lai=La.subs({ar:0,ai:1})
Lbr=Lb.subs({bi:0,br:1})
Lbi=Lb.subs({br:0,bi:1})
#print(La)

LGamma=L-LH
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
#L_real=sym.re(CtoR*L*CtoR.inv())
L_real=(CtoR*L*CtoR.inv())

L0_real=sym.simplify(sym.expand(L_real.subs({a:0,b:0})))
#L0_real=(L_real.subs({a:0,b:0}))

La_real=sym.simplify(sym.expand((L_real-L0_real).subs({a:ar+I*ai,b:0})))
Lb_real=sym.simplify(sym.expand((L_real-L0_real).subs({b:br+I*bi,a:0})))
Lar_real=La_real.subs({ar:1,ai:0})
Lai_real=La_real.subs({ai:1,ar:0})
Lbr_real=Lb_real.subs({br:1,bi:0})
Lbi_real=Lb_real.subs({bi:1,br:0})
#print(Lb1*Lb1)
#print(sym.simplify(L0.inv()))
print(L_real)
print(L0_real)
print(La_real)
print(Lb_real)
print(Lar_real)
print(Lai_real)
print(Lbr_real)
print(Lbi_real)
# L0_real_inv=L0_real.inv()
# print(L0_real_inv)
# print(sym.simplify(CtoR*La*CtoR.inv()))#.subs({ai:0,ar:1}))
# print(CtoR*Lar*CtoR.inv())
# print(CtoR*Lai*CtoR.inv())
# print(CtoR*CtoR)
#print(CtoR*Lbr*CtoR.inv())
#print(CtoR*Lbi*CtoR.inv())
#print(CtoR.inv())
#print(sym.simplify(CtoR*Lb*CtoR.inv()))
# L0_fun=sym.lambdify((delo,delm,gamma13, gamma23, gamma2d, gamma3d, nbath,gammamu,Omega,go,gm),L0)
# La_fun=sym.lambdify((go),La)
# Lb_fun=sym.lambdify((gm),Lb)
# Lac_fun=sym.lambdify((go),Lac)
# Lbc_fun=sym.lambdify((gm),Lbc)
# LGamma_fun=sym.lambdify((gamma13, gamma23, gamma2d, gamma3d, nbath,gammamu,Omega,go,gm),LGamma)
#def L0inv(deloval,delmval,p):
#    return Matrix(L0_fun(deloval,delmval,p['gamma13'],p['gamma23'],p['gamma2d'],p['gamma3d'], p['nbath'],p['gammamu'],p['Omega'],p['go'],p['gm'])).inv()
#print(L0inv(1e8,-2e8,p))
def print_Lreal(L, Lname):
#    f1=open('./Linear1/Ls.txt', 'a')
    f1=open('Ls.txt', 'a')

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
    Lreal=sym.simplify(CtoR*L*CtoR.inv())
    Lflat = 1*Lreal.T
    Lflat = Lflat[:]
    for ii in range(len(L)):
        f1.write(Lname+ '['+str(ii)+'] = ' +str(Lflat[ii])+'\n')
        print(Lname+ '['+str(ii)+'] = ' +str(Lflat[ii]))
    f1.close()
def print_L(L, Lname):
#    f1=open('./Linear1/Ls.txt', 'a')
    f1=open('Ls_excited1.txt', 'a')

    # CtoR = Matrix([[2,0,0,0,0,0,0,0,0],
    #            [0,0,0,0,2,0,0,0,0],
    #            [0,0,0,0,0,0,0,0,2],
    #            [0,1,0,1,0,0,0,0,0],
    #            [0,I,0,-I,0,0,0,0,0],
    #            [0,0,1,0,0,0,1,0,0],
    #            [0,0,I,0,0,0,-I,0,0],
    #            [0,0,0,0,0,1,0,1,0],
    #            [0,0,0,0,0,I,0,-I,0]
    #           ])
    # CtoR=CtoR/2
    # Lreal=sym.simplify(CtoR*L*CtoR.inv())
    Lflat = 1*L.T
    Lflat = Lflat[:]
    for ii in range(len(L)):
        f1.write(Lname+ '['+str(ii)+'] = ' +str(Lflat[ii])+'\n')
        print(Lname+ '['+str(ii)+'] = ' +str(Lflat[ii]))
    f1.close()
#print_Lreal(L0,'L0')
#print_Lreal(Lar,'Lar')
#print_Lreal(Lai,'Lai')
#print_Lreal(Lbr,'Lbr')
#print_Lreal(Lbi,'Lbi')
# print_Lreal(Lb,'Lb')
# print_Lreal(Lac,'Lac')
# print_Lreal(Lbc,'Lbc')


# print_L(L0_real,'L0')
# print_L(Lar_real,'Lar')
# print_L(Lai_real,'Lai')
# print_L(Lbr_real,'Lbr')
# print_L(Lbi_real,'Lbi')
