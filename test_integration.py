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
from output_calcs.c_funs_test3 import steady_rho_single_c, gauss_fun_1d
# steady_rho_single_c(double delta_a_o, double delta_a_mu, double complex aval,double complex bval,double delta_o,double delta_mu,p):
#gauss_fun_1d(double delval,double mean_del, double sd):
def steady_rho_gauss_single(delao,delam,aval,bval,delo,delm,p):
    S_out_full=steady_rho_single_c(delao,delam,aval,bval,delo,delm,p)
    rho1=[[S_out_full[0],S_out_full[3]+1j*S_out_full[4],S_out_full[5]+1j*S_out_full[6]],[S_out_full[3]-1j*S_out_full[4],S_out_full[1],S_out_full[7]+1j*S_out_full[8]],[S_out_full[5]-1j*S_out_full[6],S_out_full[7]-1j*S_out_full[8],S_out_full[2]]]

    return np.array(rho1)#*gauss_fun_1d(delam,p['mean_delam'],p['sd_delam'])*gauss_fun_1d(delao,p['mean_delao'],p['sd_delao'])*1e12
def steady_rho_gauss(delaovals,delamvals,aval,bval,delo,delm,p):
    rhovals=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
    for ii, delao in enumerate(delaovals):
        for jj, delam in enumerate(delamvals):
            rhovals[:,:,ii,jj]=steady_rho_gauss_single(delao,delam,aval,bval,delo,delm,p)
    return rhovals

B_vals=np.linspace(0.205,0.210,15)
deltamvals=np.linspace(-50e6,50e6,15)*2*np.pi
P_pump = 10*np.log10(10.74) #in dBm, 1.74 mW are going into the resonator
P_mu = 3-30 # in dBm
T=190e-3

p={}
p['freqmu']=5015e6#4733e6 #this is the microwave cavity frequency
p['freq_pump'] = 1951170.00e9 #pump frequency
#p['freqo']=p['freqmu']+p['freq_pump']

#p['Gg']=0.024085156244027e12
#p['Ge']=0.017976119414574e12
p['Ge']=0.020275e12
p['Gg']=0.0241886e12
p['f0_no_B']=195.1167943776907e12
p['d13'] = 2e-32*math.sqrt(1/3)
p['d23'] = 2e-32*math.sqrt(2/3)
p['gamma13'] = p['d13']**2/(p['d13']**2+p['d23']**2)*1/11e-3
p['gamma23'] = p['d23']**2/(p['d13']**2+p['d23']**2)*1/11e-3
p['gamma2d'] = 1e6
p['gamma3d'] = 1e6
#p['nbath'] = 20
p['gammamu'] = 1/11#e3
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
#p['No'] = 1.3e15 # number of atoms in the optical mode
#p['Nm'] = 2e16  #toal number of atoms

p['gm'] = 1.04 #coupling between atoms and microwave field

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
    return 2*pi*(p['freqmu']+p['freq_pump'])-omegaao_from_I(B_mag,p)
B_val=0.2
p['Omega']= Omega_from_PdBm(P_pump,p)
p['mean_delam']=deltamac_from_I(B_val,p)
p['nbath']=nbath_from_T(T,p,deltamacval=p['mean_delam'])
p['mean_delao']=deltaao_from_I(B_val,p)
p['sd_delao']=sd_delao_from_I(B_val,p)
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
def lines_m(delaovals,delo,delmval,bval,p):
    mlines=np.zeros((13,len(delaovals)))
    for ii, delao in enumerate(delaovals):
        ds_m_val=find_dressed_states_m(delao,delo,delmval,bval,p)
        mlines[:,ii]=sorted([-50*p['sd_delam']+p['mean_delam'],50*p['sd_delam']+p['mean_delam'],p['mean_delam'],p['sd_delam']+p['mean_delam'] ,-p['sd_delam']+p['mean_delam'],delmval,delmval+2*p['gamma2d']
                        ,delmval-2*p['gamma2d'],ds_m_val, ds_m_val+2*p['gammamu'],ds_m_val-2*p['gammamu'], ds_m_val+2*p['gamma2d'],ds_m_val-2*p['gamma2d']])
    return mlines.T
def lines_m(delaovals,delo,delmval,bval,p):
    mlines=np.zeros((9,len(delaovals)))
    for ii, delao in enumerate(delaovals):
        ds_m_val=find_dressed_states_m(delao,delo,delmval,bval,p)
        mlines[:,ii]=sorted([delmval,delmval+2*p['gamma2d']
                        ,delmval-2*p['gamma2d'],ds_m_val, ds_m_val+2*p['gammamu'],ds_m_val-2*p['gammamu'], ds_m_val+2*p['gamma2d'],ds_m_val-2*p['gamma2d'],delao-delm])
    return mlines.T
def lines_m(delaovals,deloval,delmval,bval,p):
    mlines=np.zeros((4,len(delaovals)))
    for ii, delaoval in enumerate(delaovals):
        ds_m_val=find_dressed_states_m(delaoval,deloval,delmval,bval,p)
        #mlines[:,ii]=[-(p['gm']*bval)**2/delao*2,delao,delao/2+np.sqrt(delao**2/4-(p['gm']*bval)**2),delao/2-np.sqrt(delao**2/4-(p['gm']*bval)**2),ds_m_val]
        #delao=delao+p['gamma2d']*10000
        mlines[:,ii]=np.array([-(p['gm']*bval)**2/(delaoval-deloval)*2+delmval,delaoval-deloval+delmval,(delaoval-deloval)/2+np.sqrt((delaoval-deloval)**2/4-(p['gm']*bval)**2)+delmval,(delaoval-deloval)/2-np.sqrt((delaoval-deloval)**2/4-(p['gm']*bval)**2)+delmval])-p['gm']*bval
    mlines[abs(mlines)>max(delaovals)]=np.nan
    return mlines.T

bval=bin_from_PdBm(P_mu,p)#/np.sqrt(p['gammamc'])*0.04
delamvals=np.linspace(-1,1,201)*1e10
delaovals=np.linspace(-1,1,201)*1e10
delm=3e9
delo=6e9
#steady_rho_gauss(delaovals,delamvals,aval,bval,delo,delm,p):
rhovals=steady_rho_gauss(delaovals,delamvals,0,bval,delo,delm,p)
mlines=lines_m(delaovals,delo,delm,bval,p)
# ds_m=np.zeros(len(delaovals))
# for ii, delao in enumerate(delaovals):
#     ds_m[ii]=find_dressed_states_m(delao,delo,delm,bval,p)
fig=plt.figure()

fig.clf()


ax=fig.add_subplot(1,2,1)
img1=ax.imshow((np.abs(rhovals[0,1,:,:])),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
plt.plot(mlines,delaovals)#,color='red')
#plt.plot(ds_m,delaovals)

plt.xlabel('del mu')
plt.ylabel('del o')
fig.colorbar(img1)

ax=fig.add_subplot(1,2,2)
img1=ax.imshow(np.angle(rhovals[0,1,:,:]),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
plt.plot(mlines,delaovals,color='red')
plt.xlabel('del mu')
plt.ylabel('del o')
fig.colorbar(img1)
plt.show()
