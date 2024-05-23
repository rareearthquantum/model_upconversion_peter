#importing python libs

# import sympy as sym
# sym.init_printing()

import numpy as np
import sys
sys.path.append('/home/peter/model_upconversion')
from math import pi
import math
import matplotlib.pyplot as plt
# from sympy import I, Matrix, symbols
# from sympy.physics.quantum import TensorProduct, Dagger
import scipy.optimize
# import scipy.integrate
# import scipy.constants as const

#import qutip

from matplotlib.colors import Normalize as Norm

import time
from Linear_excited1.c_linear_excited3 import rho_broad_full
# from c_linear_excited1 import rho_broad_full

from Thesis_figs.Excited_params import p, sd_delao_from_B, deltaao_from_B

filename='Linear_effic_excited_sameN_'

#I_vals=np.linspace(7.43,7.55,31)
delta2vals=np.linspace(-1000e6,1000e6,31)
delta3vals=np.linspace(-15e10,15e10,31)
delta2vals=np.linspace(-250e6,250e6,21)*10
delta3vals=np.linspace(-1e10,1e10,21)
# delta2vals=np.linspace(-250e6,250e6,11)#[:-1]
# delta3vals=np.linspace(-45e9,45e9,11)#[:-1]
# delta2vals=np.linspace(-10e7,-2.5e7,20)
# delta3vals=np.linspace(-2.8e10,-0.6e10,20)

#delta2vals=np.linspace(-300e6,000e6,11)[:-1]/1000
#delta3vals=np.linspace(-45e9,0e10,11)[:-1]/1000
# delta2vals=np.linspace(-1000e6,000e6,11)[:-1]/10*3
# delta3vals=np.linspace(-4e10,0e10,11)[:-1]
# delta2vals=np.linspace(-200e6,000e6,31)[:-1]
# delta3vals=np.linspace(-25e9,0e10,31)[:-1]
delta2vals=np.linspace(-250e6,0,21)[:-1]*10*3
delta3vals=np.linspace(-1e10,0,21)[:-1]
# delta2vals=np.linspace(-1.2e8,-0.5e8,40)#[:-1]?
# delta3vals=np.linspace(-1.7e10,-1.45e10,10)#[:-1]
P_pump_mW=10.74
P_pump = 10*np.log10(P_pump_mW) #in dBm, 1.74 mW are going into the resonator
T=100e-3
Q=1e9
# p['gammaoi']=2*pi*p['freq_pump']/Q
print('gammaoi = ' + str(p['gammaoi']))
# p['Nm']=p['No']
# p['No']=p['Nm']
N=1e16
p['Nm']=N
p['No']=N
p['gammami']=650e3*2*pi*1e0

filename=filename+'T='+str(T)+'_Q='+str(Q)+'_Ppump='+str(P_pump_mW)
del3_start=-5.5 # *1e9
del2_start=-1.25 # *1e9

filename=filename+'T='+str(T)+'_Q='+str(Q)+'_Ppump='+str(P_pump_mW)


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
print(p['Omega'])
# p['Omega']= Omega_from_PdBm(P_pump,p)*10
print(p['Omega'])

def calc_efficiency(deloval,delmval,delocval,delmucval,p):
    #aout=Caa*ain+Cba*bin
    #bout=Cab*ain+Cbb*bin
    rho0,rhoa,rhoac,rhob,rhobc=rho_broad_full(deloval,delmval, 0,p)

    Sa13=rhoa[0,2]*p['No']*p['go']
    Sb13=rhob[0,2]*p['Nm']*p['go']

    Sa23=rhoa[1,2]*p['No']*p['gm']
    Sb23=rhob[1,2]*p['Nm']*p['gm']

    Caa=p['gammaoc']*(1j*Sb23-1j*(delmval-delmucval)+(p['gammami']+p['gammamc']*2)/2)/(Sa23*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc']*2)/2)*(1j*Sb23-1j*(delmval-delmucval)+(p['gammami']+p['gammamc']*2)/2))
    Cab=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa23*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc']*2)/2)*(1j*Sb23-1j*(delmval-delmucval)+(p['gammami']+p['gammamc']*2)/2))
    Cba=           -1j*np.sqrt(p['gammaoc'])*np.sqrt(p['gammamc'])*Sa23/(Sa23*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc']*2)/2)*(1j*Sb23-1j*(delmval-delmucval)+(p['gammami']+p['gammamc']*2)/2))
    Cbb=p['gammamc']*(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc']*2)/2)/(Sa23*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc']*2)/2)*(1j*Sb23-1j*(delmval-delmucval)+(p['gammami']+p['gammamc']*2)/2))

    return Caa, Cba, Cab, Cbb
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

    Sa13=rhoa[0,2]*p['No']*p['go']
    Sb13=rhob[0,2]*p['Nm']*p['go']

    Sa23=rhoa[1,2]*p['No']*p['gm']
    Sb23=rhob[1,2]*p['Nm']*p['gm']
    Cab=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa23*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc']*2)/2)*(1j*Sb23-1j*(delmval-delmucval)+(p['gammami']+p['gammamc']*2)/2))
    effic=-np.abs(Cab)**2
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

    Sa13=rhoa[0,2]*p['No']*p['go']
    Sb13=rhob[0,2]*p['Nm']*p['go']

    Sa23=rhoa[1,2]*p['No']*p['gm']
    Sb23=rhob[1,2]*p['Nm']*p['gm']
    Cab=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa23*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc']*2)/2)*(1j*Sb23-1j*(delmval-delmucval)+(p['gammami']+p['gammamc']*2)/2))
    effic=-np.abs(Cab)**2
    return effic

def effic_fun5(x, p):
#x=[deloval,delmval,delocval,delmucval,mean_delao,mean_delao,gammaoc]
    deloval=x[0]*1e9
    delmval=x[1]*1e9
    delocval=x[2]*1e9
    delmucval=x[3]*1e9
    p['mean_delao']=x[4]*1e9
    p['mean_delam']=x[5]*1e9
    p['gammaoc']=x[6]*1e6
    p['nbath']=nbath_from_T(T,p)
    p['gammamc']=x[7]*1e6
    rho0,rhoa,rhoac,rhob,rhobc=rho_broad_full(deloval,delmval, 0,p)

    Sa13=rhoa[0,2]*p['No']*p['go']
    Sb13=rhob[0,2]*p['Nm']*p['go']

    Sa23=rhoa[1,2]*p['No']*p['gm']
    Sb23=rhob[1,2]*p['Nm']*p['gm']
    Cab=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa23*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc']*2)/2)*(1j*Sb23-1j*(delmval-delmucval)+(p['gammami']+p['gammamc']*2)/2))
    effic=-np.abs(Cab)**2
    return effic

def effic_fun6(x, p):
#x=[deloval,delmval,delocval,delmucval,mean_delao,mean_delao,gammaoc]
    # deloval=x[0]*1e10
    # delmval=x[1]*1e8
    deloval=0
    delmval=0
    delocval=x[0]*1e9
    delmucval=x[1]*1e9
    p['mean_delao']=x[2]*1e9
    p['mean_delam']=x[3]*1e9
    p['gammaoc']=x[4]*1e6
    p['nbath']=nbath_from_T(T,p)
    p['gammamc']=x[5]*1e6
    rho0,rhoa,rhoac,rhob,rhobc=rho_broad_full(deloval,delmval, 0,p)

    Sa13=rhoa[0,2]*p['No']*p['go']
    Sb13=rhob[0,2]*p['Nm']*p['go']

    Sa23=rhoa[1,2]*p['No']*p['gm']
    Sb23=rhob[1,2]*p['Nm']*p['gm']
    Cab=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa23*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc']*2)/2)*(1j*Sb23-1j*(delmval-delmucval)+(p['gammami']+p['gammamc']*2)/2))
    effic=-np.abs(Cab)**2
    return effic
def effic_fun_Ppump(x, p):
#x=[deloval,delmval,delocval,delmucval,mean_delao,mean_delao,gammaoc]
    # deloval=x[0]*1e10
    # delmval=x[1]*1e8
    deloval=0
    delmval=0
    delocval=x[0]*1e9
    delmucval=x[1]*1e9
    p['mean_delao']=x[2]*1e9
    p['mean_delam']=x[3]*1e9
    p['gammaoc']=x[4]*1e6
    p['nbath']=nbath_from_T(T,p)
    p['gammamc']=x[5]*1e6
    p['Omega']=Omega_cavity_from_PdBm_resonance(10*np.log10(x[6]),p)
    rho0,rhoa,rhoac,rhob,rhobc=rho_broad_full(deloval,delmval, 0,p)

    Sa13=rhoa[0,2]*p['No']*p['go']
    Sb13=rhob[0,2]*p['Nm']*p['go']

    Sa23=rhoa[1,2]*p['No']*p['gm']
    Sb23=rhob[1,2]*p['Nm']*p['gm']
    Cab=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa23*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+p['gammaoc']*2)/2)*(1j*Sb23-1j*(delmval-delmucval)+(p['gammami']+p['gammamc']*2)/2))
    effic=-np.abs(Cab)**2
    return effic
p['sd_delao']=sd_delao_from_B(0.2,p)
#[-3.88189826e+00, -5.07617478e-05, -5.47922352e+00, -5.12990082e-01,  2.72373617e+01,  2.01075238e+00,  1.07572598e+01]
x_bounds7=scipy.optimize.Bounds([-np.inf,-np.inf,-np.inf,-np.inf,0,0,0],
                               [np.inf,np.inf,0,0, np.inf,np.inf,np.inf])
x_init7=[-0.21213398, -1.2439472 , -1.30044741, -0.87462797,  3.84276068,5.11996611,10.74]
x_init7=[-0.20483753, -0.94468276, -1.35354733, -1.14431014,  3.91142365,5.19283831, 10.76086523]
x_init7=[-3.89236980e+00, -6.42610807e-05, -5.48630094e+00, -4.77968745e-01, 2.72342916e+01,  2.00916849e+00, 10.76086523]
x_init7=[-3.87946779e+00, -3.57419988e-05, -5.47553082e+00, -5.13381287e-01,2.72343327e+01,  2.00917272e+00,  1.07600636e+01]
minimizer_kwargs={"args":p,"bounds":x_bounds7,"method":'TNC'}
print(effic_fun_Ppump(x_init7,p))
effic_result=scipy.optimize.basinhopping(effic_fun_Ppump,x_init7,minimizer_kwargs=minimizer_kwargs,niter=100)#,bounds=x_bounds7)
print(effic_result)
print('=======')
print(effic_fun_Ppump(effic_result.x,p))
effic_result=scipy.optimize.basinhopping(effic_fun_Ppump,effic_result.x,minimizer_kwargs=minimizer_kwargs,niter=1000)#,bounds=x_bounds7)
print(effic_result)
print('=======')
print(effic_fun_Ppump(effic_result.x,p))
effic_result=scipy.optimize.basinhopping(effic_fun_Ppump,effic_result.x,minimizer_kwargs=minimizer_kwargs,niter=1000)#,bounds=x_bounds7)
print(effic_result)
print('=======')
print(effic_fun_Ppump(effic_result.x,p))
effic_result=scipy.optimize.basinhopping(effic_fun_Ppump,effic_result.x,minimizer_kwargs=minimizer_kwargs,niter=1000)#,bounds=x_bounds7)
print(effic_result)
print('=======')
# p['gammami']=650e3*2*pi
x_bounds=scipy.optimize.Bounds([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,0],
                               [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
x_bounds=scipy.optimize.Bounds([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,0],
                               [0,0,0,0,0,0, np.inf])
x_bounds5=scipy.optimize.Bounds([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,0,0],
                               [0,0,0,0,0,0, np.inf,np.inf])
x_bounds5=scipy.optimize.Bounds([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,0,0],
                               [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf, np.inf,np.inf])
x_init= [0,0,p['No']*p['go']**2/-1.9e10*1e-10,p['No']*p['gm']**2/-0.6e8*1e-8,-1.9,-0.6,2*pi*1.495]
# x_init2=[0,0,p['No']*p['go']**2/-1.9e10*1e-10,p['No']*p['gm']**2/-0.6e8*1e-8,-1.9,-0.6]
x_init5= [0,0,p['No']*p['go']**2/-1.6e10*1e-10,p['Nm']*p['gm']**2/-0.67e8*1e-8,-1.9,-0.6,2*pi*1.7,2*pi*1.495]

x_init5= [0,0,p['No']*p['go']**2/(del3_start*1e9)*1e-9,0,del3_start,del2_start,2*pi*1.7,2*pi*1.24]
x_init5=[-0.0188012 , -0.04345091, -4.21304757, -0.04358456, -5.90210988, -0.8551584 , 11.90889144,  6.85664497]
x_init5=[-0.04078435, -0.08394382, -4.21621165, -0.0839935 , -5.39586323,-0.62303809, 13.04960999,  5.24651269]
# x_init5=[ 8.44748325e-03, -2.88543496e-05, -7.75123905e-01, -2.11970458e-06,-3.62182316e-01, -1.998634062*pi*1.24e6e+01,  1.06822739e+01,  5.90530069e+00]
# x_init5=[-2.41440879e-02, -2.86967503e-04, -7.53504439e-01, -5.04264743e+00, -1.83670464e+00, -1.06379858e+00,  9.08692977e+00,  1.21162981e+01]
# x_init5=[-0.36652207, -0.01312022, -0.42073976, -0.47262512, -1.52882879,-0.6001007 ,  3.96190541,  9.26707012]
# x_init5=[-0.0307309 ,  0.        , -0.92903791, -7.43879804, -1.81110662, -0.87703357,  9.26503864, 10.49590959]
# x_init5=[0,0,-0.21213398, -1.2439472 , -1.30044741, -0.87462797,  3.84276068,5.11996611]
# ########################
# print(effic_fun5(x_init5,p))
# effic_result=scipy.optimize.minimize(effic_fun5,x_init5,args=(p),bounds=x_bounds5)
# print(effic_result)
# print('=======')
# print(effic_fun5(effic_result.x,p))
# effic_result=scipy.optimize.minimize(effic_fun5,effic_result.x,args=(p),bounds=x_bounds5)
# print(effic_result)
# print('=======')
# print(effic_fun5(effic_result.x,p))
# effic_result=scipy.optimize.minimize(effic_fun5,effic_result.x,args=(p),method='L-BFGS-B',bounds=x_bounds5)
# print(effic_result)
# print('=======')
# effic_result=scipy.optimize.minimize(effic_fun5,effic_result.x,args=(p),method='TNC',bounds=x_bounds5)
# print(effic_fun5(effic_result.x,p))
# print(effic_result)
# print('=======')
# #####################
# effic_result=scipy.optimize.minimize(effic_fun5,effic_result.x,args=(p),method='TNC',bounds=x_bounds5,options = {'maxiter' : 400})
# print(effic_fun5(x_init5,p))
# print(effic_result)
# print('=======')
# ###########################

# x_init6=[-3.89236980e+00, -6.42610807e-05, -5.48630094e+00, -4.77968745e-01, 2.72342916e+01,  2.00916849e+00]
# x_init6= [p['No']*p['go']**2/(del3_start*1e9)*1e-9,0,del3_start,del2_start,2*pi*1.7,2*pi*1.24]
x_bounds6=scipy.optimize.Bounds([-np.inf,-np.inf,-np.inf,-np.inf,0,0],
                               [np.inf,np.inf,0,0, np.inf,np.inf])
print(effic_fun6(x_init6,p))
effic_result=scipy.optimize.minimize(effic_fun6,x_init6,args=(p),bounds=x_bounds6)
print(effic_result)
# print('=======')
# print(effic_fun6(effic_result.x,p))
# effic_result=scipy.optimize.minimize(effic_fun6,effic_result.x,args=(p),bounds=x_bounds6)
# print(effic_result)
# print('=======')
# print(effic_fun6(effic_result.x,p))
# effic_result=scipy.optimize.minimize(effic_fun6,effic_result.x,args=(p),method='L-BFGS-B',bounds=x_bounds6)
# print(effic_result)
# print('=======')
print(effic_fun6(effic_result.x,p))
effic_result=scipy.optimize.minimize(effic_fun6,effic_result.x,args=(p),method='TNC',bounds=x_bounds6)
print(effic_result)
print('=======')
print(effic_fun6(effic_result.x,p))
effic_result=scipy.optimize.minimize(effic_fun6,effic_result.x,args=(p),method='TNC',bounds=x_bounds6)
print(effic_result)
print('=======')
print(effic_fun6(effic_result.x,p))
effic_result=scipy.optimize.minimize(effic_fun6,effic_result.x,args=(p),method='TNC',bounds=x_bounds6)
print(effic_result)
print('=======')
print(effic_fun6(effic_result.x,p))
effic_result=scipy.optimize.minimize(effic_fun6,effic_result.x,args=(p),method='TNC',bounds=x_bounds6)
print(effic_result)
print('=======')
print(effic_fun6(effic_result.x,p))
effic_result=scipy.optimize.minimize(effic_fun6,effic_result.x,args=(p),method='TNC',bounds=x_bounds6)
print(effic_result)
print('=======')
###############################
# x_bounds7=scipy.optimize.Bounds([-np.inf,-np.inf,-np.inf,-np.inf,0,0,0],
#                                [np.inf,np.inf,0,0, np.inf,np.inf,np.inf])
# x_init7=[-0.21213398, -1.2439472 , -1.30044741, -0.87462797,  3.84276068,5.11996611,10.74]
# x_init7=[-0.20483753, -0.94468276, -1.35354733, -1.14431014,  3.91142365,5.19283831, 10.76086523]
# print(effic_fun_Ppump(x_init7,p))
# effic_result=scipy.optimize.minimize(effic_fun_Ppump,x_init7,args=(p),bounds=x_bounds7)
# print(effic_result)
# print('=======')
#################


# effic_result=scipy.optimize.minimize(effic_fun5,effic_result.x,args=(p),method='TNC',bounds=x_bounds5,options = {'maxiter' : 400})
# print(effic_fun5(x_init5,p))
# print(effic_result)
# print('=======')
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
