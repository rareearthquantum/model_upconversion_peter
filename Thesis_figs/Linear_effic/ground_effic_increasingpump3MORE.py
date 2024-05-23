#importing python libs

# import sympy as sym
# sym.init_printing()

import numpy as np
import sys
sys.path.append('/home/peter/model_upconversion')
from math import pi
# import math
import matplotlib.pyplot as plt
# from sympy import I, Matrix, symbols
# from sympy.physics.quantum import TensorProduct, Dagger
import scipy.optimize
# import scipy.integrate
# import scipy.constants as const

#import qutip

from matplotlib.colors import Normalize as Norm

import time
from Linear1.c_linear_ground5 import rho_broad_full
from Linear1.c_linear_ground5big import rho_broad_full as rho_broad_fullbig

from Thesis_figs.Ground_params2 import p, sd_delao_from_B, deltaao_from_B

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
P_pump = 10*np.log10(100.74) #in dBm, 1.74 mW are going into the resonator
T=100e-3
Q=1e9
p['gammaoi']=2*pi*p['freq_pump']/Q
print('gammaoi = ' + str(p['gammaoi']))
# p['Nm']=p['No']
# p['No']=p['Nm']
N=1e16
N=p['No']
p['Nm']=N
p['No']=N
p['gammami']=650e3*2*pi*1e0
del3_start=-1.57 # *1e10
del2_start=-0.85 # *1e8
def Omega_from_PdBm(PdBm,p):
    mu0=4*np.pi*1e-7
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


def effic_fun6(x, p):
#x=[deloval,delmval,delocval,delmucval,mean_delao,mean_delao,gammaoc]
    # deloval=x[0]*1e10
    # delmval=x[1]*1e8
    deloval=0
    delmval=0
    delocval=x[0]*1e10
    delmucval=x[1]*1e8
    p['mean_delao']=x[2]*1e10
    p['mean_delam']=x[3]*1e8
    p['gammaoc']=x[4]*1e6
    p['nbath']=nbath_from_T(T,p)
    p['gammamc']=x[5]*1e6
    rho0,rhoa,rhoac,rhob,rhobc=rho_broad_full(deloval,delmval, 0,p)
    rho0=-rho0

    Sa13=rhoa[0,2]*p['No']*p['go']
    #Sb13=rhob[0,2]*p['Nm']*p['go']
    Sb13=rhob[0,2]*p['Nm']*p['go']

    Sa12=rhoa[0,1]*p['No']*p['gm']
    #Sb12=rhob[0,1]*p['Nm']*p['gm']
    Sb12=rhob[0,1]*p['Nm']*p['gm']
    Cab=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    effic=-np.abs(Cab)**2
    return effic

def effic_fun6big(x, p):
#x=[deloval,delmval,delocval,delmucval,mean_delao,mean_delao,gammaoc]
    # deloval=x[0]*1e10
    # delmval=x[1]*1e8
    deloval=0
    delmval=0
    delocval=x[0]*1e10
    delmucval=x[1]*1e8
    p['mean_delao']=x[2]*1e10
    p['mean_delam']=x[3]*1e8
    p['gammaoc']=x[4]*1e6
    p['nbath']=nbath_from_T(T,p)
    p['gammamc']=x[5]*1e6
    rho0,rhoa,rhoac,rhob,rhobc=rho_broad_fullbig(deloval,delmval, 0,p)
    rho0=-rho0

    Sa13=rhoa[0,2]*p['No']*p['go']
    #Sb13=rhob[0,2]*p['Nm']*p['go']
    Sb13=rhob[0,2]*p['Nm']*p['go']

    Sa12=rhoa[0,1]*p['No']*p['gm']
    #Sb12=rhob[0,1]*p['Nm']*p['gm']
    Sb12=rhob[0,1]*p['Nm']*p['gm']
    Cab=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    effic=-np.abs(Cab)**2
    return effic

def effic_fun2(x, p):
#x=[deloval,delmval,delocval,delmucval,mean_delao,mean_delao,gammaoc]
    # deloval=x[0]*1e10
    # delmval=x[1]*1e8
    deloval=0
    delmval=0
    p['mean_delao']=x[0]*1e10
    p['mean_delam']=x[1]*1e8
    p['nbath']=nbath_from_T(T,p)
    delmucval=N_frac*p['Nm']*p['gm']**2/p['mean_delam']
    delocval=p['No']*p['go']**2/p['mean_delao']
    rho0,rhoa,rhoac,rhob,rhobc=rho_broad_full(deloval,delmval, 0,p)
    rho0=-rho0

    Sa13=rhoa[0,2]*p['No']*p['go']
    #Sb13=rhob[0,2]*p['Nm']*p['go']
    Sb13=rhob[0,2]*p['Nm']*p['go']

    Sa12=rhoa[0,1]*p['No']*p['gm']
    #Sb12=rhob[0,1]*p['Nm']*p['gm']
    Sb12=rhob[0,1]*p['Nm']*p['gm']
    Cab=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    effic=-np.abs(Cab)**2
    return effic

def effic_fun2_gamma(x, p):
#x=[deloval,delmval,delocval,delmucval,mean_delao,mean_delao,gammaoc]
    # deloval=x[0]*1e10
    # delmval=x[1]*1e8
    deloval=0
    delmval=0
    p['gammaoc']=x[0]*1e6
    p['gammamc']=x[1]*1e6
    p['nbath']=nbath_from_T(T,p)
    delmucval=N_frac*p['Nm']*p['gm']**2/p['mean_delam']
    delocval=p['No']*p['go']**2/p['mean_delao']
    rho0,rhoa,rhoac,rhob,rhobc=rho_broad_full(deloval,delmval, 0,p)
    rho0=-rho0

    Sa13=rhoa[0,2]*p['No']*p['go']
    #Sb13=rhob[0,2]*p['Nm']*p['go']
    Sb13=rhob[0,2]*p['Nm']*p['go']

    Sa12=rhoa[0,1]*p['No']*p['gm']
    #Sb12=rhob[0,1]*p['Nm']*p['gm']
    Sb12=rhob[0,1]*p['Nm']*p['gm']
    Cab=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    effic=-np.abs(Cab)**2
    return effic
p['sd_delao']=sd_delao_from_B(0.2,p)
# p['gammami']=650e3*2*pi

Pump_vals=[1,10,100,1000,10000]
Pump_vals=[1,2,3,4,5,10,15,20,50,100]
Pump_vals=[1,2,5,10,20,50,100]
Pump_vals=[1,5,20,50,100,500,1000]
Pump_vals=[1,2,3,4,5,10,15,20,25,35,50,75,100,150,200,300,400,500,700,1000]
# Pump_vals=[100,1000]

x_init=[-0.20483753, -0.94468276, -1.35354733, -1.14431014,  3.91142365,5.19283831]
x_bounds6=scipy.optimize.Bounds([-np.inf,-np.inf,-np.inf,-np.inf,0,0],
                               [np.inf,np.inf,0,0, np.inf,np.inf])
delta3_start=[-1.0421  , -1.1543  , -1.5070  , -1.8597   ,-2.2124,-3.9760]
delta2_start=[-0.8100 ,  -0.8200   ,-0.8300  , -0.8300 ,  -0.9800  , -1.8000]
x_found6=np.zeros((6,len(Pump_vals)))
fun_found=np.zeros(len(Pump_vals))
fun_found6=np.zeros(len(Pump_vals))

x_found2=np.zeros((2,len(Pump_vals)))
x_found_gamma=np.zeros((2,len(Pump_vals)))

fun_found2=np.zeros(len(Pump_vals))
N_frac=(np.exp((6.63e-34*p['freqmu'])/(1.38e-23*T))-1)/(np.exp((6.63e-34*p['freqmu'])/(1.38e-23*T))+1)
p['nbath']=nbath_from_T(T,p)
x_bounds2=scipy.optimize.Bounds([-np.inf,-np.inf],
                               [0,0])
x_bounds_gamma=scipy.optimize.Bounds([0,0],
                               [np.inf,np.inf])
p['gammaoc']=2*pi*1.7e6
p['gammamc']=2*pi*1.495e6

x_found6MORE=np.zeros((6,len(Pump_vals)))
fun_found6MORE=np.zeros(len(Pump_vals))
filename_in='Thesis_figs/Linear_effic/increasingpump3_test2_N2.2MORE3'
npzfile=np.load(filename_in+'.npz')
p=npzfile['p'][()]
Pump_vals=npzfile['Pump_vals']
x_found6=npzfile['x_found6MORE']

name_add='small'
for ii,Pump_val in enumerate(Pump_vals):
    p['Omega']=Omega_cavity_from_PdBm_resonance(10*np.log10(Pump_val),p)
    print(Pump_val)
    x_init6= x_found6[:,ii]
    x_init=x_init6
    # effic_result6=scipy.optimize.minimize(effic_fun6big,x_init6,args=(p),bounds=x_bounds6)
    # print(effic_result6)
    # if not np.isnan(effic_result6.fun):
    #     x_init=effic_result6.x
    # effic_result6=scipy.optimize.minimize(effic_fun6big,x_init,args=(p),bounds=x_bounds6,method='TNC',options = {'maxiter' : 100})
    # print(effic_result6)
    # if not np.isnan(effic_result6.fun):
    #     x_init=effic_result6.x
    # effic_result6=scipy.optimize.minimize(effic_fun6big,x_init,args=(p),bounds=x_bounds6,method='L-BFGS-B',options = {'maxiter' : 100})
    # print(effic_result6)
    # if not np.isnan(effic_result6.fun):
    #     x_init=effic_result6.x
    # effic_result6=scipy.optimize.minimize(effic_fun6big,x_init,args=(p),bounds=x_bounds6,method='TNC',options = {'maxiter' :100})
    # print(effic_result6)
    # if not np.isnan(effic_result6.fun):
    #     x_init=effic_result6.x

    effic_result6=scipy.optimize.minimize(effic_fun6big,x_init,args=(p),bounds=x_bounds6)
    print(effic_result6)
    if not np.isnan(effic_result6.fun):
        x_init=effic_result6.x
    effic_result6=scipy.optimize.minimize(effic_fun6big,x_init,args=(p),bounds=x_bounds6,method='L-BFGS-B',options = {'maxiter' : 200})
    print(effic_result6)
    if not np.isnan(effic_result6.fun):
        x_init=effic_result6.x
    effic_result6=scipy.optimize.minimize(effic_fun6big,x_init,args=(p),bounds=x_bounds6,method='TNC',options = {'maxiter' : 400})
    print(effic_result6)
    if not np.isnan(effic_result6.fun):
        x_init=effic_result6.x
    effic_result6=scipy.optimize.minimize(effic_fun6big,x_init,args=(p),bounds=x_bounds6,method='TNC',options = {'maxiter' : 400})
    print(effic_result6)
    if not np.isnan(effic_result6.fun):
        x_init=effic_result6.x
    for jj in range(6):
        if not effic_result6.success:
            effic_result6=scipy.optimize.minimize(effic_fun6big,x_init,args=(p),bounds=x_bounds6,method='TNC',options = {'maxiter' : 400})
            print(effic_fun6(effic_result6.x,p))
            print(effic_result6)
            if not np.isnan(effic_result6.fun):
                x_init=effic_result6.x
            print('=======')

    x_found6MORE[:,ii]=effic_result6.x
    fun_found6MORE[ii]=effic_result6.fun

    # print(Pump_val)
    # print(effic_fun6(x_init6,p))
    # effic_result=scipy.optimize.basinhopping(effic_fun6,x_init6,minimizer_kwargs={"args":p,"bounds":x_bounds,"method":'TNC'},niter=100)#,bounds=x_bounds7)
    # print(effic_result)


    # x_init=effic_result.x
    # print(Pump_val)
    # print(effic_fun6(x_init,p))
    # effic_result=scipy.optimize.basinhopping(effic_fun6,x_init,minimizer_kwargs={"args":p,"bounds":x_bounds,"method":'TNC'},niter=400)#,bounds=x_bounds7)
    # print(effic_result)
    # x_init=effic_result.x
    # print(Pump_val)
    # print(effic_fun6(x_init,p))
    # effic_result=scipy.optimize.basinhopping(effic_fun6,x_init,minimizer_kwargs={"args":p,"bounds":x_bounds,"method":'L-BFGS-B'},niter=100)#,bounds=x_bounds7)
    # print(effic_result)
    # x_init=effic_result.x
    # print(Pump_val)
    # print(effic_fun6(x_init,p))
    # effic_result=scipy.optimize.basinhopping(effic_fun6,x_init,minimizer_kwargs={"args":p,"bounds":x_bounds,"method":'TNC'},niter=400)#,bounds=x_bounds7)
    # print(Pump_val)
    # print(effic_fun6(x_init,p))
    # x_init=effic_result.x

    np.savez(filename_in+name_add,x_found6MORE=x_found6MORE,fun_found6MORE=fun_found6MORE,p=p,Pump_vals=Pump_vals)
plt.plot(Pump_vals,fun_found6MORE)


plt.show()
