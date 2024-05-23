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

def calc_efficiency(deloval,delmval,delocval,delmucval,p):

    rho0,rhoa,rhoac,rhob,rhobc=rho_broad_full(deloval,delmval, 0,p)
    if p['Nm']==p['No']:
        rho0_NoPump=0
        rhoa_NoPump=0
        rhoac_NoPump=0
        rhob_NoPump=0
        rhobc_NoPump=0
    else:
        Omega=p['Omega']
        p['Omega']=0
        rho0_NoPump,rhoa_NoPump,rhoac_NoPump,rhob_NoPump,rhobc_NoPump=rho_broad_full(deloval,delmval,0,p)
        p['Omega']=Omega
    rho0=-rho0
    rho0_NoPump=-rho0_NoPump
    Nm_eff=np.real(p['No']*(rho0[0,0]-rho0[1,1])+(p['Nm']-p['No'])*(rho0_NoPump[0,0]-rho0_NoPump[1,1]))
    # print([rho0[0,0],rho0_NoPump[0,0]])
    # print([rho0[1,1],rho0_NoPump[1,1]])
    # print([rho0[2,2],rho0_NoPump[2,2]])
    # print('===')
    # delocval=p['No']*p['go']**2/p['mean_delao']
    # delmucval=Nm_eff*p['gm']**2/p['mean_delam']

    Sa13=rhoa[0,2]*p['No']*p['go']#+rhoa_NoPump[0,2]*(p['Nm']-p['No'])*p['go']
    #Sb13=rhob[0,2]*p['Nm']*p['go']
    Sb13=rhob[0,2]*p['No']*p['go']+rhob_NoPump[0,2]*(p['Nm']-p['No'])*p['go']

    Sa12=rhoa[0,1]*p['No']*p['gm']#+rhoa_NoPump[0,1]*(p['Nm']-p['No'])*p['gm']
    #Sb12=rhob[0,1]*p['Nm']*p['gm']
    Sb12=rhob[0,1]*p['No']*p['gm']+rhob_NoPump[0,1]*(p['Nm']-p['No'])*p['gm']


    Caa=p['gammaoc']*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2)/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    Cab=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    Cba=           -1j*np.sqrt(p['gammaoc'])*np.sqrt(p['gammamc'])*Sa12/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    Cbb=p['gammamc']*(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    return Caa, Cba, Cab, Cbb#, N_frac

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
    Cab=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
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
    Cab=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    effic=-np.abs(Cab)**2
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
    p['gammamc']=x[7]*1e6
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
def effic_fun_Ppump(x, p):
#x=[deloval,delmval,delocval,delmucval,mean_delao,mean_delao,gammaoc]
    # deloval=x[0]*1e10
    # delmval=x[1]*1e8
    deloval=0
    delmval=0
    delocval=x[0]*1e10
    delmucval=x[1]*1e8
    p['mean_delao']=x[2]*1e10
    p['mean_delam']=x[3]*1e8
    p['gammaoc']=np.abs(x[4]*1e6)
    p['nbath']=nbath_from_T(T,p)
    p['gammamc']=np.abs(x[5]*1e6)
    p['Omega']=Omega_cavity_from_PdBm_resonance(10*np.log10(np.abs(x[6])),p)
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

Pump_vals=[0.1,1,10,100,1000,10000]
# Pump_vals=np.linspace(1,50000,20)
Pump_vals=[1,10]
Pump_vals=[1,2,3,4,5,10,15,20,25,50,100]
x_init=[-0.20483753, -0.94468276, -1.35354733, -1.14431014,  3.91142365,5.19283831]
x_bounds=scipy.optimize.Bounds([-np.inf,-np.inf,-np.inf,-np.inf,0,0],
                               [np.inf,np.inf,0,0, np.inf,np.inf])
x_found=np.zeros((len(x_init),len(Pump_vals)))
fun_found=np.zeros(len(Pump_vals))
for ii,Pump_val in enumerate(Pump_vals):
    print(Pump_val)
    p['Omega']=Omega_cavity_from_PdBm_resonance(10*np.log10(Pump_val),p)
    print(effic_fun6(x_init,p))
    effic_result=scipy.optimize.basinhopping(effic_fun6,x_init,minimizer_kwargs={"args":p,"bounds":x_bounds,"method":'TNC'},niter=100)#,bounds=x_bounds7)
    print(effic_result)
    x_init=effic_result.x
    print(Pump_val)
    print(effic_fun6(x_init,p))
    effic_result=scipy.optimize.basinhopping(effic_fun6,x_init,minimizer_kwargs={"args":p,"bounds":x_bounds,"method":'TNC'},niter=400)#,bounds=x_bounds7)
    print(effic_result)
    x_init=effic_result.x
    print(Pump_val)
    print(effic_fun6(x_init,p))
    effic_result=scipy.optimize.basinhopping(effic_fun6,x_init,minimizer_kwargs={"args":p,"bounds":x_bounds,"method":'L-BFGS-B'},niter=100)#,bounds=x_bounds7)
    print(effic_result)
    x_init=effic_result.x
    print(Pump_val)
    print(effic_fun6(x_init,p))
    effic_result=scipy.optimize.basinhopping(effic_fun6,x_init,minimizer_kwargs={"args":p,"bounds":x_bounds,"method":'TNC'},niter=400)#,bounds=x_bounds7)
    print(Pump_val)
    print(effic_fun6(x_init,p))
    x_init=effic_result.x
    x_found[:,ii]=x_init

    fun_found[ii]=effic_result.fun
np.savez('Thesis_figs/Linear_effic/increasingpump10',x_found=x_found,fun_found=fun_found,p=p,Pump_vals=Pump_vals)
plt.plot(Pump_vals,fun_found)
plt.show()
