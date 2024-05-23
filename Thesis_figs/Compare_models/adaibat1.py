
import sympy as sym
sym.init_printing()

import numpy as np
import sys
sys.path.append('/home/peter/model_upconversion')
from math import pi
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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
from c_funs_ds3 import rho_broad_full as rho_broad_full_ds3

#filename='Linear_figs1/data_temp1'
#filename='Linear_effic3_sim_test_sameN_'
filename='Adiabat_sim1'
#I_vals=np.linspace(7.43,7.55,31)
delta2vals=np.linspace(-1000e6,1000e6,301)
delta3vals=np.linspace(-15e10,15e10,301)
delta2vals=np.linspace(-250e6,000e6,21)[:-1]
delta3vals=np.linspace(-45e9,0e10,21)[:-1]
# delta2vals=np.linspace(-10e7,-2.5e7,20)
# delta3vals=np.linspace(-2.8e10,-0.6e10,20)
deltamuvals=np.linspace(-1e9,1e9,21)
deltaovals =np.linspace(-1e10,1e10,31)
deltamuvals=np.linspace(1e7,1e8,21)
deltaovals =np.linspace(1e10,9e10,31)

#deltamucvals=np.linspace(-250e6,250e6,21)
P_pump = 10*np.log10(10.74) #in dBm, 1.74 mW are going into the resonator
#P_mu = -35-30 # in dBm
#P_pump = 0#10*np.log10(10.74) #in dBm, 1.74 mW are going into the resonator

T=100e-3


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
Q=1e8
#p['gammaoi']=2*pi*p['freq_pump']/Q
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
p['mean_delao']=0#deltaao_from_I(0,p)
p['sd_delao']=sd_delao_from_I(0,p)
p['mean_delam']=0e8
print('Omega = ' + str(p['Omega']))
print('mean_delao = ' +str(p['mean_delao']) )
print('  sd_delao = ' +str(p['sd_delao']) )

print('mean_delam = ' +str(p['mean_delam']) )
print('  sd_delam = ' +str(p['sd_delam']) )

class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def S_fun(deloval,delmval,p):
    good_val=1
    if np.abs(p['mean_delam']-delmval)<6*p['sd_delam']:
        # print('Not in the large delta limit for optical')
        # print('mean_delam - delmval = ' + str(p['mean_delam']-delmval))
        # print('sd_delam =             ' + str(p['sd_delam']))
        good_val=0
    if np.abs(p['mean_delao']-deloval)<6*p['sd_delao']:
        # print('Not in the large delta limit for optical')
        # print('mean_delao - deloval = ' + str(p['mean_delao']-deloval))
        # print('sd_delao =             ' + str(p['sd_delao']))
        good_val=0

    func_to_int=lambda delaval,delval,mu,sd: np.exp(-(delaval-mu)**2/(2*sd**2))/(np.sqrt(2*pi)*sd*(delaval-delval))

    # mu_integral=scipy.integrate.quad(func_to_int,delmval,np.inf,args=(delmval,p['mean_delam'],p['sd_delam']))
    # o_integral= scipy.integrate.quad(func_to_int,deloval,np.inf,args=(deloval,p['mean_delao'],p['sd_delao']))
    int_lim=50
    mu_integral=scipy.integrate.quad(func_to_int,p['mean_delam']-p['sd_delam']*5,p['mean_delam'],args=(delmval,p['mean_delam'],p['sd_delam']),limit=int_lim)[0]+scipy.integrate.quad(func_to_int,p['mean_delam'],p['mean_delam']+p['sd_delam']*5,args=(delmval,p['mean_delam'],p['sd_delam']),limit=int_lim)[0]
    o_integral =scipy.integrate.quad(func_to_int,p['mean_delao']-p['sd_delao']*5,p['mean_delao'],args=(deloval,p['mean_delao'],p['sd_delao']),limit=int_lim)[0]+scipy.integrate.quad(func_to_int,p['mean_delao'],p['mean_delao']+p['sd_delao']*5,args=(deloval,p['mean_delao'],p['sd_delao']),limit=int_lim)[0]

    #print(mu_integral)
    #print(o_integral)
    S_val=p['Omega']*p['gm']*p['go']*p['No']*mu_integral*o_integral
    return S_val, good_val
def S_simple(deloval,delmval,p):
    S_val=p['Omega']*p['go']*p['gm']*p['No']/(deloval*delmval)
    return S_val
# def calc_efficiency(deloval,delmval,delocval,delmucval,p):
#     SS,good_val=S_fun(deloval,delmval,p)
#     effic=np.abs(1j*SS*np.sqrt(p['gammaoc']*p['gammamc'])/(np.abs(SS)**2+(-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2)*(-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)))
#     Caa = (np.abs(SS)**2-(-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2)*(1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2))/(np.abs(SS)**2+(-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2)*(-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2))
#     Cbb = (np.abs(SS)**2-(1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2)*(-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2))/(np.abs(SS)**2+(-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2)*(-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2))
#     return effic, Caa, Cbb,SS,good_val
def calc_efficiency_adiabatic(deloval,delmval,delocval,delmucval,p):
    SS,good_val=S_fun(deloval,delmval,p)
    #SS=S_simple(deloval,delmval,p)
    #good_val=1
    effic=np.abs(1j*SS*np.sqrt(p['gammaoc']*p['gammamc'])/(np.abs(SS)**2+(-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2)*(-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)))**2
    Caa = p['gammaoc']*(-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2)/(np.abs(SS)**2+(-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2)*(-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2))
    Cbb = p['gammamc']*(-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2) /(np.abs(SS)**2+(-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2)*(-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2))
    return effic, Caa, Cbb,SS,good_val
def calc_efficiency_adiabatic_no_int(deloval,delmval,delocval,delmucval,p):
    #SS,good_val=S_fun(deloval,delmval,p)
    SS=S_simple(deloval,delmval,p)
    good_val=1
    effic=np.abs(1j*SS*np.sqrt(p['gammaoc']*p['gammamc'])/(np.abs(SS)**2+(-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2)*(-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)))**2
    Caa = p['gammaoc']*(-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2)/(np.abs(SS)**2+(-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2)*(-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2))
    Cbb = p['gammamc']*(-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2) /(np.abs(SS)**2+(-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2)*(-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2))
    return effic, Caa, Cbb,SS,good_val

def calc_efficiency(deloval,delmval,delocval,delmucval,p):

    rho0,rhoa,rhoac,rhob,rhobc=rho_broad_full(deloval,delmval, 0,p)
    rho0=-rho0
    N_frac = (rho0[0,0]-rho0[1,1]) #effective fraction of atoms due to Temperature
    assert N_frac<=1
    delmucval=delmucval*N_frac
    Sa13=rhoa[0,2]*p['No']*p['go']
    Sb13=rhob[0,2]*p['Nm']*p['go']
    Sa12=rhoa[0,1]*p['No']*p['gm']
    Sb12=rhob[0,1]*p['Nm']*p['gm']
    Caa=p['gammaoc']*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2)/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    Cba=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    Cab=           -1j*np.sqrt(p['gammaoc'])*np.sqrt(p['gammamc'])*Sa12/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    Cbb=p['gammamc']*(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    return Caa, Cba, Cab, Cbb, N_frac


start_time=time.time()
delta2vals=np.linspace(-2000e6,2000e6,11)*1/1
deltamuvals=np.linspace(-1500e6,1500e6,11)*1/1
deltaocval=1e10
deltamucval=2e8
deloval=-1e10#0e10

# delta2vals=np.linspace(-2000e6,2000e6,51)*1/5
# deltamuvals=np.linspace(-1500e6,1500e6,51)*1/2
# deltaocval=1e10
# deltamucval=2e7
# deloval=-0e10#0e10

effic_abia=np.zeros((len(delta2vals),len(deltamuvals)))
Caa_abia=np.zeros((len(delta2vals),len(deltamuvals)))
Cbb_abia=np.zeros((len(delta2vals),len(deltamuvals)))
S_vals_abia=np.zeros((len(delta2vals),len(deltamuvals)))
good_vals_abia=np.zeros((len(delta2vals),len(deltamuvals)))

effic_abia_no_int=np.zeros((len(delta2vals),len(deltamuvals)))
Caa_abia_no_int=np.zeros((len(delta2vals),len(deltamuvals)))
Cbb_abia_no_int=np.zeros((len(delta2vals),len(deltamuvals)))
S_vals_abia_no_int=np.zeros((len(delta2vals),len(deltamuvals)))
good_vals_abia_no_int=np.zeros((len(delta2vals),len(deltamuvals)))

Caa=np.zeros((len(delta2vals),len(deltamuvals)),dtype=np.complex_)
Cba=np.zeros((len(delta2vals),len(deltamuvals)),dtype=np.complex_)
Cab=np.zeros((len(delta2vals),len(deltamuvals)),dtype=np.complex_)
Cbb=np.zeros((len(delta2vals),len(deltamuvals)),dtype=np.complex_)
N_frac=np.zeros((len(delta2vals),len(deltamuvals)),dtype=np.complex_)

p['mean_delao']=deltaao_from_I(0,p)
p['sd_delao']=sd_delao_from_I(0,p)

# for ii,delta2val in enumerate(delta2vals):
#     p['mean_delam']=delta2val#deltamac_from_I(I_val,p)
#     p['nbath']=nbath_from_T(T,p)
#     for jj, delmval in enumerate(deltamuvals):
#         time1=time.time()
#         effic_abia[ii,jj], Caa_abia[ii,jj], Cbb_abia[ii,jj], S_vals_abia[ii,jj],good_vals_abia[ii,jj] = calc_efficiency_adiabatic(deloval,delmval,deltaocval,deltamucval,p)
#         effic_abia_no_int[ii,jj], Caa_abia_no_int[ii,jj], Cbb_abia_no_int[ii,jj], S_vals_abia_no_int[ii,jj],good_vals_abia_no_int[ii,jj] = calc_efficiency_adiabatic_no_int(deloval,delmval,deltaocval,deltamucval,p)
#         Caa[ii,jj], Cba[ii,jj], Cab[ii,jj], Cbb[ii,jj], N_frac[ii,jj]=calc_efficiency(deloval,delmval,deltaocval,deltamucval,p)
#
#     elapsed_time=time.time()-start_time
#     print('    ' + str(ii) +', Time: ' + time.ctime() +', Elapsed: '+ str(elapsed_time))
#     #np.savez(filename,delta2vals=delta2vals,p=p,deltamucvals=deltamucvals,P_pump=P_pump,T=T,elapsed_time=elapsed_time,deltaocval=deltaocval,calc_time=calc_time,Caa=Caa, Cba=Cba,Cab=Cab,Cbb=Cbb)
# print('    ===========================Complete===========================')
T=0
for ii,delta2val in enumerate(delta2vals):
    p['mean_delam']=delta2val#deltamac_from_I(I_val,p)
    p['nbath']=nbath_from_T(T,p)
    for jj, delmval in enumerate(deltamuvals):
        time1=time.time()
        effic_abia[ii,jj], Caa_abia[ii,jj], Cbb_abia[ii,jj], S_vals_abia[ii,jj],good_vals_abia[ii,jj] = calc_efficiency_adiabatic(deltaocval,deltamucval,deloval,delmval,p)
        effic_abia_no_int[ii,jj], Caa_abia_no_int[ii,jj], Cbb_abia_no_int[ii,jj], S_vals_abia_no_int[ii,jj],good_vals_abia_no_int[ii,jj] = calc_efficiency_adiabatic_no_int(deltaocval,deltamucval,deloval,delmval,p)
        Caa[ii,jj], Cba[ii,jj], Cab[ii,jj], Cbb[ii,jj], N_frac[ii,jj]=calc_efficiency(deltaocval,deltamucval,deloval,delmval,p)

    elapsed_time=time.time()-start_time
    print('    ' + str(ii) +', Time: ' + time.ctime() +', Elapsed: '+ str(elapsed_time))
    #np.savez(filename,delta2vals=delta2vals,p=p,deltamucvals=deltamucvals,P_pump=P_pump,T=T,elapsed_time=elapsed_time,deltaocval=deltaocval,calc_time=calc_time,Caa=Caa, Cba=Cba,Cab=Cab,Cbb=Cbb)
print('    ===========================Complete===========================')

fig=plt.figure(filename+'effic_adiabat')

fig.clf()

ax=fig.add_subplot(3,1,1)
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu')
img1=ax.imshow(np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu',norm=MidpointNormalize(midpoint=0))
#img1=ax.imshow(np.abs(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2))),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
#plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)
plt.title(' |C|^2 (dB scale)')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,2)
img1=ax.imshow((np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('|C|^2 ')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,1,3)
img1=ax.imshow(good_vals_abia,extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('good?')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

fig.suptitle('effic_adiabat')

fig=plt.figure(filename+'effic_adiabat_no_int')

fig.clf()

ax=fig.add_subplot(3,1,1)
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu')
img1=ax.imshow(np.log10(np.abs(effic_abia_no_int)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu',norm=MidpointNormalize(midpoint=0))
#img1=ax.imshow(np.abs(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2))),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
#plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)
plt.title(' |C|^2 (dB scale)')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,2)
img1=ax.imshow((np.abs(effic_abia_no_int)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('|C|^2 ')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,1,3)
img1=ax.imshow((np.angle(effic_abia_no_int)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' C phase')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

fig.suptitle('effic_adiabat_no_int')
fig=plt.figure(filename+'effic_adiabat_no_int/effic_abia')

ax=fig.add_subplot(3,1,1)
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu')
img1=ax.imshow(np.log10(np.abs(effic_abia_no_int)/np.abs(effic_abia)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu',norm=MidpointNormalize(midpoint=0))
#img1=ax.imshow(np.abs(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2))),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
#plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)
plt.title(' |C|^2 (dB scale)')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,2)
img1=ax.imshow((np.abs(effic_abia_no_int)/np.abs(effic_abia)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('|C|^2 ')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,1,3)
img1=ax.imshow((np.angle(effic_abia_no_int)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' C phase')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

fig.suptitle('effic_adiabat_no_int/effic_abia')

fig=plt.figure(filename+'effic_adiabat1')

fig.clf()

ax=fig.add_subplot(3,1,1)
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu')
img1=ax.imshow(10*np.log10(np.abs(effic_abia)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')#,cmap='RdBu')
#img1=ax.imshow(np.abs(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2))),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
#plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)
plt.title(' |C|^2 (dB scale)')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,2)
img1=ax.imshow((np.abs(effic_abia)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('|C|^2 ')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1, format='%.00e')

ax=fig.add_subplot(3,1,3)
img1=ax.imshow(good_vals_abia,extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('good?')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

fig.suptitle('effic_adiabat1')

fig=plt.figure(filename+'effic_adiabat_no_int1')

fig.clf()

ax=fig.add_subplot(3,1,1)
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu')
img1=ax.imshow(10*np.log10(np.abs(effic_abia_no_int)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')#,cmap='RdBu')
#img1=ax.imshow(np.abs(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2))),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
#plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)
plt.title(' |C|^2 (dB scale)')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,2)
img1=ax.imshow((np.abs(effic_abia_no_int)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('|C|^2 ')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,1,3)
img1=ax.imshow(np.angle(effic_abia_no_int),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('good?')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

fig.suptitle('effic_adiabat1_no_int')

#
# fig=plt.figure(filename+'_Caa_adiabat')
#
# fig.clf()
#
# ax=fig.add_subplot(3,1,1)
# img1=ax.imshow(10*np.log10(np.abs(Caa_abia)**2),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# #plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)
#
# plt.title(' |C|^2 (dB scale)')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1)
#
# ax=fig.add_subplot(3,1,2)
# img1=ax.imshow((np.abs(Caa_abia)**2),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# plt.title('|C|^2 ')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1, format='%.0e')
#
# ax=fig.add_subplot(3,1,3)
# img1=ax.imshow((np.angle(Caa_abia)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='hsv')
# plt.title(' C phase')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1)
#
# fig.suptitle('_Cbb_adiabat')
# fig=plt.figure(filename+'_Cbb_adiabat')
#
# fig.clf()
#
# ax=fig.add_subplot(3,1,1)
# img1=ax.imshow(10*np.log10(np.abs(Cbb_abia)**2),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# #plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)
#
# plt.title(' |C|^2 (dB scale)')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1)
#
# ax=fig.add_subplot(3,1,2)
# img1=ax.imshow((np.abs(Cbb_abia)**2),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# plt.title('|C|^2 ')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1, format='%.0e')
#
# ax=fig.add_subplot(3,1,3)
# img1=ax.imshow((np.angle(Cbb_abia)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='hsv')
# plt.title(' C phase')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1)
#
# fig.suptitle('_Cbb_adiabat')
#
# fig=plt.figure(filename+'_Caa')
#
# fig.clf()
#
# ax=fig.add_subplot(3,1,1)
# img1=ax.imshow(10*np.log10(np.abs(Caa)**2),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# #plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)
#
# plt.title(' |C|^2 (dB scale)')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1)
#
# ax=fig.add_subplot(3,1,2)
# img1=ax.imshow((np.abs(Caa)**2),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# plt.title('|C|^2 ')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1, format='%.0e')
#
# ax=fig.add_subplot(3,1,3)
# img1=ax.imshow((np.angle(Caa)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='hsv')
# plt.title(' C phase')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1)
#
# fig.suptitle('Caa')
#
# fig=plt.figure(filename+'_Cba')
#
# fig.clf()
#
# ax=fig.add_subplot(3,1,1)
# img1=ax.imshow(10*np.log10(np.abs(Cba)**2),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# plt.title(' |C| (dB scale)')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1)
#
# ax=fig.add_subplot(3,1,2)
# img1=ax.imshow((np.abs(Cba)**2),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# plt.title('|C|')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1, format='%.0e')
#
# ax=fig.add_subplot(3,1,3)
# img1=ax.imshow((np.angle(Cba)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='hsv')
# plt.title(' C phase')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1)
#
# fig.suptitle('Cba')
#
# fig=plt.figure(filename+'_Cab')
#
# fig.clf()
#
# ax=fig.add_subplot(3,1,1)
# img1=ax.imshow(10*np.log10(np.abs(Cab)**2),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# plt.title(' |C| (dB scale)')
# plt.xlabel('delta_o')
# plt.ylabel('I')
# fig.colorbar(img1)
#
# ax=fig.add_subplot(3,1,2)
# img1=ax.imshow((np.abs(Cab)**2),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# plt.title('|C|')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1, format='%.0e')
#
# ax=fig.add_subplot(3,1,3)
# img1=ax.imshow((np.angle(Cab)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='hsv')
# plt.title(' C phase')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1)
#
# fig.suptitle('Cab')
#
# fig=plt.figure(filename+'_Cbb')
#
# fig.clf()
#
# ax=fig.add_subplot(3,1,1)
# img1=ax.imshow(10*np.log10(np.abs(Cbb)**2),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# plt.title(' |C|^2 (dB scale)')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1)
#
# ax=fig.add_subplot(3,1,2)
# img1=ax.imshow((np.abs(Cbb)**2),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# plt.title('|C|')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1, format='%.0e')
#
# ax=fig.add_subplot(3,1,3)
# img1=ax.imshow((np.angle(Cbb)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='hsv')
# plt.title('C phase')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1)
#
# fig.suptitle('Cbb')
#plt.show()

# deltaocval=0
# p['mean_delao']=deltaao_from_I(0,p)
# p['sd_delao']=sd_delao_from_I(0,p)
# delta2vals=np.linspace(-200e6,200e6,11)
# deltamucvals=np.linspace(-150e6,150e6,201)
# effic=np.zeros((len(delta2vals),len(deltamucvals)))
# Caa=np.zeros((len(delta2vals),len(deltamucvals)))
# Cbb=np.zeros((len(delta2vals),len(deltamucvals)))
# S_vals=np.zeros((len(delta2vals),len(deltamucvals)))
# good_vals=np.zeros((len(delta2vals),len(deltamucvals)))
# for ii,delta2val in enumerate(delta2vals):
#     p['mean_delam']=delta2val#deltamac_from_I(I_val,p)
#     p['nbath']=nbath_from_T(T,p)
#     for jj, deltamucval in enumerate(deltamucvals):
#
#         effic[ii,jj], Caa[ii,jj], Cbb[ii,jj],S_vals[ii,jj], good_vals[ii,jj]=calc_efficiency(0,0,deltaocval,deltamucval,p)
#
#
# fig=plt.figure('effic')
#
# fig.clf()
#
# ax=fig.add_subplot(3,1,1)
# img1=ax.imshow(10*np.log10(np.abs(effic)),extent=(np.min(deltamucvals),np.max(deltamucvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# #plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)
# plt.title(' |C|^2 (dB scale)')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1)
#
# ax=fig.add_subplot(3,1,2)
# img1=ax.imshow((np.abs(effic)),extent=(np.min(deltamucvals),np.max(deltamucvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# #plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)
#
# plt.title('|C|')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1, format='%.0e')
#
# ax=fig.add_subplot(3,1,3)
# img1=ax.imshow(good_vals,extent=(np.min(deltamucvals),np.max(deltamucvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# plt.title('good?')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1)
#
# fig.suptitle('effic')
# #plt.show()
#
# fig=plt.figure('Cbb')
#
# fig.clf()
#
# ax=fig.add_subplot(3,1,1)
# img1=ax.imshow(10*np.log10(np.abs(Cbb)**2),extent=(np.min(deltamucvals),np.max(deltamucvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# #plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)
# plt.title(' |C|^2 (dB scale)')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1)
#
# ax=fig.add_subplot(3,1,2)
# img1=ax.imshow((np.abs(Cbb)**2),extent=(np.min(deltamucvals),np.max(deltamucvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# #plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)
#
# plt.title('|C|')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1, format='%.0e')
#
# ax=fig.add_subplot(3,1,3)
# img1=ax.imshow(good_vals,extent=(np.min(deltamucvals),np.max(deltamucvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# plt.title('good?')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1)
#
# fig.suptitle('Cbb')
# plt.show()


# start_time=time.time()
# deltaoval=0#10.e9
#
# start_time=time.time()
# deltaocval=0
# p['mean_delao']=deltaao_from_I(0,p)
# p['sd_delao']=sd_delao_from_I(0,p)
# for ii,delta2val in enumerate(delta2vals):
#     p['mean_delam']=delta2val#deltamac_from_I(I_val,p)
#     deltamucval=p['No']*p['gm']**2/delta2val
#     p['nbath']=nbath_from_T(T,p)
#     for jj, delta3val in enumerate(delta3vals):
#         p['mean_delao']=delta3val
#         deltaocval=p['No']*p['go']**2/delta3val
#
#         #time1=time.time()
#         effic[ii,jj], Caa[ii,jj], Cbb[ii,jj],good_vals[ii,jj]=calc_efficiency(0,0,deltaocval,deltamucval,p)
#
# fig=plt.figure()
#
# fig.clf()
#
# ax=fig.add_subplot(2,1,1)
# img1=ax.imshow((effic),extent=(np.min(deltaovals),np.max(deltaovals),np.min(deltamuvals),np.max(deltamuvals)),aspect='auto',origin='lower')
# plt.title('effic')
# plt.xlabel('delta_o')
# plt.ylabel('delta_mu')
# fig.colorbar(img1)
# ax=fig.add_subplot(2,1,2)
#
# img1=ax.imshow((good_vals),extent=(np.min(deltaovals),np.max(deltaovals),np.min(deltamuvals),np.max(deltamuvals)),aspect='auto',origin='lower')
# plt.title('effic')
# plt.xlabel('delta_o')
# plt.ylabel('delta_mu')
# fig.colorbar(img1)


# for ii, delmuval in enumerate(deltamuvals):
#     for jj,deloval in enumerate(deltaovals):
#         S_vals[ii,jj],good_vals[ii,jj]=S_fun(deloval,delmuval,p)

# fig=plt.figure()
#
# fig.clf()
#
# ax=fig.add_subplot(3,1,1)
# img1=ax.imshow((S_vals),extent=(np.min(deltaovals),np.max(deltaovals),np.min(deltamuvals),np.max(deltamuvals)),aspect='auto',origin='lower')
# plt.title('S')
# plt.xlabel('delta_o')
# plt.ylabel('delta_mu')
# fig.colorbar(img1)
#
# ax=fig.add_subplot(3,1,2)
# img1=ax.imshow((S_vals*good_vals),extent=(np.min(deltaovals),np.max(deltaovals),np.min(deltamuvals),np.max(deltamuvals)),aspect='auto',origin='lower')
# plt.title('S good')
# plt.xlabel('delta_o')
# plt.ylabel('delta_mu')
# fig.colorbar(img1)
#
# ax=fig.add_subplot(3,1,3)
# img1=ax.imshow((good_vals),extent=(np.min(deltaovals),np.max(deltaovals),np.min(deltamuvals),np.max(deltamuvals)),aspect='auto',origin='lower')
# plt.title('good?')
# plt.xlabel('delta_o')
# plt.ylabel('delta_mu')
#fig.colorbar(img1)
plt.show()
# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
