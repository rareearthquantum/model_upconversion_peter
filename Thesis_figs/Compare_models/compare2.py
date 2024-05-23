
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
from Linear1.c_linear_ground5 import rho_broad_full as rho_broad_full_lin
from c_funs_ds3 import rho_broad_full as rho_broad_full

#filename='Linear_figs1/data_temp1'
#filename='Linear_effic3_sim_test_sameN_'
filename='Adiabat_sim4'
save_dir='Thesis_figs/Compare_models/'
P_pump = 10*np.log10(1.74) #in dBm, 1.74 mW are going into the resonator
P_mu = -35-10 # in dBm
#P_pump = 0#10*np.log10(10.74) #in dBm, 1.74 mW are going into the resonator
P_mu = -300# in dBm

T=50e-3
from Thesis_figs.Ground_params2 import p, sd_delao_from_B, deltaao_from_B


p['Nm']=p['No']
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
print('bin   = ' + str(bin_from_PdBm(P_mu,p)))

#===================================================
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

    rho0,rhoa,rhoac,rhob,rhobc=rho_broad_full_lin(deloval,delmval, 0,p)
    rho0=-rho0
    N_frac = (rho0[0,0]-rho0[1,1]) #effective fraction of atoms due to Temperature
    # assert N_frac<=1
    # delmucval=delmucval*N_frac
    Sa13=rhoa[0,2]*p['No']*p['go']
    Sb13=rhob[0,2]*p['Nm']*p['go']
    Sa12=rhoa[0,1]*p['No']*p['gm']
    Sb12=rhob[0,1]*p['Nm']*p['gm']
    Caa=p['gammaoc']*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2)/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    Cba=           -1j*np.sqrt(p['gammamc'])*np.sqrt(p['gammaoc'])*Sb13/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    Cab=           -1j*np.sqrt(p['gammaoc'])*np.sqrt(p['gammamc'])*Sa12/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    Cbb=p['gammamc']*(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)/(Sa12*Sb13+(1j*Sa13-1j*(deloval-delocval)+(p['gammaoi']+2*p['gammaoc'])/2)*(1j*Sb12-1j*(delmval-delmucval)+(p['gammami']+2*p['gammamc'])/2))
    return Caa, Cba, Cab, Cbb, N_frac

def fields_vec_fun(fields_vec,ainval,binval,deloval,delmval,delocval,delmucval,p):
    aval=fields_vec[0]+1j*fields_vec[1]
    bval=fields_vec[2]+1j*fields_vec[3]

    rho=np.array(rho_broad_full(aval,bval,deloval,delmval,p))
    if p['Nm']!=p['No']:
        Omega=p['Omega']
        p['Omega']=0
        rho_only_b=np.array(rho_broad_full(0,bval,0,delmval,p))
        p['Omega']=Omega
        S21_onlyb_val=rho_only_b[1,0]*(p['Nm']-p['No'])*p['gm']
    else:
        S21_onlyb_val=0


    S12val=rho[1,0]*p['No']*p['gm']
    S13val=rho[2,0]*p['No']*p['go']
    bval1=(-1j*S12val-1j*S21_onlyb_val+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*(delmval-delmucval))
    #bval1=(-1j*S12val+np.sqrt(p['gammamc'])*binval)/((p['gammamc']+p['gammami'])/2-1j*delmval)
    aval1=(-1j*S13val+np.sqrt(p['gammaoc'])*ainval)/((2*p['gammaoc']+p['gammaoi'])/2-1j*(deloval-delocval))
    return [aval1.real,aval1.imag,bval1.real,bval1.imag]-fields_vec
# def fields_abs_fun(fields_vec,ainval,binval,deloval,delmval,p):
#     aval=fields_vec[0]+1j*fields_vec[1]
#     bval=fields_vec[2]+1j*fields_vec[3]
#     Omega=p['Omega']
#     rho=np.array(rho_broad_full(aval,bval,deloval,delmval,p))
#     p['Omega']=0
#     rho_only_b=np.array(rho_broad_full(0,bval,0,delmval,p))
#     p['Omega']=Omega
#     S12val=rho[1,0]*p['No']*p['gm']
#     S13val=rho[2,0]*p['No']*p['go']
#     S21_onlyb_val=rho_only_b[1,0]*(p['Nm']-p['No'])*p['gm']
#     bval1=(-1j*S12val-1j*S21_onlyb_val+np.sqrt(p['gammamc'])*binval)/((2*p['gammamc']+p['gammami'])/2-1j*delmval)
#     #bval1=(-1j*S12val+np.sqrt(p['gammamc'])*binval)/((p['gammamc']+p['gammami'])/2-1j*delmval)
#     aval1=(-1j*S13val+np.sqrt(p['gammaoc'])*ainval)/((2*p['gammaoc']+p['gammaoi'])/2-1j*deloval)
#     out_vec=np.array([aval1.real,aval1.imag,bval1.real,bval1.imag]-fields_vec)
#     out_abs=np.linalg.norm(out_vec)
#     return out_abs

def find_fields(ainval,binval,deloval,delmval,delocval,delmucval,start_guess_vec,p):
    #fields_found=scipy.optimize.root(fields_vec_fun,start_guess_vec,args=(ainval,binval,deloval,delmval,p),method='hybr',options={'xtol':1.49012e-08,'maxfev':5000})
    #fields_found=scipy.optimize.root(fields_vec_fun,start_guess_vec,args=(ainval,binval,deloval,delmval,p),method='broyden1',options={'maxiter':10000})
    #fields_found=scipy.optimize.root(fields_vec_fun,start_guess_vec,args=(ainval,binval,deloval,delmval,p),method='lm',options={})
    fields_found=scipy.optimize.root(fields_vec_fun,start_guess_vec,args=(ainval,binval,deloval,delmval,delocval,delmucval,p),method='hybr')
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



start_time=time.time()
delta2vals=np.linspace(-2000e6,2000e6,31)*1/1
deltamuvals=np.linspace(-1500e6,1500e6,11)*1/1
deltaocval=1e10
deltamucval=2e8
deloval=-1e10#0e10

# delta2vals=np.linspace(-2000e6,2000e6,51)*1/5
# deltamuvals=np.linspace(-1500e6,1500e6,51)*1/2
# deltaocval=1e10
# deltamucval=2e7
# deloval=-0e10#0e10

delta2vals=np.linspace(-2000e6,2000e6,201)*2/1
deltamuvals=np.linspace(-1500e6,1500e6,201)*2/1
deltaocval=1e9*0
deltamucval=2e7*0
deloval=-1e9#0e10

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

bvals=np.zeros((len(delta2vals),len(deltamuvals)),dtype=np.complex_)
avals=np.zeros((len(delta2vals),len(deltamuvals)),dtype=np.complex_)
rho_out=np.zeros((3,3,len(delta2vals),len(deltamuvals)),dtype=np.complex_)
binvals=np.zeros((len(delta2vals),len(deltamuvals)),dtype=np.complex_)
N=1e16
p['Nm']=N
p['No']=N
p['mean_delao']=deltaao_from_B(0,0,p)*10
p['sd_delao']=sd_delao_from_B(0,p)
T=50e-3
P_mu=-100
filename='compare2_notzoom'
filename=filename+'T = '+ str(T) + ', P_mu = ' + str(P_mu) +' '
# filename=filename+'T = '+ str(T) +'bin=0.01'
ainval=0
for ii,delta2val in enumerate(delta2vals):
    p['mean_delam']=delta2val#deltamac_from_B(I_val,p)
    p['nbath']=nbath_from_T(T,p)
    for jj, delmval in enumerate(deltamuvals):
        time1=time.time()

        binval=bin_from_PdBm(P_mu,p,delmval)
        binvals[ii,jj]=binval


        effic_abia[ii,jj], Caa_abia[ii,jj], Cbb_abia[ii,jj], S_vals_abia[ii,jj],good_vals_abia[ii,jj] = calc_efficiency_adiabatic(deloval,delmval,deltaocval,deltamucval,p)
        #effic_abia_no_int[ii,jj], Caa_abia_no_int[ii,jj], Cbb_abia_no_int[ii,jj], S_vals_abia_no_int[ii,jj],good_vals_abia_no_int[ii,jj] = calc_efficiency_adiabatic_no_int(deloval,delmval,deltaocval,deltamucval,p)
        Caa[ii,jj], Cba[ii,jj], Cab[ii,jj], Cbb[ii,jj], N_frac[ii,jj]=calc_efficiency(deloval,delmval,deltaocval,deltamucval,p)

        start_guess_vec=[0,0,0,0]

        avals[ii,jj], bvals[ii,jj] = find_fields(ainval,binval,deloval,delmval,deltaocval,deltamucval,start_guess_vec,p)
        rho_out[:,:,ii,jj]=rho_broad_full(avals[ii,jj],bvals[ii,jj],deloval,delmval,p)
    elapsed_time=time.time()-start_time
    print('    ' + str(ii) +', Time: ' + time.ctime() +', Elapsed: '+ str(elapsed_time))
    #np.savez(filename,delta2vals=delta2vals,p=p,deltamucvals=deltamucvals,P_pump=P_pump,T=T,elapsed_time=elapsed_time,deltaocval=deltaocval,calc_time=calc_time,Caa=Caa, Cba=Cba,Cab=Cab,Cbb=Cbb)
    np.savez(save_dir+filename,P_pump=P_pump,T=T,P_mu=P_mu,p=p,binvals=binvals,ainval=ainval,avals=avals,bvals=bvals, rho_out=rho_out,deloval=deloval,deltaocval=deltaocval,
            deltamuvals=deltamuvals, deltamucval=deltamucval, Caa=Caa, Cba=Cba,Cab=Cab,Cbb=Cbb,effic_abia=effic_abia,Caa_abia=Caa_abia,Cbb_abia=Cbb_abia,S_vals_abia=S_vals_abia,good_vals_abia=good_vals_abia,
            delta2vals=delta2vals)
print('    ===========================Complete===========================')

# for ii,delta2val in enumerate(delta2vals):
#     p['mean_delam']=delta2val#deltamac_from_B(B_val,p)
#     p['nbath']=nbath_from_T(T,p)
#     for jj, delmval in enumerate(deltamuvals):
#         time1=time.time()
#         effic_abia[ii,jj], Caa_abia[ii,jj], Cbb_abia[ii,jj], S_vals_abia[ii,jj],good_vals_abia[ii,jj] = calc_efficiency_adiabatic(deltaocval,deltamucval,deloval,delmval,p)
#         effic_abia_no_int[ii,jj], Caa_abia_no_int[ii,jj], Cbb_abia_no_int[ii,jj], S_vals_abia_no_int[ii,jj],good_vals_abia_no_int[ii,jj] = calc_efficiency_adiabatic_no_int(deltaocval,deltamucval,deloval,delmval,p)
#         Caa[ii,jj], Cba[ii,jj], Cab[ii,jj], Cbb[ii,jj], N_frac[ii,jj]=calc_efficiency(deltaocval,deltamucval,deloval,delmval,p)
#
#     elapsed_time=time.time()-start_time
#     print('    ' + str(ii) +', Time: ' + time.ctime() +', Elapsed: '+ str(elapsed_time))
#     #np.savez(filename,delta2vals=delta2vals,p=p,deltamucvals=deltamucvals,P_pump=P_pump,T=T,elapsed_time=elapsed_time,deltaocval=deltaocval,calc_time=calc_time,Caa=Caa, Cba=Cba,Cab=Cab,Cbb=Cbb)
# print('    ===========================Complete===========================')

effic_full=np.abs(avals*np.sqrt(p['gammaoc'])/binvals)**2

fig=plt.figure(filename+'effic_adiabat/effic_linear')

fig.clf()

ax=fig.add_subplot(3,1,1)
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu')
img1=ax.imshow(np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu_r',norm=MidpointNormalize(midpoint=0))
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
fig.colorbar(img1, format='%.1e')

ax=fig.add_subplot(3,1,3)
img1=ax.imshow(good_vals_abia,extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('good?')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

fig.suptitle('effic_adiabat/effic_linear')

fig=plt.figure(filename+'effic_adiabat/full')

fig.clf()

ax=fig.add_subplot(3,1,1)
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu')
img1=ax.imshow(np.log10(np.abs(effic_abia)/np.abs(effic_full)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu_r',norm=MidpointNormalize(midpoint=0))
#img1=ax.imshow(np.abs(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2))),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
#plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)
plt.title(' |C|^2 (dB scale)')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,2)
img1=ax.imshow((np.abs(effic_abia)/np.abs(effic_full)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('|C|^2 ')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1, format='%.1e')

ax=fig.add_subplot(3,1,3)
img1=ax.imshow(good_vals_abia,extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('good?')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

fig.suptitle('effic_adiabat/effic_full')


fig=plt.figure(filename+'effic_lin/effic_full')

fig.clf()

ax=fig.add_subplot(2,1,1)
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu')
img1=ax.imshow(np.log10(np.abs(Cba**2)/np.abs(effic_full)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu_r',norm=MidpointNormalize(midpoint=0))
#img1=ax.imshow(np.abs(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2))),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
#plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)
plt.title(' |C|^2 (dB scale)')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

ax=fig.add_subplot(2,1,2)
img1=ax.imshow((np.abs(Cba**2)/np.abs(effic_full)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('|C|^2 ')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1, format='%.1e')


fig.suptitle('effic_linear/effic_full')


fig=plt.figure(filename+'Compare thingzz log')

fig.clf()

ax=fig.add_subplot(3,1,1)
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu')
img1=ax.imshow(np.log10(np.abs(effic_abia)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title(' Adiabat')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,2)
img1=ax.imshow(np.log10(np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title(' Linear')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,3)
img1=ax.imshow(np.log10(np.abs(effic_full)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('Full')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

fig.suptitle('Comparing things -- log scale')



# fig=plt.figure(filename+'Compare thingzz lin')
#
# fig.clf()
#
# ax=fig.add_subplot(3,1,1)
# #img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu')
# img1=ax.imshow((np.abs(effic_abia)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# plt.title(' Adiabat')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1)
#
# ax=fig.add_subplot(3,1,2)
# img1=ax.imshow((np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# plt.title(' Linear')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1)
#
# ax=fig.add_subplot(3,1,3)
# img1=ax.imshow((np.abs(effic_full)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# plt.title('Full')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1)
#
# fig.suptitle('Comparing things -- lin scale')


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
