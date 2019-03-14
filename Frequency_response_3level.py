
# coding: utf-8

# In[44]:


#importing python libs

import sympy as sym
sym.init_printing()

import numpy as np
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

#get_ipython().run_line_magic('load_ext', 'cython')
import time


# In[45]:


#define some s pre/post operators

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
delao, delam =sym.symbols('delta_a_o delta_a_mu') #detunings between atom and cavity
gamma13,gamma23,gamma2d,gamma3d,nbath,gammamu=sym.symbols('gamma_13 gamma_23 gamma_2d gamma_3d n_b gamma_mu', real=True, negative=False) #energy decay for atom levels
Omega=sym.symbols('Omega', real=False, negative=False) #pump Rabi frequency
rho11, rho12, rho13, rho21, rho22, rho23, rho31, rho32, rho33=sym.symbols('rho_11 rho_12 rho_13 rho_21 rho_22 rho_23 rho_31 rho_32 rho_33') #Density matrix elements
a, b = sym.symbols('a b') #classical amplitudes of the optical and microwave fields
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
rhoflat = 1*rho.T
rhoflat = rhoflat[:]

Lfunc = sym.lambdify((a,b,delo, delm,delao, delam, gamma13, gamma23, gamma2d, gamma3d, nbath,gammamu,Omega,go,gm),L)

def steady_rho_single(delaoval,delamval,aval, bval,deloval,delmval,p):

    L_mat=Lfunc(aval, bval,deloval,delmval, delaoval,delamval,p['gamma13'],p['gamma23'],p['gamma2d'],p['gamma3d'], p['nbath'],p['gammamu'],p['Omega'],p['go'],p['gm'])
    rho = np.linalg.solve(L_mat.astype('complex128'),np.matrix([[1,0,0,0,0,0,0,0,0]]).T)
    return np.reshape(rho,(3,3),order='F')
def steady_rho_ensemble(delaovals,delamvals,aval, bval,deloval,delmval,p):
    ndelao=len(delaovals)
    ndelam=len(delamvals)
    coh_vals=np.zeros((3,3,ndelao,ndelam), dtype=np.complex_)
    for ii in range(ndelao):
        for jj in range(ndelam):
            coh_vals[:,:,ii,jj]=steady_rho_single(delaovals[ii],delamvals[jj],aval,bval,deloval,delmval,p)
    return coh_vals

def integrate1d_peaks(func,x,ylims, peak_func,aval,bval,deloval,delmval,p):
    peak_yval=peak_func(x)

    if (np.isnan(peak_yval) or peak_yval>ylims[1] or peak_yval<ylims[0]):
        funcval=scipy.integrate.quad(func, ylims[0], ylims[1], args=(x,aval,bval,deloval,delmval,p))[0]

    else:
        funcval1=scipy.integrate.quad(func, ylims[0], peak_yval, args=(x,aval,bval,deloval,delmval,p))[0]
        funcval2=scipy.integrate.quad(func, peak_yval, ylims[1], args=(x,aval,bval,deloval,delmval,p))[0]
        funcval=funcval1+funcval2
    return funcval

def integrate2d_peaks(func, ylims, xlims, peak_func,aval,bval,deloval,delmval,p):
    temp_fun=lambda x: integrate1d_peaks(func, x, ylims, peak_func,aval,bval,deloval,delmval,p)
    inte=scipy.integrate.quad(temp_fun,xlims[0], xlims[1])
    return inte


coh13_Rfun=lambda delaoval, delamval,aval, bval,deloval,delmval,p: np.real(steady_rho_single(delaoval,delamval,aval,bval,deloval,delmval,p)[0,2])
coh13_Ifun=lambda delaoval, delamval,aval, bval,deloval,delmval,p: np.imag(steady_rho_single(delaoval,delamval,aval,bval,deloval,delmval,p)[0,2])
gauss_fun=lambda delaoval, delamval, mo,mm,sdo,sdm: 1.0/(2*pi*sdo*sdm)*np.exp(-(delaoval-mo)**2/(2*sdo**2)
                                                                  -(delamval-mm)**2/(2*sdm**2))
coh13_Rguass_fun=lambda delaoval, delamval,aval, bval,deloval,delmval,p: coh13_Rfun(delaoval, delamval,aval, bval,deloval,delmval,p)*gauss_fun(delaoval,delamval,p['mean_delao'], p['mean_delam'],p['sd_delao'],p['sd_delam'])
coh13_Iguass_fun=lambda delaoval, delamval,aval, bval,deloval,delmval,p: coh13_Ifun(delaoval, delamval,aval, bval,deloval,delmval,p)*gauss_fun(delaoval,delamval,p['mean_delao'], p['mean_delam'],p['sd_delao'],p['sd_delam'])

coh12_Rfun=lambda delaoval, delamval,aval, bval,deloval,delmval,p: np.real(steady_rho_single(delaoval,delamval,aval,bval,deloval,delmval,p)[0,1])
coh12_Ifun=lambda delaoval, delamval,aval, bval,deloval,delmval,p: np.imag(steady_rho_single(delaoval,delamval,aval,bval,deloval,delmval,p)[0,1])
coh12_Rguass_fun=lambda delaoval, delamval,aval, bval,deloval,delmval,p: coh12_Rfun(delaoval, delamval,aval, bval,deloval,delmval,p)*gauss_fun(delaoval,delamval,p['mean_delao'], p['mean_delam'],p['sd_delao'],p['sd_delam'])
coh12_Iguass_fun=lambda delaoval, delamval,aval, bval,deloval,delmval,p: coh12_Ifun(delaoval, delamval,aval, bval,deloval,delmval,p)*gauss_fun(delaoval,delamval,p['mean_delao'], p['mean_delam'],p['sd_delao'],p['sd_delam'])


def rho13_integrated(delao_lims, delam_lims,aval,bval,deloval,delmval,p, peak_func):
    coh13_R_int=integrate2d_peaks(coh13_Rguass_fun, delao_lims, delam_lims, peak_func,aval,bval,deloval,delmval,p)
    coh13_I_int=integrate2d_peaks(coh13_Rguass_fun, delao_lims, delam_lims, peak_func,aval,bval,deloval,delmval,p)
    #return [coh13_R_int, coh13_I_int]#coh13_R_int+1.j*coh13_I_int\
    #return coh13_R_int+1.j*coh13_I_int
    return coh13_R_int[0]+1.j*coh13_I_int[0],coh13_R_int[1]+1.j*coh13_I_int[1]
def rho13_integrated_quad(delao_lims, delam_lims,aval,bval,deloval,delmval,p):
    coh13_R_int=scipy.integrate.nquad(coh13_Rguass_fun, [delao_lims,delam_lims],args=(aval,bval,deloval,delmval,p))[0]
    coh13_I_int=scipy.integrate.nquad(coh13_Iguass_fun, [delao_lims,delam_lims],args=(aval,bval,deloval,delmval,p))[0]
    #return [coh13_R_int, coh13_I_int]#coh13_R_int+1.j*coh13_I_int\
    return coh13_R_int+1.j*coh13_I_int


def rho12_integrated(delao_lims, delam_lims,aval,bval,deloval,delmval,p, peak_func):
    coh12_R_int=integrate2d_peaks(coh12_Rguass_fun, delao_lims, delam_lims, peak_func,aval,bval,deloval,delmval,p)
    coh12_I_int=integrate2d_peaks(coh12_Rguass_fun, delao_lims, delam_lims, peak_func,aval,bval,deloval,delmval,p)
    #return [coh13_R_int, coh13_I_int]#coh13_R_int+1.j*coh13_I_int\
    #return coh13_R_int+1.j*coh13_I_int
    return coh12_R_int[0]+1.j*coh12_I_int[0],coh12_R_int[1]+1.j*coh12_I_int[1]
def rho12_integrated_quad(delao_lims, delam_lims,aval,bval,deloval,delmval,p):
    coh12_R_int=scipy.integrate.nquad(coh12_Rguass_fun, [delao_lims,delam_lims],args=(aval,bval,deloval,delmval,p))[0]
    coh12_I_int=scipy.integrate.nquad(coh12_Iguass_fun, [delao_lims,delam_lims],args=(aval,bval,deloval,delmval,p))[0]
    #return [coh13_R_int, coh13_I_int]#coh13_R_int+1.j*coh13_I_int\
    return coh12_R_int+1.j*coh12_I_int


# calculate optical rabi frequency from power in in dBm
def omega_from_Pin(Pin,p):
    epsilon0=8.854187817e-12
    hbar=1.05457e-34; # in J*s

    optP = 1e-3 * 10**(Pin/10) #incident optical power in W
    pflux = optP/(2*pi*p['freq_pump']*hbar) #photon flux (photons/sec)
    n_in = pflux * p['kappaoc']*4/(p['kappaoc']+p['kappaoi'])**2 # num intracavity photons
                                                #is this right????
    Sspot = pi*p['Wcavity']**2 #cross sectional area of
                                #optical mode
    V_cav = (Sspot*p['Lcavity_vac']+Sspot*p['Lsample']*p['nYSO']**3)/2;
    optEfield = math.sqrt(n_in*hbar*2*pi*p['freq_pump']/2/epsilon0/V_cav);
    p['Omega'] = p['d23']*optEfield/hbar*(-1);
    return p['Omega']


p = {}

p['deltamu'] = 0.
p['deltao'] = 0.


p['d13'] = 2e-32*math.sqrt(1/3)
p['d23'] = 2e-32*math.sqrt(2/3)
p['gamma13'] = p['d13']**2/(p['d13']**2+p['d23']**2)*1/11e-3
p['gamma23'] = p['d23']**2/(p['d13']**2+p['d23']**2)*1/11e-3
p['gamma2d'] = 1e6
p['gamma3d'] = 1e6
p['nbath'] = 20
p['gammamu'] = 1/(p['nbath']+1) * 1e3

p['go'] = 51.9  #optical coupling

p['No'] = 1.28e15 # number of atoms in the optical mode

p['deltac']=0 #detuning for
p['kappaoi']=2*pi*7.95e6 # intrinsic loss for optical resonator
p['kappaoc']=2*pi*1.7e6 # coupling loss for optical resonator
#p['df']=0.1e6 # how small descretisation step to take when integrating over the
            # inhomogeneous lines

p['mean_delam']=0
p['sd_delam']=2*pi*25e6/2.355  #microwave inhomogeneous broadening
                                #2.355is to turn FWHM into standard deviation
p['mean_delao']=0
p['sd_delao']=2*pi*170e6/2.355 #optical inhomogeneous broadening

p['kappami'] = 650e3*2*pi # intrinsic loss for microwave cavity
p['kappamc'] = 70e3*2*pi  # coupling loss for optical cavity
                        # this is for one of the two output ports
p['Nm'] = 2.22e16  #toal number of atoms
p['gm'] = 1.04 #coupling between atoms and microwave field

p['gammaoc']=2*pi*1.7e6
p['gammaoi']=2*pi*7.95e6
p['gammamc']=2*pi*70e3
p['gammami']=2*pi*650e3


muBohr=927.4009994e-26; # Bohr magneton in J/T in J* T^-1
p['mu12'] = 4.3803*muBohr # transition dipole moment for microwave cavity (J T^-1)

p['Lsample']=12e-3 # the length of the sample, in m
p['dsample']=5e-3 # the diameter of the sample, in m

p['fillfactor']=0.8 #microwave filling factor
p['freqmu'] = 5.186e9
p['freq_pump'] = 195113.36e9 #pump frequency
p['freqo']=p['freqmu']+p['freq_pump']

p['Lcavity_vac'] = 49.5e-3 # length of the vacuum part of the optical
                           # Fabry Perot (m)
p['Wcavity'] =  0.6e-3# width of optical resonator beam in sample (m)
p['nYSO'] = 1.76
p['Omega']=-492090.88755145477
#omega_from_Pin(0,p)


S13= lambda aval,bval,delo,delm,p: p['go']*p['No']*rho13_integrated_quad([-np.inf, np.inf],[-np.inf,np.inf],aval,bval,delo,delm,p)
S12= lambda aval,bval,delo,delm,p : p['gm']*p['Nm']*rho12_integrated_quad([-np.inf, np.inf],[-np.inf,np.inf],aval,bval,delo,delm,p)

def output_vec_func(out_vec,ainval,binval,deloval,delmval,p):
    aout1=out_vec[0]+1j*out_vec[1]
    bout1=out_vec[2]+1j*out_vec[3]
    aval=(ainval+aout1)/np.sqrt(p['gammaoc'])
    bval=(binval+bout1)/np.sqrt(p['gammamc'])

    aout=(-1j*np.sqrt(p['gammaoc'])*S13(aval,bval,deloval,delmval,p)+((p['gammaoc']-p['gammaoi'])/2+1j*deloval)*ainval)/((p['gammaoc']+p['gammaoi'])/2-1j*deloval)

    bout=(-1j*np.sqrt(p['gammamc'])*S12(aval,bval,deloval,delmval,p)+((p['gammamc']-p['gammami'])/2+1j*delmval)*binval)/((p['gammamc']+p['gammami'])/2-1j*delmval)

    output_out=[aout.real, aout.imag, bout.real, bout.imag]

    return output_out

def find_output(ainval,binval,deloval,delmval,p):
    output_zero= lambda x : output_vec_func(x, ainval,binval,deloval,delmval,p)-x
    output_found=scipy.optimize.fsolve(output_zero,[0,0,0,0])

    aout_found=output_found[0]+1j*output_found[1]
    bout_found=output_found[2]+1j*output_found[3]

    return aout_found, bout_found




def find_outputs(ainvals,binvals,delovals,delmvals,p):
    aoutvals=np.zeros((len(ainvals),len(binvals),len(delovals),len(delmvals)), dtype=np.complex_)
    effic_a=np.zeros((len(ainvals),len(binvals),len(delovals),len(delmvals)), dtype=np.complex_)
    boutvals=np.zeros((len(ainvals),len(binvals),len(delovals),len(delmvals)), dtype=np.complex_)
    effic_b=np.zeros((len(ainvals),len(binvals),len(delovals),len(delmvals)), dtype=np.complex_)

    start_time=time.time()
    for ii, ainval in enumerate(ainvals):
        for jj, binval in enumerate(binvals):

            if (ainval+binval) ==0:
                print('zero lol')
            else:
                for kk, deloval in enumerate(delovals):
                    for ll, delmval in enumerate(delmvals):
                            aoutvals[ii,jj,kk,ll],boutvals[ii,jj,kk,ll]=find_output(ainval,binval,deloval,delmval,p)
                            effic_a[ii,jj,kk,ll]= aoutvals[ii,jj,kk,ll]/binval
                            effic_b[ii,jj,kk,ll]= boutvals[ii,jj,kk,ll]/ainval
                            elapsed_time=time.time()-start_time
                            print('    ' + str(ii) +', '+ str(jj)+ ', ' +str(kk) + ', ' +str(ll)+ ', Time: ' + str(elapsed_time))
                            print('    '+ 'aout = ' +str(aoutvals[ii,jj,kk,ll]) + ', bout = '+str(boutvals[ii,jj,kk,ll]))
    return aoutvals, boutvals,effic_a,effic_b

S_no_detuning_integrand= lambda deloval, delmval: 1/(deloval*delmval-np.abs(p['Omega'])**2)*gauss_fun(deloval,delmval,p['mean_delao'], p['mean_delam'],p['sd_delao'],p['sd_delam'])
S_2=p['No']*p['Omega']*p['gm']*p['go']*scipy.integrate.nquad(S_no_detuning_integrand,[[-np.inf,np.inf],[-np.inf,np.inf]])[0]

effic_2= lambda omega: np.abs(4*S_2*np.sqrt(p['gammamc']*p['gammaoc'])/(4*np.abs(S_2)**2(p['gammaoc']-2j*omega)*(p['gammamc']-2j*omega)))**2
