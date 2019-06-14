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

import time


#==============================================
#define simulation parameters
filename='output_calcs/three_lvl_out_realtest3_1'
deltamacvals=np.linspace(0,2000e6,10)
deltamvals=np.linspace(0,1000e6,10)
deltaoval=1e7
binval=50
p = {}

p['deltamu'] = 0.
p['deltao'] = 0.


p['d13'] = 2e-32*math.sqrt(1/3)
p['d23'] = 2e-32*math.sqrt(2/3)
p['gamma13'] = p['d13']**2/(p['d13']**2+p['d23']**2)*1/11e-3
p['gamma23'] = p['d23']**2/(p['d13']**2+p['d23']**2)*1/11e-3
p['gamma2d'] = 1e6
p['gamma3d'] = 1e6
p['nbath'] = 0.2
p['gammamu'] = 1/(p['nbath']+1) * 1e3

p['go'] = 51.9  #optical coupling

p['No'] = 1.28e15 # number of atoms in the optical mode

p['deltac']=0 #detuning for
p['kappaoi']=2*pi*7.95e6 # intrinsic loss for optical resonator
p['kappaoc']=2*pi*1.7e6 # coupling loss for optical resonator
#p['df']=0.1e6 # how small descretisation step to take when integrating over the
            # inhomogeneous lines

p['mean_delam']=0
p['sd_delam']=2*pi*2e6  #microwave inhomogeneous broadening
                                #2.355is to turn FWHM into standard deviation
p['mean_delao']=0
p['sd_delao']=2*pi*1000e6 #optical inhomogeneous broadening

p['kappami'] = 650e3*2*pi # intrinsic loss for microwave cavity
p['kappamc'] = 70e3*2*pi  # coupling loss for optical cavity
                        # this is for one of the two output ports
p['Nm'] = 2.22e16  #toal number of atoms
p['gm'] = 1.04 #coupling between atoms and microwave field

p['gammaoc']=2*pi*1.7e6
p['gammaoi']=2*pi*7.95e6
p['gammamc']=2*pi*0.0622e6
p['gammami']=2*pi*5.69e6


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

#=====================================================
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



def gauss_fun_1d(x, m,sd):
    return 1.0/(np.sqrt(2*pi)*sd)*np.exp(-(x-m)**2/(2*sd**2))

rho12_broadened_no_a=lambda x, bval, deltaval,p:(steady_rho_single(0, x,0,bval,0,deltaval,p))[1,0]*gauss_fun_1d(x,p['mean_delam'],p['sd_delam'])

def S12_fun_no_a(bval, deltaval,p):
    deltaacsd=p['sd_delam']
    deltaacmean=p['mean_delam']
    S12_R_fun=lambda x: np.real(rho12_broadened_no_a(x,bval,deltaval,p))
    S12_I_fun=lambda x: np.imag(rho12_broadened_no_a(x,bval,deltaval,p))
    S12_real=scipy.integrate.quad(S12_R_fun,-100*deltaacsd+deltaacmean, 100*deltaacsd+deltaacmean,points=(deltaval,p['mean_delam'],deltaacsd+p['mean_delam'],-deltaacsd+p['mean_delam'],10*deltaacsd+p['mean_delam'],-10*deltaacsd+p['mean_delam']))
    S12_imag=scipy.integrate.quad(S12_I_fun,-100*deltaacsd+deltaacmean, 100*deltaacsd+deltaacmean,points=(deltaval,p['mean_delam'],deltaacsd+p['mean_delam'],-deltaacsd+p['mean_delam'],10*deltaacsd+p['mean_delam'],-10*deltaacsd+p['mean_delam']))

    return (S12_real[0]+1.0j*S12_imag[0])*p['Nm']*p['gm']


def _cached_roots_legendre(n):
    """
    Cache roots_legendre results to speed up calls of the fixed_quad
    function.
    """
    if n in _cached_roots_legendre.cache:
        return _cached_roots_legendre.cache[n]

    _cached_roots_legendre.cache[n] = scipy.special.roots_legendre(n)
    return _cached_roots_legendre.cache[n]


_cached_roots_legendre.cache = dict()
def rho_m_broad_single(delaoval,aval,bval,deloval,delmval, p,delamlims,n=15):
    xvals, wvals = _cached_roots_legendre(n)
    yvals = (delamlims[1]-delamlims[0])*(xvals+1)/2.0 + delamlims[0]
    S_out=np.zeros((3,3),dtype=np.complex)
    for ii, yval in enumerate(yvals):
        S_out=S_out+(steady_rho_single(delaoval,yval,aval, bval,deloval,delmval,p))*gauss_fun_1d(yval,p['mean_delam'],p['sd_delam'])*wvals[ii]
    return (delamlims[1]-delamlims[0])/2.0*S_out

def rho_m_broad_full(delaoval,aval,bval,deloval,delmval, p,n=15):
    delamlims=[-50*p['sd_delam']+p['mean_delam'],50*p['sd_delam']+p['mean_delam']]
    #splitpoints=[p['mean_delam'],delmval,-11*p['sd_delam']+p['mean_delam'],11*p['sd_delam']+p['mean_delam']]
    splitpoints=[p['mean_delam'],p['sd_delam']+p['mean_delam'] ,-p['sd_delam']+p['mean_delam'],delmval,delmval+2*p['gamma2d'],delmval-2*p['gamma2d']]#,p['sd_delam']+p['mean_delam']]

    points=np.hstack((delamlims[0],sorted(splitpoints),delamlims[1]))
    S_out_full=np.zeros((3,3),dtype=np.complex)
    for ii in range(len(points)-1):
        S_out_full=S_out_full+rho_m_broad_single(delaoval,aval,bval,deloval,delmval, p,[points[ii], points[ii+1]],n=n)
    return S_out_full

def rho_o_broad_single(aval,bval,deloval,delmval, p,delaolims,n=15):
    xvals, wvals = _cached_roots_legendre(n)
    yvals = (delaolims[1]-delaolims[0])*(xvals+1)/2.0 + delaolims[0]
    S_out=np.zeros((3,3),dtype=np.complex)
    for ii, yval in enumerate(yvals):
        S_out=S_out+(rho_m_broad_full(yval,aval, bval,deloval,delmval,p,n=n))*gauss_fun_1d(yval,p['mean_delao'],p['sd_delao'])*wvals[ii]
    return (delaolims[1]-delaolims[0])/2.0*S_out

def rho_broad_full(aval,bval,deloval,delmval, p,n=15):
    delaolims=[-50*p['sd_delao']+p['mean_delao'],50*p['sd_delao']+p['mean_delao']]
    splitpoints=[p['mean_delao'],-3*p['sd_delao']+p['mean_delao'],3*p['sd_delao']+p['mean_delao'],-p['sd_delao']+p['mean_delao'],p['sd_delao']+p['mean_delao'],deloval,deloval+2*p['gamma3d'],deloval-2*p['gamma3d']]
    #splitpoints=[p['mean_delao'],deloval]#,-11*p['sd_delao']+p['mean_delao'],11*p['sd_delao']+p['mean_delao']]
    #splitpoints=[p['mean_delao'],-2*p['sd_delao']+p['mean_delao'],2*p['sd_delao']+p['mean_delao'],deloval,deloval+2*p['gamma3d'],deloval-2*p['gamma3d']]

    points=np.hstack((delaolims[0],sorted(splitpoints),delaolims[1]))
    S_out_full=np.zeros((3,3),dtype=np.complex)
    for ii in range(len(points)-1):
        S_out_full=S_out_full+rho_o_broad_single(aval,bval,deloval,delmval, p,[points[ii], points[ii+1]],n=n)
    return S_out_full

def output_vec_func(out_vec,ainval,binval, deloval,delmval,p,gauss_n=15):
    #the vector looks like [aout.real, aout.imag,bout.real, bout.imag]
    aout1=out_vec[0]+1j*out_vec[1]
    bout1=out_vec[2]+1j*out_vec[3]
    aval=aout1/np.sqrt(p['gammaoc'])
    bval=bout1/np.sqrt(p['gammamc'])
    rho=rho_broad_full(aval,bval, deloval,delmval,p,n=gauss_n)
    S12val=rho[1,0]*p['Nm']*p['gm']
    S13val=rho[2,0]*p['No']*p['go']
    #aout=(-1j*np.sqrt(p['gammaoc'])*integrate_rho13_full(aval,bval, deloval,delmval,p)+p['gammaoc']*ainval)/((p['gammaoc']+p['gammaoi'])/2-1j*deloval)
    #bout=(-1j*np.sqrt(p['gammamc'])*integrate_rho12_full(aval,bval, deloval,delmval,p)+p['gammamc']*binval)/((p['gammamc']+p['gammami'])/2-1j*delmval)
    aout=(-1j*np.sqrt(p['gammaoc'])*S13val+p['gammaoc']*ainval)/((p['gammaoc']+p['gammaoi'])/2-1j*deloval)
    bout=(-1j*np.sqrt(p['gammamc'])*S12val+p['gammamc']*binval)/((p['gammamc']+p['gammami'])/2-1j*delmval)

    return [aout.real,aout.imag,bout.real,bout.imag]

def find_output(ainval,binval,deloval,delmval,p,gauss_n=15):
    output_zero= lambda x : output_vec_func(x, ainval,binval,deloval,delmval,p,gauss_n=gauss_n)-x
    output_found=scipy.optimize.fsolve(output_zero,[ainval.real,ainval.imag,binval.real,binval.imag])

    aout_found=output_found[0]+1j*output_found[1]
    bout_found=output_found[2]+1j*output_found[3]

    return aout_found, bout_found

def output_vec_func_no_a(out_vec,binval,delmval,p):
    #the vector looks like [aout.real, aout.imag,bout.real, bout.imag]
    bout1=out_vec[0]+1j*out_vec[1]
    bval=bout1/np.sqrt(p['gammamc'])

    bout=(-1j*np.sqrt(p['gammamc'])*S12_fun_no_a(bval,delmval,p)+p['gammamc']*binval)/((p['gammamc']+p['gammami'])/2-1j*delmval)

    return [bout.real,bout.imag]

def find_output_no_a(binval,delmval,p):
    output_zero= lambda x : output_vec_func_no_a(x,binval,delmval,p)-x
    output_found=scipy.optimize.fsolve(output_zero,[binval.real,binval.imag])

    bout_found=output_found[0]+1j*output_found[1]

    return bout_found




rho_out=np.zeros((3,3,len(deltamacvals),len(deltamvals)),dtype=np.complex_)
aoutvals=np.zeros((len(deltamacvals),len(deltamvals)),dtype=np.complex_)
boutvals=np.zeros((len(deltamacvals),len(deltamvals)),dtype=np.complex_)


start_time=time.time()
for ii, deltamacval in enumerate(deltamacvals):
    p['mean_delam']=deltamacval


    for jj, deltamval in enumerate(deltamvals):
        rho_out[:,:,ii,jj]=rho_broad_full(0,binval/np.sqrt(p['gammamc']), deltaoval,deltamval,p,n=15)
        #aoutvals[ii,jj], boutvals[ii,jj]=find_output(0,binval/np.sqrt(p['gammamc']),deltaoval,deltamval,p,gauss_n=15)
        elapsed_time=time.time()-start_time

    print('    ' + str(ii) +', Time: ' + time.ctime() +', Elapsed: '+ str(elapsed_time))

    np.savez(filename,aoutvals=aoutvals,boutvals=boutvals,binval=binval,deltamacvals=deltamacvals,deltamvals=deltamvals,p=p, deltaoval=deltaoval,rho_out=rho_out)
