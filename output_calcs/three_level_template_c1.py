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
from c_funs_test import steady_rho_single_c

#==============================================
#define simulation parameters
filename='output_calcs/three_lvl_testc1'
deltamacvals=np.linspace(0,0.4e9,6)
deltamvals=np.linspace(0,2e8,6)
deltaoval=1e7
binval=50
p = {}
p['nbath']=1

p['deltamu'] = 0.
p['deltao'] = 0.


p['d13'] = 2e-32*math.sqrt(1/3)
p['d23'] = 2e-32*math.sqrt(2/3)
p['gamma13'] = p['d13']**2/(p['d13']**2+p['d23']**2)*1/11e-3
p['gamma23'] = p['d23']**2/(p['d13']**2+p['d23']**2)*1/11e-3
p['gamma2d'] = 1e6
p['gamma3d'] = 1e6
#p['nbath'] = 20
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
# def steady_rho_single_c(double delta_a_o, double delta_a_mu, double complex aval,double complex bval,double delta_o,double delta_mu,p):
#
#     cdef double Lreal[81]
#     cdef double V[9]
#     cdef int workspacec[9]
#     cdef int info
#     cdef int lda = 9
#     cdef int ldb = 9
#     cdef int nrhs = 1
#     cdef int n = 9
#     cdef double gamma_13 = p['gamma13']
#     cdef double gamma_23 = p['gamma23']
#     cdef double gamma_2d = p['gamma2d']
#     cdef double gamma_3d = p['gamma3d']
#     cdef double n_b = p['nbath']
#     cdef double gamma_mu = p['gammamu']
#     cdef double g_mu = p['gm']
#     cdef double g_o = p['go']
#     cdef double Omega = p['Omega']
#     cdef double a_r=aval.real
#     cdef double a_i=aval.imag
#     cdef double b_r=bval.real
#     cdef double b_i=bval.imag
#     V[:]=[1,0,0, 0,0,0, 0,0,0]
#
#     Lreal[0] =1
#     Lreal[1] =gamma_mu*n_b
#     Lreal[2] =0
#     Lreal[3] =-b_i*g_mu
#     Lreal[4] =b_r*g_mu
#     Lreal[5] =-a_i*g_o
#     Lreal[6] =a_r*g_o
#     Lreal[7] =0
#     Lreal[8] =0
#     Lreal[9] =1
#     Lreal[10] =-gamma_mu*(n_b + 1)
#     Lreal[11] =0
#     Lreal[12] =b_i*g_mu
#     Lreal[13] =-b_r*g_mu
#     Lreal[14] =0
#     Lreal[15] =0
#     Lreal[16] =0
#     Lreal[17] =Omega
#     Lreal[18] =1
#     Lreal[19] =gamma_23
#     Lreal[20] =-gamma_13 - gamma_23
#     Lreal[21] =0
#     Lreal[22] =0
#     Lreal[23] =a_i*g_o
#     Lreal[24] =-a_r*g_o
#     Lreal[25] =0
#     Lreal[26] =-Omega
#     Lreal[27] =0
#     Lreal[28] =-2*b_i*g_mu
#     Lreal[29] =0
#     Lreal[30] =-gamma_2d/2 - gamma_mu*n_b - gamma_mu/2
#     Lreal[31] =delta_a_mu - delta_mu
#     Lreal[32] =0
#     Lreal[33] =Omega
#     Lreal[34] =-a_i*g_o
#     Lreal[35] =a_r*g_o
#     Lreal[36] =0
#     Lreal[37] =2*b_r*g_mu
#     Lreal[38] =0
#     Lreal[39] =-delta_a_mu + delta_mu
#     Lreal[40] =-gamma_2d/2 - gamma_mu*n_b - gamma_mu/2
#     Lreal[41] =-Omega
#     Lreal[42] =0
#     Lreal[43] =a_r*g_o
#     Lreal[44] =a_i*g_o
#     Lreal[45] =0
#     Lreal[46] =0
#     Lreal[47] =-2*a_i*g_o
#     Lreal[48] =0
#     Lreal[49] =Omega
#     Lreal[50] =-gamma_13/2 - gamma_23/2 - gamma_3d/2 - gamma_mu*n_b/2
#     Lreal[51] =delta_a_o - delta_o
#     Lreal[52] =-b_i*g_mu
#     Lreal[53] =-b_r*g_mu
#     Lreal[54] =0
#     Lreal[55] =0
#     Lreal[56] =2*a_r*g_o
#     Lreal[57] =-Omega
#     Lreal[58] =0
#     Lreal[59] =-delta_a_o + delta_o
#     Lreal[60] =-gamma_13/2 - gamma_23/2 - gamma_3d/2 - gamma_mu*n_b/2
#     Lreal[61] =b_r*g_mu
#     Lreal[62] =-b_i*g_mu
#     Lreal[63] =0
#     Lreal[64] =0
#     Lreal[65] =0
#     Lreal[66] =a_i*g_o
#     Lreal[67] =-a_r*g_o
#     Lreal[68] =b_i*g_mu
#     Lreal[69] =-b_r*g_mu
#     Lreal[70] =-gamma_13/2 - gamma_23/2 - gamma_2d/2 - gamma_3d/2 - gamma_mu*(n_b + 1)/2
#     Lreal[71] =-delta_a_mu + delta_a_o + delta_mu - delta_o
#     Lreal[72] =0
#     Lreal[73] =-2*Omega
#     Lreal[74] =2*Omega
#     Lreal[75] =-a_r*g_o
#     Lreal[76] =-a_i*g_o
#     Lreal[77] =b_r*g_mu
#     Lreal[78] =b_i*g_mu
#     Lreal[79] =delta_a_mu - delta_a_o - delta_mu + delta_o
#     Lreal[80] =-gamma_13/2 - gamma_23/2 - gamma_2d/2 - gamma_3d/2 - gamma_mu*(n_b + 1)/2
#
#     #rho = np.linalg.solve(L_mat.astype('complex128'),np.matrix([[1,0,0,0,0,0,0,0,0]]).T)
#     lapack.dgesv(&n,&nrhs,Lreal,&lda,workspacec,V,&ldb, &info)
#     #lapack.dgesv(&n,&nrhs,L,&lda,workspace,V,&ldb,&info)
#     #return V
#     #return [V[3]+1j*V[4],V[5]+1j*V[6]]
#     return [[V[0],V[3]+1j*V[4],V[5]+1j*V[6]],[V[3]-1j*V[4],V[1],V[7]+1j*V[8]],[V[5]-1j*V[6],V[7]-1j*V[8],V[2]]]
print(steady_rho_single_c(0, 0, 5,50,10e7,3e8,p))

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


def gauss_fun_1d(x, m,sd):
    return 1.0/(np.sqrt(2*pi)*sd)*np.exp(-(x-m)**2/(2*sd**2))

rho12_broadened_no_a=lambda x, bval, deltaval,p:(steady_rho_single_c(0, x,0,bval,0,deltaval,p))[1,0]*gauss_fun_1d(x,p['mean_delam'],p['sd_delam'])

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
        S_out=S_out+(steady_rho_single_c(delaoval,yval,aval, bval,deloval,delmval,p))*gauss_fun_1d(yval,p['mean_delam'],p['sd_delam'])*wvals[ii]
    return (delamlims[1]-delamlims[0])/2.0*S_out

def rho_m_broad_full(delaoval,aval,bval,deloval,delmval, p,n=15):
    delamlims=[-50*p['sd_delam']+p['mean_delam'],50*p['sd_delam']+p['mean_delam']]
    #splitpoints=[p['mean_delam'],delmval,-11*p['sd_delam']+p['mean_delam'],11*p['sd_delam']+p['mean_delam']]
    ds_m=find_dressed_states_m(delaoval, deloval,delmval,bval,p)[0].real
    #ds_test=(steady_rho_single(delaoval,ds_m,aval, bval,deloval,delmval,p))
    splitpoints=[p['mean_delam'],p['sd_delam']+p['mean_delam'] ,-p['sd_delam']+p['mean_delam'],delmval,delmval+2*p['gamma2d'],delmval-2*p['gamma2d']]#,p['sd_delam']+p['mean_delam']]
    if not np.isnan(ds_m) and not ds_m in splitpoints:
        splitpoints.append(ds_m)
    points=np.array(sorted(np.hstack((delamlims[0],(splitpoints),delamlims[1]))))
    #print(ds_m)
    #print(type(ds_m))
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
        aoutvals[ii,jj], boutvals[ii,jj]=find_output(0,binval/np.sqrt(p['gammamc']),deltaoval,deltamval,p,gauss_n=15)
        elapsed_time=time.time()-start_time
        print('aout = ' +str(aoutvals[ii,jj])+', bout = ' + str(boutvals[ii,jj]))

    print('    ' + str(ii) +', Time: ' + time.ctime() +', Elapsed: '+ str(elapsed_time))

    np.savez(filename,aoutvals=aoutvals,boutvals=boutvals,binval=binval,deltamacvals=deltamacvals,deltamvals=deltamvals,p=p, deltaoval=deltaoval)
