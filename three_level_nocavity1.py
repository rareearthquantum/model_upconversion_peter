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
from output_calcs.c_funs_test3 import rho_broad_full

#==============================================
#define simulation parameters
filename='three_lvl_nocavity_out7'
deltamacvals=np.linspace(-5e8,5e8,41)
deltamvals=np.linspace(-5e8,5e8,41)
binval=5e-6
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
p['gm'] =  1.04 #coupling between atoms and microwave field

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


def output_vec_func(out_vec,ainval,binval, deloval,delmval,p):
    #the vector looks like [aout.real, aout.imag,bout.real, bout.imag]
    aout1=out_vec[0]+1j*out_vec[1]
    bout1=out_vec[2]+1j*out_vec[3]
    aval=aout1/np.sqrt(p['gammaoc'])
    bval=bout1/np.sqrt(p['gammamc'])
    rho=np.array(rho_broad_full(aval,bval, deloval,delmval,p))
    S12val=rho[1,0]*p['Nm']*p['gm']
    S13val=rho[2,0]*p['No']*p['go']
    #aout=(-1j*np.sqrt(p['gammaoc'])*integrate_rho13_full(aval,bval, deloval,delmval,p)+p['gammaoc']*ainval)/((p['gammaoc']+p['gammaoi'])/2-1j*deloval)
    #bout=(-1j*np.sqrt(p['gammamc'])*integrate_rho12_full(aval,bval, deloval,delmval,p)+p['gammamc']*binval)/((p['gammamc']+p['gammami'])/2-1j*delmval)
    aout=(-1j*np.sqrt(p['gammaoc'])*S13val+p['gammaoc']*ainval)/((p['gammaoc']+p['gammaoi'])/2-1j*deloval)
    bout=(-1j*np.sqrt(p['gammamc'])*S12val+p['gammamc']*binval)/((p['gammamc']+p['gammami'])/2-1j*delmval)

    return [aout.real,aout.imag,bout.real,bout.imag]

def find_output(ainval,binval,deloval,delmval,p,start_guess_vec):#,start_guess_vec=[ainval.real,ainval.imag,binval.real,binval.imag]):
    output_zero= lambda x : output_vec_func(x, ainval,binval,deloval,delmval,p)-x
    #output_found=scipy.optimize.fsolve(output_zero,[ainval.real,ainval.imag,binval.real,binval.imag])
    output_found=scipy.optimize.fsolve(output_zero,start_guess_vec)

    aout_found=output_found[0]+1j*output_found[1]
    bout_found=output_found[2]+1j*output_found[3]

    return aout_found, bout_found

def output_vec_func_no_a(out_vec,binval,delmval,p):
    #the vector looks like [aout.real, aout.imag,bout.real, bout.imag]
    bout1=out_vec[0]+1j*out_vec[1]+binval
    bval=bout1/np.sqrt(p['gammamc'])
    rho=np.array(rho_broad_full(0,bval, 0,delmval,p))

    #S12val=rho[1,0]*p['Nm']*p['gm']
    S12val=rho[1,0]*p['Nm']*p['gm']

    bout=(-1j*np.sqrt(p['gammamc'])*S12val+p['gammamc']*binval)/((p['gammamc']+p['gammami'])/2-1j*delmval)

    return [bout.real,bout.imag]-out_vec

def find_output_no_a(binval,delmval,p, start_guess_vec):
    #output_zero= lambda x : output_vec_func_no_a(x,binval,delmval,p)-x
    #output_found=scipy.optimize.fsolve(output_zero,start_guess_vec)
    output_found=scipy.optimize.fsolve(output_vec_func_no_a,start_guess_vec,args=(binval,delmval,p))

    bout_found=output_found[0]+1j*output_found[1]

    return bout_found

def b_vec_fun_no_a(b_vec,binval,delmval,p):
    bval=b_vec[0]+1j*b_vec[1]
    rho=np.array(rho_broad_full(0,bval, 0,delmval,p))
    #S12val=rho[1,0]*p['Nm']*p['gm']
    S12val=rho[1,0]*p['Nm']*p['gm']
    bval1=(-1j*S12val+np.sqrt(p['gammamc']*binval))/((p['gammamc']+p['gammami'])/2-1j*delmval)
    return [bval1.real, bval1.imag]-b_vec

def find_b_no_a(binval,delmval,p,start_guess_vec):
    #b_zero = lambda x: b_vec_fun_no_a(x,binval,delmval,p)-x
    #b_found=scipy.optimize.fsolve(b_zero,start_guess_vec)
    b_found=scipy.optimize.fsolve(b_vec_fun_no_a,start_guess_vec,args=(binval,delmval,p))
    return b_found[0]+1j*b_found[1]

def b_vec_fun_rho13_no_a(b_vec,binval,delmval,p):
    bval=b_vec[0]+1j*b_vec[1]
    rho=np.array(rho_broad_full(0,bval, 0,delmval,p))
    #S12val=rho[1,0]*p['Nm']*p['gm']
    S12val=rho[1,0]*p['Nm']*p['gm']
    bval1=(-1j*S12val+np.sqrt(p['gammamc']*binval))/((p['gammamc']+p['gammami'])/2-1j*delmval)
    return [bval1.real, bval1.imag,rho[2,0].real,rho[2,0].imag]-b_vec

def find_b_rho13_no_a(binval,delmval,p,start_guess_vec):
    #b_zero = lambda x: b_vec_fun_no_a(x,binval,delmval,p)-x
    #b_found=scipy.optimize.fsolve(b_zero,start_guess_vec)
    start_guess_vec=[start_guess_vec[0], start_guess_vec[1],0,0]
    out_found=scipy.optimize.fsolve(b_vec_fun_rho13_no_a,start_guess_vec,args=(binval,delmval,p))
    return out_found[0]+1j*out_found[1], out_found[2]+1j*out_found[3]

def b_vec_fun_a_rho13(b_vec,binval,delmval,p):
    bval=b_vec[0]+1j*b_vec[1]
    aval=b_vec[2]+1j*b_vec[3]
    rho=np.array(rho_broad_full(aval,bval, 0,delmval,p))
    #S12val=rho[1,0]*p['Nm']*p['gm']
    S12val=rho[1,0]*p['Nm']*p['gm']
    aval1=1j*rho[2,0]*p['No']*p['go']/delmval
    #print(aval1)
    bval1=(-1j*S12val+np.sqrt(p['gammamc']*binval))/((p['gammamc']+p['gammami'])/2-1j*delmval)
    #return [bval1.real, bval1.imag,aval.real,aval.imag, rho[2,0].real,rho[2,0].imag]-b_vec
    return [bval1.real, bval1.imag,aval1.real,aval1.imag]-b_vec

def find_b_a_rho13(binval,delmval,p,start_guess_vec):
    #b_zero = lambda x: b_vec_fun_no_a(x,binval,delmval,p)-x
    #b_found=scipy.optimize.fsolve(b_zero,start_guess_vec)
    #start_guess_vec=[start_guess_vec[0], start_guess_vec[1],0,0,0,0]
#    start_guess_vec=[start_guess_vec[0], start_guess_vec[1],0,0,0,0]

    out_found=scipy.optimize.fsolve(b_vec_fun_a_rho13,start_guess_vec,args=(binval,delmval,p))
    return out_found[0]+1j*out_found[1], out_found[2]+1j*out_found[3]#, out_found[4]+1j*out_found[5]

rho_out=np.zeros((3,3,len(deltamacvals),len(deltamvals)),dtype=np.complex_)
rho_outm=np.zeros((3,3,len(deltamacvals),len(deltamvals)),dtype=np.complex_)
rho_outd=np.zeros((3,3,len(deltamacvals),len(deltamvals)),dtype=np.complex_)

boutvals=np.zeros((len(deltamacvals),len(deltamvals)),dtype=np.complex_)
bvals=np.zeros((len(deltamacvals),len(deltamvals)),dtype=np.complex_)
rho_out_bout=np.zeros((3,3,len(deltamacvals),len(deltamvals)),dtype=np.complex_)
rho_out_b=np.zeros((3,3,len(deltamacvals),len(deltamvals)),dtype=np.complex_)
bvals2=np.zeros((len(deltamacvals),len(deltamvals)),dtype=np.complex_)
rho13_out=np.zeros((len(deltamacvals),len(deltamvals)),dtype=np.complex_)
avals=np.zeros((len(deltamacvals),len(deltamvals)),dtype=np.complex_)

start_time=time.time()
for ii, deltamacval in enumerate(deltamacvals):
    p['mean_delam']=deltamacval
    for jj, deltamval in enumerate(deltamvals):
        if jj==0:
            start_guess_vec_bout=[binval.real,binval.imag]
            start_guess_vec_b=[binval.real/p['gammamc'],binval.imag/p['gammamc']]
            #start_guess_vec_b2=[binval.real/p['gammamc'],binval.imag/p['gammamc'],0,0]

        else:
            start_guess_vec_bout=[boutvals[ii,jj-1].real,boutvals[ii,jj-1].imag]
            start_guess_vec_b=[bvals[ii,jj-1].real,bvals[ii,jj-1].imag]
            #start_guess_vec_b2=[bvals2[ii,jj-1].real,bvals2[ii,jj-1].imag,avals[ii,jj-1].real,avals[ii,jj-1].imag]

        boutvals[ii,jj]=find_output_no_a(binval,deltamval,p,start_guess_vec_bout)#=start_guess_vec)
        #bvals[ii,jj]=find_b_no_a(binval,deltamval,p,start_guess_vec_b)
        rho_out_bout[:,:,ii,jj]=rho_broad_full(0,(boutvals[ii,jj])/np.sqrt(p['gammamc']), 0,deltamval,p)
        #rho_out_b[:,:,ii,jj]=rho_broad_full(0,bvals[ii,jj]+binval/np.sqrt(p['gammamc']), 0,deltamval,p)
        #bvals2[ii,jj], avals[ii,jj]=find_b_a_rho13(binval,deltamval,p,start_guess_vec_b2)
        #rho_outm[:,:,ii,jj]=rho_broad_full(0,boutvals[ii,jj]*np.sqrt(p['gammamc']), 0,deltamval,p)
        #rho_outd[:,:,ii,jj]=rho_broad_full(0,boutvals[ii,jj], 0,deltamval,p)

        elapsed_time=time.time()-start_time
        #print('aout = ' +str(aoutvals[ii,jj])+', bout = ' + str(boutvals[ii,jj]))

    print('    ' + str(ii) +', Time: ' + time.ctime() +', Elapsed: '+ str(elapsed_time))

    #np.savez(filename,boutvals=boutvals,binval=binval,deltamacvals=deltamacvals,deltamvals=deltamvals,p=p,rho_out_b=rho_out_b, rho_out_bout=rho_out_bout,bvals=bvals,bvals2=bvals2,rho13_out_b2=rho13_out_b2)
    np.savez(filename,binval=binval,deltamacvals=deltamacvals,deltamvals=deltamvals,p=p,boutvals=boutvals,avals=avals,rho_out_bout=rho_out_bout,bvals=bvals,rho_out_b=rho_out_b)
