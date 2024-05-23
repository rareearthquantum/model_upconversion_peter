
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
#import c_linear_ground4
#import c_linear_ground_testbounds as c_linear_ground
import c_linear_ground4 as c_linear_ground

import time
delaovals=np.linspace(-1e10,1e10,21)
delamvals=np.linspace(-1e10,1e10,21)
p = {}
p['nbath']=20

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
def lines_m(delaovals,deloval,delmval,p):
    mlines=np.zeros((3,len(delaovals)))
    for ii, delaoval in enumerate(delaovals):
        #ds_m_val=find_dressed_states_m(delaoval,deloval,delmval,bval,p)
        #mlines[:,ii]=[-(p['gm']*bval)**2/delao*2,delao,delao/2+np.sqrt(delao**2/4-(p['gm']*bval)**2),delao/2-np.sqrt(delao**2/4-(p['gm']*bval)**2),ds_m_val]
        #delao=delao+p['gamma2d']*10000
        mlines[:,ii]=np.array([delaoval-delo+delm,delaoval-delo+delm+p['gamma2d']*50,delaoval-delo+delm+p['gamma3d']*1000])
    return mlines.T
rho0=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhoa=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhoac=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhob=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhobc=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
delo=0e8
delm=-0e8
t1=time.time()
for ii, deloval in enumerate(delaovals):
    print(ii)
    for jj, delmval in enumerate(delamvals):
        rho0[:,:,ii,jj],rhoa[:,:,ii,jj],rhoac[:,:,ii,jj],rhob[:,:,ii,jj],rhobc[:,:,ii,jj]=c_linear_ground.rho_broad_full_real_imag(deloval,delmval, 0,p)
        #print(time.time()-t1)
        t1tot=time.time()-t1
print(t1tot)

# fig=plt.figure()
# pltnum=1
# fig.clf()matmul
# for ii in range(3):
#     for jj in range(3):
#
#         ax=fig.add_subplot(3,3,pltnum)
#         img1=ax.imshow((np.abs(rhoa[ii,jj,:,:])),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
#
#         plt.xlabel('del mu')
#         plt.ylabel('del o')
#         plt.title(str(ii)+', ' +str(jj))
#         fig.colorbar(img1)
#         pltnum=pltnum+1
titles=['rho0','rhoar','rhoai','rhobr', 'rhobi']
for kk, rhox in enumerate([rho0,rhoa,rhoac,rhob, rhobc]):
    fig=plt.figure(titles[kk])
    pltnum=1
    fig.clf()
    for ii in range(3):
        for jj in range(3):

            ax=fig.add_subplot(3,3,pltnum)
            img1=ax.imshow((np.abs(rhox[ii,jj,:,:])),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')

            plt.xlabel('del mu')
            plt.ylabel('del o')
            plt.title(str(ii)+', ' +str(jj))
            fig.colorbar(img1)
            pltnum=pltnum+1
# for kk, rhox in enumerate([rho0,rhoa,rhoac,rhob, rhobc]):
#     fig=plt.figure(titles[kk]+' phase')
#     pltnum=1
#     fig.clf()
#     for ii in range(3):
#         for jj in range(3):
#
#             ax=fig.add_subplot(3,3,pltnum)
#             img1=ax.imshow((np.angle(rhox[ii,jj,:,:])),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
#
#             plt.xlabel('del mu')
#             plt.ylabel('del o')
#             plt.title(str(ii)+', ' +str(jj))
#             fig.colorbar(img1)
#             pltnum=pltnum+1
#
plt.show()
