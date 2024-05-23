
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
import c_linear_ground4
import c_linear_ground_slow_af
import c_linear_ground5

import time
delaovals=np.linspace(-2e8,2e8,1)
delamvals=np.linspace(-2e8,2e8,1)
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
    mlines=np.zeros((11,len(delaovals)))
    for ii, delaoval in enumerate(delaovals):
        #ds_m_val=find_dressed_states_m(delaoval,deloval,delmval,bval,p)
        #mlines[:,ii]=[-(p['gm']*bval)**2/delao*2,delao,delao/2+np.sqrt(delao**2/4-(p['gm']*bval)**2),delao/2-np.sqrt(delao**2/4-(p['gm']*bval)**2),ds_m_val]
        #delao=delao+p['gamma2d']*10000
        # mlines[:,ii]=np.array([delaoval-deloval+delmval,delaoval-deloval+delmval+p['gamma2d'],delaoval-deloval+delmval-p['gamma2d'],
        # delaoval-deloval+delmval+p['gamma2d']*70,delaoval-deloval+delmval-p['gamma2d']*70,
        # delaoval-deloval+delmval+p['gamma2d']*200,delaoval-deloval+delmval-p['gamma2d']*200,
        # delaoval-deloval+delmval+p['gamma2d']*700,delaoval-deloval+delmval-p['gamma2d']*700,
        # delmval])
    #     mlines[:,ii]=np.array([delaoval-deloval+delmval,delaoval-deloval+delmval+p['gamma2d'],delaoval-deloval+delmval-p['gamma2d'],
    # delaoval-deloval+delmval+p['gamma2d']*70,delaoval-deloval+delmval-p['gamma2d']*70,
    # delaoval-deloval+delmval+p['gamma2d']*200,delaoval-deloval+delmval-p['gamma2d']*200,
    # delaoval-deloval+delmval+p['gamma2d']*700,delaoval-deloval+delmval-p['gamma2d']*700,
    # delmval,delmval+p['gamma2d'],delmval+p['gamma2d']*100,delmval+p['gamma2d']*500,
    # delmval-p['gamma2d'],delmval-p['gamma2d']*100,delmval-p['gamma2d']*500,
    # -50*p['sd_delam']+p['mean_delam'],50*p['sd_delam']+p['mean_delam'],p['mean_delam'],
    # p['sd_delam']+p['mean_delam'] ,-p['sd_delam']+p['mean_delam'],3*p['sd_delam']+p['mean_delam'] ,-3*p['sd_delam']+p['mean_delam'],0])
        mlines[:,ii]=np.array([
    delmval,delmval+p['gamma2d'],delmval+p['gamma2d']*100,delmval+p['gamma2d']*500,
    delmval-p['gamma2d'],delmval-p['gamma2d']*100,delmval-p['gamma2d']*500,delmval+p['gamma2d']*3,delmval+p['gamma2d']*10,delmval-p['gamma2d']*3,delmval-p['gamma2d']*10])
    return mlines.T
def lines_o(delamvals,deloval,delmval,p):
    olines=np.zeros((19,len(delamvals)))
    for ii, delamval in enumerate(delamvals):
        olines[:,ii]=np.array([-50*p['sd_delao']+p['mean_delao'],50*p['sd_delao']+p['mean_delao'],p['mean_delao'],-3*p['sd_delao']+p['mean_delao'],3*p['sd_delao']+p['mean_delao'],
    -p['sd_delao']+p['mean_delao'],p['sd_delao']+p['mean_delao'],
    deloval,deloval+p['gamma3d'],deloval+p['gamma3d']*5,deloval+p['gamma3d']*50,
    deloval-p['gamma3d'],deloval-p['gamma3d']*5,deloval-p['gamma3d']*50,
    deloval-p['gamma3d']*250,deloval+p['gamma3d']*250,deloval-p['gamma3d']*750,deloval+p['gamma3d']*750,0])
    return olines.T
rho0=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhoa=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhoac=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhob=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhobc=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)

rho02=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhoa2=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhoac2=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhob2=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhobc2=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)

delo=.0e8
delm=-0e8
t1=time.time()
for ii, deloval in enumerate(delaovals):
    #print(ii)
    for jj, delmval in enumerate(delamvals):
        S_out_full=c_linear_ground5.steady_rhos(deloval,delmval, delo,delm,p)
        #print(S_out_full)
        S_out_fullpy=np.zeros(45)
        for ll,newind in enumerate([ 0,  1,  2,  7,  8, 12, 13, 14, 15, 21, 22, 23, 24, 30, 31, 32, 33,39, 40, 41, 42]):
            S_out_fullpy[newind]=S_out_full[ll]
        #print(S_out_fullpy)
        CtoRinv=np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, -1j, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1j, 0, 0],
        [0, 0, 0, 1, 1j, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, -1j],
        [0, 0, 0, 0, 0, 1, 1j, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1j],
        [0, 0, 1, 0, 0, 0, 0, 0, 0]])

        rho0[:,:,ii,jj]=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[0:9])),(3,3))
        rhoar=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[9:18])),(3,3))
        rhoai=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[18:27])),(3,3))
        rhobr=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[27:36])),(3,3))
        rhobi=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[36:45])),(3,3))
        rhoa[:,:,ii,jj]=rhoar + 1j*rhoai
        rhoac[:,:,ii,jj]=rhoar - 1j*rhoai
        rhob[:,:,ii,jj]=rhobr + 1j*rhobi
        rhobc[:,:,ii,jj]=rhobr - 1j*rhobi
        t1tot=time.time()-t1
        #print(t1tot)
print(t1tot)
for ii, deloval in enumerate(delaovals):
    #print(ii)
    for jj, delmval in enumerate(delamvals):
        S_out_full=c_linear_ground_slow_af.steady_rhos(deloval,delmval, delo,delm,p)
        #print(S_out_full)
        S_out_fullpy=np.zeros(45)
        for ll,newind in enumerate([ 0,  1,  2,  7,  8, 12, 13, 14, 15, 21, 22, 23, 24, 30, 31, 32, 33,39, 40, 41, 42]):
            S_out_fullpy[newind]=S_out_full[ll]
        #print(S_out_fullpy)
        CtoRinv=np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, -1j, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1j, 0, 0],
        [0, 0, 0, 1, 1j, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, -1j],
        [0, 0, 0, 0, 0, 1, 1j, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1j],
        [0, 0, 1, 0, 0, 0, 0, 0, 0]])

        rho02[:,:,ii,jj]=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[0:9])),(3,3))
        rhoar=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[9:18])),(3,3))
        rhoai=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[18:27])),(3,3))
        rhobr=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[27:36])),(3,3))
        rhobi=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[36:45])),(3,3))
        rhoa2[:,:,ii,jj]=rhoar + 1j*rhoai
        rhoac2[:,:,ii,jj]=rhoar - 1j*rhoai
        rhob2[:,:,ii,jj]=rhobr + 1j*rhobi
        rhobc2[:,:,ii,jj]=rhobr - 1j*rhobi
        t1tot=time.time()-t1
print(t1tot)
mlines=lines_m(delaovals,delo,delm,p)
olines=lines_o(delamvals,delo,delm,p)
# fig=plt.figure()
# pltnum=1
# fig.clf()
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
# titles=['rho0','rhoa','rhoac','rhob', 'rhobc']
# for kk, rhox in enumerate([rho0,rhoa,rhoac,rhob, rhobc]):
#     fig=plt.figure(titles[kk])
#     pltnum=1
#     fig.clf()
#     for ii in range(3):
#         for jj in range(3):
#
#             ax=fig.add_subplot(3,3,pltnum)
#             img1=ax.imshow((np.abs(rhox[ii,jj,:,:])),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
#             #plt.plot(mlines,delaovals)
#             #plt.plot(delamvals,olines)
#             plt.xlabel('del mu')
#             plt.ylabel('del o')
#             plt.title(str(ii)+', ' +str(jj))
#             fig.colorbar(img1)
#             pltnum=pltnum+1

# titles=['rho02','rhoa2','rhoac2','rhob2', 'rhobc2']
# for kk, rhox in enumerate([rho02,rhoa2,rhoac2,rhob2, rhobc2]):
#     fig=plt.figure(titles[kk])
#     pltnum=1
#     fig.clf()
#     for ii in range(3):
#         for jj in range(3):
#
#             ax=fig.add_subplot(3,3,pltnum)
#             img1=ax.imshow((np.abs(rhox[ii,jj,:,:])),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
#             #plt.plot(mlines,delaovals)
#             #plt.plot(delamvals,olines)
#             plt.xlabel('del mu')
#             plt.ylabel('del o')
#             plt.title(str(ii)+', ' +str(jj))
#             fig.colorbar(img1)
#             pltnum=pltnum+1
# for kk, rhox in enumerate([rho02,rhoa2,rhoac2,rhob2, rhobc2]):
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
# plt.figure()
# fig=plt.figure('rhoa_diff')
# pltnum=1
# fig.clf()
# for ii in range(3):
#     for jj in range(3):
#
#         ax=fig.add_subplot(3,3,pltnum)
#         img1=ax.imshow((np.abs(rhoa[ii,jj,:,:]-np.conj(rhoac[jj,ii,:,:]))),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
#         plt.xlabel('del mu')
#         plt.ylabel('del o')
#         plt.title(str(ii)+', ' +str(jj))
#         fig.colorbar(img1)
#         pltnum=pltnum+1
        # plt.plot(mlines,delaovals)
# plt.figure()
# plt.plot(delamvals,olines)

plt.show()
