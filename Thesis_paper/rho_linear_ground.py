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
import scipy.io as sio
from matplotlib.colors import Normalize as Norm

import time
import Linear1.c_linear_ground5 as c_linear_ground5

from Thesis_figs.Ground_params2 import p, sd_delao_from_B, deltaao_from_B
P_pump = 10*np.log10(1.74) #in dBm, 1.74 mW are going into the resonator
P_mu = -30 # in dBm
T=10e-3
delaovals=np.linspace(-1,1,551)*1e7
delamvals=np.linspace(-1,1,551)*1e7

Q=1e9
# Q=2*pi*p['freq_pump']/p['gammaoi']

p['gammaoi']=2*pi*p['freq_pump']/Q
print('gammaoi = ' + str(p['gammaoi']))

def Omega_from_PdBm(PdBm,p):
    mu0=4*np.pi*1e-7
    c=3e8
    hbar=1.05457e-34; # in J*s
    P=1e-3*10**(PdBm/10)
    Abeam=pi*p['Wbeam']**2/4
    Efield=np.sqrt(2*mu0*c*P/Abeam)
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
def nbath_from_T(T,p,deltamacval=0):
    omega=np.abs(2*pi*p['freqmu']+deltamacval)
    nbath = 1/(np.exp(1.0545718e-34*omega/1.38064852e-23/T)-1)
    return nbath
p['Omega']= Omega_cavity_from_PdBm_resonance(P_pump,p)
p['nbath']=nbath_from_T(T,p)
rho0=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhoa=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhoac=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhob=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)
rhobc=np.zeros((3,3,len(delaovals),len(delamvals)),dtype=np.complex_)

delo=.2e8*0
delm=-0.4e8*0
t1=time.time()
for ii, deloval in enumerate(delaovals):
    #print(ii)
    for jj, delmval in enumerate(delamvals):
        # p['nbath']=nbath_from_T(T,p,deltamacval=delmval)
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

# filename='Linear_rho_ground1'
# np.savez('Thesis_figs/Linear_coherences/'+filename,delamvals=delamvals,delaovals=delaovals,p=p,P_pump=P_pump,T=T,
#         rho0=rho0,rhoa=rhoa,rhoac=rhoac,rhob=rhob,rhobc=rhobc,delo=delo,delm=delm)#,deltaoval=deltaoval)

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
titles=['rho0','rhoa','rhoac','rhob', 'rhobc']
titles=['rho0','rhoa','rhob']

# for kk, rhox in enumerate([rho0,rhoa,rhoac,rhob, rhobc]):
# for kk, rhox in enumerate([rho0,rhoa,rhob]):
#
#     fig=plt.figure(titles[kk]+'T='+str(T))
#     pltnum=1
#     fig.clf()
#     for ii in range(3):
#         for jj in range(3):
#             rhox[abs(rhox)<1e-16]=0
#             ax=fig.add_subplot(3,3,pltnum)
#             img1=ax.imshow(np.log10((np.abs(rhox[ii,jj,:,:]))),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
#             #plt.plot(mlines,delaovals)
#             #plt.plot(delamvals,olines)
#             plt.xlabel('del mu')
#             plt.ylabel('del o')
#             plt.title(str(ii)+', ' +str(jj))
#             fig.colorbar(img1)
#             pltnum=pltnum+1
#

fig=plt.figure('rho0_T='+str(T))
pltnum=1
for ii in range(3):
    for jj in range(3):
        rho0[abs(rho0)<1e-20]=0
        ax=fig.add_subplot(3,3,pltnum)
        img1=ax.imshow(((np.abs(rho0[ii,jj,:,:]))),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
        #plt.plot(mlines,delaovals)
        #plt.plot(delamvals,olines)
        plt.xlabel('del mu')
        plt.ylabel('del o')
        plt.title(str(ii)+', ' +str(jj))
        fig.colorbar(img1)
        pltnum=pltnum+1

fig=plt.figure('log10_rhob_rhoa_T='+str(T))
ax=fig.add_subplot(2,2,1)
img1=ax.imshow(np.log10((np.abs(rhob[1,0,:,:]))),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
plt.title('rhob12')
fig.colorbar(img1)
ax=fig.add_subplot(2,2,2)
img1=ax.imshow(np.log10((np.abs(rhob[2,0,:,:]))),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
plt.title('rhob13')
fig.colorbar(img1)

ax=fig.add_subplot(2,2,3)
img1=ax.imshow(np.log10((np.abs(rhoa[1,0,:,:]))),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
plt.title('rhoa12')
fig.colorbar(img1)
ax=fig.add_subplot(2,2,4)
img1=ax.imshow(np.log10((np.abs(rhoa[2,0,:,:]))),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
plt.title('rhoa13')
fig.colorbar(img1)

fig=plt.figure('lin_rhob_rhoa_T='+str(T))
ax=fig.add_subplot(2,2,1)
img1=ax.imshow(((np.abs(rhob[1,0,:,:]))),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
plt.title('rhob12')
fig.colorbar(img1)
ax=fig.add_subplot(2,2,2)
img1=ax.imshow(((np.abs(rhob[2,0,:,:]))),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
plt.title('rhob13')
fig.colorbar(img1)

ax=fig.add_subplot(2,2,3)
img1=ax.imshow(((np.abs(rhoa[1,0,:,:]))),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
plt.title('rhoa12')
fig.colorbar(img1)
ax=fig.add_subplot(2,2,4)
img1=ax.imshow(((np.abs(rhoa[2,0,:,:]))),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
plt.title('rhoa13')
fig.colorbar(img1)


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
# plt.figure()
# plt.plot(mlines,delaovals)
# plt.figure()
# plt.plot(delamvals,olines)

plt.show()
