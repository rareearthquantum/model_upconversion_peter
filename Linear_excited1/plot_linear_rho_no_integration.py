
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
import c_linear_excited3 as c_linear_excited

import time
delaovals=np.linspace(-1e8,1e8,251)/3
delamvals=np.linspace(-1e8,1e8,251)/3
delaovals=np.linspace(-1e8,1e8,21)*1
delamvals=np.linspace(-1e8,1e8,201)*100
P_pump = 10*np.log10(100.74) #in dBm, 1.74 mW are going into the resonator
P_pump = 10*np.log10(100.74) #in dBm, 1.74 mW are going into the resonator
P_pump_mW=100
P_pump = 10*np.log10(P_pump_mW)
P_mu = -15-30 # in dBm
T=100e-3*0
P_mu = 156000 # in dBm


p={}
p['freqmu']=5015e6#4733e6 #this is the microwave cavity frequency
p['freq_pump'] = 195116.71e9 #pump frequency
#p['freqo']=p['freqmu']+p['freq_pump']

#p['nbath']=nbath_from_T(T,p)
p['d13'] = 2e-32*math.sqrt(2/3)
p['d12'] = 2e-32*math.sqrt(1/3)
p['gamma13'] = p['d13']**2/(p['d13']**2+p['d12']**2)*1/11e-3
p['gamma12'] = p['d12']**2/(p['d13']**2+p['d12']**2)*1/11e-3
p['gamma2d'] = 1e6
p['gamma3d'] = 1e6
#p['nbath'] = 20
p['gamma23'] = 1/11#1/(p['nbath']+1) * 1e3
p['sd_delam']=2*pi*2e6
p['gammaoc']=2*pi*1.7e6
p['gammaoi']=2*pi*7.95e6
p['gammamc']=2*pi*0.0622e6
p['gammami']=2*pi*5.69e6
p['gammaoc']=2*pi*1.7e6
p['gammaoi']=2*pi*7.95e6
p['gammamc']=70e3*2*pi
p['gammami']=650e3*2*pi
Q=1e9
p['gammaoi']=2*pi*p['freq_pump']/Q
print('gammaoi = ' + str(p['gammaoi']))


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
p['nYSO'] = 1.76
#functions to define simulation parameters which change as we change the frequencies
def Omega_from_PdBm(PdBm,p):
    mu0=4*pi*1e-7
    c=3e8
    hbar=1.05457e-34; # in J*s
    P=1e-3*10**(PdBm/10)
    Abeam=pi*p['Wbeam']**2/4
    Efield=np.sqrt(2*mu0*c*P/Abeam)
    Omega=p['d12']*Efield/hbar
    return Omega
def nbath_from_T(T,p,deltamacval=0):
    omega=np.abs(2*pi*p['freqmu']-deltamacval)
    nbath = 1/(np.exp(1.0545718e-34*omega/1.38064852e-23/T)-1)
    return nbath

p['nbath']=nbath_from_T(T,p)

p['Omega']= Omega_from_PdBm(P_pump,p)

print(p)
def lines_m(delaovals,deloval,delmval,p):
    mlines=np.zeros((6,len(delaovals)))
    for ii, delaoval in enumerate(delaovals):
        #ds_m_val=find_dressed_states_m(delaoval,deloval,delmval,bval,p)
        #mlines[:,ii]=[-(p['gm']*bval)**2/delao*2,delao,delao/2+np.sqrt(delao**2/4-(p['gm']*bval)**2),delao/2-np.sqrt(delao**2/4-(p['gm']*bval)**2),ds_m_val]
        #delao=delao+p['gamma2d']*10000
        # mlines[:,ii]=np.array([delaoval-deloval+delmval,delaoval-deloval+delmval+p['gamma2d'],delaoval-deloval+delmval-p['gamma2d'],
        # delaoval-deloval+delmval+p['gamma2d']*70,delaoval-deloval+delmval-p['gamma2d']*70,
        # delaoval-deloval+delmval+p['gamma2d']*200,delaoval-deloval+delmval-p['gamma2d']*200,
        # delaoval-deloval+delmval+p['gamma2d']*700,delaoval-deloval+delmval-p['gamma2d']*700,
        # delmval])
        mlines[:,ii]=np.array([0.25*p['Omega']**2/p['gamma12'],0.25*p['Omega']**2/p['gamma23'],0.25*p['Omega']**2/p['gamma13'],0.25*p['Omega']**2/p['gamma2d'],0.25*p['Omega']**2/p['gamma3d'],
                p['Omega']])
    #     mlines[:,ii]=np.array([delaoval-deloval+delmval,delaoval-deloval+delmval+p['gamma2d'],delaoval-deloval+delmval-p['gamma2d'],
    # delaoval-deloval+delmval+p['gamma2d']*70,delaoval-deloval+delmval-p['gamma2d']*70,
    # delaoval-deloval+delmval+p['gamma2d']*200,delaoval-deloval+delmval-p['gamma2d']*200,
    # delaoval-deloval+delmval+p['gamma2d']*700,delaoval-deloval+delmval-p['gamma2d']*700,
    # delmval,delmval+p['gamma2d'],delmval+p['gamma2d']*100,delmval+p['gamma2d']*500,
    # delmval-p['gamma2d'],delmval-p['gamma2d']*100,delmval-p['gamma2d']*500,
    # -50*p['sd_delam']+p['mean_delam'],50*p['sd_delam']+p['mean_delam'],p['mean_delam'],
    # p['sd_delam']+p['mean_delam'] ,-p['sd_delam']+p['mean_delam'],3*p['sd_delam']+p['mean_delam'] ,-3*p['sd_delam']+p['mean_delam'],0])
        # mlines[:,ii]=np.array([delaoval-deloval+delmval,delaoval-deloval+delmval+p['gamma2d'],delaoval-deloval+delmval-p['gamma2d'],
        #                       delaoval-deloval+delmval-p['gamma2d']*3,delaoval-deloval+delmval+p['gamma2d']*3,
        #                       delaoval-deloval+delmval-p['gamma2d']*6,delaoval-deloval+delmval+p['gamma2d']*6,
        #                       delmval, delmval+p['gamma2d'],delmval-p['gamma2d']*1])
        # mlines[:,ii]=np.array([c_linear_excited.ds_m1_aprx(delaoval,deloval,delmval,p),c_linear_excited.ds_m2_aprx(delaoval,deloval,delmval,p)])
    return mlines.T
def lines_o(delamvals,deloval,delmval,p):
    olines=np.zeros((5,len(delamvals)))
    for ii, delamval in enumerate(delamvals):
    #     olines[:,ii]=np.array([p['mean_delao'],-3*p['sd_delao']+p['mean_delao'],3*p['sd_delao']+p['mean_delao'],
    # -p['sd_delao']+p['mean_delao'],p['sd_delao']+p['mean_delao'],
    # deloval,deloval+p['gamma3d'],deloval+p['gamma3d']*5,
    # deloval-p['gamma3d'],deloval-p['gamma3d']*5,0])
        olines[:,ii]=np.array([deloval,deloval-p['gamma3d'],deloval+p['gamma3d']
                ,deloval+p['gamma3d']*3,deloval-p['gamma3d']*3])
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

delo=0e8
delm=0e8
t1=time.time()
for ii, deloval in enumerate(delaovals):
    #print(ii)
    for jj, delmval in enumerate(delamvals):
        S_out_full=c_linear_excited.steady_rhos(deloval,delmval, delo,delm,p)
        #print(S_out_full)
        S_out_fullpy=np.zeros(45)
#        for ll,newind in enumerate([ 0,  1,  2,  7,  8, 12, 13, 14, 15, 21, 22, 23, 24, 30, 31, 32, 33,39, 40, 41, 42]):
        for ll,newind in enumerate([0, 1, 2, 3, 4,14, 15, 16, 17,23, 24, 25, 26,32, 33, 34, 35,41, 42, 43, 44]):

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

mlines=lines_m(delaovals,delo,delm,p)
olines=lines_o(delamvals,delo,delm,p)

titles=['rho0','rhoa','rhoac','rhob', 'rhobc']
titles=['rho0','rhoa','rhob']

for kk, rhox in enumerate([rho0,rhoa,rhob]):
#for kk, rhox in enumerate([rho0,rhoa,rhoac,rhob,rhobc]):

    fig=plt.figure(titles[kk])
    pltnum=1
    fig.clf()
    for ii in range(3):
        for jj in range(3):

            ax=fig.add_subplot(3,3,pltnum)
            img1=ax.imshow((np.abs(rhox[ii,jj,:,:])),extent=(min(delamvals),max(delamvals),min(delaovals),max(delaovals)),aspect='auto',origin='lower')
            plt.plot(mlines,delaovals)

            # plt.plot(delamvals,olines)
            ax.set_xlim(min(delamvals),max(delamvals))

            plt.xlabel('del mu')
            plt.ylabel('del o')
            plt.title(str(ii)+', ' +str(jj))
            fig.colorbar(img1)
            pltnum=pltnum+1
    plt.legend(['W2/g12','W2/g23','W2/g13','W2/g2d','W2/g3d','W'])
    plt.title(str(P_pump_mW))
    # mlines[:,ii]=0.25*p['Omega']**2/np.array([p['gamma12'],p['gamma23'],p['gamma13'],p['gamma2d'],p['gamma3d']])

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
