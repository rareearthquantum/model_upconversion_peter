import sys
sys.path.append('/home/peter/model_upconversion')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from c_funs_test5 import rho_broad_full

filename='microwave_cavity_poster3'


npzfile=np.load(filename+'.npz')
print(npzfile)
p_m=npzfile['p'][()]
bvals=npzfile['bvals']#/np.sqrt(p['gammamc'])
avals=npzfile['avals']
binvals = npzfile['binvals']
deltamucvals=npzfile['deltamucvals']
delta2vals=npzfile['delta2vals']
P_mu=npzfile['P_mu']#/np.sqrt(p['gammamc'])
T=npzfile['T']
boutvals=bvals*np.sqrt(p_m['gammamc'])
deltamucvals2=np.linspace(np.min(deltamucvals),np.max(deltamucvals),51)
delta2vals2=np.linspace(np.min(delta2vals),np.max(delta2vals),51)
ind1=round(len(delta2vals)/2)
N_frac=np.zeros(len(delta2vals))
def nbath_from_T(T,p,deltamacval=0):
    omega=np.abs(2*np.pi*p['freqmu']-deltamacval)
    nbath = 1/(np.exp(1.0545718e-34*omega/1.38064852e-23/T)-1)
    return nbath
for ii, delta2val in enumerate(delta2vals):
    p_m['mean_delam']=delta2val
    p_m['nbath']=nbath_from_T(T,p_m)
    rho=np.array(rho_broad_full(avals[ii,ind1],bvals[ii,ind1],0,0,p_m))
    N_frac[ii]=rho[0,0]-rho[1,1]

# plt.figure('N_frac')
# plt.plot(delta2vals,N_frac)
# plt.show()
filename='optical_cavity_poster1'

npzfile=np.load(filename+'.npz')
print(npzfile)
p_o=npzfile['p'][()]

ainvals = 1#npzfile['ainvals']
avals=npzfile['avals']#/np.sqrt(p['gammamc'])
#binvals = npzfile['ainval']
deltaocvals=npzfile['deltaocvals']
delta3vals=npzfile['delta3vals']
P_o=npzfile['P_o']#/np.sqrt(p['gammamc'])

aoutvals=avals*np.sqrt(p_o['gammaoc'])
deltaocvals2=np.linspace(np.min(deltaocvals),np.max(deltaocvals),51)
delta3vals2=np.linspace(np.min(delta3vals),np.max(delta3vals),51)

colormap='viridis'
colormap='cividis'
fig=plt.figure(filename)

fig.clf()

ax=fig.add_subplot(1,2,1)
img1=ax.imshow(10*np.log10(np.abs(boutvals/binvals)**2).T,extent=(np.min(delta2vals)*1e-6,np.max(delta2vals)*1e-6,np.min(deltamucvals)*1e-6,np.max(deltamucvals)*1e-6),aspect='auto',origin='lower',cmap=colormap)
deltamu_peak=p_m['No']*p_m['gm']**2/delta2vals*N_frac
deltamu_peak[np.abs(deltamu_peak)>np.max(deltamucvals)]=np.nan
deltamu_peak_line=plt.plot(delta2vals*1e-6,deltamu_peak*1e-6,color='red')
#plt.title('|bout/bin|^2 (dB)')
plt.ylabel('Microwave Cavity \n Detuning (MHz)')
plt.xlabel('Microwave Transition Detuning (MHz)')
#fig.colorbar(img1)

ax=fig.add_subplot(1,2,2)
img1=ax.imshow(10*np.log10(np.abs(aoutvals/ainvals)**2).T,extent=(np.min(delta3vals)*1e-9,np.max(delta3vals)*1e-9,np.min(deltaocvals)*1e-6,np.max(deltaocvals)*1e-6),aspect='auto',origin='lower',cmap=colormap)
deltao_peak=p_o['No']*p_o['go']**2/delta3vals
deltao_peak[np.abs(deltao_peak)>np.max(deltaocvals)]=np.nan
deltao_peak_line=plt.plot(delta3vals*1e-9,deltao_peak*1e-6,color='red')
#plt.title(' |aout/ain|^2 (dB scale)')
plt.ylabel('Optical Cavity \n Detuning (MHz)')
plt.xlabel('Optical Transition Detuning (THz)')
plt.tight_layout()
fig.patch.set_facecolor('blue')
fig.patch.set_alpha(0)

plt.show()
