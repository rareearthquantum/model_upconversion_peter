import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import colormaps as cmaps

filename='ground_params_test3_big'
npzfile=np.load(filename+'.npz')
print(npzfile)
rho_out_b=npzfile['rho_out_b']
p=npzfile['p'][()]
bvals=npzfile['bvals']#/np.sqrt(p['gammamc'])
binvals = npzfile['binvals']
deltamvals=npzfile['deltamvals']
I_vals=npzfile['I_vals']
B_vals=(0.027684*I_vals*1e3-0.056331)*1e-3


P_mu=npzfile['P_mu']
print(p)

freqmu_vals=deltamvals/(2*np.pi)+p['freqmu']
rho13=rho_out_b[0,2,:,:]
boutvals=bvals*np.sqrt(p['gammamc'])

fig=plt.figure(filename)

fig.clf()

#ax=fig.add_subplot(3,2,1)
plt.imshow(np.log10(np.abs(rho13[:,:])**2),extent=(np.min(freqmu_vals)/1e6,np.max(freqmu_vals)/1e6,np.min(B_vals)*1e3,np.max(B_vals)*1e3),aspect='auto',origin='lower')
#ax1.Aspect(aspect=1)
#plt.title('Upconversion',fontsize=15)
plt.xlabel('$\delta_\mu \ (MHz)$',fontsize=15)
plt.ylabel('$Magnetic \ Field \ (mT)$',fontsize=15)
#fig.colorbar(img1)
fig.tight_layout()
plt.show()
