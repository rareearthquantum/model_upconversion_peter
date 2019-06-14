import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import colormaps as cmaps

filename='ground_params_test3'
npzfile=np.load(filename+'.npz')
print(npzfile)
rho_out_b=npzfile['rho_out_b']
p=npzfile['p'][()]
bvals=npzfile['bvals']#/np.sqrt(p['gammamc'])
binvals = 1#npzfile['binvals']
deltamvals=npzfile['deltamvals']
I_vals=npzfile['I_vals']
print(p)


freqmu_vals=deltamvals/(2*np.pi)+p['freqmu']
rho13=rho_out_b[0,2,:,:]
boutvals=bvals*np.sqrt(p['gammamc'])
fig=plt.figure(filename)

fig.clf()

ax=fig.add_subplot(3,2,1)
img1=ax.imshow(np.log10(np.abs(rho13[:,:])**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,2)
img1=ax.imshow(10*np.log10(np.abs(boutvals[:,:]/binvals)**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower')
plt.title('bout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,3)
img1=ax.imshow((np.abs(rho13[:,:])**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,4)
img1=ax.imshow((np.abs(boutvals[:,:])**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower')
plt.title('bout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

#matplotlib.cm.register_cmap(name='twilight',cmap=twilight)


ax=fig.add_subplot(3,2,5)
img1=ax.imshow((np.angle(rho13[:,:])),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)
ax=fig.add_subplot(3,2,6)
img1=ax.imshow((np.angle(boutvals[:,:])),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title('bout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)
plt.show()
