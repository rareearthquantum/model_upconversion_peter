import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import colormaps as cmaps

filename='excited_params_diptest1'
npzfile=np.load(filename+'.npz')
print(npzfile)
rho_out_b=npzfile['rho_out_b']
p=npzfile['p'][()]
bvals=npzfile['bvals']#/np.sqrt(p['gammamc'])
binvals = npzfile['binvals']
deltamvals=npzfile['deltamvals']
I_vals=npzfile['I_vals']
P_mu=npzfile['P_mu']
print(p)

freqmu_vals=deltamvals/(2*np.pi)+p['freqmu']
rho13=rho_out_b[0,2,:,:]
boutvals=bvals*np.sqrt(p['gammamc'])

fig=plt.figure(filename)

fig.clf()

ax=fig.add_subplot(3,2,1)
img1=ax.imshow(np.log10(np.abs(rho13[:,:])**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower')
plt.title(' rho13 (log scale)')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,2)
img1=ax.imshow(10*np.log10(np.abs(boutvals[:,:]/binvals)**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower')
plt.title('|bout/bin| (dB)')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,3)
img1=ax.imshow((np.abs(rho13[:,:])**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower')
plt.title('rho13')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,4)
img1=ax.imshow((np.abs(boutvals[:,:])**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower')
plt.title('|bout/bin|')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

#matplotlib.cm.register_cmap(name='twilight',cmap=twilight)


ax=fig.add_subplot(3,2,5)
img1=ax.imshow((np.angle(rho13[:,:])),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' rho13 phase')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)
ax=fig.add_subplot(3,2,6)
img1=ax.imshow((np.angle(boutvals[:,:])),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title('bout phase')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

fig.suptitle('P_mu = ' + str(P_mu)+ ' dBm')

#input('Hmm.. ')

fig=plt.figure(filename+'_2')

fig.clf()

ax=fig.add_subplot(3,2,1)
img1=ax.plot(np.log10(np.abs(rho13[:,:])**2).T)
plt.title(' rho13 (log scale)')
plt.xlabel('delta_mu')

ax=fig.add_subplot(3,2,2)
img1=ax.imshow(10*np.log10(np.abs(boutvals[:,:]/binvals)**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower')
plt.title('|bout/bin| (dB)')
plt.xlabel('delta_mu')
plt.ylabel('I')

ax=fig.add_subplot(3,2,3)
img1=ax.plot((np.abs(rho13[:,:])**2).T)
plt.xlabel('delta_mu')
plt.ylabel('I')

ax=fig.add_subplot(3,2,4)
img1=ax.imshow((np.abs(boutvals[:,:])**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower')
plt.title('|bout/bin|')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

#matplotlib.cm.register_cmap(name='twilight',cmap=twilight)


ax=fig.add_subplot(3,2,5)
img1=ax.plot((np.angle(rho13[:,:])).T)
plt.title(' rho13 phase')
plt.xlabel('delta_mu')
plt.ylabel('I')

ax=fig.add_subplot(3,2,6)
img1=ax.imshow((np.angle(boutvals[:,:])),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title('bout phase')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

fig.suptitle('P_mu = ' + str(P_mu)+ ' dBm')
plt.show()
