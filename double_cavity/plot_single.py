import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import colormaps as cmaps

filename='ground_params2_test20'
npzfile=np.load(filename+'.npz')
print(npzfile)
rho_out=npzfile['rho_out']
p=npzfile['p'][()]
bvals=npzfile['bvals']#/np.sqrt(p['gammamc'])
avals=npzfile['avals']
binvals = npzfile['binvals']
deltamvals=npzfile['deltamvals']
I_vals=npzfile['I_vals']
P_mu=npzfile['P_mu']
calc_time=npzfile['calc_time']
print(p)
print('This simulation took ' + str(npzfile['elapsed_time']))
freqmu_vals=deltamvals/(2*np.pi)+p['freqmu']
rho13=rho_out[0,2,:,:]
boutvals=bvals*np.sqrt(p['gammamc'])
aoutvals=avals*np.sqrt(p['gammaoc'])

fig=plt.figure(filename)

fig.clf()

ax=fig.add_subplot(3,2,1)
img1=ax.imshow(10*np.log10(np.abs(aoutvals)/np.abs(binvals)),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower')
plt.title(' |aout/bin| (dB scale)')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,2)
img1=ax.imshow(10*np.log10(np.abs(boutvals/binvals)**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower')
plt.title('|bout/bin|^2 (dB)')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,3)
img1=ax.imshow((np.abs(aoutvals/binvals)),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower')
plt.title('|aout/bin|')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,2,4)
img1=ax.imshow((np.abs(boutvals/binvals)**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower')
plt.title('|bout/bin|^2')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

#matplotlib.cm.register_cmap(name='twilight',cmap=twilight)


ax=fig.add_subplot(3,2,5)
img1=ax.imshow((np.angle(aoutvals)),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower',cmap='hsv')
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

fig=plt.figure(filename+'time')
plt.imshow(calc_time,extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower')
plt.title('Calc time')
plt.xlabel('delta_mu')
plt.ylabel('I')
plt.colorbar()

plt.show()
