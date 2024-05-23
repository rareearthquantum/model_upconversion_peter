
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

filename='effic_sim_nods2'
filename='effic_sim2_ds_test_T0'
filename='effic_sim2_nods_BIG_Q'
npzfile=np.load('cavity_shift/'+filename+'.npz')
print(npzfile)
p=npzfile['p'][()]
bvals=npzfile['bvals']#/np.sqrt(p['gammamc'])
binvals = npzfile['binvals']
avals=npzfile['avals']#/np.sqrt(p['gammamc'])
#binvals = npzfile['ainval']
delta3vals=npzfile['delta3vals']
delta2vals=npzfile['delta2vals']
P_mu=npzfile['P_mu']#/np.sqrt(p['gammamc'])

boutvals=bvals*np.sqrt(p['gammamc'])
aoutvals=avals*np.sqrt(p['gammaoc'])

effic=p['gammaoc']*np.abs(avals)**2/np.abs(binvals)**2

fig=plt.figure(filename+'_effic')
fig.clf()

ax=fig.add_subplot(1,1,1)
img1=ax.imshow((np.abs(effic)),extent=(np.min(delta3vals),np.max(delta3vals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title(' effic')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1, format='%.0e')

fig=plt.figure(filename)
fig.clf()

ax=fig.add_subplot(3,2,1)
img1=ax.imshow(10*np.log10(np.abs(aoutvals/binvals)**2),extent=(np.min(delta3vals),np.max(delta3vals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title(' |aout/bin|^2 (dB scale)')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,2)
img1=ax.imshow(10*np.log10(np.abs(boutvals/binvals)**2),extent=(np.min(delta3vals),np.max(delta3vals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('|bout/bin|^2 (dB)')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,3)
img1=ax.imshow((np.abs(aoutvals/binvals)**2),extent=(np.min(delta3vals),np.max(delta3vals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('|aout/bin|^2')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,2,4)
img1=ax.imshow((np.abs(boutvals/binvals)**2),extent=(np.min(delta3vals),np.max(delta3vals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('|bout/bin|^2')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1)

#matplotlib.cm.register_cmap(name='twilight',cmap=twilight)


ax=fig.add_subplot(3,2,5)
img1=ax.imshow((np.angle(aoutvals)),extent=(np.min(delta3vals),np.max(delta3vals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' rho13 phase')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1)
ax=fig.add_subplot(3,2,6)
img1=ax.imshow((np.angle(boutvals[:,:])),extent=(np.min(delta3vals),np.max(delta3vals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title('bout phase')
plt.xlabel('delta_3')
plt.ylabel('delta_2')
fig.colorbar(img1)
fig.suptitle('P_mu = ' + str(P_mu)+ ' dBm')
plt.show()
