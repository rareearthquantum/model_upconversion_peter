import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import colormaps as cmaps

filename='three_lvl_nocavity_out_bintest7'
bin_num=2
npzfile=np.load(filename+'.npz')
rho_out=npzfile['rho_out_b']
p=npzfile['p'][()]
bvals=npzfile['bvals']#/np.sqrt(p['gammamc'])
boutvals=bvals[bin_num,:,:]*np.sqrt(p['gammamc'])
#binvals=npzfile['binvals']
#binvals = 10**np.linspace(-3,10,50)
#binvals = 10**np.linspace(10,20,3)
#binvals = 10**np.linspace(10,20,3)*(1+1j)
binvals = npzfile['binvals']
binval=binvals[bin_num]
deltamvals=npzfile['deltamvals']
deltamacvals=npzfile['deltamacvals']
print(p)
rho13=rho_out[0,2,bin_num,:,:]

fig=plt.figure(filename)

fig.clf()
#binval=1
ax=fig.add_subplot(3,2,1)
img1=ax.imshow(np.log10(np.abs(rho13[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower')
plt.title('aout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,2)
img1=ax.imshow(10*np.log10(np.abs(boutvals[:,:]/binval)**2),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower')
plt.title('bout, bin = ' + str(np.log10(binval.real)))
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,3)
img1=ax.imshow((np.abs(rho13[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower')
#plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,4)
img1=ax.imshow((np.abs(boutvals[:,:]/binval)),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower')
#plt.title('bout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

#matplotlib.cm.register_cmap(name='twilight',cmap=twilight)


ax=fig.add_subplot(3,2,5)
img1=ax.imshow((np.angle(rho13[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower',cmap='hsv')
#plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)
ax=fig.add_subplot(3,2,6)
img1=ax.imshow((np.angle(boutvals[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower',cmap='hsv')
#plt.title('bout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)
# filename='output_calcs/three_lvl_out2_n0'
# npzfile=np.load(filename+'.npz')
# aoutvals=npzfile['aoutvals']
# boutvals=npzfile['boutvals']
#
# deltamvals=npzfile['deltamvals']
# deltamacvals=npzfile['deltamacvals']
#
# ax=fig.add_subplot(3,2,3)
# img1=ax.imshow(np.log10(np.abs(aoutvals[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower')
# plt.title('aout n0' )
# plt.xlabel('delta_mu')
# plt.ylabel('deltaac_mu')
# fig.colorbar(img1)
#
# ax=fig.add_subplot(3,2,4)
# img1=ax.imshow(np.log10(np.abs(boutvals[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower')
# plt.title('bout')
# plt.xlabel('delta_mu')
# plt.ylabel('deltaac_mu')
# fig.colorbar(img1)
#
# filename='output_calcs/three_lvl_out2_n10'
# npzfile=np.load(filename+'.npz')
# aoutvals=npzfile['aoutvals']
# boutvals=npzfile['boutvals']
#
# deltamvals=npzfile['deltamvals']
# deltamacvals=npzfile['deltamacvals']
#
# ax=fig.add_subplot(3,2,5)
# img1=ax.imshow(np.log10(np.abs(aoutvals[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower')
# plt.title('aout n10')
# plt.xlabel('delta_mu')
# plt.ylabel('deltaac_mu')
# fig.colorbar(img1)
#
# ax=fig.add_subplot(3,2,6)
# img1=ax.imshow(np.log10(np.abs(boutvals[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower')
# plt.title('bout')
# plt.xlabel('delta_mu')
# plt.ylabel('deltaac_mu')
# fig.colorbar(img1)
plt.show()
