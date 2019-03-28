import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import colormaps as cmaps

filename='three_lvl_nocavity_out7'
npzfile=np.load(filename+'.npz')
rho_out=npzfile['rho_out_bout']
p=npzfile['p'][()]
boutvals=npzfile['boutvals']#/np.sqrt(p['gammamc'])
binval=npzfile['binval']
deltamvals=npzfile['deltamvals']
deltamacvals=npzfile['deltamacvals']
print(p)
rho13=rho_out[0,2,:,:]

fig=plt.figure(filename)

fig.clf()

ax=fig.add_subplot(3,2,1)
img1=ax.imshow(np.log10(np.abs(rho13[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='equal',origin='lower')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,2)
img1=ax.imshow(10*np.log10(np.abs(boutvals[:,:]/binval)),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='equal',origin='lower')
plt.title('bout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,3)
img1=ax.imshow((np.abs(rho13[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='equal',origin='lower')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,4)
img1=ax.imshow((np.abs(boutvals[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='equal',origin='lower')
plt.title('bout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

#matplotlib.cm.register_cmap(name='twilight',cmap=twilight)


ax=fig.add_subplot(3,2,5)
img1=ax.imshow((np.angle(rho13[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='equal',origin='lower',cmap='hsv')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)
ax=fig.add_subplot(3,2,6)
img1=ax.imshow((np.angle(boutvals[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='equal',origin='lower',cmap='hsv')
plt.title('bout')
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
# img1=ax.imshow(np.log10(np.abs(aoutvals[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='equal',origin='lower')
# plt.title('aout n0' )
# plt.xlabel('delta_mu')
# plt.ylabel('deltaac_mu')
# fig.colorbar(img1)
#
# ax=fig.add_subplot(3,2,4)
# img1=ax.imshow(np.log10(np.abs(boutvals[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='equal',origin='lower')
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
# img1=ax.imshow(np.log10(np.abs(aoutvals[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='equal',origin='lower')
# plt.title('aout n10')
# plt.xlabel('delta_mu')
# plt.ylabel('deltaac_mu')
# fig.colorbar(img1)
#
# ax=fig.add_subplot(3,2,6)
# img1=ax.imshow(np.log10(np.abs(boutvals[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='equal',origin='lower')
# plt.title('bout')
# plt.xlabel('delta_mu')
# plt.ylabel('deltaac_mu')
# fig.colorbar(img1)
plt.show()
