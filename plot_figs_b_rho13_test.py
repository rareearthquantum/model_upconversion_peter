import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import colormaps as cmaps

filename='three_lvl_nocavity_out_b_bout1'
npzfile=np.load(filename+'.npz')
rho_out_bout=npzfile['rho_out_bout']
rho_out_b=npzfile['rho_out_b']

p=npzfile['p'][()]
boutvals=npzfile['boutvals']#/np.sqrt(p['gammamc'])
bvals=npzfile['bvals']
binval=npzfile['binval']
deltamvals=npzfile['deltamvals']
deltamacvals=npzfile['deltamacvals']
print(p)
rho13_b=rho_out_b[0,2,:,:]
rho13_bout=rho_out_bout[0,2,:,:]

fig=plt.figure(filename+'test')

fig.clf()

ax=fig.add_subplot(3,2,1)
img1=ax.imshow(np.log10(np.abs(rho13_bout/rho13_b)),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='equal',origin='lower')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,2)
img1=ax.imshow(10*np.log10(np.abs(boutvals[:,:]/bvals)),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='equal',origin='lower')
plt.title('bout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,3)
img1=ax.imshow((np.abs(rho13_bout/rho13_b)),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='equal',origin='lower')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,4)
img1=ax.imshow((np.abs(boutvals[:,:]/bvals)),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='equal',origin='lower')
plt.title('bout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)
print(bvals/boutvals)
#matplotlib.cm.register_cmap(name='twilight',cmap=twilight)


# ax=fig.add_subplot(3,2,5)
# img1=ax.imshow((np.angle(rho13[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='equal',origin='lower',cmap='hsv')
# plt.title(' n1 aout')
# plt.xlabel('delta_mu')
# plt.ylabel('deltaac_mu')
# fig.colorbar(img1)
# ax=fig.add_subplot(3,2,6)
# img1=ax.imshow((np.angle(boutvals[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='equal',origin='lower',cmap='hsv')
# plt.title('bout')
# plt.xlabel('delta_mu')
# plt.ylabel('deltaac_mu')
# fig.colorbar(img1)
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
