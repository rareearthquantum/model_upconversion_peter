import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import colormaps as cmaps

filename='three_lvl_nocavity_doublecross1'
npzfile=np.load(filename+'.npz')
rho_out1=npzfile['rho_out_bout1']
rho_out2=npzfile['rho_out_bout2']
p=npzfile['p'][()]
boutvals=npzfile['bvals']*np.sqrt(p['gammamc'])
binval=npzfile['binval']
deltamvals=npzfile['deltamvals']
deltamacvals=npzfile['deltamacvals']
print(p)
rho13_1=rho_out1[0,2,:,:]+rho_out2[0,2,:,:]
rho13_2=rho_out2[0,2,:,:]

fig=plt.figure(filename)

fig.clf()

ax=fig.add_subplot(3,3,1)
img1=ax.imshow(np.log10(np.abs(rho13_1[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)
ax=fig.add_subplot(3,3,2)
img1=ax.imshow(np.log10(np.abs(rho13_2[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

ax=fig.add_subplot(3,3,3)
img1=ax.imshow(10*np.log10(np.abs(boutvals[:,:]/binval)**2),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower')
plt.title('bout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

ax=fig.add_subplot(3,3,4)
img1=ax.imshow((np.abs(rho13_1[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)
ax=fig.add_subplot(3,3,5)
img1=ax.imshow((np.abs(rho13_2[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

ax=fig.add_subplot(3,3,6)
img1=ax.imshow((np.abs(boutvals[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower')
plt.title('bout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

#matplotlib.cm.register_cmap(name='twilight',cmap=twilight)


ax=fig.add_subplot(3,3,7)
img1=ax.imshow((np.angle(rho13_1[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)
ax=fig.add_subplot(3,3,8)
img1=ax.imshow((np.angle(rho13_2[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)
ax=fig.add_subplot(3,3,9)
img1=ax.imshow((np.angle(boutvals[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower',cmap='hsv')
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
