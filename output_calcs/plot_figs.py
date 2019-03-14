import numpy as np
import matplotlib.pyplot as plt
filename='output_calcs/three_lvl_out_big21'
npzfile=np.load(filename+'.npz')
aoutvals=npzfile['aoutvals']
boutvals=npzfile['boutvals']

deltamvals=npzfile['deltamvals']
deltamacvals=npzfile['deltamacvals']



fig=plt.figure(filename)

fig.clf()

ax=fig.add_subplot(1,2,1)
img1=ax.imshow(np.log10(np.abs(aoutvals[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='equal',origin='lower')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

ax=fig.add_subplot(1,2,2)
img1=ax.imshow(np.log10(np.abs(boutvals[:,:])),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='equal',origin='lower')
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
