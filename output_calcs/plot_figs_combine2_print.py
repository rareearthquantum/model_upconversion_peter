import numpy as np
import matplotlib.pyplot as plt

filename_start='output_calcs/three_lvl_out_realtest3_'

npzfile=np.load(filename_start + '1.npz')
binval=npzfile['binval']
#ainval=npzfile['ainval']
aoutvals1=npzfile['aoutvals']
boutvals1=npzfile['boutvals']

deltamvals1=npzfile['deltamvals']
deltamacvals1=npzfile['deltamacvals']
deltamvals=np.hstack((-deltamvals1[::-1], deltamvals1[1:]))
deltamacvals=np.hstack((-deltamacvals1[::-1], deltamacvals1[1:]))
aoutvals=np.zeros((2*len(deltamacvals1)-1, 2*len(deltamvals1)-1),dtype=np.complex_)
boutvals=np.zeros((2*len(deltamacvals1)-1, 2*len(deltamvals1)-1),dtype=np.complex_)
#print(deltamacvals.shape)
#print(aoutvals.shape)
aoutvals[(len(deltamacvals1)-1):(2*len(deltamacvals1)-1),(len(deltamvals1)-1):(2*len(deltamvals1)-1)]=aoutvals1
boutvals[(len(deltamacvals1)-1):(2*len(deltamacvals1)-1),(len(deltamvals1)-1):(2*len(deltamvals1)-1)]=boutvals1
#print(deltamvals[len(deltamvals1)-1])
#print(deltamacvals[len(deltamacvals1)-1])

npzfile=np.load(filename_start + '2.npz')
aoutvals2=npzfile['aoutvals']
boutvals2=npzfile['boutvals']
aoutvals[:(len(deltamacvals1)),(len(deltamvals1)-1):(2*len(deltamvals1)-1)]=aoutvals2
boutvals[:(len(deltamacvals1)),(len(deltamvals1)-1):(2*len(deltamvals1)-1)]=boutvals2

npzfile=np.load(filename_start + '4.npz')
aoutvals1=npzfile['aoutvals']
boutvals1=npzfile['boutvals']
aoutvals[:(len(deltamacvals1)),:(len(deltamvals1))]=aoutvals1
boutvals[:(len(deltamacvals1)),:(len(deltamvals1))]=boutvals1
binval=npzfile['binval']

# npzfile=np.load(filename_start + '2.npz')
# aoutvals1=(npzfile['aoutvals'][::-1,::-1])
# boutvals1=(npzfile['boutvals'][::-1,::-1])
# aoutvals[(len(deltamacvals1)-1):(2*len(deltamacvals1)-1),:(len(deltamvals1))]=aoutvals1
# boutvals[(len(deltamacvals1)-1):(2*len(deltamacvals1)-1),:(len(deltamvals1))]=boutvals1

npzfile=np.load(filename_start + '3.npz')
aoutvals1=npzfile['aoutvals']
boutvals1=npzfile['boutvals']
aoutvals[(len(deltamacvals1)-1):(2*len(deltamacvals1)-1),:(len(deltamvals1))]=aoutvals1
boutvals[(len(deltamacvals1)-1):(2*len(deltamacvals1)-1),:(len(deltamvals1))]=boutvals1




fig=plt.figure(filename_start)

fig.clf()

ax=fig.add_subplot(3,2,1)
img1=ax.imshow(np.log10(np.abs(aoutvals[:,:]/binval))*10,extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower')
plt.title('aout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,2)
img1=ax.imshow(np.log10(np.abs(boutvals[:,:]/binval))*10,extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower')
plt.title('bout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)



ax=fig.add_subplot(3,2,3)
img1=ax.imshow((np.abs(aoutvals[:,:]))*np.log10(np.abs(aoutvals[:,:]/binval))/np.log10(np.abs(aoutvals[:,:]/binval)),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower')
plt.title('aout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,4)
img1=ax.imshow((np.abs(boutvals[:,:]))*np.log10(np.abs(aoutvals[:,:]/binval))/np.log10(np.abs(aoutvals[:,:]/binval)),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower')
plt.title('bout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)





ax=fig.add_subplot(3,2,5)
img1=ax.imshow((np.angle(aoutvals[:,:]))*np.log10(np.abs(aoutvals[:,:]/binval))/np.log10(np.abs(aoutvals[:,:]/binval)),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower')
plt.title('aout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,6)
img1=ax.imshow((np.angle(boutvals[:,:]))*np.log10(np.abs(aoutvals[:,:]/binval))/np.log10(np.abs(aoutvals[:,:]/binval)),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='auto',origin='lower')
plt.title('bout')
plt.xlabel('delta_mu')
plt.ylabel('deltaac_mu')
fig.colorbar(img1)


#fig.tight_layout()
#plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

plt.show()
