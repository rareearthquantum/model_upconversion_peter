import numpy as np
import matplotlib.pyplot as plt
npzfile=np.load('output_bin2.npz')
aoutvals=npzfile['aoutvals']
boutvals=npzfile['boutvals']
effic_a=npzfile['effic_a']
effic_b=npzfile['effic_b']
delovals=npzfile['delovals']
delmvals=npzfile['delmvals']
ainvals=npzfile['ainvals']
binvals=npzfile['binvals']
print(delovals.shape)
print(delmvals.shape)
print(effic_a.shape)
#plt.figure('honhonhon')
fig=plt.subplots()

plt.suptitle('ain = ' +str(ainvals[0]) + ', bin = ' + str(binvals[0]))
sbplt1=plt.subplot(2,2,1)
plt1=sbplt1.imshow(np.abs(aoutvals[0,0,:,:]),extent=(np.min(delmvals),np.max(delmvals),np.min(delovals),np.max(delovals)),aspect='auto')
plt.title('aout')
plt.xlabel('delta_mu')
plt.ylabel('delta_o')
plt.colorbar(plt1)

sbplt1=plt.subplot(2,2,2)
plt2=sbplt1.imshow(np.abs(boutvals[0,0,:,:]),extent=(np.min(delmvals),np.max(delmvals),np.min(delovals),np.max(delovals)),aspect='auto')
plt.title('bout')
plt.xlabel('delta_mu')
plt.ylabel('delta_o')
plt.colorbar(plt2)

sbplt1=plt.subplot(2,2,3)
plt3=sbplt1.imshow(np.abs(effic_a[0,0,:,:]),extent=(np.min(delmvals),np.max(delmvals),np.min(delovals),np.max(delovals)),aspect='auto')
plt.title('effic_a')
plt.xlabel('delta_mu')
plt.ylabel('delta_o')
plt.colorbar(plt3)

sbplt1=plt.subplot(2,2,4)
plt4=sbplt1.imshow(np.abs(effic_b[0,0,:,:]),extent=(np.min(delmvals),np.max(delmvals),np.min(delovals),np.max(delovals)),aspect='auto')
plt.title('effic_b')
plt.xlabel('delta_mu')
plt.ylabel('delta_o')
plt.colorbar(plt4)
#fig=pl.figure('honhon')
#fig.clf()
#ax=fig.add_subplot(2,2,1)

fig=plt.figure(1)
fig.clf()
ax=fig.add_subplot(2,2,1)
img1=ax.
