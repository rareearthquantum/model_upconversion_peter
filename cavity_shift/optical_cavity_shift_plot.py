import numpy as np
import matplotlib.pyplot as plt
import matplotlib

filename='optical_cavity_sim5'

# npzfile=np.load(filename+'.npz')
npzfile=np.load('cavity_shift/'+filename+'.npz')

print(npzfile)
p=npzfile['p'][()]
bvals=npzfile['bvals']#/np.sqrt(p['gammamc'])
ainvals = 1#npzfile['ainvals']
avals=npzfile['avals']#/np.sqrt(p['gammamc'])
#binvals = npzfile['ainval']
deltaocvals=npzfile['deltaocvals']
delta3vals=npzfile['delta3vals']
P_o=npzfile['P_o']#/np.sqrt(p['gammamc'])

boutvals=bvals*np.sqrt(p['gammamc'])
aoutvals=avals*np.sqrt(p['gammaoc'])
deltaocvals2=np.linspace(np.min(deltaocvals),np.max(deltaocvals),51)
delta3vals2=np.linspace(np.min(delta3vals),np.max(delta3vals),51)

fig=plt.figure(filename)

fig.clf()


ax=fig.add_subplot(3,2,1)
img1=ax.imshow(10*np.log10(np.abs(aoutvals/ainvals)),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower')
plt.plot(p['No']*p['go']**2/delta3vals2,delta3vals2)

plt.title(' |aout/bin| (dB scale)')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,2)
img1=ax.imshow(10*np.log10(np.abs(boutvals/ainvals)**2),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower')
plt.plot(p['No']*p['go']**2/delta3vals2,delta3vals2)
plt.title('|bout/bin|^2 (dB)')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,3)
img1=ax.imshow((np.abs(aoutvals/ainvals)),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower')
plt.plot(p['No']*p['go']**2/delta3vals2,delta3vals2)
plt.title('|aout/bin|')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,2,4)
img1=ax.imshow((np.abs(boutvals/ainvals)**2),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower')
plt.plot(p['No']*p['go']**2/delta3vals2,delta3vals2)
plt.title('|bout/bin|^2')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

#matplotlib.cm.register_cmap(name='twilight',cmap=twilight)


ax=fig.add_subplot(3,2,5)
img1=ax.imshow((np.angle(aoutvals)),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' rho13 phase')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)
ax=fig.add_subplot(3,2,6)
img1=ax.imshow((np.angle(boutvals[:,:])),extent=(np.min(deltaocvals),np.max(deltaocvals),np.min(delta3vals),np.max(delta3vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title('bout phase')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)
fig.suptitle('P_o = ' + str(P_o)+ ' dBm')
plt.show()
