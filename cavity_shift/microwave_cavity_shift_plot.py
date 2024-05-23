import numpy as np
import matplotlib.pyplot as plt
import matplotlib

filename='microwave_cavity_sim6'

# npzfile=np.load(filename+'.npz')
npzfile=np.load('cavity_shift/'+filename+'.npz')

print(npzfile)
p=npzfile['p'][()]
bvals=npzfile['bvals']#/np.sqrt(p['gammamc'])
binvals = npzfile['binvals']
avals=npzfile['avals']#/np.sqrt(p['gammamc'])
#binvals = npzfile['ainval']
deltamucvals=npzfile['deltamucvals']
delta2vals=npzfile['delta2vals']
P_mu=npzfile['P_mu']#/np.sqrt(p['gammamc'])

boutvals=bvals*np.sqrt(p['gammamc'])
aoutvals=avals*np.sqrt(p['gammaoc'])
deltamucvals2=np.linspace(np.min(deltamucvals),np.max(deltamucvals),51)
delta2vals2=np.linspace(np.min(delta2vals),np.max(delta2vals),51)


fig=plt.figure(filename)

fig.clf()


ax=fig.add_subplot(3,2,1)
img1=ax.imshow(10*np.log10(np.abs(aoutvals/binvals)),extent=(np.min(deltamucvals),np.max(deltamucvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)

plt.title(' |aout/bin| (dB scale)')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,2)
img1=ax.imshow(10*np.log10(np.abs(boutvals/binvals)**2),extent=(np.min(deltamucvals),np.max(deltamucvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)
plt.title('|bout/bin|^2 (dB)')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,3)
img1=ax.imshow((np.abs(aoutvals/binvals)),extent=(np.min(deltamucvals),np.max(deltamucvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)
plt.title('|aout/bin|')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,2,4)
img1=ax.imshow((np.abs(boutvals/binvals)**2),extent=(np.min(deltamucvals),np.max(deltamucvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)
plt.title('|bout/bin|^2')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)

#matplotlib.cm.register_cmap(name='twilight',cmap=twilight)


ax=fig.add_subplot(3,2,5)
img1=ax.imshow((np.angle(aoutvals)),extent=(np.min(deltamucvals),np.max(deltamucvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' rho13 phase')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)
ax=fig.add_subplot(3,2,6)
img1=ax.imshow((np.angle(boutvals[:,:])),extent=(np.min(deltamucvals),np.max(deltamucvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title('bout phase')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1)
fig.suptitle('P_mu = ' + str(P_mu)+ ' dBm')
plt.show()
