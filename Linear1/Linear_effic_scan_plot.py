
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

filename='Linear_effic_max_test2'
npzfile=np.load(filename+'.npz')


delta3vals=npzfile['delta3vals']
delta2vals=npzfile['delta2vals']
Cba=npzfile['Cba']

fig=plt.figure('dB plot')
plt.imshow(10*np.log10(np.abs(Cba)**2),extent=(np.min(delta3vals)*1e-9,np.max(delta3vals)*1e-9,np.min(delta2vals)*1e-6,np.max(delta2vals)*1e-6),aspect='auto',origin='lower')
plt.xlabel('$\delta_3$ (THz)')
plt.ylabel('$\delta_2$ (MHz)')
plt.colorbar()
plt.tight_layout()
fig.patch.set_facecolor('blue')
fig.patch.set_alpha(0)


fig=plt.figure('linear plot')
plt.imshow((np.abs(Cba)**2),extent=(np.min(delta3vals)*1e-9,np.max(delta3vals)*1e-9,np.min(delta2vals)*1e-6,np.max(delta2vals)*1e-6),aspect='auto',origin='lower')
plt.xlabel('$\delta_3$ (THz)')
plt.ylabel('$\delta_2$ (MHz)')
plt.colorbar()
plt.tight_layout()
fig.patch.set_facecolor('blue')
fig.patch.set_alpha(0)
plt.show()
