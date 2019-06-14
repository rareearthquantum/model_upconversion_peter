import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import colormaps as cmaps

filename='Paper_figs1/sim_fig4_data2_'
npzfile=np.load(filename+'1.npz')
print(npzfile)
rho_out_b=npzfile['rho_out_b']
p=npzfile['p'][()]
deltamvals=npzfile['deltamvals']
freq_pump_vals=npzfile['freq_pump_vals']
delfreqmu_vals=deltamvals/(2*np.pi)

for ii in [2,3,4]:
    npzfile=np.load(filename+str(ii)+'.npz')
    rho_out_b=rho_out_b+npzfile['rho_out_b']
    #rho_out_b=np.abs(rho_out_b)+np.abs(npzfile['rho_out_b'])

rho13=rho_out_b[0,2,:,:]

fig=plt.figure(filename)

fig.clf()

ax=fig.add_subplot(3,1,1)
img1=ax.imshow(np.log10(np.abs(rho13[:,:])**2),extent=(np.min(delfreqmu_vals),np.max(delfreqmu_vals),np.min(freq_pump_vals),np.max(freq_pump_vals)),aspect='auto',origin='lower')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('$\nu$')
fig.colorbar(img1)


ax=fig.add_subplot(3,1,2)
img1=ax.imshow((np.abs(rho13[:,:])**2),extent=(np.min(delfreqmu_vals),np.max(delfreqmu_vals),np.min(freq_pump_vals),np.max(freq_pump_vals)),aspect='auto',origin='lower')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('$\nu$')
fig.colorbar(img1)


ax=fig.add_subplot(3,1,3)
img1=ax.imshow((np.angle(rho13[:,:])),extent=(np.min(delfreqmu_vals),np.max(delfreqmu_vals),np.min(freq_pump_vals),np.max(freq_pump_vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('$\nu$')
fig.colorbar(img1)


fig=plt.figure(filename+'line')
plt.plot((np.abs(rho13[:,:])**2),freq_pump_vals)
plt.show()
