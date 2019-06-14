import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import colormaps as cmaps

filename='Paper_figs1/sim_fig4_data2_all1'
filename='Paper_figs1/sim_fig5_data1'
filename='Paper_figs1/sim_fig5_data_biggish1'
filename='Paper_figs1/sim_fig4_data_ds2'
filename='Paper_figs1/sim_fig5_data_temp8'

npzfile=np.load(filename+'.npz')
print(npzfile)
print('Simuation took ' + str (npzfile['elapsed_time']))
rho_out=npzfile['rho_out_b']
p=npzfile['p'][()]
bvals=npzfile['bvals']#/np.sqrt(p['gammamc'])
#avals=npzfile['avals']
binvals = npzfile['binvals']
deltamvals=npzfile['deltamvals']
#I_vals=npzfile['I_vals']
P_mu=npzfile['P_mu']
#calc_time=npzfile['calc_time']
freq_pump_vals=npzfile['freq_pump_vals']#+p['freqmu']

print(p)
print('This simulation took ' + str(npzfile['elapsed_time']))
freqmu_vals=deltamvals/(2*np.pi)+p['freqmu']
rho13=rho_out[0,2,:,:]
boutvals=bvals*np.sqrt(p['gammamc'])


delfreqmu_vals=deltamvals/(2*np.pi)
rho13=rho_out[0,2,:,:]
rho13_background=10.0**(11)
rho13_background=10.0**(-18.5)
#rho13_background=0

boutvals=bvals*np.sqrt(p['gammamc'])
fig=plt.figure(filename)
rho_abs2=np.abs(rho13)**2
rho_abs2[rho_abs2<rho13_background]=rho13_background
fig.clf()

ax=fig.add_subplot(3,2,1)
img1=ax.imshow(np.log10(rho_abs2),extent=(np.min(delfreqmu_vals),np.max(delfreqmu_vals),np.min(freq_pump_vals),np.max(freq_pump_vals)),aspect='auto',origin='lower',cmap='viridis_r')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('$\nu$')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,2)
img1=ax.imshow(10*np.log10(np.abs(boutvals[:,:]/binvals)**2),extent=(np.min(delfreqmu_vals),np.max(delfreqmu_vals),np.min(freq_pump_vals),np.max(freq_pump_vals)),aspect='auto',origin='lower')
plt.title('bout')
plt.xlabel('delta_mu')
plt.ylabel('$\nu$')

fig.colorbar(img1)

ax=fig.add_subplot(3,2,3)
img1=ax.imshow(rho_abs2,extent=(np.min(delfreqmu_vals),np.max(delfreqmu_vals),np.min(freq_pump_vals),np.max(freq_pump_vals)),aspect='auto',origin='lower')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('$\nu$')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,4)
img1=ax.imshow((np.abs(boutvals[:,:])**2),extent=(np.min(delfreqmu_vals),np.max(delfreqmu_vals),np.min(freq_pump_vals),np.max(freq_pump_vals)),aspect='auto',origin='lower')
plt.title('bout')
plt.xlabel('delta_mu')
plt.ylabel('$\nu$')
fig.colorbar(img1)

#matplotlib.cm.register_cmap(name='twilight',cmap=twilight)


ax=fig.add_subplot(3,2,5)
img1=ax.imshow((np.angle(rho13[:,:])),extent=(np.min(delfreqmu_vals),np.max(delfreqmu_vals),np.min(freq_pump_vals),np.max(freq_pump_vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' n1 aout')
plt.xlabel('delta_mu')
plt.ylabel('$\nu$')
fig.colorbar(img1)
ax=fig.add_subplot(3,2,6)
img1=ax.imshow((np.angle(boutvals[:,:])),extent=(np.min(delfreqmu_vals),np.max(delfreqmu_vals),np.min(freq_pump_vals),np.max(freq_pump_vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title('bout')
plt.xlabel('delta_mu')
plt.ylabel('$\nu$')
fig.colorbar(img1)
fig=plt.figure(filename+'line')
#plt.plot(np.log10(np.abs(rho13[:,:])**2/10),freq_pump_vals)
plt.plot((np.abs(rho13[:,:])**2/10),freq_pump_vals)
fig=plt.figure(filename+'line2')
#plt.plot(np.log10(np.abs(rho13[:,:])**2/10),freq_pump_vals)
plt.plot(delfreqmu_vals,(np.abs(rho13[:,:].T)**2/10))
plt.show()
