import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import colormaps as cmaps

filename='Paper_figs1/sim_fig6_data_dB-15_test3'
#filename='Paper_figs1/sim_fig6_data_dB3_meantimetest2'
#filename='Paper_figs1/sim_fig6_data_dB3_test3'
#filename='Paper_figs1/sim_fig7_data_temp2'
#filename='Paper_figs1/sim_fig7_data_temp_cfe2'
#filename='Paper_figs1/sim_fig6_data_dB-15_test_ds2_biggish15_broy1'
#filename='Paper_figs1/sim_fig6_data_dB-15_test_ds2_biggish7_hybr'
#filename='Paper_figs1/sim_fig7_data_ithinkitwork2s'
#filename='Paper_figs1/sim_fig7_data_temp_cfe2'
#filename='Paper_figs1/sim_fig7_data15_do1e10'
#filename='Paper_figs1/sim_fig7_data_temp7_cfe2'
filename='Paper_figs1/sim_fig6_data_dB3_test_ds2_temp1'
npzfile=np.load(filename+'.npz')
print(npzfile)
rho_out=npzfile['rho_out']
p=npzfile['p'][()]
bvals=npzfile['bvals']#/np.sqrt(p['gammamc'])
#avals=npzfile['avals']
binvals = npzfile['binvals']
deltamvals=npzfile['deltamvals']
#I_vals=npzfile['I_vals']
P_mu=npzfile['P_mu']
#calc_time=npzfile['calc_time']
B_vals=npzfile['B_vals']
calc_time=npzfile['calc_time']

print(p)
print('This simulation took ' + str(npzfile['elapsed_time']))
freqmu_vals=deltamvals/(2*np.pi)#+p['freqmu']
rho13=rho_out[0,2,:,:]
boutvals=bvals*np.sqrt(p['gammamc'])


delfreqmu_vals=deltamvals/(2*np.pi)
rho13=rho_out[0,2,:,:]
rho13_background=0
#rho13_background=10.0**(-19)
#rho13_background=10.0**(-21.5)
boutvals=bvals*np.sqrt(p['gammamc'])
fig=plt.figure(filename)
rho_abs2=np.abs(rho13**2)
rho_abs2[rho_abs2<rho13_background]=rho13_background
fig.clf()

ax=fig.add_subplot(3,2,1)
img1=ax.imshow(np.log10(rho_abs2),extent=(np.min(freqmu_vals)*1e-6,np.max(freqmu_vals)*1e-6,np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower',cmap='viridis_r')
plt.title(' |rho13|^2 (log scale)')
plt.xlabel('delta_mu (MHz)')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,2)
img1=ax.imshow(10*np.log10(np.abs(boutvals/binvals)**2),extent=(np.min(freqmu_vals)*1e-6,np.max(freqmu_vals)*1e-6,np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower',cmap='viridis_r')
plt.title('|bout/bin|^2 (dB)')
plt.xlabel('delta_mu (MHz)')
plt.ylabel('I')
fig.colorbar(img1)

ax=fig.add_subplot(3,2,3)
img1=ax.imshow((rho_abs2),extent=(np.min(freqmu_vals)*1e-6,np.max(freqmu_vals)*1e-6,np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower')
plt.title('|rho13|^2 ')
plt.xlabel('delta_mu')
plt.ylabel('I')
fig.colorbar(img1, format='%.0e')

ax=fig.add_subplot(3,2,4)
img1=ax.imshow((np.abs(boutvals/binvals)**2),extent=(np.min(freqmu_vals)*1e-6,np.max(freqmu_vals)*1e-6,np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower')
plt.title('|bout/bin|^2')
plt.xlabel('delta_mu (MHz)')
plt.ylabel('I')
fig.colorbar(img1)

#matplotlib.cm.register_cmap(name='twilight',cmap=twilight)


ax=fig.add_subplot(3,2,5)
img1=ax.imshow((np.angle(rho13)),extent=(np.min(freqmu_vals)*1e-6,np.max(freqmu_vals)*1e-6,np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title(' rho13 phase')
plt.xlabel('delta_mu (MHz)')
plt.ylabel('I')
fig.colorbar(img1)
ax=fig.add_subplot(3,2,6)
img1=ax.imshow((np.angle(boutvals[:,:])),extent=(np.min(freqmu_vals)*1e-6,np.max(freqmu_vals)*1e-6,np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower',cmap='hsv')
plt.title('bout phase')
plt.xlabel('delta_mu (MHz)')

plt.ylabel('I')
fig.colorbar(img1)
fig.suptitle('P_mu = ' + str(P_mu)+ ' dBm')


fig=plt.figure(filename+'time')
plt.imshow(calc_time,extent=(np.min(freqmu_vals)*1e-6,np.max(freqmu_vals)*1e-6,np.min(B_vals),np.max(B_vals)),aspect='auto',origin='lower')
plt.title('Calc time')
plt.xlabel('delta_mu (MHz)')
plt.ylabel('I')
plt.colorbar()

plt.show()
