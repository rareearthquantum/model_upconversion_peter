import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import colormaps as cmaps

filename='excited_params_testbroadening7'
npzfile=np.load(filename+'.npz')
print(npzfile)
rho_out_b=npzfile['rho_out_b']
p=npzfile['p'][()]
bvals=npzfile['bvals']#/np.sqrt(p['gammamc'])
binvals = npzfile['binvals']
deltamvals=npzfile['deltamvals']
I_vals=npzfile['I_vals']
P_mu=npzfile['P_mu']
sd_delam_vals=npzfile['sd_delam_vals']#[0:2]
sd_delao_fac_vals=npzfile['sd_delao_fac_vals']#[0:3]

print(p)

freqmu_vals=deltamvals/(2*np.pi)+p['freqmu']
rho13=rho_out_b[0,2,:,:,:,:]
boutvals=bvals*np.sqrt(p['gammamc'])
fig=plt.figure(filename+'_rho13')

fig.clf()
pltcount=1
for ii, sd_delam_val in enumerate(sd_delam_vals):
    for jj, sd_delao_fac_val in enumerate(sd_delao_fac_vals):

        ax=fig.add_subplot(len(sd_delam_vals),len(sd_delao_fac_vals),pltcount)
        img1=ax.imshow(np.log10(np.abs(rho13[ii,jj,:,:])**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower')
        if jj==0:
            plt.ylabel('$\sigma_\mu = 4\pi e$' + str(np.log10(sd_delam_val/(4*np.pi))))
        if ii==0:
            plt.title('$C\sigma_o = 1e$' +str(np.log10(sd_delao_fac_val)))
        #plt.title('sd_delam = ' + str(sd_delam_val)+', sd_delao_fac = ' +str(sd_delao_fac_val))
        #plt.xlabel('delta_mu')
        #plt.ylabel('I')
        #fig.colorbar(img1)

        pltcount=pltcount+1

fig=plt.figure(filename+'_b')

fig.clf()
pltcount=1
for ii, sd_delam_val in enumerate(sd_delam_vals):
    for jj, sd_delao_fac_val in enumerate(sd_delao_fac_vals):

        ax=fig.add_subplot(len(sd_delam_vals),len(sd_delao_fac_vals),pltcount)
        img1=ax.imshow(np.log10(np.abs(boutvals[ii,jj,:,:]/binvals[ii,jj,:,:])**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower')
        if jj==0:
            plt.ylabel('$\sigma_\mu = 4\pi e$' + str(np.log10(sd_delam_val/(4*np.pi))))
        if ii==0:
            plt.title('$C\sigma_o = 1e$' +str(np.log10(sd_delao_fac_val)))
        #plt.title('sd_delam = ' + str(sd_delam_val)+', sd_delao_fac = ' +str(sd_delao_fac_val))
        #plt.xlabel('delta_mu')
        #plt.ylabel('I')
        #fig.colorbar(img1)

        pltcount=pltcount+1
fig=plt.figure(filename+'_rho13_phase')

fig.clf()
pltcount=1
for ii, sd_delam_val in enumerate(sd_delam_vals):
    for jj, sd_delao_fac_val in enumerate(sd_delao_fac_vals):

        ax=fig.add_subplot(len(sd_delam_vals),len(sd_delao_fac_vals),pltcount)
        img1=ax.imshow((np.angle(rho13[ii,jj,:,:])**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower',cmap='hsv')
        if jj==0:
            plt.ylabel('$\sigma_\mu = 4\pi e$' + str(np.log10(sd_delam_val/(4*np.pi))))
        if ii==0:
            plt.title('$C\sigma_o = 1e$' +str(np.log10(sd_delao_fac_val)))
        #plt.title('sd_delam = ' + str(sd_delam_val)+', sd_delao_fac = ' +str(sd_delao_fac_val))
        #plt.xlabel('delta_mu')
        #plt.ylabel('I')
        #fig.colorbar(img1)

        pltcount=pltcount+1


plt.show()
