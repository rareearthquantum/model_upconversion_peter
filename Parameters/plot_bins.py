import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import colormaps as cmaps

filename='ground_params_bintest1'
npzfile=np.load(filename+'.npz')
rho_out_b=npzfile['rho_out_b']
p=npzfile['p'][()]
bvals=npzfile['bvals']#/np.sqrt(p['gammamc'])
binvals_all = npzfile['binvals']
deltamvals=npzfile['deltamvals']
I_vals=npzfile['I_vals']
P_mu_vals=npzfile['P_mu_vals']
print(p)

freqmu_vals=deltamvals/(2*np.pi)+p['freqmu']
rho13_all=rho_out_b[0,2,:,:,:]
boutvals_all=bvals*np.sqrt(p['gammamc'])
for ii, P_mu in enumerate(P_mu_vals):
    rho13=rho13_all[:,:,ii]
    boutvals=boutvals_all[:,:,ii]
    binvals=binvals_all[:,:,ii]
    fig=plt.figure(filename + str(ii))

    fig.clf()

    ax=fig.add_subplot(3,2,1)
    img1=ax.imshow(np.log10(np.abs(rho13[:,:])**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower')
    plt.title(' rho13 (log scale)')
    plt.xlabel('delta_mu')
    plt.ylabel('I')
    fig.colorbar(img1)

    ax=fig.add_subplot(3,2,2)
    img1=ax.imshow(10*np.log10(np.abs(boutvals[:,:]/binvals)**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower')
    plt.title('|bout/bin| (dB)')
    plt.xlabel('delta_mu')
    plt.ylabel('I')
    fig.colorbar(img1)

    ax=fig.add_subplot(3,2,3)
    img1=ax.imshow((np.abs(rho13[:,:])**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower')
    plt.title('rho13')
    plt.xlabel('delta_mu')
    plt.ylabel('I')
    fig.colorbar(img1)

    ax=fig.add_subplot(3,2,4)
    img1=ax.imshow((np.abs(boutvals[:,:])**2),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower')
    plt.title('|bout/bin|')
    plt.xlabel('delta_mu')
    plt.ylabel('I')
    fig.colorbar(img1)

    #matplotlib.cm.register_cmap(name='twilight',cmap=twilight)


    ax=fig.add_subplot(3,2,5)
    img1=ax.imshow((np.angle(rho13[:,:])),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower',cmap='hsv')
    plt.title(' rho13 phase')
    plt.xlabel('delta_mu')
    plt.ylabel('I')
    fig.colorbar(img1)
    ax=fig.add_subplot(3,2,6)
    img1=ax.imshow((np.angle(boutvals[:,:])),extent=(np.min(freqmu_vals),np.max(freqmu_vals),np.min(I_vals),np.max(I_vals)),aspect='auto',origin='lower',cmap='hsv')
    plt.title('bout phase')
    plt.xlabel('delta_mu')
    plt.ylabel('I')
    fig.colorbar(img1)

    fig.suptitle('P_mu = ' + str(P_mu))
plt.show()
